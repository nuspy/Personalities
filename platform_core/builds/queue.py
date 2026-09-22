"""La coda delle realizzazioni.

**Perché una coda su Redis e non Celery.** Celery porta un broker, un
protocollo, un pool di processi e un modello di ritentativo — tutto utile
quando i job sono molti, brevi e indipendenti. Qui sono pochi, lunghissimi
(un addestramento dura minuti o ore) e ne gira **uno per volta per
acceleratore**: una GPU non si divide fra due training. Di tutta quella
macchina servirebbe la lista, e la lista Redis ce l'ha già.

La scelta va rivista se un giorno i job diventassero tanti e corti — ma quel
giorno si riconosce da sé, e finché non arriva un sistema più piccolo è più
facile da capire quando qualcosa non va.

**Sul recapito.** `BRPOPLPUSH` sposta il job dalla coda a una lista di lavori
presi in carico, in una sola operazione atomica. Se il worker muore mentre
elabora, il job resta lì invece di svanire: lo si ritrova, e si decide cosa
farne. Una `BRPOP` semplice lo toglierebbe dalla coda senza lasciare traccia,
e un worker ucciso a metà porterebbe con sé il lavoro senza che nessuno sappia
che era in corso.

**Due code, non una.** Gli addestramenti vogliono un acceleratore; digestione
e ingestione no — interrogano un modello o leggono file. Con una coda sola un
worker senza GPU avrebbe preso anche gli addestramenti, facendoli fallire uno
dopo l'altro: la coda si svuota, i job sembrano lavorati, e nessuno ha
addestrato nulla. Ciascun job va nella coda del suo tipo, e ciascun worker
legge solo le code di ciò che sa fare: quello con la GPU entrambe, quello
senza solo la sua.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)

#: La coda dei job che richiedono un acceleratore. Il nome è quello di prima
#: della separazione: i job già accodati allora erano tutti di questo tipo.
CODA = "builds:in_coda"

#: La coda dei job che un nodo senza acceleratore sa eseguire.
CODA_SENZA_ACCELERATORE = "builds:in_coda:cpu"

#: I tipi che non chiedono una GPU.
TIPI_SENZA_ACCELERATORE = frozenset({"digestione", "ingestione"})

#: Tutte le code, nell'ordine in cui le legge un worker con acceleratore:
#: prima il lavoro che solo lui sa fare.
TUTTE_LE_CODE = (CODA, CODA_SENZA_ACCELERATORE)


def coda_per(kind: str) -> str:
    """La coda in cui va un job di questo tipo."""
    return CODA_SENZA_ACCELERATORE if kind in TIPI_SENZA_ACCELERATORE else CODA

#: I job presi in carico, per worker: `builds:in_carico:<worker_id>`.
PREFISSO_IN_CARICO = "builds:in_carico:"

#: Quanto aspettare un job prima di tornare al chiamante. Non è un timeout del
#: lavoro: è la frequenza con cui il worker riprende il controllo per
#: verificare se gli è stato chiesto di fermarsi.
ATTESA_SECONDI = 5


@dataclass(frozen=True)
class JobBuild:
    """Il minimo che serve a un worker per cominciare.

    Solo identificativi: i parametri stanno sulla riga `builds`, e rileggerli
    dal database garantisce che il worker lavori sugli stessi valori che la
    console mostra. Duplicarli qui significherebbe due verità, e la seconda
    sbagliata il giorno in cui qualcuno modifica la build prima che parta.
    """

    build_id: uuid.UUID
    kind: str
    personality_id: uuid.UUID

    def to_json(self) -> str:
        return json.dumps({
            "build_id": str(self.build_id),
            "kind": self.kind,
            "personality_id": str(self.personality_id),
        })

    @classmethod
    def from_json(cls, grezzo: str) -> "JobBuild":
        d = json.loads(grezzo)
        return cls(
            build_id=uuid.UUID(d["build_id"]),
            kind=d["kind"],
            personality_id=uuid.UUID(d["personality_id"]),
        )


class SupportoCoda(Protocol):
    """Il minimo che serve alla coda. Redis lo soddisfa."""

    def lpush(self, name: str, *values: Any) -> int: ...
    def brpoplpush(self, src: str, dst: str, timeout: int = 0) -> Any: ...
    def lrem(self, name: str, count: int, value: Any) -> int: ...
    def llen(self, name: str) -> int: ...
    def lrange(self, name: str, start: int, end: int) -> List[Any]: ...


class CodaBuild:
    def __init__(self, supporto: SupportoCoda) -> None:
        self._supporto = supporto

    @classmethod
    def per_consumatore(cls, url: Optional[str] = None) -> "CodaBuild":
        """Una coda con un client adatto ai comandi bloccanti.

        `brpoplpush` tiene il socket fermo finché non arriva un job o scade
        l'attesa, e un client con il timeout di lettura predefinito lo
        interpreta come una connessione morta: il worker cade con
        «Timeout reading from socket» al primo giro a vuoto — cioè quasi
        subito, perché la coda è vuota quasi sempre.

        Il margine sopra l'attesa serve al viaggio di andata e ritorno: senza,
        i due timeout scadrebbero insieme e la gara la vincerebbe a volte uno,
        a volte l'altro.
        """
        import redis

        from ..settings import get_settings

        client = redis.Redis.from_url(
            url or get_settings().redis_url,
            decode_responses=True,
            socket_timeout=ATTESA_SECONDI + 10,
        )
        return cls(client)

    def accoda(self, job: JobBuild) -> None:
        coda = coda_per(job.kind)
        self._supporto.lpush(coda, job.to_json())
        logger.info("Job accodato in %s: %s (%s)", coda, job.build_id, job.kind)

    def prendi(
        self,
        worker_id: str,
        *,
        attesa: int = ATTESA_SECONDI,
        code: Sequence[str] = TUTTE_LE_CODE,
    ) -> Optional[JobBuild]:
        """Prende un job da una delle code, spostandolo fra le prese in carico.

        Restituisce `None` allo scadere dell'attesa: non è un errore, è la
        condizione normale di una coda vuota — e restituirla permette al
        worker di controllare se deve fermarsi.

        Le code si guardano una dopo l'altra, dividendo l'attesa: `BRPOPLPUSH`
        ne legge una sola, e l'alternativa con più chiavi (`BLMPOP`) non sposta
        il job fra le prese in carico — che è proprio ciò che lo salva quando
        il worker muore.
        """
        grezzo = None
        porzione = max(1, attesa // max(1, len(code)))
        for coda in code:
            grezzo = self._supporto.brpoplpush(
                coda, self._in_carico(worker_id), timeout=porzione,
            )
            if grezzo is not None:
                break
        if grezzo is None:
            return None

        testo = grezzo.decode() if isinstance(grezzo, bytes) else grezzo
        try:
            return JobBuild.from_json(testo)
        except (ValueError, KeyError) as exc:
            # Un messaggio malformato non deve bloccare la coda per sempre: si
            # toglie dalle prese in carico e si prosegue, perché lasciarlo lì
            # significherebbe ritrovarselo a ogni ispezione senza che nessuno
            # possa farci nulla.
            logger.error("Messaggio illeggibile, scartato: %s — %s", testo[:120], exc)
            self._supporto.lrem(self._in_carico(worker_id), 1, grezzo)
            return None

    def completa(self, worker_id: str, job: JobBuild) -> None:
        """Toglie il job dalle prese in carico. Da chiamare a lavoro finito."""
        self._supporto.lrem(self._in_carico(worker_id), 1, job.to_json())

    def in_attesa(self, *, code: Sequence[str] = TUTTE_LE_CODE) -> int:
        return sum(int(self._supporto.llen(c)) for c in code)

    def in_carico(self, worker_id: str) -> List[JobBuild]:
        """I job che questo worker aveva preso.

        Non vuoto all'avvio significa che il worker precedente è morto mentre
        lavorava: quei job vanno riportati in coda o dichiarati falliti, e
        l'unica cosa da non fare è ignorarli.
        """
        grezzi = self._supporto.lrange(self._in_carico(worker_id), 0, -1)
        risultato: List[JobBuild] = []
        for g in grezzi:
            testo = g.decode() if isinstance(g, bytes) else g
            try:
                risultato.append(JobBuild.from_json(testo))
            except (ValueError, KeyError):
                continue
        return risultato

    @staticmethod
    def _in_carico(worker_id: str) -> str:
        return f"{PREFISSO_IN_CARICO}{worker_id}"


class CodaInMemoria:
    """Coda di prova, con la stessa semantica di quella su Redis.

    Una lista per nome, come Redis: con una lista sola per tutte le chiavi le
    code separate sarebbero indistinguibili, e la prova che un worker senza
    acceleratore non prende gli addestramenti passerebbe per caso.
    """

    def __init__(self) -> None:
        self._liste: Dict[str, List[str]] = {}

    def lpush(self, name: str, *values: Any) -> int:
        lista = self._liste.setdefault(name, [])
        lista[0:0] = [str(v) for v in values]
        return len(lista)

    def brpoplpush(self, src: str, dst: str, timeout: int = 0) -> Any:
        sorgente = self._liste.get(src, [])
        if not sorgente:
            return None
        valore = sorgente.pop()
        self._liste.setdefault(dst, []).insert(0, valore)
        return valore

    def lrem(self, name: str, count: int, value: Any) -> int:
        lista = self._liste.get(name, [])
        testo = str(value)
        if testo in lista:
            lista.remove(testo)
            return 1
        return 0

    def llen(self, name: str) -> int:
        return len(self._liste.get(name, []))

    def lrange(self, name: str, start: int, end: int) -> List[Any]:
        return list(self._liste.get(name, []))
