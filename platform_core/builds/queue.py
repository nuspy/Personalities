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
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

#: La coda dei job da fare.
CODA = "builds:in_coda"

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
        self._supporto.lpush(CODA, job.to_json())
        logger.info("Job accodato: %s (%s)", job.build_id, job.kind)

    def prendi(self, worker_id: str, *, attesa: int = ATTESA_SECONDI) -> Optional[JobBuild]:
        """Prende un job, spostandolo fra le prese in carico.

        Restituisce `None` allo scadere dell'attesa: non è un errore, è la
        condizione normale di una coda vuota — e restituirla permette al
        worker di controllare se deve fermarsi.
        """
        grezzo = self._supporto.brpoplpush(
            CODA, self._in_carico(worker_id), timeout=attesa,
        )
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

    def in_attesa(self) -> int:
        return int(self._supporto.llen(CODA))

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
    """Coda di prova, con la stessa semantica di quella su Redis."""

    def __init__(self) -> None:
        self._coda: List[str] = []
        self._prese: Dict[str, List[str]] = {}

    def lpush(self, name: str, *values: Any) -> int:
        self._coda[0:0] = [str(v) for v in values]
        return len(self._coda)

    def brpoplpush(self, src: str, dst: str, timeout: int = 0) -> Any:
        if not self._coda:
            return None
        valore = self._coda.pop()
        self._prese.setdefault(dst, []).insert(0, valore)
        return valore

    def lrem(self, name: str, count: int, value: Any) -> int:
        lista = self._prese.get(name, [])
        testo = str(value)
        if testo in lista:
            lista.remove(testo)
            return 1
        return 0

    def llen(self, name: str) -> int:
        return len(self._coda)

    def lrange(self, name: str, start: int, end: int) -> List[Any]:
        return list(self._prese.get(name, []))
