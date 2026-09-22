"""Accensione a richiesta di un motore locale.

Un modello che gira su una GPU occupa quella GPU anche mentre nessuno lo
interroga. Tenerlo acceso ventiquattr'ore per due digestioni al giorno
significa pagarlo — in corrente, o in memoria video sottratta a chi altro usa
quella macchina. Tenerlo spento significa che il primo lavoro fallisce.

La terza via è accenderlo quando serve e spegnerlo quando non serve più, ed è
quello che fa questo modulo. Tre scelte lo definiscono:

**Il comando arriva dalla configurazione, mai dai dati.** `avvio` e `arresto`
sono righe di comando lette dall'ambiente — in Kubernetes da un Secret — e
nessun endpoint, nessuna tabella e nessuna console può cambiarle. Un comando
che si potesse modificare dall'interfaccia sarebbe esecuzione di codice
arbitrario con i diritti del worker, indipendentemente da quanto bene sia
protetto l'accesso a quell'interfaccia. Si esegue senza shell
(`create_subprocess_exec` su `shlex.split`), così nemmeno un valore di
configurazione sbagliato può concatenare un secondo comando.

**Chi accende è il worker, non l'API.** Il processo che risponde alle
richieste pubbliche non deve poter eseguire comandi né raggiungere in SSH la
macchina dei modelli: chiede, e basta. La richiesta passa per il supporto
condiviso (`CHIAVE_RICHIESTA`), il worker la raccoglie e agisce.

**Lo stato ha una scadenza.** Si pubblica con TTL: se il worker che lo
manteneva muore, dopo poco la console non dice più «acceso» — dice che nessuno
lo sta gestendo. L'assenza è l'informazione, come per il battito dei worker.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Awaitable, Callable, Deque, Optional

logger = logging.getLogger(__name__)

#: Dove il worker pubblica lo stato e dove l'API deposita le richieste.
CHIAVE_STATO = "motore:locale"
CHIAVE_RICHIESTA = "motore:richiesta"

#: Quanto dura lo stato pubblicato. Più lungo dell'intervallo di sorveglianza,
#: o uno stato valido scadrebbe fra un aggiornamento e l'altro.
TTL_STATO = 120
INTERVALLO_SORVEGLIANZA = 15.0

#: Ogni quanto si richiede la salute mentre si attende l'accensione.
PASSO_ATTESA = 2.0

#: Quante righe dell'output del comando conservare per spiegare un fallimento.
#: Le ultime, non le prime: il motivo di un errore sta in fondo.
RIGHE_DIAGNOSTICA = 20


#: I compiti che svuotano l'output dei comandi avviati e ancora vivi.
_LETTORI: set = set()


class MotoreNonDisponibile(RuntimeError):
    """Il motore locale non si è acceso, o non è raggiungibile."""


@dataclass(frozen=True)
class StatoMotore:
    """Cosa sta facendo il motore, nella forma che la console legge."""

    #: `spento`, `in_accensione`, `acceso`, `in_spegnimento`, `guasto`,
    #: `non_gestito` (nessun comando configurato).
    stato: str = "non_gestito"
    #: Da quando dura questo stato, in secondi dall'epoca.
    da: float = 0.0
    #: Ultima volta che un lavoro l'ha usato. Zero: mai, da quando gira il
    #: worker.
    ultimo_uso: float = 0.0
    #: Quanti lavori lo stanno usando adesso. Finché è maggiore di zero lo
    #: spegnimento non avviene, nemmeno se richiesto a mano.
    in_corso: int = 0
    #: Dopo quanta inattività si spegne da solo.
    inattivita_s: float = 0.0
    #: Perché è in questo stato, quando la ragione non è ovvia. Il testo di un
    #: errore di accensione finisce qui: è l'unica cosa che la console possa
    #: mostrare a chi non ha accesso ai log del worker.
    motivo: str = ""
    #: Chi lo sta gestendo, per distinguere due worker.
    worker_id: str = ""

    def to_dict(self) -> dict:
        return {
            "stato": self.stato,
            "da": self.da,
            "ultimo_uso": self.ultimo_uso,
            "in_corso": self.in_corso,
            "inattivita_s": self.inattivita_s,
            "motivo": self.motivo,
            "worker_id": self.worker_id,
        }

    @staticmethod
    def from_json(raw: str | bytes) -> "StatoMotore":
        dati = json.loads(raw)
        return StatoMotore(
            stato=dati.get("stato", "non_gestito"),
            da=float(dati.get("da", 0)),
            ultimo_uso=float(dati.get("ultimo_uso", 0)),
            in_corso=int(dati.get("in_corso", 0)),
            inattivita_s=float(dati.get("inattivita_s", 0)),
            motivo=dati.get("motivo", ""),
            worker_id=dati.get("worker_id", ""),
        )


#: Come si esegue un comando: restituisce `(codice_uscita_o_None, output)`.
#: `None` significa che il processo era ancora vivo quando si è smesso di
#: aspettarlo — il caso normale per un avvio che resta in primo piano.
Esecutore = Callable[[str, float], Awaitable[tuple[Optional[int], str]]]
#: Come si chiede se il motore risponde.
Sonda = Callable[[], Awaitable[bool]]


def _argomenti(comando: str) -> list[str]:
    """Spezza la riga di comando secondo le convenzioni del sistema.

    Non `shlex.split(comando)` e basta: in modo POSIX la barra rovesciata è un
    carattere di fuga, e `C:\\Projects\\bonsai\\start.ps1` diventerebbe
    `C:Projectsbonsaistart.ps1` — un eseguibile che non esiste, e un errore
    che non somiglia affatto alla sua causa. Su Windows si spezza in modo non
    POSIX e si tolgono le virgolette dai pezzi, che è ciò che fa la shell là.

    In nessuno dei due casi si passa da una shell: un `&&` nel comando resta
    testo, non diventa un secondo programma.
    """
    if os.name == "nt":
        return [pezzo.strip('"') for pezzo in shlex.split(comando, posix=False) if pezzo]
    return shlex.split(comando)


async def _esegui(comando: str, attesa: float) -> tuple[Optional[int], str]:
    """Esegue una riga di comando senza shell, senza aspettarne la fine.

    Non si aspetta la fine perché i due casi d'uso hanno durate opposte: un
    `ssh` a comando forzato ritorna in un secondo, uno script che avvia il
    server in primo piano non ritorna mai. Chi chiama non aspetta il processo:
    aspetta che il motore risponda.

    L'output si legge comunque, in un compito a parte e tenendone solo le
    ultime righe. Non leggerlo sarebbe peggio che perderlo: una pipe piena
    blocca il processo che ci scrive, e un server avviato così si fermerebbe
    dopo qualche migliaio di righe di log.
    """
    argv = _argomenti(comando)
    if not argv:
        raise MotoreNonDisponibile("comando vuoto")

    try:
        processo = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
        )
    except NotImplementedError:
        # Su Windows i sottoprocessi asincroni vogliono il ciclo Proactor, e
        # il backend sceglie il Selector perché psycopg 3 non sa usare
        # l'altro (vedi `platform_core/__init__.py`). Non è un caso di
        # nicchia: è esattamente lo sviluppo su questa macchina, dove il
        # modello e il worker stanno insieme. Si ripiega su un sottoprocesso
        # sincrono in un thread, che non dipende dal ciclo.
        return await asyncio.get_running_loop().run_in_executor(
            None, _esegui_in_thread, argv, attesa,
        )

    righe: Deque[str] = deque(maxlen=RIGHE_DIAGNOSTICA)

    async def drena() -> None:
        assert processo.stdout is not None
        while True:
            riga = await processo.stdout.readline()
            if not riga:
                return
            righe.append(riga.decode("utf-8", "replace").rstrip())

    lettore = asyncio.ensure_future(drena())
    # Un riferimento forte finché vive: un compito conosciuto solo dal loop
    # può essere raccolto dal garbage collector a metà, e la pipe tornerebbe
    # a riempirsi in silenzio.
    _LETTORI.add(lettore)
    lettore.add_done_callback(_LETTORI.discard)
    try:
        await asyncio.wait_for(processo.wait(), timeout=attesa)
    except asyncio.TimeoutError:
        # Vivo: è il caso dell'avvio in primo piano. Il lettore resta a
        # svuotare la pipe finché il processo campa.
        return None, "\n".join(righe)
    finally:
        if processo.returncode is not None:
            await lettore

    return processo.returncode, "\n".join(righe)


def _esegui_in_thread(argv: list[str], attesa: float) -> tuple[Optional[int], str]:
    """Lo stesso di `_esegui`, con `subprocess` invece di asyncio.

    Il thread resta occupato al massimo `attesa` secondi: se il comando è un
    avvio che non ritorna, si smette di aspettarlo e si lascia solo il
    lettore — un thread di servizio, marcato come demone, che muore con il
    processo.
    """
    import subprocess
    import threading

    processo = subprocess.Popen(  # noqa: S603 — argv, mai una shell
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )

    righe: Deque[str] = deque(maxlen=RIGHE_DIAGNOSTICA)

    def drena() -> None:
        assert processo.stdout is not None
        for riga in processo.stdout:
            righe.append(riga.decode("utf-8", "replace").rstrip())

    lettore = threading.Thread(target=drena, daemon=True, name="motore-output")
    lettore.start()

    try:
        processo.wait(timeout=attesa)
    except subprocess.TimeoutExpired:
        return None, "\n".join(righe)

    lettore.join(timeout=2.0)
    return processo.returncode, "\n".join(righe)


def _sonda_http(url: str, timeout: float = 5.0) -> Sonda:
    """Chiede al motore se è pronto. Qualunque risposta 2xx vale sì."""

    async def sonda() -> bool:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                risposta = await client.get(url)
            return risposta.is_success
        except Exception:  # noqa: BLE001 — irraggiungibile è «non pronto»
            return False

    return sonda


class MotoreLocale:
    """Accende, tiene acceso finché serve, spegne quando non serve più."""

    def __init__(
        self,
        *,
        avvio: str = "",
        arresto: str = "",
        salute_url: str = "",
        attesa_s: float = 180.0,
        inattivita_s: float = 900.0,
        store=None,
        worker_id: str = "",
        esecutore: Optional[Esecutore] = None,
        sonda: Optional[Sonda] = None,
        adesso: Callable[[], float] = time.time,
    ) -> None:
        self._avvio = avvio.strip()
        self._arresto = arresto.strip()
        self._attesa_s = attesa_s
        self._inattivita_s = inattivita_s
        self._store = store
        self._worker_id = worker_id
        self._esecutore = esecutore or _esegui
        self._sonda = sonda or (_sonda_http(salute_url) if salute_url else None)
        self._adesso = adesso

        self._in_corso = 0
        self._ultimo_uso = 0.0
        #: Se è stato questo processo ad accenderlo. Distingue «acceso» da
        #: «acceso da noi», e solo il secondo autorizza a spegnerlo uscendo:
        #: un motore che qualcun altro stava usando non si spegne perché noi
        #: abbiamo finito.
        self._acceso_da_noi = False
        self._stato = "non_gestito" if not self.gestito else "spento"
        self._da = adesso()
        self._motivo = "" if self.gestito else "nessun comando di avvio configurato"
        self._serratura = asyncio.Lock()

    # ------------------------------------------------------------- proprietà

    @property
    def gestito(self) -> bool:
        """Vero se questo processo sa accendere il motore.

        Senza comando di avvio l'oggetto resta utilizzabile e non fa nulla: è
        il caso dello sviluppo con il modello già acceso a mano, e di un
        worker che parla a un fornitore remoto.
        """
        return bool(self._avvio)

    def stato(self) -> StatoMotore:
        return StatoMotore(
            stato=self._stato,
            da=self._da,
            ultimo_uso=self._ultimo_uso,
            in_corso=self._in_corso,
            inattivita_s=self._inattivita_s if self.gestito else 0.0,
            motivo=self._motivo,
            worker_id=self._worker_id,
        )

    # ----------------------------------------------------------- accensione

    async def accendi(self, *, motivo: str = "") -> None:
        """Assicura che il motore risponda, accendendolo se occorre.

        Idempotente: se risponde già, non fa nulla. Due lavori che partono
        insieme non lanciano due avvii — la serratura li mette in fila, e il
        secondo trova il motore acceso dal primo.
        """
        if not self.gestito:
            return

        async with self._serratura:
            if self._sonda is not None and await self._sonda():
                self._cambia("acceso", motivo)
                return

            self._cambia("in_accensione", motivo)
            logger.info("Accendo il motore locale%s", f": {motivo}" if motivo else "")
            codice, output = await self._esecutore(self._avvio, self._attesa_s)

            if self._sonda is None:
                # Senza sonda l'unico segnale è il comando: uscita pulita, o
                # processo ancora vivo (avvio in primo piano).
                if codice not in (0, None):
                    raise MotoreNonDisponibile(self._guasto(
                        f"il comando di avvio è uscito con {codice}", output,
                    ))
                self._acceso_da_noi = True
                self._cambia("acceso", motivo)
                return

            scadenza = self._adesso() + self._attesa_s
            while self._adesso() < scadenza:
                if await self._sonda():
                    self._acceso_da_noi = True
                    self._cambia("acceso", motivo)
                    logger.info("Motore locale pronto")
                    return
                await asyncio.sleep(PASSO_ATTESA)

            dettaglio = (
                f"il comando di avvio è uscito con {codice}"
                if codice not in (0, None)
                else f"non ha risposto entro {self._attesa_s:.0f} s"
            )
            raise MotoreNonDisponibile(self._guasto(dettaglio, output))

    async def spegni(self, *, motivo: str = "", forza: bool = False) -> bool:
        """Spegne il motore. Restituisce falso se non si poteva.

        Non spegne mentre un lavoro lo sta usando: una digestione di tre ore
        interrotta al secondo perché è scaduto un timer perderebbe tre ore di
        analisi. `forza` serve a chi deve liberare la GPU comunque, e resta
        una decisione esplicita di chi la prende.
        """
        if not self.gestito or not self._arresto:
            return False

        async with self._serratura:
            if self._in_corso > 0 and not forza:
                logger.info(
                    "Spegnimento rimandato: %d lavori stanno usando il motore",
                    self._in_corso,
                )
                return False

            self._cambia("in_spegnimento", motivo)
            logger.info("Spengo il motore locale%s", f": {motivo}" if motivo else "")
            codice, output = await self._esecutore(self._arresto, 60.0)
            if codice not in (0, None):
                self._guasto(f"il comando di arresto è uscito con {codice}", output)
                return False

            self._acceso_da_noi = False
            self._cambia("spento", motivo)
            return True

    @property
    def acceso_da_noi(self) -> bool:
        """Vero se l'accensione in corso è opera di questo processo."""
        return self._acceso_da_noi

    @asynccontextmanager
    async def in_uso(self, *, motivo: str = ""):
        """Tiene il motore acceso per la durata del blocco.

        Il conteggio si alza **prima** di accendere e si abbassa alla fine:
        così un lavoro che parte mentre la sorveglianza sta valutando lo
        spegnimento non si ritrova il motore spento sotto i piedi.
        """
        self._in_corso += 1
        self._ultimo_uso = self._adesso()
        try:
            await self.accendi(motivo=motivo)
            yield self
        finally:
            self._in_corso -= 1
            self._ultimo_uso = self._adesso()

    # ---------------------------------------------------------- sorveglianza

    async def sorveglia(self, fermarsi: asyncio.Event) -> None:
        """Pubblica lo stato, raccoglie le richieste, spegne se inattivo.

        Un solo compito per le tre cose perché condividono il ritmo e l'ordine
        conta: si raccoglie una richiesta prima di valutare l'inattività, o un
        «accendi» premuto in console potrebbe essere annullato un istante dopo
        dal timer.
        """
        while not fermarsi.is_set():
            try:
                await self._raccogli_richiesta()
                await self._spegni_se_inattivo()
                self.pubblica()
            except Exception:  # noqa: BLE001
                # La sorveglianza non deve morire: è ciò che spegne la GPU.
                logger.exception("Sorveglianza del motore: giro fallito")
            try:
                await asyncio.wait_for(
                    fermarsi.wait(), timeout=INTERVALLO_SORVEGLIANZA,
                )
            except asyncio.TimeoutError:
                pass

    async def _raccogli_richiesta(self) -> None:
        if self._store is None:
            return
        grezza = self._store.get(CHIAVE_RICHIESTA)
        if not grezza:
            return
        self._store.delete(CHIAVE_RICHIESTA)
        richiesta = grezza.decode() if isinstance(grezza, bytes) else str(grezza)
        if richiesta == "accendi":
            await self.accendi(motivo="richiesta dalla console")
        elif richiesta == "spegni":
            await self.spegni(motivo="richiesta dalla console")
        else:
            logger.warning("Richiesta sconosciuta ignorata: %r", richiesta)

    async def _spegni_se_inattivo(self) -> None:
        if not self.gestito or self._stato != "acceso" or self._inattivita_s <= 0:
            return
        if self._in_corso > 0:
            return
        # Mai usato da quando questo worker gira: il motore era acceso prima,
        # e non è questo processo a decidere di spegnere ciò che non ha acceso.
        if self._ultimo_uso == 0.0:
            return
        if self._adesso() - self._ultimo_uso < self._inattivita_s:
            return
        await self.spegni(
            motivo=f"inattivo da {self._inattivita_s:.0f} s",
        )

    def pubblica(self) -> None:
        """Scrive lo stato dove la console lo legge, con scadenza."""
        if self._store is None:
            return
        self._store.set(
            CHIAVE_STATO, json.dumps(self.stato().to_dict()), ex=TTL_STATO,
        )

    # ------------------------------------------------------------- interno

    def _cambia(self, stato: str, motivo: str = "") -> None:
        if stato != self._stato:
            self._da = self._adesso()
        self._stato = stato
        self._motivo = motivo
        self.pubblica()

    def _guasto(self, dettaglio: str, output: str) -> str:
        """Registra il guasto e restituisce il messaggio da riportare.

        Registra e non solleva: l'accensione solleva — chi chiede il motore
        deve saperlo subito — mentre lo spegnimento risponde falso, perché una
        GPU che non si è liberata non è una ragione per far fallire il lavoro
        che aveva appena finito.
        """
        coda = output.strip().splitlines()[-3:]
        messaggio = dettaglio + (f" — {' / '.join(coda)}" if coda else "")
        self._cambia("guasto", messaggio)
        logger.error("Motore locale: %s", messaggio)
        return messaggio


def motore_da_impostazioni(store=None, worker_id: str = "") -> MotoreLocale:
    """Costruisce il motore leggendo la configurazione del servizio."""
    from ..settings import get_settings

    s = get_settings()
    return MotoreLocale(
        avvio=s.local_engine_start,
        arresto=s.local_engine_stop,
        salute_url=s.local_engine_health_url,
        attesa_s=s.local_engine_wait_s,
        inattivita_s=s.local_engine_idle_s,
        store=store,
        worker_id=worker_id,
    )


def chiedi_accensione(store, richiesta: str) -> None:
    """Deposita una richiesta per il worker. Usata dall'API.

    L'API non esegue comandi e non raggiunge la macchina dei modelli: scrive
    una parola — `accendi` o `spegni` — e il worker la raccoglie. La scadenza
    breve è deliberata: una richiesta che nessun worker ha raccolto entro due
    minuti è una richiesta caduta nel vuoto, e deve sparire invece di essere
    eseguita mezz'ora dopo da un worker che riparte.
    """
    if richiesta not in ("accendi", "spegni"):
        raise ValueError(f"richiesta non ammessa: {richiesta!r}")
    store.set(CHIAVE_RICHIESTA, richiesta, ex=120)


def stato_pubblicato(store) -> StatoMotore:
    """Lo stato che il worker ha pubblicato, o «nessuno lo gestisce»."""
    grezzo = store.get(CHIAVE_STATO)
    if not grezzo:
        return StatoMotore(
            stato="non_gestito",
            motivo=(
                "nessun worker sta gestendo l'accensione del motore locale: "
                "o non è configurata, o il worker che la gestiva non risponde"
            ),
        )
    try:
        return StatoMotore.from_json(grezzo)
    except (ValueError, KeyError):
        return StatoMotore(stato="non_gestito", motivo="stato illeggibile")
