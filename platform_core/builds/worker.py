"""Il consumatore delle realizzazioni.

Gira sul nodo con l'acceleratore e prende un job per volta. **Uno per volta
non è una limitazione da superare**: una GPU non si divide fra due
addestramenti, e due job concorrenti sullo stesso acceleratore finiscono
entrambi fuori memoria — più lentamente che se fossero stati in fila.

All'avvio il worker guarda cosa aveva preso in carico prima di morire. Quei
job non sono persi, ma nemmeno ripartibili alla cieca: un addestramento
interrotto a metà ha lasciato file parziali, e riprenderlo come se nulla fosse
produrrebbe un artefatto che nessuno sa da dove venga. Si dichiarano falliti,
dicendo perché — chi guarda decide se rilanciarli.
"""
from __future__ import annotations

import asyncio
import logging
import signal
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from ..capabilities.probe import probe
from ..capabilities.registry import CapabilityRegistry, Feature
from ..domain.session import dispose_engine, get_session_factory
from .queue import CODA_SENZA_ACCELERATORE, TUTTE_LE_CODE, CodaBuild, JobBuild
from .repository import BuildRepository
from .runner import BuildFallita, EsitoBuild, Runner

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from ..knowledge.digest_runner import DigestioneCorpus
    from ..knowledge.ingestione import IngestioneCaricamenti

logger = logging.getLogger(__name__)

#: Ogni quanti avanzamenti scrivere sul database.
#:
#: Un addestramento ne emette centinaia — uno per passo — e una scrittura per
#: ciascuno riempirebbe la tabella e rallenterebbe il lavoro vero. Uno su
#: cinque basta a far muovere una barra, e l'ultimo si scrive sempre.
OGNI = 5

#: Come si costruisce la digestione, data una sessione.
#:
#: Iniettabile perché altrimenti il ramo che la esegue sarebbe verificabile
#: solo leggendone il sorgente: con il classificatore cablato dentro, provare
#: che gli avanzamenti arrivino davvero al database richiederebbe un modello
#: vero e un corpus vero.
FabbricaDigestione = Callable[["AsyncSession"], "DigestioneCorpus"]


def _digestione_predefinita(session: "AsyncSession") -> "DigestioneCorpus":
    """Il classificatore vero. Importato qui perché tira dentro il provider."""
    from ..knowledge.digest_runner import DigestioneCorpus
    from ..knowledge.digestion import Digestore
    from ..knowledge.embedding import OpenAICompatibleEmbedder
    from ..llm.openai_compatible import OpenAICompatibleProvider

    return DigestioneCorpus(
        session,
        Digestore(OpenAICompatibleProvider()),
        embedder=OpenAICompatibleEmbedder(),
    )


#: Come si costruisce l'ingestione, data una sessione. Iniettabile per la
#: stessa ragione della digestione: provarne il percorso senza un modello di
#: embedding vero e senza Whisper.
FabbricaIngestione = Callable[["AsyncSession"], "IngestioneCaricamenti"]


def _ingestione_predefinita(session: "AsyncSession") -> "IngestioneCaricamenti":
    from ..knowledge.embedding import OpenAICompatibleEmbedder
    from ..knowledge.ingestione import IngestioneCaricamenti

    return IngestioneCaricamenti(session, OpenAICompatibleEmbedder())


class WorkerBuild:
    def __init__(
        self,
        worker_id: str,
        *,
        coda: CodaBuild,
        registry: Optional[CapabilityRegistry] = None,
        runner: Optional[Runner] = None,
        digestione: Optional[FabbricaDigestione] = None,
        ingestione: Optional[FabbricaIngestione] = None,
        code: Sequence[str] = TUTTE_LE_CODE,
    ) -> None:
        self.worker_id = worker_id
        self._coda = coda
        self._registry = registry
        self._runner = runner or Runner()
        self._digestione = digestione or _digestione_predefinita
        self._ingestione = ingestione or _ingestione_predefinita
        #: Le code che questo worker legge: senza acceleratore, solo quella
        #: dei lavori che non ne chiedono uno.
        self._code = tuple(code)
        self._fermarsi = asyncio.Event()

    def ferma(self) -> None:
        """Chiede di fermarsi dopo il job in corso.

        Non interrompe il lavoro: un addestramento ucciso a metà lascia file
        parziali e una riga che dice «in corso» per sempre. Si finisce quello
        che si è cominciato, poi si esce.
        """
        self._fermarsi.set()

    async def gira(self) -> None:
        await self._recupera_orfani()

        logger.info("Worker %s in ascolto sulla coda", self.worker_id)
        while not self._fermarsi.is_set():
            # In un thread: `brpoplpush` è bloccante, e nel loop fermerebbe
            # tutto — compreso il battito che annuncia le capacità.
            job = await asyncio.get_running_loop().run_in_executor(
                None, lambda: self._coda.prendi(self.worker_id, code=self._code),
            )
            if job is None:
                continue

            try:
                await self._esegui(job)
            finally:
                self._coda.completa(self.worker_id, job)

        logger.info("Worker %s fermato", self.worker_id)

    async def _recupera_orfani(self) -> None:
        orfani = self._coda.in_carico(self.worker_id)
        if not orfani:
            return

        logger.warning(
            "%d job erano in carico da un'esecuzione precedente", len(orfani),
        )
        async with get_session_factory()() as session:
            repo = BuildRepository(session)
            for job in orfani:
                build = await repo.per_id(job.build_id)
                if build is not None and not build.conclusa:
                    await repo.fallita(
                        build,
                        "il worker si è interrotto mentre la eseguiva: "
                        "i file parziali non sono utilizzabili, rilanciala",
                    )
                self._coda.completa(self.worker_id, job)
            await session.commit()

    async def _esegui(self, job: JobBuild) -> None:
        async with get_session_factory()() as session:
            repo = BuildRepository(session)
            build = await repo.per_id(job.build_id)

            if build is None:
                logger.warning("Build %s sparita dal database", job.build_id)
                return

            if not await repo.prendi_in_carico(build, self.worker_id):
                await session.commit()
                return

            await session.commit()
            parametri = dict(build.params or {})
            tipo = build.kind
            build_id = build.id

        contatore = {"n": 0}

        async def annota(
            percentuale: int, messaggio: str, *, sempre: bool = False,
        ) -> None:
            # `sempre` per la digestione, che filtra già alla sorgente: là un
            # messaggio esce solo quando la percentuale cambia o quando una
            # passata finisce, e scartarne quattro su cinque farebbe sparire
            # proprio gli avvisi che escono una volta sola.
            contatore["n"] += 1
            if not sempre and contatore["n"] % OGNI and percentuale not in (100, -1):
                return
            async with get_session_factory()() as s:
                r = BuildRepository(s)
                b = await r.per_id(build_id)
                if b is not None:
                    await r.avanza(
                        b, max(percentuale, 0), messaggio,
                        livello="errore" if percentuale < 0 else "info",
                    )
                    await s.commit()

        loop = asyncio.get_running_loop()

        def avanzamento(percentuale: int, messaggio: str) -> None:
            # Dal thread di lavoro al loop: `call_soon_threadsafe` è l'unico
            # modo di programmare lavoro asincrono da un altro thread.
            loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(annota(percentuale, messaggio))
            )

        try:
            if tipo == "digestione":
                # La digestione non passa dal `Runner`: quello esegue codice
                # sincrono in un thread perché gli stadi della pipeline sono
                # sincroni, mentre qui è già tutto asincrono — e interroga il
                # database, che da un altro thread non si può.
                esito = await self._digerisci(build_id, parametri, annota)
            elif tipo == "ingestione":
                esito = await self._ingerisci(build_id, parametri, annota)
            else:
                esito = await self._runner.esegui(
                    build_id=build_id, kind=tipo, params=parametri,
                    avanzamento=avanzamento,
                )
        except BuildFallita as exc:
            logger.error("Build %s fallita: %s", build_id, exc)
            await self._chiudi(build_id, errore=str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Build %s: errore inatteso", build_id)
            await self._chiudi(build_id, errore=f"errore inatteso: {exc}")
            return

        await self._chiudi(build_id, esito=esito)
        logger.info("Build %s completata: %s", build_id, esito.artifact_path)

    async def _digerisci(self, build_id, parametri, annota):
        """Analizza un corpus. Nessun acceleratore richiesto.

        Non passa dal `Runner`: quello esegue codice sincrono in un thread
        perche' gli stadi della pipeline lo sono, mentre qui e' gia' tutto
        asincrono — e interroga il database, che da un altro thread non si
        puo'.
        """
        import uuid as _uuid

        from ..domain.knowledge_models import KnowledgeBase

        kb_id = parametri.get("kb_id")
        if not kb_id:
            raise BuildFallita("nessuna knowledge base indicata")

        # La digestione annuncia l'avanzamento da codice sincrono, e scriverlo
        # con un `ensure_future` per volta lascerebbe righe fuori ordine e
        # eccezioni che nessuno vede. Una coda e un solo lettore danno
        # entrambe le cose: ordine, e un errore che arriva dove si guarda.
        avvisi: asyncio.Queue = asyncio.Queue()

        async def scrivi_avvisi() -> None:
            while True:
                avviso = await avvisi.get()
                if avviso is None:
                    return
                percentuale, messaggio = avviso
                try:
                    await annota(percentuale, messaggio, sempre=True)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "Build %s: avanzamento non scritto", build_id,
                    )

        def avanzamento(fatti: int, totale: int, messaggio: str) -> None:
            avvisi.put_nowait(
                (int(fatti * 100 / totale) if totale else 0, messaggio)
            )

        scrittore = asyncio.ensure_future(scrivi_avvisi())
        try:
            async with get_session_factory()() as session:
                kb = await session.get(KnowledgeBase, _uuid.UUID(kb_id))
                if kb is None:
                    raise BuildFallita(f"knowledge base {kb_id} non trovata")

                esito = await self._digestione(session).digerisci(
                    kb,
                    chi=parametri.get("chi", ""),
                    rifai=bool(parametri.get("rifai")),
                    avanzamento=avanzamento,
                )
                await session.commit()
        finally:
            # Anche quando la digestione fallisce: le righe già in coda
            # dicono fin dove era arrivata, ed è l'unica cosa che resta da
            # leggere quando un lavoro di tre ore muore al secondo.
            await avvisi.put(None)
            await scrittore

        return EsitoBuild(artifact_path=None, meta=esito.to_dict())

    async def _ingerisci(self, build_id, parametri, annota):
        """Legge i file caricati dalla console e li aggiunge al corpus.

        Come la digestione, fuori dal `Runner`: scrive sul database a ogni
        documento, e lo fa dal loop. I file si tolgono dall'archivio a lavoro
        finito, riuscito o no: il testo vive nei passaggi, e una copia del
        documento originale tenuta «per sicurezza» è una copia di cui nessuno
        risponde — spesso di un'opera coperta da diritti.
        """
        import uuid as _uuid

        from ..domain.knowledge_models import KnowledgeBase

        kb_id = parametri.get("kb_id")
        file = parametri.get("file") or []
        if not kb_id or not file:
            raise BuildFallita("nessun file da ingerire")

        try:
            async with get_session_factory()() as session:
                kb = await session.get(KnowledgeBase, _uuid.UUID(kb_id))
                if kb is None:
                    raise BuildFallita(f"knowledge base {kb_id} non trovata")

                ingestione = self._ingestione(session)

                async def avanzamento(fatti: int, totale: int, messaggio: str) -> None:
                    await annota(
                        int(fatti * 100 / totale) if totale else 0,
                        messaggio, sempre=True,
                    )

                esito = await ingestione.ingerisci(
                    kb, file, lingua=parametri.get("lingua"),
                    avanzamento=avanzamento,
                )
                await session.commit()
        finally:
            from ..knowledge.archivio import archivio_caricamenti

            archivio = archivio_caricamenti()
            for voce in file:
                archivio.elimina(voce.get("riferimento", ""))

        if esito.documenti == 0 and esito.falliti:
            # Nessun file è entrato: è un fallimento, non un successo vuoto.
            # Il motivo del primo basta a capire, gli altri stanno nell'esito.
            nome, motivo = esito.falliti[0]["nome"], esito.falliti[0]["motivo"]
            raise BuildFallita(
                f"nessun documento aggiunto: {nome} — {motivo}"
                + (f" (e altri {len(esito.falliti) - 1})" if len(esito.falliti) > 1 else "")
            )
        return EsitoBuild(artifact_path=None, meta=esito.to_dict())

    async def _chiudi(self, build_id, *, esito=None, errore: Optional[str] = None) -> None:
        async with get_session_factory()() as session:
            repo = BuildRepository(session)
            build = await repo.per_id(build_id)
            if build is None:
                return
            if errore is not None:
                await repo.fallita(build, errore)
            else:
                await repo.conclusa(
                    build,
                    artifact_path=esito.artifact_path if esito else None,
                    artifact_meta=esito.meta if esito else None,
                )
            await repo.potatura_eventi(build)
            await session.commit()


async def avvia(worker_id: str, *, solo_digestione: bool = False) -> int:
    """Avvia il consumatore.

    Con un acceleratore prende qualunque lavoro; senza, **solo digestioni e
    ingestioni** — interrogano un modello o leggono file, e non addestrano
    nulla. Senza acceleratore il worker legge soltanto la loro coda: gli
    addestramenti non li vede nemmeno.

    Il controllo è qui e non solo nell'API: un worker senza acceleratore che
    prendesse addestramenti li farebbe fallire uno dopo l'altro, svuotando la
    coda senza produrre nulla — il modo peggiore di non funzionare, perché i
    job spariscono e sembrano lavorati.
    """
    capacita = probe()
    if not capacita.can_train_lora and not solo_digestione:
        logger.error(
            "Questo nodo non può addestrare (%s). Con --solo-digestione "
            "prenderebbe comunque le analisi dei corpora, che non richiedono "
            "un acceleratore.",
            capacita.reasons.get("can_train_lora", "motivo non disponibile"),
        )
        return 1

    # Un client proprio, non quello condiviso con il registro delle
    # capacita': i comandi bloccanti vogliono un timeout di lettura che
    # sopravviva all'attesa.
    coda = CodaBuild.per_consumatore()
    worker = WorkerBuild(
        worker_id, coda=coda,
        code=(CODA_SENZA_ACCELERATORE,) if solo_digestione else TUTTE_LE_CODE,
    )

    for segnale in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
        try:
            asyncio.get_running_loop().add_signal_handler(segnale, worker.ferma)
        except NotImplementedError:
            # Windows non li supporta sul loop: ci pensa il gestore sincrono
            # installato da chi avvia il processo.
            signal.signal(segnale, lambda *_: worker.ferma())

    if solo_digestione:
        logger.info(
            "Consumatore avviato su %s: solo digestioni e ingestioni", worker_id,
        )
    else:
        logger.info(
            "Consumatore avviato su %s (%s, %d MB)",
            worker_id, capacita.gpu_name, capacita.vram_total_mb,
        )

    try:
        await worker.gira()
    finally:
        await dispose_engine()
    return 0
