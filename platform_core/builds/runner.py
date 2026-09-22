"""Esecuzione di una realizzazione.

Il ponte fra un job accodato e la pipeline che fa il lavoro vero. Tre cose
meritano una spiegazione.

**Gli stadi girano in un thread separato.** Sono codice sincrono che occupa la
CPU e la GPU per minuti: eseguirli nel loop di eventi lo bloccherebbe, e il
worker smetterebbe di rispondere — compreso a chi gli chiede di fermarsi.

**L'avanzamento attraversa il confine.** Lo stadio emette da un thread, il
database si scrive dal loop: fra i due c'è una coda, perché chiamare codice
asincrono da un thread qualunque non è possibile e provarci produce errori che
compaiono solo sotto carico.

**La pipeline si importa qui e non in cima al modulo.** Trascina torch, e
torch pesa gigabyte: un'API che non addestra nulla non deve pagarlo
all'avvio. L'import tardivo è ciò che permette allo stesso pacchetto di
girare su un nodo senza acceleratore.
"""
from __future__ import annotations

import asyncio
import logging
import pathlib
import queue
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from ..paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

#: Dove finiscono gli artefatti. Un percorso sul nodo che li produce: con più
#: worker, un artefatto è raggiungibile solo da chi l'ha scritto finché non
#: viene caricato altrove.
CARTELLA_BUILD = pathlib.Path(
    __import__("os").getenv("PERSONA_BUILD_DIR", str(PROJECT_ROOT / "builds"))
)


class BuildFallita(Exception):
    """La realizzazione non è andata a buon fine."""


@dataclass
class EsitoBuild:
    artifact_path: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


#: Come un avanzamento arriva da qui a chi guarda: (percentuale, messaggio).
Avanzamento = Callable[[int, str], None]


class Runner:
    """Esegue una build, riportando l'avanzamento.

    Non sa nulla di database né di code: riceve i parametri, fa il lavoro,
    riporta. È ciò che permette di provarlo senza infrastruttura.
    """

    def __init__(self, cartella: Optional[pathlib.Path] = None) -> None:
        self._cartella = cartella or CARTELLA_BUILD

    async def esegui(
        self,
        *,
        build_id: uuid.UUID,
        kind: str,
        params: Dict[str, Any],
        avanzamento: Avanzamento,
    ) -> EsitoBuild:
        destinazione = self._cartella / str(build_id)
        destinazione.mkdir(parents=True, exist_ok=True)

        if kind == "lora":
            return await self._in_thread(
                self._addestra, params, destinazione, avanzamento, modo="lora",
            )
        if kind == "finetune":
            return await self._in_thread(
                self._addestra, params, destinazione, avanzamento, modo="finetune",
            )
        if kind == "gguf":
            return await self._in_thread(
                self._converti, params, destinazione, avanzamento,
            )

        raise BuildFallita(f"tipo di realizzazione sconosciuto: {kind}")

    async def _in_thread(
        self,
        funzione: Callable[..., EsitoBuild],
        *args: Any,
        **kwargs: Any,
    ) -> EsitoBuild:
        """Esegue il lavoro sincrono fuori dal loop, inoltrando l'avanzamento.

        L'avanzamento passa da una coda: lo stadio emette dal thread di lavoro
        e il loop svuota la coda, perché programmare un `await` da un thread
        qualunque non è possibile.
        """
        annunci: "queue.Queue[Optional[Tuple[int, str]]]" = queue.Queue()
        avanzamento = args[2] if len(args) > 2 else kwargs.get("avanzamento")
        args = (*args[:2], lambda p, m: annunci.put((p, m)), *args[3:])

        risultato: Dict[str, Any] = {}

        def lavora() -> None:
            try:
                risultato["esito"] = funzione(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - riportata al chiamante
                risultato["errore"] = exc
            finally:
                # La sentinella sveglia il loop: senza, resterebbe in attesa
                # di un avanzamento che non arriverà mai.
                annunci.put(None)

        thread = threading.Thread(target=lavora, name="build", daemon=True)
        thread.start()

        loop = asyncio.get_running_loop()
        while True:
            annuncio = await loop.run_in_executor(None, annunci.get)
            if annuncio is None:
                break
            if avanzamento is not None:
                percentuale, messaggio = annuncio
                avanzamento(percentuale, messaggio)

        thread.join(timeout=5)

        if (errore := risultato.get("errore")) is not None:
            raise BuildFallita(str(errore)) from errore
        return risultato.get("esito") or EsitoBuild()

    # -- lavoro vero -------------------------------------------------------

    def _addestra(
        self,
        params: Dict[str, Any],
        destinazione: pathlib.Path,
        avanzamento: Avanzamento,
        *,
        modo: str,
    ) -> EsitoBuild:
        """Addestra un adapter o fa un fine-tuning completo.

        L'import sta qui dentro e non in cima: `training_stage` trascina torch,
        e un'API che non addestra nulla non deve pagare gigabyte all'avvio.
        """
        from historical_persona_pipeline.pipeline.stage4_training.training_stage import (
            TrainingStage,
        )

        dataset = params.get("dataset_path")
        if not dataset or not pathlib.Path(dataset).exists():
            raise BuildFallita(
                f"dataset non trovato: {dataset or '(non indicato)'}. "
                f"Una build parte da conversazioni già generate."
            )

        from historical_persona_pipeline.pipeline.stage4_training.training_modes import (
            TrainingPlan,
        )

        # La traduzione fra il vocabolario della piattaforma e quello della
        # pipeline avviene qui, in un punto solo. I parametri dell'adapter
        # stanno in una sezione annidata `lora` e con nomi propri (`r`,
        # `alpha`): passarli piatti non dà errore — vengono semplicemente
        # ignorati, e l'adapter esce con i valori predefiniti. È il genere di
        # difetto che si scopre guardando `adapter_config.json` e chiedendosi
        # perché il rango non sia quello chiesto.
        rango = int(params.get("lora_rank", 16))
        configurazione = {
            "training": {
                "base_model": params.get("base_model", ""),
                "mode": "lora_new" if modo == "lora" else "full_finetune",
                "epochs": params.get("epochs", 3),
                "learning_rate": params.get("learning_rate", 2e-4),
                "batch_size": params.get("batch_size", 1),
                "max_seq_length": params.get("max_seq_length", 2048),
                "lora": {
                    "r": rango,
                    "alpha": int(params.get("lora_alpha", rango * 2)),
                    "dropout": float(params.get("lora_dropout", 0.0)),
                    **params.get("lora", {}),
                },
                **params.get("extra", {}),
            }
        }

        piano = TrainingPlan.from_config(configurazione)
        problemi = piano.validate()
        if problemi:
            # Prima di caricare gigabyte di pesi: un piano incoerente si vede
            # subito, e scoprirlo dopo dieci minuti di download è tempo buttato.
            raise BuildFallita("; ".join(problemi))

        stadio = TrainingStage(configurazione, destinazione)
        stadio.progress_update.connect(avanzamento)
        stadio.error_occurred.connect(
            lambda messaggio: avanzamento(-1, f"errore: {messaggio}")
        )

        avanzamento(1, f"Avvio {modo} su {piano.base_model}")
        risultato = stadio.run(pathlib.Path(dataset), piano)

        percorso = _percorso_da(risultato) or str(destinazione)
        return EsitoBuild(
            artifact_path=percorso,
            meta={"modo": modo, "base_model": params.get("base_model")},
        )

    def _converti(
        self,
        params: Dict[str, Any],
        destinazione: pathlib.Path,
        avanzamento: Avanzamento,
    ) -> EsitoBuild:
        """Esporta in GGUF."""
        from historical_persona_pipeline.pipeline.conversion.conversion_stage import (
            ConversionStage,
        )

        adapter = params.get("adapter_path")
        if not adapter or not pathlib.Path(adapter).exists():
            raise BuildFallita(
                f"adapter non trovato: {adapter or '(non indicato)'}"
            )

        stadio = ConversionStage(
            {"conversion": params.get("conversion", {})}, destinazione,
        )
        stadio.progress_update.connect(avanzamento)

        avanzamento(1, "Avvio della conversione")
        risultato = stadio.run({
            "adapter_path": adapter,
            "base_model": params.get("base_model", ""),
            "export_format": params.get("format", "gguf"),
        })

        return EsitoBuild(
            artifact_path=_percorso_da(risultato) or str(destinazione),
            meta={"formato": params.get("format", "gguf")},
        )


def _percorso_da(risultato: Any) -> Optional[str]:
    """Il percorso dell'artefatto, comunque lo stadio l'abbia restituito.

    Gli stadi della pipeline non concordano sulla forma del risultato — chi un
    dizionario, chi un oggetto, chi una stringa — e uniformarli richiederebbe
    di toccarli tutti. Qui si accettano le forme che usano.
    """
    if risultato is None:
        return None
    if isinstance(risultato, (str, pathlib.Path)):
        return str(risultato)
    if isinstance(risultato, dict):
        for chiave in ("output_dir", "adapter_path", "output_path", "path"):
            if (valore := risultato.get(chiave)):
                return str(valore)
        return None
    for attributo in ("output_dir", "adapter_path", "output_path", "path"):
        if (valore := getattr(risultato, attributo, None)):
            return str(valore)
    return None
