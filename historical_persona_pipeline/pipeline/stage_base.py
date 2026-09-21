"""Base comune degli stadi della pipeline.

Era `QObject` con `pyqtSignal`, ed è la ragione per cui importare uno stadio
tirava dentro PyQt6: centinaia di megabyte di interfaccia grafica dentro un
worker che gira in un container senza schermo.

I segnali ora sono oggetti Python (`pipeline/signals.py`) con la **stessa
interfaccia** — `connect`, `emit` — così nessuno stadio è dovuto cambiare. La
pipeline funziona ed è coperta da test: modificarne le chiamate per una
ragione architetturale sarebbe stato rischiare di romperla per nulla.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from .signals import Segnale


class PipelineStage(ABC):
    """Base astratta di tutti gli stadi.

    I tre segnali sono gli stessi di prima:

    - `progress_update(percentuale, messaggio)`
    - `error_occurred(messaggio)`
    - `stage_completed(risultato)`
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Per istanza e non per classe: con `pyqtSignal` la dichiarazione sta
        # sulla classe e Qt crea la copia per ogni oggetto. Qui la copia la
        # facciamo noi — senza, due stadi dello stesso tipo condividerebbero
        # gli ascoltatori, e l'avanzamento di uno comparirebbe sotto l'altro.
        self.progress_update = Segnale("progress_update")
        self.error_occurred = Segnale("error_occurred")
        self.stage_completed = Segnale("stage_completed")

    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """Esegue la logica dello stadio."""

    def validate_config(self) -> bool:
        """Verifica la configurazione specifica dello stadio."""
        return True

    def osserva(
        self,
        avanzamento: Optional[Callable[[int, str], None]] = None,
        errore: Optional[Callable[[str], None]] = None,
    ) -> "PipelineStage":
        """Aggancia gli ascoltatori in una riga, restituendo lo stadio.

            stadio = TrainingStage(config).osserva(avanzamento=stampa)

        Comodità per chi invoca uno stadio come job: senza, ogni chiamante
        ripete tre righe di `connect` — e chi le ripete prima o poi ne
        dimentica una, perdendo proprio la segnalazione degli errori.
        """
        if avanzamento is not None:
            self.progress_update.connect(avanzamento)
        if errore is not None:
            self.error_occurred.connect(errore)
        return self
