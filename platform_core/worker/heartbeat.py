"""Battito con cui un worker dichiara alla piattaforma cosa sa fare.

L'annuncio ha una scadenza piu' lunga dell'intervallo con cui viene rinnovato:
un worker momentaneamente lento resta vivo, uno fermo sparisce da solo. Non
serve nessun meccanismo che accorga la morte — e' l'assenza a dirlo.

Alla chiusura ordinata l'annuncio viene ritirato subito, senza attendere la
scadenza: fra l'arresto e il ritiro c'e' una finestra in cui la piattaforma
offrirebbe una funzione che nessuno raccoglierebbe piu'.
"""
from __future__ import annotations

import logging
import os
import socket
import threading
from typing import Optional

from ..capabilities.probe import HardwareCapabilities, probe
from ..capabilities.registry import (
    HEARTBEAT_INTERVAL_SECONDS, CapabilityRegistry,
)

logger = logging.getLogger(__name__)


def default_worker_id() -> str:
    """Identificativo stabile per il processo.

    In Kubernetes il nome del pod e' gia' univoco e sopravvive al riavvio del
    processo dentro lo stesso pod, che e' esattamente il comportamento voluto:
    un riavvio rapido non deve far comparire un secondo worker fantasma.
    """
    return os.getenv("HOSTNAME") or f"{socket.gethostname()}-{os.getpid()}"


class CapabilityHeartbeat:
    """Rinnova l'annuncio finche' il worker e' vivo."""

    def __init__(
        self,
        registry: CapabilityRegistry,
        *,
        worker_id: Optional[str] = None,
        role: str = "cpu",
        interval: int = HEARTBEAT_INTERVAL_SECONDS,
        capabilities: Optional[HardwareCapabilities] = None,
    ):
        self._registry = registry
        self._worker_id = worker_id or default_worker_id()
        self._role = role
        self._interval = interval
        self._capabilities = capabilities or probe()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def capabilities(self) -> HardwareCapabilities:
        return self._capabilities

    def start(self) -> None:
        """Annuncia subito, poi rinnova in un thread di servizio.

        Il primo annuncio e' sincrono: al ritorno da `start()` la piattaforma
        sa gia' del worker, e un test non deve attendere il primo intervallo.
        """
        if self._thread is not None:
            return

        self._announce()
        self._thread = threading.Thread(
            target=self._loop, name=f"heartbeat-{self._worker_id}", daemon=True
        )
        self._thread.start()

        logger.info(
            "Worker %s annunciato (ruolo=%s, GPU=%s, LoRA=%s)",
            self._worker_id, self._role,
            self._capabilities.gpu_name or "nessuna",
            self._capabilities.can_train_lora,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

        try:
            self._registry.withdraw(self._worker_id)
            logger.info("Worker %s ritirato", self._worker_id)
        except Exception as exc:
            # Il ritiro e' un'ottimizzazione: senza, la voce scade da sola.
            # Un errore qui non deve impedire la chiusura del processo.
            logger.warning("Ritiro dell'annuncio non riuscito: %s", exc)

    def __enter__(self) -> "CapabilityHeartbeat":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    # ------------------------------------------------------------ interno

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._announce()

    def _announce(self) -> None:
        try:
            self._registry.announce(
                self._worker_id, self._capabilities, role=self._role
            )
        except Exception as exc:
            # Redis irraggiungibile per un istante non deve fermare il worker:
            # smette di annunciarsi, la sua voce scade, e quando torna
            # raggiungibile si riannuncia da solo al battito successivo.
            logger.warning("Annuncio delle capacita' non riuscito: %s", exc)
