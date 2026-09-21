"""Punto d'ingresso del worker.

    python -m platform_core.worker --role gpu

Un solo eseguibile per due ruoli, distinti dalla configurazione: `worker-cpu`
indicizza le knowledge base e consolida le memorie, `worker-gpu` addestra e
converte. Stessa immagine, deployment diversi — e `worker-gpu` puo' stare a
zero repliche, che e' uno stato valido, non un guasto.

In questa fase il worker annuncia soltanto le proprie capacita': le code
arrivano con la fase 3b. Serve gia' ora perche' e' cio' che rende osservabile
la degradazione, e perche' senza di esso l'API non ha nulla da riportare.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading

from ..api.deps import get_key_value_store
from ..capabilities.probe import probe
from ..capabilities.registry import CapabilityRegistry
from ..settings import get_settings
from .heartbeat import CapabilityHeartbeat, default_worker_id

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="platform_core.worker")
    parser.add_argument(
        "--role", choices=["cpu", "gpu"], default=None,
        help="ruolo del worker; per default si deduce dall'hardware presente",
    )
    parser.add_argument("--worker-id", default=None)
    parser.add_argument(
        "--once", action="store_true",
        help="annuncia una volta sola e termina (diagnosi)",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    capabilities = probe()

    # Il ruolo si deduce da cio' che il processo sa fare davvero, non da cio'
    # che gli e' stato detto: un worker avviato come "gpu" su un nodo senza
    # acceleratore annuncerebbe una capacita' inesistente, e la piattaforma
    # accoderebbe lavori che quel worker non puo' eseguire.
    role = args.role or ("gpu" if capabilities.can_train_lora else "cpu")
    if args.role == "gpu" and not capabilities.can_train_lora:
        logger.warning(
            "Avviato come 'gpu' ma l'addestramento non e' possibile (%s): "
            "annuncio il ruolo effettivo 'cpu'",
            capabilities.reasons.get("can_train_lora", "motivo non disponibile"),
        )
        role = "cpu"

    registry = CapabilityRegistry(get_key_value_store())
    heartbeat = CapabilityHeartbeat(
        registry,
        worker_id=args.worker_id or default_worker_id(),
        role=role,
        capabilities=capabilities,
    )

    if args.once:
        heartbeat.start()
        heartbeat.stop()
        print(f"Annunciato e ritirato: {heartbeat.worker_id} (ruolo {role})")
        return 0

    stopping = threading.Event()

    def shutdown(signum, _frame):
        logger.info("Ricevuto segnale %s: arresto in corso", signum)
        stopping.set()

    signal.signal(signal.SIGINT, shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown)

    with heartbeat:
        logger.info(
            "Worker %s in ascolto (ruolo=%s). Ctrl+C per fermarlo.",
            heartbeat.worker_id, role,
        )
        stopping.wait()

    return 0


if __name__ == "__main__":
    sys.exit(main())
