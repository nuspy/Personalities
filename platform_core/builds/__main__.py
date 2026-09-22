"""Punto d'ingresso del consumatore di realizzazioni.

    python -m platform_core.builds --worker-id w-gpu-1

Gira sul nodo con l'acceleratore, accanto al worker che annuncia le capacita'.
Sono due processi e non uno: l'annuncio deve continuare anche mentre un
addestramento occupa la GPU per venti minuti, e un processo solo lo
interromperebbe proprio quando serve di piu' saperlo vivo.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from ..settings import get_settings
from ..worker.heartbeat import default_worker_id
from .worker import avvia


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="platform_core.builds")
    parser.add_argument("--worker-id", default=None)
    parser.add_argument(
        "--solo-digestione", "--senza-acceleratore", action="store_true",
        dest="solo_digestione",
        help=(
            "prende solo i lavori che non chiedono una GPU — digestione e "
            "ingestione dei corpora"
        ),
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    return asyncio.run(avvia(
        args.worker_id or default_worker_id(),
        solo_digestione=args.solo_digestione,
    ))


if __name__ == "__main__":
    sys.exit(main())
