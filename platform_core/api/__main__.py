"""Avvio del servizio API.

    python -m platform_core.api --port 8000

Esiste per una ragione sola, ed è Windows. uvicorn sceglie da sé come costruire
l'event loop e su Windows costruisce un `ProactorEventLoop`, che psycopg 3 non
sa usare in modalità asincrona: ogni accesso al database fallisce. Non è una
policy globale — quella la si può impostare all'import, e infatti
`platform_core/__init__.py` lo fa — ma una *factory* passata al momento
dell'avvio, che sovrascrive qualunque cosa sia stata decisa prima.

Su Linux non cambia nulla rispetto a `uvicorn platform_core.api.app:app`, e in
container si può continuare a usare quel comando. Qui però il modo giusto di
avviare il servizio è uno solo, ed è questo: chi sviluppa su Windows non deve
scoprire da un errore di connessione che gli serviva un flag.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Callable, Optional

import uvicorn

from ..settings import get_settings


def loop_factory() -> Optional[Callable[[], asyncio.AbstractEventLoop]]:
    """Il costruttore di event loop adatto alla piattaforma.

    `None` significa «vada uvicorn per la sua strada»: su Linux sceglie uvloop
    se c'è, che è più veloce e perfettamente compatibile.
    """
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="platform_core.api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()

    if sys.platform != "win32":
        uvicorn.run(
            "platform_core.api.app:app",
            host=args.host, port=args.port, reload=args.reload,
            log_level=settings.log_level.lower(),
        )
        return 0

    if args.reload:
        # Il ricaricamento automatico gira in un sottoprocesso, e in quel caso
        # uvicorn sceglie già il selettore da sé: si può usare la via normale.
        uvicorn.run(
            "platform_core.api.app:app",
            host=args.host, port=args.port, reload=True,
            log_level=settings.log_level.lower(),
        )
        return 0

    from .app import app

    config = uvicorn.Config(
        app,
        host=args.host, port=args.port,
        log_level=settings.log_level.lower(),
        # «none» non significa senza loop: significa che il loop lo fornisce
        # chi avvia, qui sotto.
        loop="none",
    )
    server = uvicorn.Server(config)

    with asyncio.Runner(loop_factory=loop_factory()) as runner:
        runner.run(server.serve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
