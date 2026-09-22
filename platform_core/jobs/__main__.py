"""Esegue la manutenzione periodica.

    python -m platform_core.jobs                 # una passata e basta
    python -m platform_core.jobs --ogni 3600     # ogni ora, finché non lo fermi

**Una passata e basta è il modo preferito.** In produzione il ciclo lo fa chi
sa farlo — `cron`, un `CronJob` di Kubernetes — che sa anche cosa fare se una
passata non finisce, cosa che un `while True` scritto qui non saprebbe. Il
ciclo esiste per lo sviluppo, dove non c'è nessuno scheduler.

Il codice di uscita distingue i due fallimenti: `1` se la passata non è
riuscita affatto, `2` se è finita lasciando errori su singole righe. Un
`CronJob` che risponde sempre zero è un lavoro che nessuno controlla.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .periodico import passata

logger = logging.getLogger(__name__)


async def _cicla(intervallo: int, limite: int) -> int:
    while True:
        esito = await passata(limite=limite)
        print(json.dumps(esito.to_dict(), ensure_ascii=False))
        await asyncio.sleep(intervallo)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ogni", type=int, default=0,
        help="secondi fra una passata e l'altra; senza, ne fa una sola",
    )
    parser.add_argument("--limite", type=int, default=100)
    parser.add_argument(
        "--senza-rinnovi", action="store_true",
        help="salta i rinnovi degli abbonamenti",
    )
    parser.add_argument(
        "--senza-memorie", action="store_true",
        help="salta il consolidamento delle memorie",
    )
    args = parser.parse_args()

    if args.ogni > 0:
        try:
            asyncio.run(_cicla(args.ogni, args.limite))
        except KeyboardInterrupt:
            return 0
        return 0

    try:
        esito = asyncio.run(passata(
            limite=args.limite,
            rinnovi=not args.senza_rinnovi,
            memorie=not args.senza_memorie,
        ))
    except Exception:
        logger.exception("La manutenzione non è riuscita")
        return 1

    print(json.dumps(esito.to_dict(), ensure_ascii=False))
    return 2 if esito.errori else 0


if __name__ == "__main__":
    sys.exit(main())
