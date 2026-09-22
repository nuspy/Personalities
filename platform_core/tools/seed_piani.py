"""Scrive il catalogo dei piani.

    python -m platform_core.tools.seed_piani

**Il piano gratuito non è opzionale.** Senza una riga `free`, chi non ha
abbonamento ricade sui valori scritti nel codice (`DIRITTI_BASE`), che sono un
ripiego pensato per non lasciare il servizio muto — non una decisione
commerciale. Con la riga, cosa riceve chi non paga si amministra invece di
doverlo ricompilare.

**I diritti sostituiscono, non si sommano.** Un piano elenca *tutte* le
categorie a cui dà accesso: `gold` deve nominare anche `free` e `base`, o chi
lo sottoscrive perde le voci che aveva prima — che è il difetto peggiore
possibile, perché si manifesta come regressione a chi ha appena pagato.

I prezzi sono in centesimi: un `float` per il denaro è il modo classico di
scoprire dopo mesi che una somma di importi non torna per un millesimo.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any, Dict, List

from sqlalchemy import select

from ..domain.billing_models import Plan
from ..domain.session import get_session_factory

logger = logging.getLogger(__name__)

#: Il catalogo. `rank` ordina la pagina dei prezzi, dal meno al più caro.
CATALOGO: List[Dict[str, Any]] = [
    {
        "slug": "free",
        "name": "Gratuito",
        "rank": 0,
        "price_monthly": 0,
        "price_yearly": 0,
        "credits_per_period": 50,
        "limits": {"messaggi_al_giorno": 20, "conversazioni": 5, "memorie": 50},
        "entitlements": {
            "categorie": ["free"],
            "voce": False,
            "corpora_propri": False,
            "personalita_proprie": 0,
        },
    },
    {
        "slug": "base",
        "name": "Base",
        "rank": 1,
        "price_monthly": 900,
        "price_yearly": 9000,
        "credits_per_period": 500,
        "limits": {"messaggi_al_giorno": 200, "conversazioni": 50, "memorie": 500},
        "entitlements": {
            "categorie": ["free", "base"],
            "voce": True,
            "corpora_propri": False,
            "personalita_proprie": 0,
        },
    },
    {
        "slug": "gold",
        "name": "Gold",
        "rank": 2,
        "price_monthly": 2400,
        "price_yearly": 24000,
        "credits_per_period": 2000,
        "limits": {
            "messaggi_al_giorno": 1000, "conversazioni": 500, "memorie": 5000,
        },
        "entitlements": {
            "categorie": ["free", "base", "gold"],
            "voce": True,
            "corpora_propri": True,
            "personalita_proprie": 3,
        },
    },
    {
        "slug": "premium",
        "name": "Premium",
        "rank": 3,
        "price_monthly": 5900,
        "price_yearly": 59000,
        "credits_per_period": 10000,
        "limits": {
            # `-1` e non un numero grandissimo: «illimitato» e «un milione» si
            # comportano allo stesso modo finché qualcuno non arriva a un
            # milione, e lì la differenza è un utente che non capisce perché
            # si è fermato.
            "messaggi_al_giorno": -1, "conversazioni": -1, "memorie": -1,
        },
        "entitlements": {
            "categorie": ["free", "base", "gold", "premium"],
            "voce": True,
            "corpora_propri": True,
            "personalita_proprie": 25,
        },
    },
]


async def scrivi(*, forza: bool = False) -> int:
    """Crea i piani mancanti; aggiorna gli esistenti solo se richiesto.

    Non aggiornare è il comportamento giusto di default: su un'installazione
    viva i prezzi e i limiti possono essere stati cambiati da chi amministra,
    e riportarli ai valori del file sarebbe un cambio commerciale deciso da
    uno script.
    """
    scritti = 0
    async with get_session_factory()() as session:
        for riga in CATALOGO:
            esistente = (await session.execute(
                select(Plan).where(Plan.slug == riga["slug"])
            )).scalar_one_or_none()

            if esistente is None:
                session.add(Plan(active=True, **riga))
                logger.info("Piano «%s» creato", riga["slug"])
                scritti += 1
            elif forza:
                for campo, valore in riga.items():
                    setattr(esistente, campo, valore)
                logger.info("Piano «%s» riportato ai valori del file", riga["slug"])
                scritti += 1
            else:
                logger.info("Piano «%s» già presente: lasciato com'è", riga["slug"])

        await session.commit()
    return scritti


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forza", action="store_true",
        help="riporta anche i piani esistenti ai valori scritti qui",
    )
    args = parser.parse_args()

    quanti = asyncio.run(scrivi(forza=args.forza))
    print(f"{quanti} piani scritti.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
