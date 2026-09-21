"""Costruisce un dataset di addestramento dai passaggi di un corpus.

    python -m platform_core.tools.dataset_di_prova --kb seneca-lettere --out dataset.jsonl

Serve alla verifica della fase 3b, non alla produzione. Il dataset vero lo
genera lo stadio 3 della pipeline, che costruisce domande e risposte
interrogando un modello: è la parte lenta e costosa, e per provare che il
ciclo di realizzazione funzioni non serve.

Qui si prende un passaggio e lo si mette come risposta a una domanda
verosimile. Il risultato **non produce una buona personalità** — le domande
sono generiche e le risposte sono citazioni — ma ha la forma giusta, ed è
quella che il ciclo deve saper digerire.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import sys
from typing import List, Optional

from sqlalchemy import select

from ..domain.knowledge_models import Chunk, KnowledgeBase
from ..domain.session import dispose_engine, get_session_factory

logger = logging.getLogger(__name__)

#: Domande generiche, ruotate sui passaggi. Non è generazione: è
#: un'impalcatura, e il commento sopra dice perché basta.
DOMANDE = [
    "Che cosa pensi di questo?",
    "Come la vedi?",
    "Dimmi come ragioni su questo punto.",
    "Che consiglio mi dai?",
    "Perché la metti così?",
    "Su cosa si fonda quello che dici?",
]

SISTEMA = (
    "Sei Lucio Anneo Seneca, che scrive a Lucilio. Periodo breve e scandito, "
    "esempio concreto invece dell'astrazione, nessuna promessa di consolazione."
)


async def costruisci(
    slug: str, destinazione: pathlib.Path, *, massimo: int = 200
) -> int:
    async with get_session_factory()() as session:
        kb = (await session.execute(
            select(KnowledgeBase).where(KnowledgeBase.slug == slug)
        )).scalar_one_or_none()

        if kb is None:
            raise SystemExit(f"Base '{slug}' non trovata")

        passaggi = (await session.execute(
            select(Chunk)
            .where(Chunk.kb_id == kb.id)
            .order_by(Chunk.ordinal)
            .limit(massimo)
        )).scalars().all()

    if not passaggi:
        raise SystemExit(f"La base '{slug}' non contiene passaggi")

    destinazione.parent.mkdir(parents=True, exist_ok=True)
    scritte = 0

    with destinazione.open("w", encoding="utf-8") as uscita:
        for i, passaggio in enumerate(passaggi):
            testo = passaggio.text.strip()
            # Passaggi troppo brevi non insegnano nulla sullo stile e
            # allungano l'addestramento senza cambiarne l'esito.
            if len(testo) < 200:
                continue

            uscita.write(json.dumps({
                "conversations": [
                    {"from": "system", "value": SISTEMA},
                    {"from": "human", "value": DOMANDE[i % len(DOMANDE)]},
                    {"from": "gpt", "value": testo},
                ]
            }, ensure_ascii=False) + "\n")
            scritte += 1

    await dispose_engine()
    return scritte


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="platform_core.tools.dataset_di_prova")
    parser.add_argument("--kb", required=True, help="slug della base")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--max", type=int, default=200)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    quante = asyncio.run(costruisci(args.kb, args.out, massimo=args.max))

    print(f"{quante} conversazioni scritte in {args.out}")
    print(
        "Impalcatura per provare il ciclo di realizzazione: le domande sono "
        "generiche e le risposte sono citazioni. Il dataset vero lo genera lo "
        "stadio 3 della pipeline."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
