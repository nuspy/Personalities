"""Aggiunge audio o video a un corpus, trascrivendoli.

    python -m platform_core.tools.ingerisci_audio --kb tizio-interviste registrazione.mp4

Estrae la traccia con ffmpeg, la trascrive con Whisper, e indicizza il testo
come un qualunque documento — con una differenza che conta: il documento
porta `registro: parlato`, e a valle quella distinzione impedisce di
addestrare un modello a scrivere come si parla.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import pathlib
import sys
from typing import List, Optional

from sqlalchemy import select

from ..domain.knowledge_models import KnowledgeBase
from ..domain.session import dispose_engine, get_session_factory
from ..knowledge.embedding import OpenAICompatibleEmbedder
from ..knowledge.indexer import Indexer
from ..knowledge.transcription import (
    MODELLO_PREDEFINITO, TrascrizioneNonDisponibile, Trascrittore,
    e_multimediale,
)

logger = logging.getLogger(__name__)


async def ingerisci(
    slug: str,
    file: List[pathlib.Path],
    *,
    modello: str,
    lingua: Optional[str],
    solo_trascrivi: bool,
) -> int:
    multimediali = [f for f in file if e_multimediale(f)]
    if not multimediali:
        raise SystemExit("Nessun file audio o video fra quelli indicati")

    trascrittore = Trascrittore(modello)
    embedder = OpenAICompatibleEmbedder()

    async with get_session_factory()() as session:
        kb = None
        if not solo_trascrivi:
            kb = (await session.execute(
                select(KnowledgeBase).where(KnowledgeBase.slug == slug)
            )).scalar_one_or_none()
            if kb is None:
                raise SystemExit(f"Base '{slug}' non trovata")

        indexer = Indexer(session, embedder) if kb else None

        for percorso in multimediali:
            print(f"\n{percorso.name}")
            try:
                # In un thread: la trascrizione occupa la GPU per minuti, e nel
                # loop bloccherebbe tutto il resto.
                trascrizione = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda p=percorso: trascrittore.trascrivi(p, lingua=lingua),
                )
            except (TrascrizioneNonDisponibile, FileNotFoundError) as exc:
                print(f"  non riuscita: {exc}")
                continue

            print(f"  lingua {trascrizione.lingua}, "
                  f"{trascrizione.durata:.0f}s, "
                  f"{len(trascrizione.segmenti)} segmenti, "
                  f"{trascrizione.parole} parole")
            if trascrizione.scartati:
                print(f"  {trascrizione.scartati} segmenti scartati "
                      f"(il modello non aveva sentito abbastanza)")

            if trascrizione.da_riascoltare:
                # Un avviso e non un rifiuto: nessuna difesa automatica
                # distingue un errore plausibile, e l'unica verifica che
                # funziona è un orecchio su un campione.
                print(f"  ATTENZIONE: fiducia media {trascrizione.fiducia_media:.2f} — "
                      f"ascolta un campione prima di fidarti di questo testo")

            if solo_trascrivi:
                print()
                print(trascrizione.testo[:600])
                continue

            esito = await indexer.indicizza(
                kb,
                titolo=percorso.stem,
                testo=trascrizione.testo,
                uri=str(percorso),
                lingua=trascrizione.lingua or lingua,
                meta=trascrizione.meta(),
            )
            if esito.saltato:
                print(f"  saltato: {esito.motivo}")
            else:
                print(f"  indicizzato: {esito.passaggi} passaggi")

        if not solo_trascrivi:
            await session.commit()

    await dispose_engine()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="platform_core.tools.ingerisci_audio")
    parser.add_argument("file", nargs="+", type=pathlib.Path)
    parser.add_argument("--kb", default="", help="slug della base")
    parser.add_argument("--modello", default=MODELLO_PREDEFINITO)
    parser.add_argument(
        "--lingua", default=None,
        help="codice della lingua; omesso, la riconosce da sé",
    )
    parser.add_argument(
        "--solo-trascrivi", action="store_true",
        help="mostra la trascrizione senza indicizzarla",
    )
    args = parser.parse_args(argv)

    if not args.kb and not args.solo_trascrivi:
        raise SystemExit("Serve --kb, oppure --solo-trascrivi per una prova")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return asyncio.run(ingerisci(
        args.kb, args.file, modello=args.modello,
        lingua=args.lingua, solo_trascrivi=args.solo_trascrivi,
    ))


if __name__ == "__main__":
    sys.exit(main())
