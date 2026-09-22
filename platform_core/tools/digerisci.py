"""Analizza un corpus già indicizzato.

    python -m platform_core.tools.digerisci --kb seneca-lettere --chi "Lucio Anneo Seneca"

Si esegue una volta per corpus, e richiede tempo: su cinquemila passaggi sono
ore. È un costo una tantum per una personalità che vivrà anni, e il confronto
giusto non è con il risparmio ma con un adapter che impara a imitare il
curatore invece dell'autore.

Riprendibile: ogni lotto viene salvato subito, e una nuova esecuzione riparte
da ciò che non è stato ancora etichettato.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import List, Optional

from sqlalchemy import select

from ..domain.knowledge_models import Chunk, KnowledgeBase
from ..domain.session import dispose_engine, get_session_factory
from ..knowledge.digest_runner import DigestioneCorpus
from ..knowledge.digestion import Digestore
from ..api.deps import provider_per
from ..llm.compiti import Compito

logger = logging.getLogger(__name__)


async def esegui(
    slug: str, *, chi: str, rifai: bool, lotto: int, campione: int,
) -> int:
    provider = provider_per(Compito.DIGESTIONE)
    factory = get_session_factory()

    async with factory() as session:
        kb = (await session.execute(
            select(KnowledgeBase).where(KnowledgeBase.slug == slug)
        )).scalar_one_or_none()

        if kb is None:
            raise SystemExit(f"Base '{slug}' non trovata")

        if campione:
            # Su un campione prima di impegnare ore: se il classificatore
            # sbaglia, si scopre in due minuti invece che a corpus finito.
            await _prova_su_campione(session, kb, provider, chi, campione)
            await dispose_engine()
            return 0

        def mostra(fatti: int, totale: int, messaggio: str) -> None:
            print(f"  [{fatti:5}/{totale}] {messaggio}")

        from ..knowledge.embedding import OpenAICompatibleEmbedder

        esito = await DigestioneCorpus(
            session, Digestore(provider, per_lotto=lotto),
            embedder=OpenAICompatibleEmbedder(), per_lotto=lotto,
        ).digerisci(kb, chi=chi, rifai=rifai, avanzamento=mostra)

    await dispose_engine()

    print()
    print(f"Base «{slug}»")
    print(f"  esaminati:      {esito.esaminati}")
    print(f"  classificati:   {esito.classificati}")
    print(f"  ripuliti:       {esito.ripuliti}")
    print(f"  scartati:       {esito.scartati} "
          f"({esito.scartati_a_vista} a vista, "
          f"{esito.scartati_dal_giudizio} dal giudizio)")
    if esito.per_categoria:
        print()
        print("  per categoria dominante:")
        for categoria, quanti in sorted(
            esito.per_categoria.items(), key=lambda kv: -kv[1]
        ):
            print(f"    {categoria:14} {quanti}")
    if esito.per_provenienza:
        print()
        print("  per provenienza:")
        for provenienza, quanti in sorted(
            esito.per_provenienza.items(), key=lambda kv: -kv[1]
        ):
            print(f"    {provenienza:14} {quanti}")
    return 0


async def _prova_su_campione(session, kb, provider, chi: str, quanti: int) -> None:
    passaggi = (await session.execute(
        select(Chunk).where(Chunk.kb_id == kb.id).order_by(Chunk.ordinal).limit(quanti)
    )).scalars().all()

    etichette = await Digestore(provider).classifica(
        [c.text for c in passaggi], chi=chi,
    )

    print(f"Campione di {len(passaggi)} passaggi da «{kb.slug}»")
    print()
    for chunk, etichettatura in zip(passaggi, etichette):
        categorie = ", ".join(
            f"{c.value} {p:.1f}"
            for c, p in sorted(
                etichettatura.categorie.items(), key=lambda kv: -kv[1]
            )[:3]
        )
        segno = "SCARTATO" if etichettatura.scartato else "tenuto  "
        print(f"  [{segno}] {etichettatura.provenienza.value:14} q={etichettatura.qualita:.2f}")
        print(f"             {categorie}")
        print(f"             {chunk.text[:100].strip()}…")
        if etichettatura.motivo_scarto:
            print(f"             motivo: {etichettatura.motivo_scarto}")
        print()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="platform_core.tools.digerisci")
    parser.add_argument("--kb", required=True, help="slug della base")
    parser.add_argument("--chi", default="", help="di chi parla il corpus")
    parser.add_argument(
        "--rifai", action="store_true",
        help="rianalizza anche i passaggi già etichettati",
    )
    parser.add_argument("--lotto", type=int, default=6)
    parser.add_argument(
        "--campione", type=int, default=0,
        help="prova su N passaggi e mostra gli esiti, senza salvare",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return asyncio.run(esegui(
        args.kb, chi=args.chi, rifai=args.rifai,
        lotto=args.lotto, campione=args.campione,
    ))


if __name__ == "__main__":
    sys.exit(main())
