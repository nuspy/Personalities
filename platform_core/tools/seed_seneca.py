"""Costruisce una personalità di prova con un corpus vero.

    python -m platform_core.tools.seed_seneca --corpus corpus_seneca.json

Serve alla verifica della fase 1 e ai test da capo a fondo: senza una
personalità con un corpus reale dietro, «risponde citando le fonti» resta
un'affermazione senza prova.

Il corpus è la traduzione italiana del 1802 delle *Lettere a Lucilio*, da
Wikisource: pubblico dominio, e con una lingua abbastanza marcata da rendere
evidente se lo stile passa o no.

**Il prompt di sistema è scritto a mano, e provvisoriamente.** Dalla fase 3b lo
genera `persona_synthesizer` dal profilo stilistico misurato sul corpus, che è
il modo giusto: un prompt scritto a occhio descrive l'idea che chi scrive ha
dell'autore, non l'autore. Qui serve solo a provare il motore.
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

from ..domain.base import utcnow
from ..domain.knowledge_models import (
    KnowledgeBase, Personality, PersonalityKnowledgeBase, PersonalityVersion,
)
from ..domain.session import dispose_engine, get_session_factory
from ..knowledge.embedding import OpenAICompatibleEmbedder
from ..knowledge.indexer import Indexer

logger = logging.getLogger(__name__)

SLUG_KB = "seneca-lettere"
SLUG_PERSONA = "seneca"

PROMPT = """Sei Lucio Anneo Seneca, che scrive a Lucilio.

Rispondi come in una lettera: ti rivolgi a chi legge in seconda persona, e ciò \
che dici nasce sempre da una difficoltà concreta — la tua o la sua. Non esponi \
dottrine: mostri come si ragiona su un caso.

Il tuo periodo è breve e scandito. Preferisci l'affermazione alla cautela, e \
l'esempio all'astrazione: un uomo che agisce, una circostanza, una scelta. \
Chiudi volentieri con una frase che si possa ricordare da sola, ma solo quando \
il ragionamento l'ha guadagnata.

Non prometti consolazione. Se la risposta è dura, la dici."""

REGOLE = [
    "Non fare riferimento a fatti, persone o invenzioni successivi al 65 d.C.",
    "Se ti viene chiesto qualcosa che non sapresti, ammettilo senza inventare.",
    "Non usare elenchi puntati: scrivi in prosa continua, come in una lettera.",
]


async def costruisci(percorso_corpus: pathlib.Path, *, ricrea: bool = False) -> None:
    corpus = json.loads(percorso_corpus.read_text(encoding="utf-8"))
    if not corpus:
        raise SystemExit(f"{percorso_corpus} non contiene testi")

    embedder = OpenAICompatibleEmbedder()
    factory = get_session_factory()

    async with factory() as session:
        kb = await _base(session, embedder.modello, ricrea=ricrea)
        indexer = Indexer(session, embedder)

        indicizzati = 0
        saltati = 0
        for voce in corpus:
            esito = await indexer.indicizza(
                kb,
                titolo=voce["titolo"],
                testo=voce["testo"],
                uri=voce.get("uri"),
                lingua="it",
                meta={"autore": "Lucio Anneo Seneca", "fonte": "Wikisource"},
            )
            if esito.saltato:
                saltati += 1
            else:
                indicizzati += 1
                print(f"  {voce['titolo']}: {esito.passaggi} passaggi")

        personalita, versione = await _personalita(session, kb)
        await session.commit()

    await dispose_engine()

    print()
    print(f"Base «{kb.slug}»: {indicizzati} documenti indicizzati, {saltati} già presenti")
    print(f"  {kb.stats}")
    print(f"Personalità «{personalita.slug}» versione {versione.version}")


async def _base(session, modello: str, *, ricrea: bool) -> KnowledgeBase:
    esistente = (
        await session.execute(select(KnowledgeBase).where(KnowledgeBase.slug == SLUG_KB))
    ).scalar_one_or_none()

    if esistente is not None and ricrea:
        await session.delete(esistente)
        await session.flush()
        esistente = None

    if esistente is not None:
        if esistente.embed_model != modello:
            raise SystemExit(
                f"la base esiste con il modello {esistente.embed_model}, "
                f"ora è configurato {modello}: usa --ricrea per rifarla, "
                f"perché vettori di modelli diversi non sono confrontabili"
            )
        return esistente

    kb = KnowledgeBase(
        slug=SLUG_KB,
        name="Lettere a Lucilio",
        description="Traduzione italiana del 1802, da Wikisource. Pubblico dominio.",
        kind="corpus",
        visibility="public",
        embed_model=modello,
    )
    session.add(kb)
    await session.flush()
    return kb


async def _personalita(session, kb: KnowledgeBase):
    personalita = (
        await session.execute(select(Personality).where(Personality.slug == SLUG_PERSONA))
    ).scalar_one_or_none()

    if personalita is None:
        personalita = Personality(
            slug=SLUG_PERSONA,
            display_name="Seneca",
            description="Lucio Anneo Seneca, come scrive nelle lettere a Lucilio.",
            status="published",
        )
        session.add(personalita)
        await session.flush()

    ultima = (
        await session.execute(
            select(PersonalityVersion.version)
            .where(PersonalityVersion.personality_id == personalita.id)
            .order_by(PersonalityVersion.version.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    versione = PersonalityVersion(
        personality_id=personalita.id,
        version=(ultima or 0) + 1,
        system_prompt=PROMPT,
        behavior_rules={"regole": REGOLE},
        llm_config={"temperature": 0.8, "max_tokens": 1200},
        rag_config={"max_chunks": 5},
        published_at=utcnow(),
    )
    session.add(versione)
    await session.flush()

    personalita.current_version_id = versione.id

    legame = (
        await session.execute(
            select(PersonalityKnowledgeBase).where(
                PersonalityKnowledgeBase.personality_id == personalita.id,
                PersonalityKnowledgeBase.kb_id == kb.id,
            )
        )
    ).scalar_one_or_none()

    if legame is None:
        session.add(PersonalityKnowledgeBase(
            personality_id=personalita.id,
            kb_id=kb.id,
            # `voice` e non `knowledge`: queste lettere sono la fonte dello
            # stile, non un archivio di fatti da verificare. La distinzione
            # conta per il groundcheck della fase 2, che non deve segnalare
            # come inventata una frase riuscita.
            role="voice",
            max_chunks=5,
        ))

    return personalita, versione


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="platform_core.tools.seed_seneca")
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument(
        "--ricrea", action="store_true",
        help="cancella la base e la ricostruisce (serve se cambia il modello di embedding)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(costruisci(args.corpus, ricrea=args.ricrea))
    return 0


if __name__ == "__main__":
    sys.exit(main())
