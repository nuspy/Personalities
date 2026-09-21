"""Stopword per lingua.

Ordine di preferenza: liste curate su disco (`data/stopwords/<lang>.txt`),
poi quelle del modello spaCy caricato, infine nessuna. Per latino e greco
antico spaCy non offre nulla, quindi le liste su disco sono l'unica fonte.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Set

from ...paths import STOPWORD_DIR
from ..data_models import SupportedLanguage

logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def _load_from_disk(language_code: str) -> frozenset[str]:
    path = STOPWORD_DIR / f"{language_code}.txt"
    if not path.exists():
        return frozenset()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        logger.error(f"Stopword illeggibili per '{language_code}': {exc}")
        return frozenset()
    return frozenset(
        line.strip().lower() for line in lines
        if line.strip() and not line.startswith("#")
    )


def get_stopwords(language: SupportedLanguage, parser=None) -> Set[str]:
    """Insieme di stopword per la lingua, unendo disco e modello spaCy."""
    words = set(_load_from_disk(language.value))

    model_stopwords = getattr(parser, "stopwords", None)
    if model_stopwords:
        words |= {w.lower() for w in model_stopwords}

    return words
