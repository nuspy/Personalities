"""Campi semantici caricati da `data/semantic_fields/*.json`.

Ogni file e' un campo (il nome del file e' il nome del campo) e contiene
`{codice_lingua: [termini]}`. I termini di tutte le lingue vengono uniti: un
corpus latino con traduzione italiana a fronte deve pesare su un solo campo.

Prima questi file esistevano ma non venivano letti: `VocabularyAnalyzer`
aveva due campi semantici scritti a mano, solo in inglese, e restituiva
comunque `semantic_field_distribution={}`.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Dict, List

from ...paths import SEMANTIC_FIELD_DIR

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_semantic_fields() -> Dict[str, List[str]]:
    """Mappa nome_campo -> termini di tutte le lingue."""
    fields: Dict[str, List[str]] = {}

    if not SEMANTIC_FIELD_DIR.exists():
        return fields

    for path in sorted(SEMANTIC_FIELD_DIR.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            logger.error(f"Campo semantico illeggibile {path.name}: {exc}")
            continue

        terms: List[str] = []
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list):
                    terms.extend(str(t).lower() for t in value)
        elif isinstance(data, list):
            terms = [str(t).lower() for t in data]

        if terms:
            # `military_terms.json` -> campo "military".
            field_name = path.stem.replace("_terms", "")
            fields[field_name] = sorted(set(terms))

    return fields


def clear_cache() -> None:
    """Da chiamare dopo aver aggiunto campi semantici a runtime."""
    load_semantic_fields.cache_clear()
