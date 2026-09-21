"""Analisi dei valori: quanto e dove il corpus tocca ciascun valore culturale.

Il matching avviene sui confini di parola: `combined_text.count("vir")`
contava anche *virtute*, *servire* e *virus*, gonfiando i valori con falsi
positivi. Qui ogni keyword diventa una regex ancorata a `\\b`, con supporto
per keyword multi-parola e per le forme flesse via prefisso opzionale.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...data_models import ValueProfile, TextSegment

logger = logging.getLogger(__name__)

# Numero di caratteri di contesto attorno a un'occorrenza citata come esempio.
CONTEXT_BEFORE = 120
CONTEXT_AFTER = 160
MAX_EXAMPLES_PER_VALUE = 3


class ValueAnalyzer:
    """Confronta il corpus con uno o piu' dizionari di valori."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.value_dicts = self._load_value_dicts(self.data_dir / "value_dictionaries")
        if not self.value_dicts:
            logger.warning(
                f"Nessun dizionario di valori in {self.data_dir / 'value_dictionaries'}"
            )

    # ------------------------------------------------------------------ IO

    def _load_value_dicts(self, path: Path) -> Dict[str, Dict[str, Any]]:
        dicts: Dict[str, Dict[str, Any]] = {}
        if not path.exists():
            return dicts
        for f in sorted(path.glob("*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    dicts[f.stem] = json.load(fh)
            except Exception as exc:
                logger.error(f"Dizionario di valori illeggibile {f.name}: {exc}")
        return dicts

    def available_dicts(self) -> List[str]:
        """Nomi dei dizionari disponibili su disco (per popolare la GUI)."""
        return sorted(self.value_dicts.keys())

    def reload(self) -> None:
        """Ricarica da disco: serve dopo la generazione di un nuovo dizionario."""
        self.value_dicts = self._load_value_dicts(self.data_dir / "value_dictionaries")

    # -------------------------------------------------------------- matching

    @staticmethod
    def _compile_keyword(keyword: str) -> Optional[re.Pattern]:
        """Regex su confini di parola, tollerante alla flessione.

        Le lingue del corpus sono flessive e cambiano la coda della parola, non
        solo vi aggiungono: `fides` compare come *fidem*, `virtus` come
        *virtute*, `dignitas` come *dignitatem*. Un match sulla forma esatta ne
        perderebbe la maggior parte.

        La keyword viene quindi ridotta a una radice (via le ultime due
        lettere, mai sotto i 4 caratteri) seguita da una desinenza libera fino
        a 4 caratteri. L'ancoraggio `\\b` in testa resta: e' cio' che impedisce
        a `vir` di contare dentro *servire*, il difetto del conteggio per
        sottostringa che sostituisce.
        """
        kw = keyword.strip().lower()
        if len(kw) < 3:
            # Keyword troppo corte generano rumore in qualsiasi modo le si tratti.
            return None

        if " " in kw:
            # Multi-parola: whitespace flessibile, nessuna flessione.
            escaped = r"\s+".join(re.escape(part) for part in kw.split())
            return re.compile(rf"\b{escaped}\b", re.IGNORECASE)

        if len(kw) >= 6:
            stem, suffix_len = kw[:-2], 4
        elif len(kw) >= 5:
            stem, suffix_len = kw[:-1], 3
        else:
            # 3-4 caratteri: la radice coinciderebbe con un frammento troppo
            # generico, quindi si resta sulla forma piena.
            stem, suffix_len = kw, 3

        return re.compile(rf"\b{re.escape(stem)}\w{{0,{suffix_len}}}\b", re.IGNORECASE)

    @staticmethod
    def _flatten_keywords(keywords: Any) -> List[str]:
        """Accetta sia {lang: [...]} sia [...] piatta."""
        if isinstance(keywords, dict):
            out: List[str] = []
            for kw_list in keywords.values():
                if isinstance(kw_list, list):
                    out.extend(str(k) for k in kw_list)
            return out
        if isinstance(keywords, list):
            return [str(k) for k in keywords]
        return []

    # --------------------------------------------------------------- analisi

    def analyze(
        self,
        segments: List[TextSegment],
        dict_name: str = "roman_values",
    ) -> ValueProfile:
        if not self.value_dicts:
            logger.error("Nessun dizionario di valori caricato: profilo valori vuoto.")
            return ValueProfile(
                value_distribution={}, dominant_values=[],
                theme_clusters=[], example_passages={},
            )

        if dict_name not in self.value_dicts:
            fallback = sorted(self.value_dicts.keys())[0]
            logger.warning(
                f"Dizionario '{dict_name}' non trovato. Uso '{fallback}'."
            )
            dict_name = fallback

        value_dict = self.value_dicts[dict_name]

        value_counts: Dict[str, float] = {}
        raw_counts: Dict[str, int] = {}
        examples: Dict[str, List[str]] = {}
        # Traccia quali segmenti toccano quale valore: base per i theme cluster.
        segments_by_value: Dict[str, List[str]] = defaultdict(list)

        for value_name, value_data in value_dict.items():
            if not isinstance(value_data, dict):
                continue
            patterns = [
                p for p in (
                    self._compile_keyword(kw)
                    for kw in self._flatten_keywords(value_data.get("keywords", {}))
                ) if p is not None
            ]
            if not patterns:
                continue

            count = 0
            value_examples: List[str] = []

            for segment in segments:
                content = segment.content
                segment_hits = 0
                for pattern in patterns:
                    for match in pattern.finditer(content):
                        segment_hits += 1
                        if len(value_examples) < MAX_EXAMPLES_PER_VALUE:
                            start = max(0, match.start() - CONTEXT_BEFORE)
                            end = min(len(content), match.end() + CONTEXT_AFTER)
                            snippet = content[start:end].replace("\n", " ").strip()
                            value_examples.append(f"...{snippet}...")
                if segment_hits:
                    count += segment_hits
                    segments_by_value[value_name].append(segment.id)

            if count > 0:
                raw_counts[value_name] = count
                weight = float(value_data.get("weight", 1.0))
                value_counts[value_name] = count * weight
                examples[value_name] = value_examples

        total = sum(value_counts.values())
        distribution = (
            {k: round(v / total * 100, 2) for k, v in value_counts.items()}
            if total > 0 else {}
        )
        dominant = [
            name for name, _ in
            sorted(distribution.items(), key=lambda kv: -kv[1])[:5]
        ]

        theme_clusters = [
            {
                "value": name,
                "raw_occurrences": raw_counts.get(name, 0),
                "segments_touched": len(set(segments_by_value.get(name, []))),
                "description": value_dict.get(name, {}).get("description", ""),
            }
            for name in dominant
        ]

        return ValueProfile(
            value_distribution=distribution,
            dominant_values=dominant,
            theme_clusters=theme_clusters,
            example_passages=examples,
        )
