"""Analisi sintattica: la forma del periodo, cioe' il ritmo del parlato.

Correzioni: l'analisi ora passa dai lotti (una chiamata per segmento era
l'operazione piu' lenta della pipeline) e la coordinazione non e' piu'
calcolata come "tutto cio' che non e' subordinato" — erano due misure
indipendenti trattate come complementari, e la somma dei due rapporti faceva
sempre 1 per costruzione.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List

import numpy as np

from ...data_models import SupportedLanguage, SyntacticProfile, TextSegment
from ..nlp_manager import NLPManager

logger = logging.getLogger(__name__)

TOP_POS_PATTERNS = 15


class SyntaxAnalyzer:
    def __init__(self, nlp_manager: NLPManager, config: Dict[str, Any] | None = None):
        self.nlp_manager = nlp_manager
        analysis = (config or {}).get("analysis", {})
        self.max_chars_per_batch = analysis.get("max_chars_per_nlp_batch", 400_000)

    def analyze(self, segments: List[TextSegment]) -> SyntacticProfile:
        by_language: Dict[SupportedLanguage, List[TextSegment]] = {}
        for segment in segments:
            by_language.setdefault(segment.language, []).append(segment)

        all_results: List[Dict[str, Any]] = []

        for language, lang_segments in by_language.items():
            if language == SupportedLanguage.UNKNOWN:
                continue
            parser = self.nlp_manager.get_parser(language)

            for batch in self._batches(lang_segments):
                try:
                    all_results.extend(parser.analyze_syntax_batch([s.content for s in batch]))
                except Exception as exc:
                    logger.warning(f"Analisi sintattica fallita su un lotto: {exc}")

        return self._aggregate(all_results)

    def _batches(self, segments: List[TextSegment]) -> List[List[TextSegment]]:
        batches: List[List[TextSegment]] = []
        current: List[TextSegment] = []
        size = 0
        for segment in segments:
            if current and size + segment.char_count > self.max_chars_per_batch:
                batches.append(current)
                current, size = [], 0
            current.append(segment)
            size += segment.char_count
        if current:
            batches.append(current)
        return batches

    def _aggregate(self, results: List[Dict[str, Any]]) -> SyntacticProfile:
        lengths: List[int] = []
        subordinate = 0
        coordinate = 0
        questions = 0
        clause_counts: List[int] = []
        person_totals: Counter[str] = Counter()
        pos_patterns: Counter[tuple] = Counter()
        total_sentences = 0
        # Vero solo se almeno un lotto e' stato analizzato con un vero modello.
        morphology_available = False

        for result in results:
            morphology_available = morphology_available or result.get("has_morphology", False)
            for sentence in result.get("sentences", []):
                lengths.append(sentence.get("length", 0))
                total_sentences += 1
                if sentence.get("has_subordinate"):
                    subordinate += 1
                if sentence.get("has_coordination"):
                    coordinate += 1
                if sentence.get("is_question"):
                    questions += 1
                if sentence.get("clause_count"):
                    clause_counts.append(sentence["clause_count"])

            for person, count in result.get("person_counts", {}).items():
                person_totals[person] += count

            pos_patterns.update(result.get("pos_sequences", []))

        total_verbs = sum(person_totals.values())

        special_patterns = {
            "interrogative_sentences": questions,
            "total_sentences": total_sentences,
            "long_sentences_over_40_words": sum(1 for l in lengths if l > 40),
            "short_sentences_under_8_words": sum(1 for l in lengths if 0 < l < 8),
        }

        return SyntacticProfile(
            avg_sentence_length=self._round_mean(lengths),
            sentence_length_std=float(round(np.std(lengths), 2)) if lengths else 0.0,
            # Subordinazione e coordinazione sono misurate separatamente:
            # una frase puo' avere entrambe, o nessuna delle due.
            subordination_ratio=self._ratio(subordinate, total_sentences),
            coordination_ratio=self._ratio(coordinate, total_sentences),
            person_distribution=(
                {k: round(v / total_verbs, 4) for k, v in person_totals.items()}
                if total_verbs else {}
            ),
            common_pos_patterns=[
                {"pattern": " ".join(pattern), "count": count}
                for pattern, count in pos_patterns.most_common(TOP_POS_PATTERNS)
            ],
            special_patterns=special_patterns,
            morphology_available=morphology_available,
        )

    @staticmethod
    def _round_mean(values: List[int]) -> float:
        return float(round(np.mean(values), 2)) if values else 0.0

    @staticmethod
    def _ratio(part: int, total: int) -> float:
        return float(round(part / total, 3)) if total else 0.0
