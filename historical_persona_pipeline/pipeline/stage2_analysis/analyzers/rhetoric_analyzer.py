"""Analisi retorica: come il personaggio costruisce un ragionamento.

Sostituisce tre liste di marcatori scritte nel codice, solo inglesi e
italiane e contate per sottostringa (`"since"` contava dentro *sincerity*).
Ora i marcatori arrivano da `data/rhetoric_markers.json`, coprono anche
latino, greco antico, tedesco, russo e ungherese, e il conteggio e'
normalizzato ogni mille parole — altrimenti un corpus piu' lungo sembrerebbe
sempre piu' argomentativo di uno breve.

Il profilo registra anche la prospettiva narrativa: se il personaggio dice
"io" o parla di se' in terza persona (il tratto piu' riconoscibile di Cesare)
e' la prima cosa che un lettore nota nella voce.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

from ....paths import DATA_DIR
from ...data_models import RhetoricalProfile, SupportedLanguage, TextSegment

logger = logging.getLogger(__name__)

MARKERS_PATH = DATA_DIR / "rhetoric_markers.json"

# Pronomi di prima persona per lingua: base della prospettiva narrativa.
FIRST_PERSON_MARKERS = {
    "it": [r"\bio\b", r"\bmi\b", r"\bmio\b", r"\bmia\b", r"\bnoi\b", r"\bnostro\b"],
    "en": [r"\bi\b", r"\bme\b", r"\bmy\b", r"\bmine\b", r"\bwe\b", r"\bour\b"],
    "la": [r"\bego\b", r"\bmihi\b", r"\bme\b", r"\bmeus\b", r"\bnos\b", r"\bnoster\b", r"\bnostri\b"],
    "grc": [r"εγω", r"μοι", r"εμου", r"ημεις", r"ημων"],
    "fr": [r"\bje\b", r"\bmoi\b", r"\bmon\b", r"\bnous\b", r"\bnotre\b"],
    "de": [r"\bich\b", r"\bmir\b", r"\bmein\b", r"\bwir\b", r"\bunser\b"],
    "es": [r"\byo\b", r"\bmi\b", r"\bnosotros\b", r"\bnuestro\b"],
    "ru": [r"\bя\b", r"\bмне\b", r"\bмой\b", r"\bмы\b", r"\bнаш\b"],
    "hu": [r"\bén\b", r"\bnekem\b", r"\bmi\b", r"\bmiénk\b"],
}


@lru_cache(maxsize=1)
def load_markers() -> Dict[str, Dict[str, List[str]]]:
    if not MARKERS_PATH.exists():
        logger.error(f"Marcatori retorici non trovati: {MARKERS_PATH}")
        return {}
    try:
        with open(MARKERS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        logger.error(f"Marcatori retorici illeggibili: {exc}")
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_")}


class RhetoricAnalyzer:
    def __init__(self, author_name: str = ""):
        self.author_name = author_name.strip()
        self.markers = load_markers()

    def analyze(self, segments: List[TextSegment]) -> RhetoricalProfile:
        if not segments:
            return RhetoricalProfile(
                argument_type_distribution={}, persuasion_techniques=[],
                narrative_perspective="unknown", self_reference_patterns=[],
            )

        counts: Counter[str] = Counter()
        examples: Dict[str, List[str]] = defaultdict(list)
        total_words = 0

        languages = {seg.language for seg in segments}
        patterns = self._compile_patterns(languages)

        for segment in segments:
            content = segment.content
            lowered = content.lower()
            total_words += segment.word_count

            for category, category_patterns in patterns.items():
                for pattern in category_patterns:
                    found = pattern.findall(lowered)
                    if not found:
                        continue
                    counts[category] += len(found)
                    if len(examples[category]) < 2:
                        match = pattern.search(content)
                        if match:
                            start = max(0, match.start() - 60)
                            end = min(len(content), match.end() + 120)
                            examples[category].append(content[start:end].replace("\n", " "))

        total_markers = sum(counts.values())
        distribution = (
            {k: round(v / total_markers * 100, 2) for k, v in counts.items()}
            if total_markers else {}
        )

        # Densita' ogni mille parole: confrontabile fra corpora di lunghezza diversa.
        per_thousand = (
            {k: round(v / total_words * 1000, 3) for k, v in counts.items()}
            if total_words else {}
        )

        techniques = [
            {
                "type": category,
                "occurrences": counts[category],
                "per_1000_words": per_thousand.get(category, 0.0),
                "examples": examples.get(category, []),
            }
            for category, _ in counts.most_common()
        ]

        perspective, self_refs = self._narrative_perspective(segments)

        return RhetoricalProfile(
            argument_type_distribution=distribution,
            persuasion_techniques=techniques,
            narrative_perspective=perspective,
            self_reference_patterns=self_refs,
        )

    # ------------------------------------------------------------ marcatori

    def _compile_patterns(
        self, languages: Sequence[SupportedLanguage]
    ) -> Dict[str, List[re.Pattern]]:
        """Compila solo i marcatori delle lingue presenti nel corpus."""
        lang_codes = {lang.value for lang in languages}
        compiled: Dict[str, List[re.Pattern]] = {}

        for category, by_language in self.markers.items():
            patterns: List[re.Pattern] = []
            for lang_code, marker_list in by_language.items():
                if lang_code not in lang_codes:
                    continue
                for marker in marker_list:
                    escaped = r"\s+".join(re.escape(p) for p in marker.lower().split())
                    patterns.append(re.compile(rf"\b{escaped}\b", re.IGNORECASE))
            if patterns:
                compiled[category] = patterns

        return compiled

    def _build_author_pattern(self, segments: List[TextSegment]) -> Optional[re.Pattern]:
        """Regex che intercetta il nome dell'autore *come compare nel corpus*.

        Il nome fornito dall'utente e quello attestato nei testi spesso non
        coincidono: si scrive "Gaio Giulio Cesare" e il corpus latino dice
        *Caesar*, quello greco Καῖσαρ. Cercare la stringa esatta fa mancare
        del tutto le auto-citazioni, e con esse il tratto piu' riconoscibile
        della voce — parlare di se' in terza persona.

        Si cercano quindi, fra le parole capitalizzate del corpus, quelle
        ortograficamente vicine a un token del nome, e si costruisce il
        pattern sulle forme trovate davvero.
        """
        if not self.author_name:
            return None

        name_tokens = [t for t in re.split(r"\s+", self.author_name) if len(t) >= 4]
        if not name_tokens:
            return None

        try:
            from rapidfuzz import fuzz
        except ImportError:
            # Senza rapidfuzz si resta sulla corrispondenza esatta del cognome.
            surname = name_tokens[-1]
            return re.compile(rf"\b{re.escape(surname)}\w{{0,3}}\b", re.IGNORECASE)

        # Campione del corpus: bastano poche migliaia di parole per trovare
        # le forme ricorrenti del nome.
        sample = " ".join(seg.content for seg in segments[:20])[:200_000]
        candidates = Counter(re.findall(r"\b[A-ZΑ-ΩА-Я][\w'-]{3,}\b", sample))

        matched_forms: set[str] = set()
        for form, count in candidates.most_common(400):
            if count < 2:
                continue
            for token in name_tokens:
                # 80: tollera Cesare/Caesar e le desinenze dei casi latini,
                # senza confondere nomi diversi che iniziano uguale.
                if fuzz.ratio(form.lower(), token.lower()) >= 80:
                    matched_forms.add(form.lower())
                    break

        if not matched_forms:
            return None

        logger.debug(
            f"Forme del nome attestate nel corpus: {', '.join(sorted(matched_forms))}"
        )
        alternatives = "|".join(re.escape(f) for f in sorted(matched_forms, key=len, reverse=True))
        return re.compile(rf"\b(?:{alternatives})\w{{0,3}}\b", re.IGNORECASE)

    # ---------------------------------------------------------- prospettiva

    def _narrative_perspective(
        self, segments: List[TextSegment]
    ) -> tuple[str, List[str]]:
        """Prima persona, terza persona, o terza persona riferita a se'.

        Un autore che parla di se' in terza persona (*Caesar imperavit*) e'
        il caso che piu' facilmente sfugge a un modello addestrato senza
        questa indicazione esplicita.
        """
        first_person_hits = 0
        author_mentions = 0
        self_refs: List[str] = []

        author_pattern = self._build_author_pattern(segments)

        total_words = 0
        for segment in segments:
            lowered = segment.content.lower()
            total_words += segment.word_count

            for pattern_str in FIRST_PERSON_MARKERS.get(segment.language.value, []):
                first_person_hits += len(re.findall(pattern_str, lowered, re.IGNORECASE))

            if author_pattern:
                matches = author_pattern.findall(segment.content)
                author_mentions += len(matches)
                if matches and len(self_refs) < 5:
                    match = author_pattern.search(segment.content)
                    if match:
                        start = max(0, match.start() - 40)
                        end = min(len(segment.content), match.end() + 100)
                        self_refs.append(segment.content[start:end].replace("\n", " "))

        if not total_words:
            return "unknown", []


        first_density = first_person_hits / total_words * 1000
        author_density = author_mentions / total_words * 1000

        if author_density > 1.0 and author_density > first_density:
            perspective = "third_person_self_reference"
        elif first_density > 5.0:
            perspective = "first_person"
        elif first_density > 1.0:
            perspective = "mixed"
        else:
            perspective = "third_person"

        return perspective, self_refs
