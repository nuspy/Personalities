"""Rilevamento della lingua, incluse quelle antiche.

`langdetect` copre solo le lingue moderne: latino e greco antico vengono
riconosciuti con euristiche dedicate prima di interpellarlo. Su testi lunghi
si vota su piu' campioni invece di analizzare tutto, che sarebbe lento e non
piu' accurato.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Tuple

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

from ..data_models import SupportedLanguage

logger = logging.getLogger(__name__)

# Risultati deterministici fra esecuzioni.
DetectorFactory.seed = 42

# Sotto questa soglia qualsiasi verdetto sarebbe rumore.
MIN_CHARS = 20

# Testi piu' lunghi vengono campionati in finestre su cui si vota.
SAMPLE_WINDOW = 1500
MAX_SAMPLES = 5


class LanguageDetector:
    LANGDETECT_MAP = {
        'it': SupportedLanguage.ITALIAN,
        'en': SupportedLanguage.ENGLISH,
        'fr': SupportedLanguage.FRENCH,
        'de': SupportedLanguage.GERMAN,
        'es': SupportedLanguage.SPANISH,
        'pt': SupportedLanguage.PORTUGUESE,
        'ru': SupportedLanguage.RUSSIAN,
        'uk': SupportedLanguage.UKRAINIAN,
        'el': SupportedLanguage.GREEK_MODERN,
        'hu': SupportedLanguage.HUNGARIAN,
    }

    # Diacritici politonici: assenti dal greco moderno monotonico.
    ANCIENT_GREEK_MARKERS = [
        'ἀ', 'ἐ', 'ἠ', 'ἰ', 'ὀ', 'ὐ', 'ὠ', 'ᾶ', 'ῆ', 'ῖ', 'ῦ', 'ῶ', 'ῃ', 'ῳ',
        'ἄ', 'ἔ', 'ἤ', 'ἴ', 'ὄ', 'ὔ', 'ὤ', 'ἅ', 'ἕ', 'ἥ', 'ἱ', 'ὅ',
    ]

    # Congiunzioni e desinenze latine ad alta frequenza.
    LATIN_MARKERS = [
        ' et ', ' sed ', ' cum ', ' quod ', ' quia ', ' autem ', ' enim ', ' nam ',
        ' atque ', ' neque ', ' igitur ', ' tamen ', ' ergo ', ' vero ',
        'atur', 'itur', 'untur', 'orum', 'arum', 'ibus', 'que ', 'ique ',
    ]

    def detect(self, text: str) -> Tuple[SupportedLanguage, float]:
        text_clean = text.strip()
        if len(text_clean) < MIN_CHARS:
            return SupportedLanguage.UNKNOWN, 0.0

        greek = self._score_ancient_greek(text_clean)
        if greek is not None:
            return greek

        latin = self._score_latin(text_clean)
        if latin is not None:
            return latin

        return self._detect_modern(text_clean)

    # ------------------------------------------------------------ euristiche

    def _score_ancient_greek(self, text: str) -> Tuple[SupportedLanguage, float] | None:
        hits = sum(1 for marker in self.ANCIENT_GREEK_MARKERS if marker in text)
        if hits >= 3:
            return SupportedLanguage.GREEK_ANCIENT, min(0.99, 0.6 + hits * 0.05)
        return None

    def _score_latin(self, text: str) -> Tuple[SupportedLanguage, float] | None:
        lowered = text.lower()
        hits = sum(1 for marker in self.LATIN_MARKERS if marker in lowered)
        if hits < 4:
            return None

        confidence = min(0.95, 0.5 + hits * 0.05)

        # Una lingua romanza moderna riconosciuta con altissima confidenza
        # indica piu' probabilmente un testo moderno che cita il latino.
        try:
            modern = detect_langs(text)
            if modern and modern[0].prob >= 0.99 and modern[0].lang in ("it", "es", "pt", "fr"):
                return None
        except LangDetectException:
            pass
        except Exception as exc:
            logger.debug(f"Controllo lingue moderne fallito: {exc}")

        return SupportedLanguage.LATIN, confidence

    # -------------------------------------------------------------- moderne

    def _detect_modern(self, text: str) -> Tuple[SupportedLanguage, float]:
        samples = self._samples(text)
        votes: Counter[str] = Counter()
        confidence_sum: dict[str, float] = {}

        for sample in samples:
            try:
                results = detect_langs(sample)
            except LangDetectException:
                continue
            except Exception as exc:
                logger.debug(f"langdetect fallito: {exc}")
                continue
            if not results:
                continue
            best = results[0]
            votes[best.lang] += 1
            confidence_sum[best.lang] = confidence_sum.get(best.lang, 0.0) + best.prob

        if not votes:
            return SupportedLanguage.UNKNOWN, 0.0

        lang_code, count = votes.most_common(1)[0]
        if lang_code not in self.LANGDETECT_MAP:
            return SupportedLanguage.UNKNOWN, 0.0

        mean_confidence = confidence_sum[lang_code] / count
        # La concordanza fra campioni e' essa stessa un segnale di affidabilita'.
        agreement = count / len(samples) if samples else 1.0
        return self.LANGDETECT_MAP[lang_code], round(mean_confidence * agreement, 4)

    @staticmethod
    def _samples(text: str) -> list[str]:
        if len(text) <= SAMPLE_WINDOW:
            return [text]
        step = max(1, (len(text) - SAMPLE_WINDOW) // max(1, MAX_SAMPLES - 1))
        return [
            text[i:i + SAMPLE_WINDOW]
            for i in range(0, len(text) - SAMPLE_WINDOW + 1, step)
        ][:MAX_SAMPLES]
