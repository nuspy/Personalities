"""Normalizzazione del testo prima dell'analisi.

La versione precedente faceva `re.sub(r'\\s+', ' ')`, collassando anche i
confini di paragrafo: si perdeva la struttura che serve alla segmentazione e
alle statistiche di stile. Qui gli spazi si normalizzano *dentro* la riga,
mentre la riga vuota resta separatore di paragrafo.
"""
from __future__ import annotations

import re
import unicodedata

# Sillabazione di fine riga sopravvissuta all'estrazione.
_HYPHEN_BREAK = re.compile(r"(\w)[-­]\s*\n\s*(\w)")

# Spazi multipli all'interno di una riga (non i ritorni a capo).
_INLINE_SPACE = re.compile(r"[ \t  - ]+")

# Tre o piu' ritorni a capo -> separatore di paragrafo unico.
_EXCESS_NEWLINES = re.compile(r"\n{3,}")

# Spazio prima della punteggiatura, tipico dell'estrazione da PDF.
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")

# Virgolette e trattini tipografici -> forma ASCII, per non moltiplicare le
# varianti dello stesso token nel conteggio del vocabolario.
_QUOTE_MAP = {
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": "'",
    "–": "-", "—": "-", "−": "-",
    "…": "...",
}


class TextNormalizer:
    def __init__(self, normalize_quotes: bool = True, unicode_form: str = "NFC"):
        self.normalize_quotes = normalize_quotes
        self.unicode_form = unicode_form

    def normalize(self, text: str) -> str:
        if not text:
            return ""

        # NFC ricompone i diacritici: senza questo "à" precomposta e "a"+accento
        # combinante risultano due token diversi. Va fatto prima di tutto il resto.
        text = unicodedata.normalize(self.unicode_form, text)

        text = _HYPHEN_BREAK.sub(r"\1\2", text)

        if self.normalize_quotes:
            for source, target in _QUOTE_MAP.items():
                text = text.replace(source, target)

        lines = [_INLINE_SPACE.sub(" ", line).strip() for line in text.split("\n")]
        text = "\n".join(lines)

        text = _EXCESS_NEWLINES.sub("\n\n", text)
        text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)

        return text.strip()
