from typing import Tuple
import langdetect
from langdetect import detect_langs, DetectorFactory
from ..data_models import SupportedLanguage
import logging

logger = logging.getLogger(__name__)

# Enforce deterministic results
DetectorFactory.seed = 42

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
    }
    
    # Ancient Greek polytonic diacritics (distinctive from Modern Greek)
    ANCIENT_GREEK_MARKERS = ['ἀ', 'ἐ', 'ἠ', 'ἰ', 'ὀ', 'ὐ', 'ὠ', 'ᾶ', 'ῆ', 'ῖ', 'ῦ', 'ῶ', 'ῃ', 'ῳ']
    
    # Latin common patterns (stopwords/endings)
    LATIN_MARKERS = [
        ' et ', ' sed ', ' cum ', ' quod ', ' quia ', ' autem ', ' enim ', ' nam ',
        'atur', 'itur', 'untur', 'orum', 'arum', 'ibus', 'que '
    ]
    
    def detect(self, text: str) -> Tuple[SupportedLanguage, float]:
        text_clean = text.strip()
        if len(text_clean) < 20:
            return SupportedLanguage.UNKNOWN, 0.0
        
        # 1. Check Ancient Greek (strong heuristic)
        greek_score = sum(1 for marker in self.ANCIENT_GREEK_MARKERS if marker in text_clean)
        if greek_score >= 3:
            return SupportedLanguage.GREEK_ANCIENT, min(0.99, 0.6 + greek_score * 0.05)
        
        # 2. Check Latin (heuristic)
        text_lower = text_clean.lower()
        latin_score = sum(1 for marker in self.LATIN_MARKERS if marker in text_lower)
        if latin_score >= 4:
            # If it looks like Latin, verify it's not detected as Italian/French/Spanish by langdetect with high confidence
            try:
                modern_check = detect_langs(text_clean)
                if modern_check:
                    top_modern = modern_check[0]
                    # If langdetect is SUPER sure it's Italian (e.g. > 0.99), it might be Italian containing Latin quotes
                    # But usually Latin text confuses langdetect into low confidence results
                    if top_modern.prob < 0.9: 
                        return SupportedLanguage.LATIN, min(0.95, 0.5 + latin_score * 0.05)
            except:
                pass
            return SupportedLanguage.LATIN, min(0.95, 0.5 + latin_score * 0.05)
            
        # 3. Use langdetect for modern languages
        try:
            results = detect_langs(text_clean)
            if results:
                best = results[0]
                if best.lang in self.LANGDETECT_MAP:
                    return self.LANGDETECT_MAP[best.lang], best.prob
        except Exception as e:
            logger.debug(f"Langdetect failed: {e}")
            pass
        
        return SupportedLanguage.UNKNOWN, 0.0
