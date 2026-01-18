import re

class TextNormalizer:
    def normalize(self, text: str) -> str:
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize quotes
        text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        
        # Fix common OCR errors (very basic)
        text = text.replace(' .', '.')
        
        # Roman numerals normalization could go here but might be risky
        
        return text
