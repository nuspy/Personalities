from typing import Dict, Any, Optional
import logging
from ..data_models import SupportedLanguage

logger = logging.getLogger(__name__)

class NLPManager:
    """Manages NLP models for multiple languages with lazy loading."""
    
    SPACY_MODELS = {
        SupportedLanguage.ITALIAN: "it_core_news_lg",
        SupportedLanguage.ENGLISH: "en_core_web_lg",
        SupportedLanguage.FRENCH: "fr_core_news_lg",
        SupportedLanguage.GERMAN: "de_core_news_lg",
        SupportedLanguage.SPANISH: "es_core_news_lg",
        SupportedLanguage.PORTUGUESE: "pt_core_news_lg",
        SupportedLanguage.RUSSIAN: "ru_core_news_lg",
        SupportedLanguage.UKRAINIAN: "uk_core_news_lg",
        SupportedLanguage.GREEK_MODERN: "el_core_news_lg",
    }
    
    CLTK_LANGUAGES = {
        SupportedLanguage.GREEK_ANCIENT: "grc",
        SupportedLanguage.LATIN: "lat",
    }
    
    def __init__(self):
        self._loaded_models: Dict[SupportedLanguage, Any] = {}
    
    def get_parser(self, language: SupportedLanguage):
        if language in self._loaded_models:
            return self._loaded_models[language]
        
        parser = self._load_parser(language)
        self._loaded_models[language] = parser
        return parser
    
    def _load_parser(self, language: SupportedLanguage):
        logger.info(f"Loading NLP model for {language}...")
        try:
            if language in self.CLTK_LANGUAGES:
                return self._load_cltk_parser(language)
            
            # Priority handling for Hungarian Transformer model (GPU optimized)
            if language == SupportedLanguage.HUNGARIAN:
                 return self._load_spacy_model_by_name("hu_core_news_trf", language)

            if language in self.SPACY_MODELS:
                return self._load_spacy_parser(language)
            
            # Fallback for unknown languages or those without specific models
            return BasicParser(language)
            
        except Exception as e:
            logger.error(f"Failed to load model for {language}: {e}")
            return BasicParser(language)

    def _load_spacy_model_by_name(self, model_name, language):
        import spacy
        try:
            nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"Model {model_name} not found. Please install it manually.")
            raise
        return SpacyWrapper(nlp, language)
    
    def _load_spacy_parser(self, language: SupportedLanguage):
        import spacy
        model_name = self.SPACY_MODELS[language]
        
        try:
            nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"Spacy model {model_name} not found. Attempting download...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            nlp = spacy.load(model_name)
        
        return SpacyWrapper(nlp, language)
    
    def _load_cltk_parser(self, language: SupportedLanguage):
        from cltk import NLP
        lang_code = self.CLTK_LANGUAGES[language]
        # Suppress interactive download prompts if possible or ensure data exists
        # Assuming data is pre-installed via setup script
        cltk_nlp = NLP(language=lang_code)
        return CLTKWrapper(cltk_nlp, language)
    
    def unload(self, language=None):
        if language:
            if language in self._loaded_models:
                del self._loaded_models[language]
                import gc
                gc.collect()
        else:
            self._loaded_models.clear()
            import gc
            gc.collect()


class BasicParser:
    """Fallback parser that does regex-based splitting."""
    def __init__(self, language):
        self.language = language
        
    def analyze_syntax(self, text: str) -> dict:
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return {
            'sentences': [{'length': len(s.split())} for s in sentences],
            'person_counts': {}, # Cannot determine without POS tagger
        }
        
    def extract_vocabulary(self, text: str) -> dict:
        from collections import Counter
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)
        return {
            'total_tokens': len(words),
            'unique_tokens': len(set(words)),
            'type_token_ratio': len(set(words)) / len(words) if words else 0,
            'word_frequencies': word_freq.most_common(100),
            'hapax_count': sum(1 for c in word_freq.values() if c == 1)
        }


class SpacyWrapper:
    def __init__(self, nlp, language):
        self.nlp = nlp
        self.language = language
    
    def analyze_syntax(self, text: str) -> dict:
        # Increase limit for large texts
        self.nlp.max_length = max(len(text) + 1000, self.nlp.max_length)
        
        doc = self.nlp(text)
        results = {
            'sentences': [],
            'person_counts': {'1': 0, '2': 0, '3': 0},
        }
        
        for sent in doc.sents:
            sent_data = {
                'length': len([t for t in sent if not t.is_punct]),
                'has_subordinate': any(t.dep_ in ['ccomp', 'xcomp', 'advcl', 'acl', 'relcl'] for t in sent),
            }
            results['sentences'].append(sent_data)
            
            for token in sent:
                if token.pos_ == 'VERB':
                    # Extract Person feature
                    person = token.morph.get('Person')
                    if person:
                        # Spacy returns list like ['1']
                        p_val = person[0]
                        if p_val in results['person_counts']:
                            results['person_counts'][p_val] += 1
        
        return results
    
    def extract_vocabulary(self, text: str) -> dict:
        from collections import Counter
        self.nlp.max_length = max(len(text) + 1000, self.nlp.max_length)
        doc = self.nlp(text)
        
        lemmas = [t.lemma_.lower() for t in doc if not t.is_punct and not t.is_space]
        lemma_freq = Counter(lemmas)
        
        return {
            'total_tokens': len(lemmas),
            'unique_lemmas': len(set(lemmas)),
            'type_token_ratio': len(set(lemmas)) / len(lemmas) if lemmas else 0,
            'lemma_frequencies': lemma_freq.most_common(100),
            'hapax_count': sum(1 for l, c in lemma_freq.items() if c == 1),
        }


class CLTKWrapper:
    def __init__(self, cltk_nlp, language):
        self.nlp = cltk_nlp
        self.language = language
    
    def analyze_syntax(self, text: str) -> dict:
        # CLTK analysis can be slow, handle with care
        try:
            doc = self.nlp.analyze(text)
            
            results = {
                'sentences': [],
                'person_counts': {'1': 0, '2': 0, '3': 0},
            }
            
            for sent in doc.sentences:
                # Word count approx
                sent_len = len(sent.words)
                results['sentences'].append({'length': sent_len})
                
                for word in sent.words:
                    if word.pos == 'verb': # CLTK POS tags might vary
                         # Feature extraction depends on CLTK version and language
                         if 'Person' in word.features:
                             p = str(word.features['Person'])
                             if p in results['person_counts']:
                                 results['person_counts'][p] += 1
            return results
            
        except Exception as e:
            logger.error(f"CLTK Analysis failed: {e}")
            return BasicParser(self.language).analyze_syntax(text)
    
    def extract_vocabulary(self, text: str) -> dict:
        # Fallback to basic for robustness if CLTK is heavy
        return BasicParser(self.language).extract_vocabulary(text)
