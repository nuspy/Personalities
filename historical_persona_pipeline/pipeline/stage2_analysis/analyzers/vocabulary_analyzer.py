from typing import List, Dict, Any
from collections import Counter
import logging

from ...pipeline.data_models import VocabularyProfile, TextSegment
from ..nlp_manager import NLPManager

logger = logging.getLogger(__name__)

class VocabularyAnalyzer:
    def __init__(self, nlp_manager: NLPManager):
        self.nlp_manager = nlp_manager
        
        # Could load semantic fields here
        self.semantic_fields = {
             "military": ["legion", "army", "sword", "battle", "general", "soldier", "war"],
             "politics": ["senate", "republic", "consul", "law", "people", "power", "state"],
        }
    
    def analyze(self, segments: List[TextSegment]) -> VocabularyProfile:
        # Group by language
        by_language = {}
        for seg in segments:
            by_language.setdefault(seg.language, []).append(seg)
        
        # For simplicity, we'll analyze the dominant language or aggregate stats naively
        # A more robust solution would analyze per language and then merge
        
        total_lemmas = []
        
        for language, lang_segments in by_language.items():
            parser = self.nlp_manager.get_parser(language)
            for segment in lang_segments:
                try:
                    result = parser.extract_vocabulary(segment.content)
                    # We can't easily merge counters without raw lemma lists, 
                    # but for this stage let's just re-extract or assume single large doc
                    # Optimization: Just process concatenated text per language?
                    pass
                except Exception as e:
                    logger.warning(f"Vocab analysis failed for segment: {e}")

        # Simpler approach: Concatenate all content per language and analyze once
        # This is better for global stats like Hapax Legomena
        
        aggregated_stats = {
            'total_tokens': 0,
            'unique_lemmas': 0,
            'hapax_count': 0,
            'frequencies': Counter()
        }
        
        for language, lang_segments in by_language.items():
            full_text = " ".join(s.content for s in lang_segments)
            parser = self.nlp_manager.get_parser(language)
            try:
                vocab_data = parser.extract_vocabulary(full_text)
                
                aggregated_stats['total_tokens'] += vocab_data['total_tokens']
                aggregated_stats['unique_lemmas'] += vocab_data['unique_lemmas']
                aggregated_stats['hapax_count'] += vocab_data.get('hapax_count', 0)
                
                # Merge frequencies (might be mixing languages, but okay for "distinctive terms")
                for term, count in vocab_data['lemma_frequencies']:
                    aggregated_stats['frequencies'][term] += count
                    
            except Exception as e:
                logger.error(f"Error processing vocabulary for language {language}: {e}")

        # Calculate metrics
        total = aggregated_stats['total_tokens']
        unique = aggregated_stats['unique_lemmas']
        ttr = unique / total if total > 0 else 0
        hapax_ratio = aggregated_stats['hapax_count'] / total if total > 0 else 0
        
        # Identify Distinctive Terms (naive freq > threshold)
        distinctive = [
            {'term': t, 'count': c, 'freq': c/total} 
            for t, c in aggregated_stats['frequencies'].most_common(50)
            if len(t) > 3 # Filter short stopwords roughly
        ]
        
        # Semantic Fields
        sem_distribution = {}
        # This part requires language-aware semantic field dictionaries which we have in data/
        # For now return placeholder
        
        return VocabularyProfile(
            distinctive_terms=distinctive,
            semantic_field_distribution=sem_distribution,
            hapax_legomena_ratio=float(round(hapax_ratio, 4)),
            type_token_ratio=float(round(ttr, 4)),
            formulas_and_collocations=[] # Needs n-gram analysis
        )
