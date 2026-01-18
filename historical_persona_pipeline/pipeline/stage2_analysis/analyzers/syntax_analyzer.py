from typing import List, Dict
from collections import Counter
import numpy as np
import logging

from ...pipeline.data_models import SyntacticProfile, TextSegment
from ..nlp_manager import NLPManager

logger = logging.getLogger(__name__)

class SyntaxAnalyzer:
    def __init__(self, nlp_manager: NLPManager):
        self.nlp_manager = nlp_manager
    
    def analyze(self, segments: List[TextSegment]) -> SyntacticProfile:
        all_results = []
        
        # Group by language to minimize model switching
        by_language = {}
        for seg in segments:
            by_language.setdefault(seg.language, []).append(seg)
        
        for language, lang_segments in by_language.items():
            parser = self.nlp_manager.get_parser(language)
            
            for i, segment in enumerate(lang_segments):
                try:
                    # Log progress for large batches?
                    result = parser.analyze_syntax(segment.content)
                    all_results.append(result)
                except Exception as e:
                    logger.warning(f"Syntax analysis failed for segment {segment.id}: {e}")
        
        return self._aggregate_results(all_results)
    
    def _aggregate_results(self, results: List[Dict]) -> SyntacticProfile:
        all_lengths = []
        subordinate, coordinate = 0, 0
        person_totals = Counter()
        
        total_sentences = 0
        
        for result in results:
            for sent in result.get('sentences', []):
                all_lengths.append(sent.get('length', 0))
                if sent.get('has_subordinate'):
                    subordinate += 1
                else:
                    coordinate += 1
                total_sentences += 1
            
            for person, count in result.get('person_counts', {}).items():
                person_totals[person] += count
        
        total_clauses = subordinate + coordinate
        total_verbs = sum(person_totals.values())
        
        return SyntacticProfile(
            avg_sentence_length=float(round(np.mean(all_lengths), 2)) if all_lengths else 0.0,
            sentence_length_std=float(round(np.std(all_lengths), 2)) if all_lengths else 0.0,
            subordination_ratio=float(round(subordinate / total_sentences, 3)) if total_sentences > 0 else 0.0,
            coordination_ratio=float(round(coordinate / total_sentences, 3)) if total_sentences > 0 else 0.0,
            person_distribution={k: float(v/total_verbs) for k, v in person_totals.items()} if total_verbs > 0 else {},
            common_pos_patterns=[], # To be implemented with more detail
            special_patterns={}
        )
