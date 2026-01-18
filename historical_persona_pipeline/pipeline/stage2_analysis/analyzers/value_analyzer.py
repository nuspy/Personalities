from typing import List, Dict
from pathlib import Path
import json
import logging
from collections import Counter

from ...pipeline.data_models import ValueProfile, TextSegment

logger = logging.getLogger(__name__)

class ValueAnalyzer:
    """Analyzes text against a set of value dictionaries."""
    
    def __init__(self, data_dir: Path):
        self.value_dicts = self._load_value_dicts(data_dir / "value_dictionaries")
    
    def _load_value_dicts(self, path: Path) -> Dict:
        dicts = {}
        if path.exists():
            for f in path.glob("*.json"):
                try:
                    with open(f, 'r', encoding='utf-8') as file:
                        dicts[f.stem] = json.load(file)
                except Exception as e:
                    logger.error(f"Error loading value dictionary {f}: {e}")
        return dicts
    
    def analyze(self, segments: List[TextSegment], dict_name: str = "roman_values") -> ValueProfile:
        # Default to roman_values if not found, or use first available if roman_values missing
        if dict_name not in self.value_dicts:
            if self.value_dicts:
                dict_name = list(self.value_dicts.keys())[0]
                logger.warning(f"Value dictionary '{dict_name}' not found. Using '{dict_name}' instead.")
            else:
                logger.error("No value dictionaries found.")
                return ValueProfile(value_distribution={}, dominant_values=[], theme_clusters=[], example_passages={})
        
        value_dict = self.value_dicts[dict_name]
        
        # Combine all text for analysis (could be optimized)
        combined_text = " ".join(seg.content for seg in segments).lower()
        
        value_counts = {}
        examples = {}
        
        for value_name, value_data in value_dict.items():
            keywords = value_data.get('keywords', {})
            # Handle multi-language keywords structure
            all_keywords = []
            if isinstance(keywords, dict):
                for kw_list in keywords.values():
                    if isinstance(kw_list, list):
                        all_keywords.extend(kw_list)
            elif isinstance(keywords, list):
                all_keywords = keywords
            
            # Simple keyword counting
            # Ideally use lemmatization but raw string matching is fast/robust for now
            count = 0
            found_keywords = set()
            
            for kw in all_keywords:
                kw_lower = kw.lower()
                matches = combined_text.count(kw_lower)
                if matches > 0:
                    count += matches
                    found_keywords.add(kw_lower)
            
            if count > 0:
                weighted_count = count * value_data.get('weight', 1.0)
                value_counts[value_name] = weighted_count
                
                # Find example context (simple window)
                # Just take one example for now
                try:
                    kw_idx = combined_text.find(next(iter(found_keywords)))
                    start = max(0, kw_idx - 50)
                    end = min(len(combined_text), kw_idx + 100)
                    examples[value_name] = ["..." + combined_text[start:end].replace('\n', ' ') + "..."]
                except:
                    pass
        
        total = sum(value_counts.values())
        distribution = {k: round(v/total*100, 2) for k, v in value_counts.items()} if total > 0 else {}
        dominant = sorted(distribution.items(), key=lambda x: -x[1])[:5]
        
        return ValueProfile(
            value_distribution=distribution,
            dominant_values=[v[0] for v in dominant],
            theme_clusters=[],
            example_passages=examples
        )
