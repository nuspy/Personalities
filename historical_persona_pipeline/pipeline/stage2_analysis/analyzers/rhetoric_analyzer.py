from typing import List, Dict, Any
import logging
from ...pipeline.data_models import RhetoricalProfile, TextSegment

logger = logging.getLogger(__name__)

class RhetoricAnalyzer:
    """Analyzes rhetorical patterns and argumentation strategies."""
    
    # Very basic markers for demonstration
    ARGUMENT_MARKERS = {
        "causal": ["because", "since", "therefore", "thus", "poiché", "quindi"],
        "contrast": ["but", "however", "although", "ma", "tuttavia", "sebbene"],
        "authority": ["as stated", "according to", "come detto", "secondo"],
    }
    
    def analyze(self, segments: List[TextSegment]) -> RhetoricalProfile:
        combined_text = " ".join(seg.content for seg in segments).lower()
        
        counts = {k: 0 for k in self.ARGUMENT_MARKERS.keys()}
        
        for cat, markers in self.ARGUMENT_MARKERS.items():
            for m in markers:
                counts[cat] += combined_text.count(m)
        
        total = sum(counts.values())
        dist = {k: v/total for k, v in counts.items()} if total > 0 else {}
        
        # Self-reference detection (Cesare style vs First person)
        # This requires checking if author name appears in subject position vs "I/me"
        # For now, placeholder logic
        narrative_perspective = "mixed" 
        
        return RhetoricalProfile(
            argument_type_distribution=dist,
            persuasion_techniques=[],
            narrative_perspective=narrative_perspective,
            self_reference_patterns=[]
        )
