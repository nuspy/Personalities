from typing import Dict, Any, List
from pathlib import Path
import logging

from ..stage_base import PipelineStage
from ..data_models import IngestionResult, CompleteStyleProfile, KnowledgeBounds, SupportedLanguage
from .nlp_manager import NLPManager
from .analyzers.syntax_analyzer import SyntaxAnalyzer
from .analyzers.vocabulary_analyzer import VocabularyAnalyzer
from .analyzers.value_analyzer import ValueAnalyzer
from .analyzers.rhetoric_analyzer import RhetoricAnalyzer
from .value_profile_generator import ValueProfileGenerator

logger = logging.getLogger(__name__)

class AnalysisStage(PipelineStage):
    def __init__(self, config: Dict[str, Any], project_dir: Path):
        super().__init__(config)
        self.project_dir = project_dir
        self.nlp_manager = NLPManager()
        
        # Initialize analyzers
        self.syntax_analyzer = SyntaxAnalyzer(self.nlp_manager)
        self.vocab_analyzer = VocabularyAnalyzer(self.nlp_manager)
        self.value_analyzer = ValueAnalyzer(project_dir / ".." / "data") # Adjust path as needed
        self.rhetoric_analyzer = RhetoricAnalyzer()
        
    def run(self, ingestion_result: IngestionResult) -> CompleteStyleProfile:
        self.progress_update.emit(0, "Starting linguistic analysis...")
        
        segments = ingestion_result.segments
        if not segments:
            raise ValueError("No segments to analyze")
            
        # Determine primary language
        # (Could be improved to handle mixed corpora)
        lang_counts = ingestion_result.language_distribution
        primary_lang = max(lang_counts, key=lang_counts.get) if lang_counts else SupportedLanguage.UNKNOWN
        
        # 1. Syntax Analysis
        self.progress_update.emit(20, "Analyzing syntax...")
        syntax_profile = self.syntax_analyzer.analyze(segments)
        
        # 2. Vocabulary Analysis
        self.progress_update.emit(40, "Analyzing vocabulary...")
        vocab_profile = self.vocab_analyzer.analyze(segments)
        
        # 3. Value Analysis
        self.progress_update.emit(60, "Analyzing values...")
        # Get configured value dict from config, default to roman
        value_dict_name = self.config.get("persona", {}).get("value_dict", "roman_values")
        value_profile = self.value_analyzer.analyze(segments, dict_name=value_dict_name)
        
        # 4. Rhetoric Analysis
        self.progress_update.emit(80, "Analyzing rhetoric...")
        rhetoric_profile = self.rhetoric_analyzer.analyze(segments)
        
        # 5. Generate System Prompt & Final Profile
        self.progress_update.emit(90, "Generating final profile...")
        
        # Basic knowledge bounds (placeholder)
        knowledge = KnowledgeBounds(
            era_start="Unknown",
            era_end="Unknown",
            known_topics=[],
            anachronistic_concepts=[]
        )
        
        # Generate a basic system prompt
        system_prompt = self._generate_system_prompt(
            self.config.get("persona", {}).get("author_name", "Unknown"),
            syntax_profile,
            value_profile
        )
        
        profile = CompleteStyleProfile(
            author_name=self.config.get("persona", {}).get("author_name", "Unknown"),
            era=self.config.get("persona", {}).get("era", "Unknown"),
            primary_language=primary_lang,
            syntax=syntax_profile,
            vocabulary=vocab_profile,
            values=value_profile,
            rhetoric=rhetoric_profile,
            knowledge=knowledge,
            generated_system_prompt=system_prompt,
            training_guidelines={}
        )
        
        self.progress_update.emit(100, "Analysis complete")
        self.stage_completed.emit(profile)
        return profile

    def _generate_system_prompt(self, author, syntax, values):
        # A simple template - normally this would be more sophisticated
        prompt = f"You are {author}.\n"
        prompt += f"Speak using sentences of approx {syntax.avg_sentence_length} words.\n"
        if values.dominant_values:
            prompt += f"Emphasize these values: {', '.join(values.dominant_values)}. \n"
        return prompt
