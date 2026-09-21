from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime

class SupportedLanguage(str, Enum):
    ITALIAN = "it"
    ENGLISH = "en"
    FRENCH = "fr"
    GERMAN = "de"
    SPANISH = "es"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    UKRAINIAN = "uk"
    HUNGARIAN = "hu"
    GREEK_MODERN = "el"
    GREEK_ANCIENT = "grc"
    LATIN = "la"
    UNKNOWN = "unknown"

class FileFormat(str, Enum):
    TXT = "txt"
    MARKDOWN = "markdown"
    PDF = "pdf"
    DOC = "doc"
    DOCX = "docx"
    RTF = "rtf"
    ODT = "odt"
    EPUB = "epub"
    HTML = "html"
    XML_TEI = "xml_tei"
    WEB = "web"
    UNKNOWN = "unknown"

class TextSegment(BaseModel):
    id: str
    content: str
    language: SupportedLanguage
    language_confidence: float = Field(ge=0.0, le=1.0)
    source_file: str
    source_format: FileFormat
    page_or_section: Optional[str] = None
    word_count: int
    char_count: int

class IngestionResult(BaseModel):
    project_id: str
    timestamp: datetime
    total_files_processed: int
    total_segments: int
    segments: List[TextSegment]
    language_distribution: Dict[SupportedLanguage, int]
    errors: List[Dict[str, Any]] = []

class SyntacticProfile(BaseModel):
    avg_sentence_length: float
    sentence_length_std: float
    subordination_ratio: float
    coordination_ratio: float
    person_distribution: Dict[str, float]
    common_pos_patterns: List[Any]  # Can be list of tuples or strings depending on serialization
    special_patterns: Dict[str, int]
    # False quando nessun modello morfologico era disponibile: i rapporti di
    # subordinazione valgono 0 perche' non misurabili, non perche' assenti.
    # Senza questo flag il profilo dichiarerebbe "periodo paratattico" a un
    # autore che non e' stato analizzato affatto.
    morphology_available: bool = True

class VocabularyProfile(BaseModel):
    distinctive_terms: List[Dict[str, Any]]
    semantic_field_distribution: Dict[str, float]
    hapax_legomena_ratio: float
    type_token_ratio: float
    formulas_and_collocations: List[str]

class ValueProfile(BaseModel):
    value_distribution: Dict[str, float]
    dominant_values: List[str]
    theme_clusters: List[Dict[str, Any]]
    example_passages: Dict[str, List[str]]

class RhetoricalProfile(BaseModel):
    argument_type_distribution: Dict[str, float]
    persuasion_techniques: List[Dict[str, Any]]
    narrative_perspective: str
    self_reference_patterns: List[str]

class VoiceProfile(BaseModel):
    """Come il personaggio parla: ritmo, incipit, vocativi, modalita'.

    Dimensione assente dal profilo originale, che descriveva la struttura
    delle frasi ma non i tratti per cui una voce si riconosce a orecchio.
    """
    signature_openers: List[Dict[str, Any]] = []
    vocatives: List[Dict[str, Any]] = []
    question_ratio: float = 0.0
    exclamation_ratio: float = 0.0
    parenthetical_per_1000_words: float = 0.0
    direct_speech_per_1000_words: float = 0.0
    avg_word_length: float = 0.0
    modality_distribution: Dict[str, float] = {}
    speech_register: str = "unknown"


class KnowledgeBounds(BaseModel):
    era_start: str
    era_end: str
    known_topics: List[str]
    anachronistic_concepts: List[str]

class CorpusStats(BaseModel):
    """Dimensione effettiva del materiale analizzato.

    Serve a interpretare il profilo: gli stessi numeri hanno un peso molto
    diverso se estratti da 2.000 o da 2.000.000 di parole.
    """
    total_files: int = 0
    total_segments: int = 0
    total_words: int = 0
    language_distribution: Dict[str, int] = {}
    sources: List[str] = []


class CompleteStyleProfile(BaseModel):
    author_name: str
    era: str
    primary_language: SupportedLanguage
    syntax: SyntacticProfile
    vocabulary: VocabularyProfile
    values: ValueProfile
    rhetoric: RhetoricalProfile
    voice: VoiceProfile = VoiceProfile()
    knowledge: Optional[KnowledgeBounds] = None
    corpus_stats: CorpusStats = CorpusStats()
    generated_system_prompt: str
    training_guidelines: Dict[str, List[str]] = {}
    generated_at: Optional[datetime] = None

class TrainingConversation(BaseModel):
    id: str
    conversation_type: str
    system_prompt: str
    turns: List[Dict[str, str]]
    metadata: Dict[str, Any] = {}

class TrainingDataset(BaseModel):
    project_id: str
    timestamp: datetime
    author_name: str
    total_conversations: int
    conversations: List[TrainingConversation]
    type_distribution: Dict[str, int]

class EvalMetric(BaseModel):
    name: str
    score: float
    max_score: float
    details: Dict[str, Any]

class EvaluationResult(BaseModel):
    project_id: str
    timestamp: datetime
    model_name: str
    metrics: List[EvalMetric]
    overall_score: float
    generated_samples: List[Dict[str, str]]
