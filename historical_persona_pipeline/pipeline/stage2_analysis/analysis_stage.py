"""Stage 2 — analisi: dal corpus al profilo di personalita'.

Coordina cinque analizzatori indipendenti (sintassi, lessico, voce, retorica,
valori) e ne sintetizza un prompt di sistema operativo.

Rispetto alla versione precedente: il percorso dei dizionari di valori e'
quello reale (prima puntava a una cartella inesistente e il profilo valori
usciva sempre vuoto), la lingua primaria e' pesata sulle parole e non sul
numero di segmenti, e il fallimento di un singolo analizzatore non azzera
l'intero profilo.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ...paths import DATA_DIR
from ..data_models import (
    CompleteStyleProfile,
    CorpusStats,
    IngestionResult,
    KnowledgeBounds,
    RhetoricalProfile,
    SupportedLanguage,
    SyntacticProfile,
    TextSegment,
    ValueProfile,
    VocabularyProfile,
    VoiceProfile,
)
from ..stage_base import PipelineStage
from .analyzers.rhetoric_analyzer import RhetoricAnalyzer
from .analyzers.syntax_analyzer import SyntaxAnalyzer
from .analyzers.value_analyzer import ValueAnalyzer
from .analyzers.vocabulary_analyzer import VocabularyAnalyzer
from .analyzers.voice_analyzer import VoiceAnalyzer
from .nlp_manager import NLPManager
from .persona_synthesizer import PersonaSynthesizer

logger = logging.getLogger(__name__)

# Sotto questa quota di parole provenienti da testi *del* personaggio, il
# profilo descrive soprattutto chi ha scritto le fonti.
MIN_PRIMARY_SOURCE_SHARE = 0.3


class AnalysisStage(PipelineStage):
    def __init__(self, config: Dict[str, Any], project_dir: Path):
        super().__init__(config)
        self.project_dir = Path(project_dir)
        self.nlp_manager = NLPManager()

        persona = config.get("persona", {})
        self.author_name = persona.get("author_name", "") or "Unknown"
        self.era = persona.get("era", "") or ""

        self.syntax_analyzer = SyntaxAnalyzer(self.nlp_manager, config)
        self.vocab_analyzer = VocabularyAnalyzer(self.nlp_manager, config)
        self.voice_analyzer = VoiceAnalyzer(config)
        self.rhetoric_analyzer = RhetoricAnalyzer(self.author_name)
        # Percorso canonico del package: non dipende dalla working directory
        # ne' dalla posizione della cartella di progetto.
        self.value_analyzer = ValueAnalyzer(DATA_DIR)
        self.synthesizer = PersonaSynthesizer(config)

    # ------------------------------------------------------------------ run

    def run(self, ingestion_result: IngestionResult) -> CompleteStyleProfile:
        segments = ingestion_result.segments
        if not segments:
            raise ValueError("Nessun segmento da analizzare")

        self.progress_update.emit(0, "Avvio analisi linguistica...")

        primary_lang = self._primary_language(segments)
        self.logger.info(
            f"Lingua primaria: {primary_lang.value} su {len(segments)} segmenti"
        )

        self._warn_if_no_primary_sources(ingestion_result)

        self.progress_update.emit(10, "Analisi della sintassi...")
        syntax = self._safe(
            lambda: self.syntax_analyzer.analyze(segments),
            SyntacticProfile(
                avg_sentence_length=0.0, sentence_length_std=0.0,
                subordination_ratio=0.0, coordination_ratio=0.0,
                person_distribution={}, common_pos_patterns=[], special_patterns={},
            ),
            "sintassi",
        )

        self.progress_update.emit(35, "Analisi del lessico...")
        vocabulary = self._safe(
            lambda: self.vocab_analyzer.analyze(segments),
            VocabularyProfile(
                distinctive_terms=[], semantic_field_distribution={},
                hapax_legomena_ratio=0.0, type_token_ratio=0.0,
                formulas_and_collocations=[],
            ),
            "lessico",
        )

        self.progress_update.emit(60, "Analisi della voce...")
        voice = self._safe(lambda: self.voice_analyzer.analyze(segments), VoiceProfile(), "voce")

        self.progress_update.emit(72, "Analisi dell'argomentazione...")
        rhetoric = self._safe(
            lambda: self.rhetoric_analyzer.analyze(segments),
            RhetoricalProfile(
                argument_type_distribution={}, persuasion_techniques=[],
                narrative_perspective="unknown", self_reference_patterns=[],
            ),
            "retorica",
        )

        self.progress_update.emit(84, "Analisi dei valori...")
        value_dict = self.config.get("persona", {}).get("value_dict", "auto")
        if value_dict == "auto":
            value_dict = self._pick_value_dict()
        values = self._safe(
            lambda: self.value_analyzer.analyze(segments, dict_name=value_dict),
            ValueProfile(
                value_distribution={}, dominant_values=[],
                theme_clusters=[], example_passages={},
            ),
            "valori",
        )

        self.progress_update.emit(92, "Sintesi del profilo...")

        profile = CompleteStyleProfile(
            author_name=self.author_name,
            era=self.era,
            primary_language=primary_lang,
            syntax=syntax,
            vocabulary=vocabulary,
            values=values,
            rhetoric=rhetoric,
            voice=voice,
            knowledge=self._knowledge_bounds(),
            corpus_stats=self._corpus_stats(ingestion_result),
            generated_system_prompt="",  # riempito subito sotto
            training_guidelines={},
            generated_at=datetime.now(),
        )

        profile.generated_system_prompt = self.synthesizer.build_system_prompt(profile)
        profile.training_guidelines = self.synthesizer.build_training_guidelines(profile)

        self._save(profile)

        self.progress_update.emit(100, "Analisi completata")
        self.stage_completed.emit(profile)
        return profile

    # -------------------------------------------------------------- supporto

    def _safe(self, operation, fallback, label: str):
        """Un analizzatore che fallisce non deve azzerare l'intero profilo."""
        try:
            return operation()
        except Exception as exc:
            self.logger.exception(f"Analisi '{label}' fallita")
            self.error_occurred.emit(f"Analisi '{label}' fallita: {exc}")
            return fallback

    @staticmethod
    def _primary_language(segments: List[TextSegment]) -> SupportedLanguage:
        """Lingua con piu' PAROLE, non con piu' segmenti."""
        words_by_lang: Dict[SupportedLanguage, int] = {}
        for segment in segments:
            if segment.language == SupportedLanguage.UNKNOWN:
                continue
            words_by_lang[segment.language] = (
                words_by_lang.get(segment.language, 0) + segment.word_count
            )
        if not words_by_lang:
            return SupportedLanguage.UNKNOWN
        return max(words_by_lang, key=words_by_lang.get)

    def _warn_if_no_primary_sources(self, ingestion_result: IngestionResult) -> None:
        """Avverte quando il corpus parla del personaggio invece di essere suo.

        E' la distinzione che decide se il profilo ha senso. Una voce
        enciclopedica *su* Seneca misura lo stile di chi l'ha scritta: frasi
        da manuale, il nome del soggetto come formula piu' frequente, nessuna
        traccia della voce di Seneca. Il profilo che ne esce e' formalmente
        valido e sostanzialmente falso, e senza questo avviso nulla lo
        distingue da uno costruito sui testi veri.
        """
        manifest_path = self.project_dir / "sources" / "online" / "manifest.json"
        if not manifest_path.exists():
            # Corpus caricato a mano: non sappiamo cosa contiene, e
            # presumerlo sarebbe peggio che tacere.
            return

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.logger.debug(f"Manifest delle fonti illeggibile: {exc}")
            return

        documents = manifest.get("documents", [])
        if not documents:
            return

        primary_words = sum(
            d.get("word_count", 0) for d in documents if d.get("is_primary_source")
        )
        total_words = sum(d.get("word_count", 0) for d in documents) or 1
        share = primary_words / total_words

        if share >= MIN_PRIMARY_SOURCE_SHARE:
            return

        self.error_occurred.emit(
            f"Avviso: solo il {share:.0%} del corpus e' costituito da testi "
            f"DEL personaggio; il resto parla DI lui. Il profilo misurera' in "
            f"buona parte lo stile di chi ha scritto quelle pagine, non quello "
            f"di {self.author_name}. Per un risultato utile servono le sue "
            f"opere: caricarle con --files, o verificare che la ricerca abbia "
            f"trovato fonti su Wikisource o Gutenberg."
        )

    def _pick_value_dict(self) -> str:
        """Sceglie il dizionario di valori piu' pertinente all'epoca indicata."""
        available = self.value_analyzer.available_dicts()
        if not available:
            return "roman_values"

        era_text = f"{self.era} {self.author_name}".lower()
        for name in available:
            # `roman_values` -> confronta "roman" con l'epoca dichiarata.
            token = name.replace("_values", "").replace("_code", "").replace("_", " ")
            if token and token in era_text:
                return name

        return available[0]

    def _knowledge_bounds(self) -> KnowledgeBounds:
        """Confini di conoscenza dalla configurazione, se indicati."""
        persona = self.config.get("persona", {})
        knowledge = persona.get("knowledge", {}) or {}
        return KnowledgeBounds(
            era_start=str(knowledge.get("era_start", self.era or "Unknown")),
            era_end=str(knowledge.get("era_end", self.era or "Unknown")),
            known_topics=list(knowledge.get("known_topics", [])),
            anachronistic_concepts=list(knowledge.get("anachronistic_concepts", [])),
        )

    @staticmethod
    def _corpus_stats(ingestion_result: IngestionResult) -> CorpusStats:
        return CorpusStats(
            total_files=ingestion_result.total_files_processed,
            total_segments=ingestion_result.total_segments,
            total_words=sum(s.word_count for s in ingestion_result.segments),
            language_distribution={
                lang.value: count
                for lang, count in ingestion_result.language_distribution.items()
            },
            sources=sorted({s.source_file for s in ingestion_result.segments}),
        )

    def _save(self, profile: CompleteStyleProfile) -> None:
        output_dir = self.project_dir / "profiles"
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "style_profile.json").write_text(
            profile.model_dump_json(indent=2), encoding="utf-8"
        )
        # Il prompt a parte: e' il prodotto che si copia altrove piu' spesso.
        (output_dir / "system_prompt.md").write_text(
            profile.generated_system_prompt, encoding="utf-8"
        )
        self.logger.info(f"Profilo salvato in {output_dir}")
