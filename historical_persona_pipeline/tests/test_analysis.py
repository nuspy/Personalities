"""Test dell'analisi: valori, lingua primaria, voce, sintesi del prompt."""
from __future__ import annotations

import uuid

import pytest

from historical_persona_pipeline.config_loader import load_config
from historical_persona_pipeline.paths import DATA_DIR
from historical_persona_pipeline.pipeline.data_models import (
    CorpusStats, FileFormat, SupportedLanguage, TextSegment,
)
from historical_persona_pipeline.pipeline.stage2_analysis.analysis_stage import AnalysisStage
from historical_persona_pipeline.pipeline.stage2_analysis.analyzers.value_analyzer import (
    ValueAnalyzer,
)
from historical_persona_pipeline.pipeline.stage2_analysis.analyzers.voice_analyzer import (
    VoiceAnalyzer,
)


def segment(content: str, language=SupportedLanguage.LATIN) -> TextSegment:
    return TextSegment(
        id=str(uuid.uuid4()),
        content=content,
        language=language,
        language_confidence=0.9,
        source_file="test.txt",
        source_format=FileFormat.TXT,
        word_count=len(content.split()),
        char_count=len(content),
    )


class TestValueAnalyzer:
    """Il percorso dei dizionari puntava a una cartella inesistente: il
    profilo valori usciva sempre vuoto e nessuno se ne accorgeva."""

    @pytest.fixture
    def analyzer(self):
        return ValueAnalyzer(DATA_DIR)

    def test_i_dizionari_vengono_trovati(self, analyzer):
        assert "roman_values" in analyzer.available_dicts()

    def test_valori_estratti_dal_testo(self, analyzer):
        text = (
            "Caesar clementia usus est erga victos. Fidem servavit et "
            "dignitatem auxit. Virtus militum maxima fuit."
        )
        profile = analyzer.analyze([segment(text)], "roman_values")

        assert profile.dominant_values, "nessun valore estratto"
        assert "clementia" in profile.value_distribution

    def test_forme_flesse_riconosciute(self, analyzer):
        """Il latino flette la radice: `fides` compare come `fidem`."""
        profile = analyzer.analyze(
            [segment("Fidem dedit et fidem servavit, ut fides maneret.")],
            "roman_values",
        )
        assert "fides" in profile.value_distribution

    def test_nessun_falso_positivo_per_sottostringa(self, analyzer):
        """`combined_text.count('vir')` contava dentro *servire* e *virus*."""
        profile = analyzer.analyze(
            [segment("Servire necesse est. Nemo vult servire semper. " * 5)],
            "roman_values",
        )
        assert "virtus" not in profile.dominant_values

    def test_dizionario_inesistente_usa_un_fallback(self, analyzer):
        profile = analyzer.analyze([segment("Caesar clementia usus est.")], "inesistente")
        assert profile is not None  # nessuna eccezione

    def test_esempi_riportano_il_contesto(self, analyzer):
        profile = analyzer.analyze(
            [segment("Caesar clementia usus est erga victos hostes in Gallia.")],
            "roman_values",
        )
        examples = profile.example_passages.get("clementia", [])
        assert examples and "clementia" in examples[0].lower()


class TestPrimaryLanguage:
    """La lingua primaria si sceglieva contando i segmenti: un unico segmento
    enorme perdeva contro molti frammenti brevi."""

    def test_vince_la_lingua_con_piu_parole(self):
        long_latin = segment("verbum " * 5000, SupportedLanguage.LATIN)
        short_italian = [
            segment("parola " * 10, SupportedLanguage.ITALIAN) for _ in range(50)
        ]
        assert AnalysisStage._primary_language([long_latin] + short_italian) == (
            SupportedLanguage.LATIN
        )

    def test_lingua_sconosciuta_ignorata(self):
        segments = [
            segment("parola " * 100, SupportedLanguage.ITALIAN),
            segment("???? " * 5000, SupportedLanguage.UNKNOWN),
        ]
        assert AnalysisStage._primary_language(segments) == SupportedLanguage.ITALIAN

    def test_corpus_tutto_sconosciuto(self):
        segments = [segment("xxx " * 10, SupportedLanguage.UNKNOWN)]
        assert AnalysisStage._primary_language(segments) == SupportedLanguage.UNKNOWN


class TestVoiceAnalyzer:
    def test_incipit_ricorrenti_rilevati(self):
        """Su un corpus sufficiente le aperture ricorrenti emergono.

        La ricorrenza da sola non basta: serve un campione abbastanza ampio
        perche' la misura significhi qualcosa (vedi
        `TestSignificanceThreshold`), e l'apertura non deve dominare il
        corpus. Qui le due condizioni sono rispettate.
        """
        frasi = (
            ["Quibus rebus cognitis Caesar copias duxit"] * 30
            + [f"Legiones numero {i} castra in finibus posuerunt" for i in range(130)]
        )
        profile = VoiceAnalyzer().analyze([segment(". ".join(frasi) + ".")])
        openers = [o["opener"] for o in profile.signature_openers]
        assert any("quibus" in o for o in openers)

    def test_tasso_di_domande_calcolato(self):
        text = "Che cosa e' la virtu'? E' forse il bene? Dimmelo. Non lo so."
        profile = VoiceAnalyzer().analyze([segment(text, SupportedLanguage.ITALIAN)])
        assert profile.question_ratio > 0

    def test_corpus_vuoto_restituisce_profilo_neutro(self):
        profile = VoiceAnalyzer().analyze([])
        assert profile.speech_register == "unknown"
        assert profile.signature_openers == []


class TestSystemPrompt:
    """Il generatore precedente produceva tre righe e ignorava quasi tutto
    quello che la pipeline aveva misurato."""

    @pytest.fixture
    def profile(self, tmp_path):
        config = load_config()
        config["persona"]["author_name"] = "Gaius Iulius Caesar"
        config["persona"]["era"] = "Repubblica romana, I secolo a.C."

        text = (
            "Caesar clementia usus est erga victos. Caesar fidem servavit. "
            "Quibus rebus cognitis Caesar profectus est. Caesar dignitatem auxit. "
            "Quibus rebus cognitis Caesar copias duxit. Virtus militum maxima fuit. "
        ) * 6
        from historical_persona_pipeline.pipeline.data_models import IngestionResult
        from datetime import datetime

        segments = [segment(text)]
        ingestion = IngestionResult(
            project_id="test",
            timestamp=datetime.now(),
            total_files_processed=1,
            total_segments=len(segments),
            segments=segments,
            language_distribution={SupportedLanguage.LATIN: 1},
        )
        return AnalysisStage(config, tmp_path).run(ingestion)

    def test_prompt_contiene_il_nome(self, profile):
        assert "Caesar" in profile.generated_system_prompt

    def test_prompt_contiene_i_valori(self, profile):
        assert "## I tuoi valori" in profile.generated_system_prompt
        assert profile.values.dominant_values

    def test_terza_persona_auto_riferita_rilevata(self, profile):
        """Il tratto piu' riconoscibile di Cesare: parla di se' in terza persona."""
        assert profile.rhetoric.narrative_perspective == "third_person_self_reference"
        assert "TERZA PERSONA" in profile.generated_system_prompt

    def test_linee_guida_prodotte(self, profile):
        assert profile.training_guidelines["do"]
        assert profile.training_guidelines["avoid"]

    def test_statistiche_del_corpus_registrate(self, profile):
        assert profile.corpus_stats.total_words > 0
        assert profile.corpus_stats.total_segments == 1

    def test_profilo_salvato_su_disco(self, profile, tmp_path):
        assert (tmp_path / "profiles" / "style_profile.json").exists()
        assert (tmp_path / "profiles" / "system_prompt.md").exists()

    def test_nessuna_affermazione_su_dati_non_misurati(self, profile):
        """Senza modello morfologico i rapporti valgono 0 per assenza di
        misura: il prompt non deve dichiarare 'periodo paratattico'."""
        prompt = profile.generated_system_prompt
        if not profile.syntax.morphology_available:
            assert "paratattico" not in prompt
            assert "ipotattico" not in prompt


class TestAuthorNameMatching:
    """Il nome fornito dall'utente e quello attestato nel corpus spesso non
    coincidono: si scrive 'Gaio Giulio Cesare' e il testo latino dice
    *Caesar*. Senza un confronto tollerante l'auto-riferimento in terza
    persona — il tratto piu' caratteristico — non viene mai rilevato."""

    def test_nome_italiano_riconosciuto_in_corpus_latino(self):
        from historical_persona_pipeline.pipeline.stage2_analysis.analyzers.rhetoric_analyzer import (
            RhetoricAnalyzer,
        )

        text = (
            "Caesar in Galliam profectus est. Caesar clementia usus est. "
            "Quibus rebus cognitis Caesar copias duxit. Caesar fidem servavit. "
        ) * 6
        profile = RhetoricAnalyzer("Gaio Giulio Cesare").analyze([segment(text)])
        assert profile.narrative_perspective == "third_person_self_reference"

    def test_prima_persona_riconosciuta(self):
        from historical_persona_pipeline.pipeline.stage2_analysis.analyzers.rhetoric_analyzer import (
            RhetoricAnalyzer,
        )

        text = (
            "Ego in Galliam profectus sum. Mihi placuit clementia. "
            "Me hortatus est. Nos copias duximus. Nostri milites fortes erant. "
        ) * 6
        profile = RhetoricAnalyzer("Gaio Giulio Cesare").analyze([segment(text)])
        assert profile.narrative_perspective in ("first_person", "mixed")

    def test_senza_nome_autore_nessun_errore(self):
        from historical_persona_pipeline.pipeline.stage2_analysis.analyzers.rhetoric_analyzer import (
            RhetoricAnalyzer,
        )

        profile = RhetoricAnalyzer("").analyze([segment("Gallia est omnis divisa. " * 20)])
        assert profile.narrative_perspective != "third_person_self_reference"

    def test_marcatori_argomentativi_latini_rilevati(self):
        """I marcatori erano solo in inglese e italiano: su un corpus latino
        il profilo retorico usciva vuoto."""
        from historical_persona_pipeline.pipeline.stage2_analysis.analyzers.rhetoric_analyzer import (
            RhetoricAnalyzer,
        )

        text = (
            "Quod ita est, igitur properandum est. Nam hostes appropinquant, "
            "sed nostri parati sunt. Quamquam pauci, tamen fortes. "
        ) * 6
        profile = RhetoricAnalyzer("Caesar").analyze([segment(text)])
        assert profile.argument_type_distribution
        assert "causal" in profile.argument_type_distribution


class TestSignificanceThreshold:
    """Un tratto misurato su un campione minuscolo non e' un tratto.

    Su 234 parole «quibus» compariva in 3 frasi su 8 e finiva nel profilo
    come apertura caratteristica. Il modello la prendeva per prescrizione e
    apriva cosi' 18 risposte su 18. Tre tentativi di correggere il problema
    a valle — linee guida piu' caute, richiesta di varieta' nel prompt,
    rigenerazione col divieto esplicito — non hanno spostato nulla: nessuna
    istruzione corregge un profilo che afferma il falso con autorevolezza.
    La cura e' non affermarlo.
    """

    def test_corpus_piccolo_non_produce_aperture_caratteristiche(self):
        text = ". ".join(["Quibus rebus cognitis Caesar profectus est"] * 6) + "."
        profile = VoiceAnalyzer().analyze([segment(text)])
        assert profile.signature_openers == []

    def test_corpus_ampio_le_produce(self):
        """Con abbastanza frasi la misura torna significativa."""
        frasi = (
            ["Quibus rebus cognitis Caesar copias duxit"] * 40
            + [f"Legiones numero {i} castra posuerunt in finibus" for i in range(120)]
        )
        profile = VoiceAnalyzer().analyze([segment(". ".join(frasi) + ".")])
        openers = [o["opener"] for o in profile.signature_openers]
        assert openers, "un corpus ampio deve produrre aperture"

    def test_apertura_che_domina_viene_esclusa(self):
        """Presente in oltre meta' delle frasi descrive il corpus, non l'autore."""
        frasi = ["Quibus rebus cognitis Caesar egit"] * 130
        profile = VoiceAnalyzer().analyze([segment(". ".join(frasi) + ".")])
        openers = [o["opener"] for o in profile.signature_openers]
        assert not any("quibus" in o for o in openers)

    def test_corpus_piccolo_non_produce_formule(self):
        from historical_persona_pipeline.pipeline.stage2_analysis.analyzers.vocabulary_analyzer import (
            MIN_TOKENS_FOR_FORMULAS, VocabularyAnalyzer,
        )
        from historical_persona_pipeline.pipeline.stage2_analysis.nlp_manager import NLPManager

        text = "Quibus rebus cognitis Caesar profectus est. " * 20
        profile = VocabularyAnalyzer(NLPManager()).analyze([segment(text)])
        assert profile.formulas_and_collocations == []
        assert MIN_TOKENS_FOR_FORMULAS >= 1000

    def test_il_prompt_tace_su_cio_che_non_ha_misurato(self, tmp_path):
        """Il profilo non deve elencare aperture che non ha potuto stabilire."""
        from datetime import datetime

        from historical_persona_pipeline.pipeline.data_models import IngestionResult

        config = load_config()
        config["persona"]["author_name"] = "Gaio Giulio Cesare"

        text = ". ".join(["Quibus rebus cognitis Caesar profectus est"] * 6) + "."
        segments = [segment(text)]
        ingestion = IngestionResult(
            project_id="t", timestamp=datetime.now(), total_files_processed=1,
            total_segments=1, segments=segments,
            language_distribution={SupportedLanguage.LATIN: 1},
        )
        profile = AnalysisStage(config, tmp_path).run(ingestion)

        prompt = profile.generated_system_prompt
        assert "Aperture di frase che ti sono proprie" not in prompt
        assert "Fra le tue aperture ricorrono" not in prompt
        # Il resto del profilo deve restare.
        assert "## Come parli" in prompt
