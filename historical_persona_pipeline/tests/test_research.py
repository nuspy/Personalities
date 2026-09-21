"""Test del filtro di pertinenza e della generazione di domande.

Nessun test tocca la rete: la ricerca vera e' verificata separatamente, qui
si fissa la logica che decide cosa entra nel corpus.
"""
from __future__ import annotations

import uuid

import pytest

from historical_persona_pipeline.config_loader import load_config
from historical_persona_pipeline.pipeline.data_models import (
    CompleteStyleProfile, FileFormat, RhetoricalProfile, SupportedLanguage,
    SyntacticProfile, TextSegment, ValueProfile, VocabularyProfile, VoiceProfile,
)
from historical_persona_pipeline.pipeline.stage0_research.relevance import (
    RelevanceFilter, filter_documents,
)
from historical_persona_pipeline.pipeline.stage0_research.sources import ResearchDocument
from historical_persona_pipeline.pipeline.stage3_dataset.question_generator import (
    ALL_CATEGORIES, CATEGORY_ANACHRONISTIC, CATEGORY_HISTORICAL, CATEGORY_VALUES,
    QuestionGenerator,
)


@pytest.fixture
def caesar_filter():
    return RelevanceFilter("Giulio Cesare", "Repubblica romana, I secolo a.C.")


class TestRelevanceFilter:
    def test_soggetto_corretto_accettato(self, caesar_filter):
        text = (
            "Gaio Giulio Cesare nacque nel 100 a.C. Cesare conquisto' la Gallia "
            "nel 58 a.C. Cesare attraverso' il Rubicone nel 49 a.C. "
        ) * 20
        assert caesar_filter.evaluate("Gaio Giulio Cesare", text).keep

    def test_omonimo_di_altro_secolo_scartato(self, caesar_filter):
        """Giulio Cesare Vanini, filosofo del Seicento: stesso nome, altra persona."""
        text = (
            "Giulio Cesare Vanini fu un filosofo nato nel 1585 e arso vivo "
            "nel 1619 a Tolosa. Nel 1615 pubblico' la sua opera. "
        ) * 20
        verdict = caesar_filter.evaluate("Giulio Cesare Vanini", text)
        assert not verdict.keep

    def test_opera_letteraria_omonima_scartata(self, caesar_filter):
        text = (
            "Giulio Cesare e' una tragedia di William Shakespeare del 1599. "
            "Nel 1623 fu pubblicata nel First Folio. "
        ) * 20
        assert not caesar_filter.evaluate("Giulio Cesare (Shakespeare)", text).keep

    def test_pagina_di_contesto_scartata(self, caesar_filter):
        """*I secolo a.C.* cita Cesare ma non tratta di lui."""
        text = (
            "Il I secolo a.C. vide le guerre civili romane. Cesare e Pompeo "
            "si scontrarono nel 49 a.C. Ottaviano vinse ad Azio nel 31 a.C. "
        ) * 20
        assert not caesar_filter.evaluate("I secolo a.C.", text).keep

    def test_fonte_primaria_accettata_anche_senza_nome_nel_titolo(self, caesar_filter):
        """*De bello Gallico* e' l'opera di Cesare: il titolo non contiene il
        suo nome, ma e' esattamente il testo che serve."""
        text = "Gallia est omnis divisa in partes tres. " * 60
        assert caesar_filter.evaluate("De bello Gallico", text, is_primary_source=True).keep

    def test_epoca_assente_non_blocca(self):
        neutral = RelevanceFilter("Giulio Cesare", "")
        text = "Giulio Cesare fu un generale romano. Cesare vinse molte battaglie. " * 20
        assert neutral.evaluate("Gaio Giulio Cesare", text).keep

    def test_secolo_dedotto_dall_epoca(self):
        assert RelevanceFilter("X", "I secolo a.C.").expected_century == -1
        assert RelevanceFilter("X", "16th century").expected_century == 16
        assert RelevanceFilter("X", "XVI secolo").expected_century == 16
        assert RelevanceFilter("X", "").expected_century is None

    def test_scarti_riportano_il_motivo(self, caesar_filter):
        documents = [
            ResearchDocument(
                title="Giulio Cesare Vanini",
                text="Vanini nacque nel 1585 e mori' nel 1619. " * 30,
                url="http://esempio/vanini",
                source="wikipedia",
            )
        ]
        kept, rejected = filter_documents(documents, "Giulio Cesare", "I secolo a.C.")
        assert kept == []
        assert rejected and rejected[0][1].reasons


@pytest.fixture
def profile():
    return CompleteStyleProfile(
        author_name="Gaio Giulio Cesare",
        era="Repubblica romana, I secolo a.C.",
        primary_language=SupportedLanguage.LATIN,
        syntax=SyntacticProfile(
            avg_sentence_length=22.0, sentence_length_std=8.0,
            subordination_ratio=0.6, coordination_ratio=0.4,
            person_distribution={"3": 0.8}, common_pos_patterns=[], special_patterns={},
        ),
        vocabulary=VocabularyProfile(
            distinctive_terms=[{"term": "legio", "count": 40, "freq": 0.01}],
            semantic_field_distribution={"military": 80.0, "politics": 20.0},
            hapax_legomena_ratio=0.3, type_token_ratio=0.15,
            formulas_and_collocations=["quibus rebus cognitis"],
        ),
        values=ValueProfile(
            value_distribution={"clementia": 50.0, "virtus": 50.0},
            dominant_values=["clementia", "virtus"],
            theme_clusters=[], example_passages={"clementia": ["...clementia usus est..."]},
        ),
        rhetoric=RhetoricalProfile(
            argument_type_distribution={"causal": 60.0},
            persuasion_techniques=[], narrative_perspective="third_person_self_reference",
            self_reference_patterns=[],
        ),
        voice=VoiceProfile(),
        generated_system_prompt="Sei Gaio Giulio Cesare.",
    )


@pytest.fixture
def segments():
    text = (
        "Caesar in Galliam profectus est. Legiones Rhenum transierunt. "
        "Vercingetorix Alesiam defendit, sed Caesar vicit. Ariovistus fugit. "
    ) * 6
    return [
        TextSegment(
            id=str(uuid.uuid4()), content=text, language=SupportedLanguage.LATIN,
            language_confidence=0.9, source_file="bg.txt", source_format=FileFormat.TXT,
            word_count=len(text.split()), char_count=len(text),
        )
    ]


class TestQuestionGenerator:
    """La generazione copriva due categorie cablate nel codice, ignorando le
    sei dichiarate in configurazione."""

    def test_tutte_le_categorie_configurate_sono_coperte(self, profile, segments):
        config = load_config()
        questions = QuestionGenerator(config).generate(profile, segments, 60)

        categories = {q.category for q in questions}
        assert len(categories) >= 5, f"solo {categories}"

    def test_il_totale_richiesto_e_rispettato(self, profile, segments):
        questions = QuestionGenerator(load_config()).generate(profile, segments, 30)
        # Alcune categorie possono produrre meno del richiesto se il corpus
        # e' povero, ma non si deve sforare verso l'alto.
        assert 0 < len(questions) <= 30

    def test_domande_storiche_portano_il_contesto(self, profile, segments):
        generator = QuestionGenerator(load_config())
        questions = generator.generate(
            profile, segments, 10, distribution={CATEGORY_HISTORICAL: 1.0}
        )
        assert questions
        assert all(q.context for q in questions)
        assert all(q.source_segment_id for q in questions)

    def test_domande_sui_valori_usano_i_valori_misurati(self, profile, segments):
        questions = QuestionGenerator(load_config()).generate(
            profile, segments, 6, distribution={CATEGORY_VALUES: 1.0}
        )
        assert questions
        assert any("clementia" in q.text or "virtus" in q.text for q in questions)

    def test_domande_anacronistiche_citano_cose_moderne(self, profile, segments):
        questions = QuestionGenerator(load_config()).generate(
            profile, segments, 6, distribution={CATEGORY_ANACHRONISTIC: 1.0}
        )
        assert questions
        assert all(q.metadata.get("concept") for q in questions)

    def test_ripartizione_senza_perdite(self):
        counts = QuestionGenerator._allocate(100, {"a": 0.33, "b": 0.33, "c": 0.34})
        assert sum(counts.values()) == 100

    def test_ripartizione_con_pesi_non_normalizzati(self):
        counts = QuestionGenerator._allocate(50, {"a": 2, "b": 3})
        assert sum(counts.values()) == 50

    def test_generazione_deterministica(self, profile, segments):
        config = load_config()
        first = QuestionGenerator(config, seed=7).generate(profile, segments, 20)
        second = QuestionGenerator(config, seed=7).generate(profile, segments, 20)
        assert [q.text for q in first] == [q.text for q in second]

    def test_corpus_vuoto_non_solleva(self, profile):
        questions = QuestionGenerator(load_config()).generate(profile, [], 20)
        # Senza corpus restano le categorie che non ne dipendono.
        assert all(q.category != CATEGORY_HISTORICAL for q in questions)


class TestDatasetQuality:
    """Difetti visibili solo leggendo un dataset generato davvero.

    Sono passati attraverso tutti i test precedenti perche' il codice
    funzionava: produceva domande valide e risposte non vuote. Erano
    *scadenti*, ed e' un'altra cosa.
    """

    def test_il_personaggio_non_diventa_entita_delle_sue_domande(self, profile, segments):
        """«Che cosa accadde a Caesar?» chiesto a Cesare non ha senso, e su un
        corpus dove l'autore parla di se' in terza persona il suo nome e'
        proprio l'entita' piu' frequente."""
        generator = QuestionGenerator(load_config())
        questions = generator.generate(
            profile, segments, 12, distribution={CATEGORY_HISTORICAL: 1.0}
        )

        assert questions
        for question in questions:
            entity = (question.metadata.get("entity") or "").lower()
            assert "caesar" not in entity
            assert "cesare" not in entity

    def test_le_formule_sono_suggerite_con_parsimonia(self, profile):
        """Un'istruzione imperativa viene applicata alla lettera: su un
        dataset reale 15 risposte su 18 aprivano con la stessa formula."""
        from historical_persona_pipeline.pipeline.stage2_analysis.persona_synthesizer import (
            PersonaSynthesizer,
        )
        from historical_persona_pipeline.pipeline.data_models import VoiceProfile

        profile.voice = VoiceProfile(
            signature_openers=[{"opener": "quibus rebus cognitis", "count": 9}]
        )
        guidelines = PersonaSynthesizer().build_training_guidelines(profile)

        rules = " ".join(guidelines["do"]).lower()
        assert "parsimonia" in rules or "non da inserire" in rules
        assert not rules.startswith("apri le frasi con")
        assert any("stesso modo" in r.lower() for r in guidelines["avoid"])

    def test_nessuna_categoria_si_allontana_di_piu_di_un_esempio(self):
        """Il resto della divisione andava tutto alla categoria col peso
        maggiore: su 18 esempi `historical_facts` ne riceveva 8 invece di 4."""
        distribution = {
            "historical_facts": 0.25, "philosophy_values": 0.20,
            "anachronistic": 0.20, "character_personality": 0.15,
            "domain_expertise": 0.15, "casual_conversation": 0.05,
        }
        for total in (18, 37, 100, 1000):
            counts = QuestionGenerator._allocate(total, distribution)
            assert sum(counts.values()) == total
            for category, weight in distribution.items():
                expected = total * weight
                assert abs(counts[category] - expected) <= 1, (
                    f"{category}: {counts[category]} contro {expected:.1f} attesi su {total}"
                )
