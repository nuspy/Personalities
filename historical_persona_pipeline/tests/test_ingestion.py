"""Test dell'ingestione: chunking, normalizzazione, rilevamento lingua.

Ogni test corrisponde a un difetto reale trovato nel codice precedente; il
nome dice quale comportamento sbagliato non deve tornare.
"""
from __future__ import annotations

import pytest

from historical_persona_pipeline.config_loader import load_config
from historical_persona_pipeline.pipeline.data_models import SupportedLanguage
from historical_persona_pipeline.pipeline.stage1_ingestion.chunker import (
    chunk_text, split_paragraphs, split_sentences,
)
from historical_persona_pipeline.pipeline.stage1_ingestion.ingestion_stage import IngestionStage
from historical_persona_pipeline.pipeline.stage1_ingestion.language_detector import LanguageDetector
from historical_persona_pipeline.pipeline.stage1_ingestion.text_normalizer import TextNormalizer

LATIN = (
    "Gallia est omnis divisa in partes tres, quarum unam incolunt Belgae, "
    "aliam Aquitani, tertiam qui ipsorum lingua Celtae, nostra Galli appellantur. "
    "Hi omnes lingua, institutis, legibus inter se differunt, et cum Germanis "
    "continenter bellum gerunt, quod fere cotidianis proeliis contendunt."
)

ITALIAN = (
    "La storia di Roma comincia con una leggenda, ma prosegue con fatti "
    "documentati che gli storici hanno ricostruito nel corso dei secoli. "
    "Il racconto delle origini serve a spiegare un'identita' collettiva."
)


class TestChunker:
    def test_paragrafi_non_vengono_spezzati(self):
        text = "\n\n".join(["Paragrafo " + "x" * 300] * 5)
        chunks = chunk_text(text, target_chars=700, min_chars=100)
        # Nessun chunk deve interrompere un paragrafo a meta'.
        for chunk in chunks:
            assert not chunk.startswith("x")

    def test_coda_corta_viene_accorpata(self):
        """Un frammento finale di 20 caratteri falserebbe il rilevamento lingua."""
        text = "a" * 1000 + "\n\n" + "coda breve"
        chunks = chunk_text(text, target_chars=900, min_chars=200)
        assert all(len(c) >= 200 for c in chunks)
        assert "coda breve" in chunks[-1]

    def test_paragrafo_monolitico_spezzato_su_confine_di_frase(self):
        sentence = "Questa e' una frase di prova sufficientemente lunga. "
        chunks = chunk_text(sentence * 100, target_chars=500, max_chars=800)
        assert len(chunks) > 1
        for chunk in chunks:
            assert not chunk.strip().startswith("e' una frase")

    def test_testo_vuoto_non_produce_chunk(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\n  ") == []

    def test_split_sentences_riconosce_punto_interrogativo_greco(self):
        # In greco il punto e virgola e' il punto interrogativo.
        frasi = split_sentences("Ποιος ειναι; Ο Σωκρατης ειναι.")
        assert len(frasi) == 2


class TestNormalizer:
    def test_paragrafi_conservati(self):
        """Il vecchio `re.sub(r'\\s+', ' ')` collassava tutto in una riga."""
        normalized = TextNormalizer().normalize("Primo paragrafo.\n\nSecondo paragrafo.")
        assert "\n\n" in normalized
        assert len(split_paragraphs(normalized)) == 2

    def test_sillabazione_ricomposta(self):
        normalized = TextNormalizer().normalize("con-\nsiderare il problema")
        assert "considerare" in normalized
        assert "con- siderare" not in normalized

    def test_virgolette_tipografiche_uniformate(self):
        normalized = TextNormalizer().normalize('Disse “parole” e ‘altro’')
        assert '"parole"' in normalized
        assert "'altro'" in normalized

    def test_accenti_precomposti_e_combinanti_coincidono(self):
        """Senza NFC 'citta' con accento precomposto e combinante sono token diversi."""
        normalizer = TextNormalizer()
        precomposed = normalizer.normalize("città")
        combining = normalizer.normalize("città")
        assert precomposed == combining

    def test_spazio_prima_della_punteggiatura_rimosso(self):
        assert TextNormalizer().normalize("parola , altra .") == "parola, altra."


class TestLanguageDetector:
    def test_latino_riconosciuto(self):
        language, confidence = LanguageDetector().detect(LATIN)
        assert language == SupportedLanguage.LATIN
        assert confidence > 0.5

    def test_italiano_riconosciuto(self):
        language, _ = LanguageDetector().detect(ITALIAN)
        assert language == SupportedLanguage.ITALIAN

    def test_greco_antico_distinto_dal_moderno(self):
        polytonic = (
            "ἀνδρῶν γὰρ ἐπιφανῶν πᾶσα γῆ τάφος, καὶ οὐ στηλῶν μόνον ἐν τῇ "
            "οἰκείᾳ σημαίνει ἐπιγραφή, ἀλλὰ καὶ ἐν τῇ μὴ προσηκούσῃ."
        )
        language, _ = LanguageDetector().detect(polytonic)
        assert language == SupportedLanguage.GREEK_ANCIENT

    def test_testo_troppo_corto_resta_sconosciuto(self):
        language, confidence = LanguageDetector().detect("ciao")
        assert language == SupportedLanguage.UNKNOWN
        assert confidence == 0.0


class TestIngestionStage:
    @pytest.fixture
    def config(self):
        return load_config()

    def test_file_txt_prodotto_in_segmenti(self, config, tmp_path):
        source = tmp_path / "testo.txt"
        source.write_text(LATIN * 3, encoding="utf-8")

        result = IngestionStage(config).run([source])

        assert result.total_files_processed == 1
        assert result.total_segments >= 1
        assert result.segments[0].language == SupportedLanguage.LATIN

    def test_file_mancante_segnalato_non_conteggiato(self, config, tmp_path):
        """`total_files_processed` contava i file selezionati, non quelli letti."""
        good = tmp_path / "buono.txt"
        good.write_text(ITALIAN * 3, encoding="utf-8")
        missing = tmp_path / "assente.txt"

        result = IngestionStage(config).run([good, missing])

        assert result.total_files_processed == 1
        assert any("non trovato" in e["error"].lower() for e in result.errors)

    def test_corpus_completamente_vuoto_solleva_errore(self, config, tmp_path):
        """Un corpus vuoto deve fermare la pipeline, non attraversarla in silenzio."""
        empty = tmp_path / "vuoto.txt"
        empty.write_text("", encoding="utf-8")

        with pytest.raises(ValueError):
            IngestionStage(config).run([empty])

    def test_codifica_non_utf8_letta_senza_perdere_accenti(self, config, tmp_path):
        source = tmp_path / "latin1.txt"
        source.write_bytes(
            ("La città eterna " + ITALIAN).encode("latin-1")
        )

        result = IngestionStage(config).run([source])
        assert "citt" in result.segments[0].content

    def test_estensione_sconosciuta_ma_testuale_viene_letta(self, config, tmp_path):
        """Prima un'estensione non mappata faceva scartare il file."""
        source = tmp_path / "note.dat"
        source.write_text(ITALIAN * 3, encoding="utf-8")

        result = IngestionStage(config).run([source])
        assert result.total_segments >= 1
