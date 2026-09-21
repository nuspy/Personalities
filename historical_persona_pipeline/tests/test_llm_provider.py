"""Test del provider LLM: negoziazione JSON, errori leggibili, ritentativi.

I server OpenAI-compatibili non concordano su come si chiede una risposta
JSON, e il disaccordo si manifesta come un 400 che — senza il corpo della
risposta — non dice nulla. Questi test fissano il comportamento verificato
contro LM Studio.
"""
from __future__ import annotations

import pytest

from historical_persona_pipeline.pipeline.utils.llm_provider import (
    LLMError, LLMFactory, OpenAICompatibleProvider,
    _is_response_format_rejection, _json_response_format, _response_detail,
)


class _FakeResponse:
    def __init__(self, payload=None, text=""):
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("non JSON")
        return self._payload


class TestResponseDetail:
    """Un 400 senza il corpo della risposta costringe a indovinare."""

    def test_messaggio_da_errore_strutturato(self):
        response = _FakeResponse({"error": {"message": "modello non caricato"}})
        assert "modello non caricato" in _response_detail(response)

    def test_messaggio_da_errore_stringa(self):
        response = _FakeResponse({"error": "'response_format.type' must be 'json_schema'"})
        assert "json_schema" in _response_detail(response)

    def test_corpo_non_json_usato_come_testo(self):
        assert "guasto" in _response_detail(_FakeResponse(None, "guasto interno"))

    def test_risposta_assente(self):
        assert _response_detail(None) == ""


class TestJsonDialects:
    def test_json_object(self):
        assert _json_response_format("json_object") == {"type": "json_object"}

    def test_json_schema_permissivo(self):
        fmt = _json_response_format("json_schema")
        assert fmt["type"] == "json_schema"
        # Lo schema non deve imporre chiavi: le risposte attese cambiano
        # a seconda del punto della pipeline che le richiede.
        assert fmt["json_schema"]["schema"]["additionalProperties"] is True

    def test_nessun_vincolo(self):
        assert _json_response_format(None) is None

    def test_rifiuto_riconosciuto(self):
        exc = LLMError("errore 400 — 'response_format.type' must be 'json_schema' or 'text'")
        assert _is_response_format_rejection(exc)

    def test_altri_errori_non_scambiati_per_rifiuti_di_formato(self):
        assert not _is_response_format_rejection(LLMError("errore 401 — chiave non valida"))


class TestNegotiation:
    """Il dialetto si scopre una volta e si riusa."""

    @pytest.fixture
    def provider(self):
        return OpenAICompatibleProvider(
            base_url="http://localhost:0/v1", api_key="x", model="m", max_retries=1
        )

    def test_ripiega_su_json_schema_quando_json_object_e_rifiutato(self, provider, monkeypatch):
        tried = []

        def fake_post(payload):
            fmt = payload.get("response_format", {}).get("type")
            tried.append(fmt)
            if fmt == "json_object":
                raise LLMError("errore 400 — 'response_format.type' must be 'json_schema' or 'text'")
            return '{"risposta": "ok"}'

        monkeypatch.setattr(provider, "_post", fake_post)
        assert provider.generate("x", json_mode=True) == '{"risposta": "ok"}'
        assert tried == ["json_object", "json_schema"]

    def test_il_dialetto_trovato_viene_riusato(self, provider, monkeypatch):
        tried = []

        def fake_post(payload):
            fmt = payload.get("response_format", {}).get("type")
            tried.append(fmt)
            if fmt == "json_object":
                raise LLMError("errore 400 — response_format non supportato")
            return "{}"

        monkeypatch.setattr(provider, "_post", fake_post)
        provider.generate("x", json_mode=True)
        provider.generate("y", json_mode=True)
        # La seconda chiamata non ripete il tentativo fallito.
        assert tried == ["json_object", "json_schema", "json_schema"]

    def test_senza_json_mode_nessun_response_format(self, provider, monkeypatch):
        seen = {}
        monkeypatch.setattr(provider, "_post", lambda p: seen.update(p) or "testo")
        provider.generate("x", json_mode=False)
        assert "response_format" not in seen

    def test_errori_non_di_formato_non_vengono_mascherati(self, provider, monkeypatch):
        def fake_post(payload):
            raise LLMError("errore 401 — chiave non valida")

        monkeypatch.setattr(provider, "_post", fake_post)
        with pytest.raises(LLMError, match="401"):
            provider.generate("x", json_mode=True)

    def test_ultimo_tentativo_senza_vincolo(self, provider, monkeypatch):
        """Se nessun formato e' accettato si prosegue senza: il prompt chiede
        comunque JSON e il parser a valle tollera il testo libero."""
        formats = []

        def fake_post(payload):
            fmt = payload.get("response_format")
            formats.append(fmt)
            if fmt is not None:
                raise LLMError("errore 400 — response_format non supportato")
            return "{}"

        monkeypatch.setattr(provider, "_post", fake_post)
        provider.generate("x", json_mode=True)
        assert formats[-1] is None


class TestFactory:
    def test_alias_riconosciuti(self):
        for name in ("lm_studio", "ollama", "openai", "nebius", "vllm"):
            provider = LLMFactory.create({"llm_provider": name, "llm_base_url": "http://x/v1"})
            assert isinstance(provider, OpenAICompatibleProvider)

    def test_provider_sconosciuto_elenca_i_validi(self):
        with pytest.raises(LLMError, match="anthropic"):
            LLMFactory.create({"llm_provider": "inventato"})

    def test_anthropic_senza_chiave(self):
        with pytest.raises(LLMError, match="chiave API"):
            LLMFactory.create({"llm_provider": "anthropic"})


class TestDurationEstimate:
    """La generazione del dataset puo' durare ore: il ritmo va detto presto.

    Col default di 1000 conversazioni e un modello grande in locale si
    superano le quindici ore. Chi non se lo aspetta lo scopre restando a
    guardare la barra di avanzamento.
    """

    @pytest.mark.parametrize("seconds,expected", [
        (45, "45 secondi"),
        (89, "89 secondi"),
        (300, "5 minuti"),
        (1800, "30 minuti"),
        (9000, "2.5 ore"),
    ])
    def test_durata_leggibile(self, seconds, expected):
        from historical_persona_pipeline.pipeline.stage3_dataset.dataset_stage import (
            _human_duration,
        )

        assert _human_duration(seconds) == expected

    def test_la_stima_viene_annunciata(self):
        import inspect

        from historical_persona_pipeline.pipeline.stage3_dataset.dataset_stage import (
            DatasetStage,
        )

        source = inspect.getsource(DatasetStage.run)
        assert "_human_duration" in source
        assert "per conversazione" in source


class TestTokenBudget:
    """I modelli di ragionamento spendono nel pensiero quasi tutto il budget.

    Con 4096 token e un prompt articolato, un `qwen3` consuma 4094 token a
    ragionare e restituisce `content` vuoto con `finish_reason: length`. Il
    codice lo leggeva come una risposta vuota e scartava l'esempio: diciotto
    generazioni su diciotto perse, senza che nulla dicesse perche'.
    """

    def test_budget_iniziale_adeguato_ai_modelli_di_ragionamento(self):
        from historical_persona_pipeline.pipeline.utils.llm_provider import (
            DEFAULT_MAX_TOKENS,
        )

        assert DEFAULT_MAX_TOKENS >= 8192

    def test_troncamento_da_ragionamento_spiegato(self):
        from historical_persona_pipeline.pipeline.utils.llm_provider import (
            _explain_empty_content,
        )

        detail = _explain_empty_content(
            {"finish_reason": "length"},
            {"reasoning_content": "..."},
            {"completion_tokens": 4096, "completion_tokens_details": {"reasoning_tokens": 4094}},
        )
        assert "4094" in detail and "ragionamento" in detail

    def test_troncamento_senza_ragionamento_spiegato(self):
        from historical_persona_pipeline.pipeline.utils.llm_provider import (
            _explain_empty_content,
        )

        detail = _explain_empty_content(
            {"finish_reason": "length"}, {}, {"completion_tokens": 512}
        )
        assert "max_tokens" in detail

    def test_contenuto_vuoto_generico_spiegato(self):
        from historical_persona_pipeline.pipeline.utils.llm_provider import (
            _explain_empty_content,
        )

        detail = _explain_empty_content({"finish_reason": "stop"}, {}, {})
        assert "vuoto" in detail

    def test_il_budget_viene_alzato_e_la_richiesta_ripetuta(self, monkeypatch):
        from historical_persona_pipeline.pipeline.utils.llm_provider import (
            OpenAICompatibleProvider, TruncatedResponse,
        )

        provider = OpenAICompatibleProvider(
            base_url="http://localhost:0/v1", api_key="x", model="m", max_tokens=1000
        )
        budgets = []

        def fake_post(payload):
            budgets.append(payload["max_tokens"])
            if len(budgets) < 2:
                raise TruncatedResponse("token esauriti nel ragionamento")
            return "ok"

        monkeypatch.setattr(provider, "_post", fake_post)
        assert provider.generate("x") == "ok"
        assert budgets[1] > budgets[0], "il budget deve crescere"

    def test_il_budget_maggiorato_resta_per_le_richieste_successive(self, monkeypatch):
        """Se e' servito una volta servira' ancora: ripartire ogni volta dal
        valore basso sprecherebbe una richiesta per ciascun esempio."""
        from historical_persona_pipeline.pipeline.utils.llm_provider import (
            OpenAICompatibleProvider, TruncatedResponse,
        )

        provider = OpenAICompatibleProvider(
            base_url="http://localhost:0/v1", api_key="x", model="m", max_tokens=1000
        )
        calls = []

        def fake_post(payload):
            calls.append(payload["max_tokens"])
            if len(calls) == 1:
                raise TruncatedResponse("token esauriti")
            return "ok"

        monkeypatch.setattr(provider, "_post", fake_post)
        provider.generate("x")
        assert provider.max_tokens > 1000

    def test_il_budget_non_cresce_senza_limite(self, monkeypatch):
        from historical_persona_pipeline.pipeline.utils.llm_provider import (
            LLMError, OpenAICompatibleProvider, TruncatedResponse,
        )

        provider = OpenAICompatibleProvider(
            base_url="http://localhost:0/v1", api_key="x", model="m", max_tokens=1000
        )
        monkeypatch.setattr(
            provider, "_post",
            lambda payload: (_ for _ in ()).throw(TruncatedResponse("sempre troncato")),
        )
        with pytest.raises(LLMError, match="troncato"):
            provider.generate("x")


class TestFailureReporting:
    """Quando tutte le generazioni falliscono, l'errore deve dire perche'."""

    def test_motivi_raggruppati_ignorando_i_numeri(self):
        from historical_persona_pipeline.pipeline.stage3_dataset.conversation_generator import (
            _normalize_reason,
        )

        a = _normalize_reason("ha speso 4094 dei 4096 token")
        b = _normalize_reason("ha speso 8190 dei 8192 token")
        assert a == b, "occorrenze dello stesso problema devono raggrupparsi"

    def test_riepilogo_ordinato_per_frequenza(self):
        from historical_persona_pipeline.pipeline.stage3_dataset.conversation_generator import (
            ConversationGenerator,
        )

        generator = ConversationGenerator.__new__(ConversationGenerator)
        import threading
        from collections import Counter

        generator._lock = threading.Lock()
        generator.last_failures = Counter()

        for _ in range(3):
            generator._record_failure("token esauriti")
        generator._record_failure("risposta vuota")

        summary = generator.failure_summary()
        assert summary.startswith("token esauriti (3x)")
        assert "risposta vuota (1x)" in summary

    def test_riepilogo_vuoto_senza_scarti(self):
        from historical_persona_pipeline.pipeline.stage3_dataset.conversation_generator import (
            ConversationGenerator,
        )
        from collections import Counter

        generator = ConversationGenerator.__new__(ConversationGenerator)
        generator.last_failures = Counter()
        assert generator.failure_summary() == ""
