"""Le tre strategie di contesto, e il fornitore Anthropic che ne consuma una.

Due invarianti prima di tutto il resto:

- **una strategia non cambia il testo.** Se potesse, due strategie diverse
  produrrebbero due prompt diversi per la stessa personalità, e il test che
  garantisce lo strato 0 byte-identico non proteggerebbe più nulla;
- **ogni motore riceve solo i campi che conosce.** `cache_prompt` a OpenAI o
  `prompt_cache_key` a vLLM sono, nel migliore dei casi, rumore; nel peggiore
  un 400 su ogni risposta.
"""
from __future__ import annotations

import json
from typing import List

import httpx
import pytest

from platform_core.llm.anthropic import AnthropicProvider
from platform_core.llm.base import GenerationRequest, Message, TruncatedResponse
from platform_core.llm.openai_compatible import OpenAICompatibleProvider, riconosci_motore
from platform_core.runtime.context_strategy import (
    KVCacheStrategy, NoCacheStrategy, PromptCacheStrategy, chiave_del_prefisso,
    scegli_strategia,
)

STABILE = "Sei Seneca. Rispondi con misura.\n\nRegole di comportamento:\n- Non inventare"
VOLATILE = "Passaggi:\n[K1] La vita non è breve, la rendiamo tale."


def richiesta(domanda: str = "Il tempo è nostro?") -> GenerationRequest:
    return GenerationRequest(
        messages=[
            Message(role="system", content=STABILE),
            Message(role="user", content="Ciao"),
            Message(role="assistant", content="Salute a te."),
            Message(role="system", content=VOLATILE),
            Message(role="user", content=domanda),
        ],
        cache_breakpoint_after=0,
    )


class TestScelta:
    @pytest.mark.parametrize("url, motore", [
        ("https://api.openai.com/v1", "openai"),
        ("http://127.0.0.1:1234/v1", "lmstudio"),
        ("http://localhost:8080/v1", "llamacpp"),
        ("http://gpu.interno:9000/v1", "other"),
    ])
    def test_il_motore_si_riconosce_dall_indirizzo_nei_casi_certi(self, url, motore):
        assert riconosci_motore(url) == motore

    def test_la_dichiarazione_vince_sull_indirizzo(self):
        """Un vLLM dietro un nome qualunque si riconosce solo se dichiarato:
        indovinare un motore con cache e sbagliare manderebbe campi che il
        server rifiuta."""
        assert riconosci_motore("http://gpu.interno:9000/v1", "vllm") == "vllm"

    @pytest.mark.parametrize("motore, attesa", [
        ("llamacpp", KVCacheStrategy),
        ("vllm", KVCacheStrategy),
        ("lmstudio", KVCacheStrategy),
        ("openai", PromptCacheStrategy),
        ("anthropic", PromptCacheStrategy),
        ("other", NoCacheStrategy),
    ])
    def test_ogni_motore_ha_la_sua_strategia(self, motore, attesa):
        class Fornitore:
            pass

        f = Fornitore()
        f.motore = motore
        assert isinstance(scegli_strategia(f), attesa)

    def test_un_fornitore_che_non_dichiara_il_motore_non_riceve_promesse(self):
        """Mai una strategia che prometta un riuso che non avverrà."""
        assert isinstance(scegli_strategia(object()), NoCacheStrategy)


class TestInvarianti:
    @pytest.mark.parametrize("strategia", [
        KVCacheStrategy("llamacpp", slot=4),
        KVCacheStrategy("vllm"),
        PromptCacheStrategy("anthropic"),
        PromptCacheStrategy("openai"),
        NoCacheStrategy(),
    ])
    def test_nessuna_strategia_tocca_il_testo(self, strategia):
        base = richiesta()
        applicata = strategia.applica(base, testo_stabile=STABILE)

        assert [m.to_dict() for m in applicata.messages] == [m.to_dict() for m in base.messages]
        assert applicata.cache_breakpoint_after == base.cache_breakpoint_after

    def test_la_chiave_e_la_stessa_per_lo_stesso_prefisso(self):
        """Due turni della stessa personalità, domande diverse: stessa chiave,
        perché è il prefisso che si riusa."""
        s = PromptCacheStrategy("openai")
        a = s.applica(richiesta("Il tempo?"), testo_stabile=STABILE).cache.chiave
        b = s.applica(richiesta("La morte?"), testo_stabile=STABILE).cache.chiave
        assert a == b

    def test_un_prompt_modificato_apre_una_cache_nuova(self):
        assert chiave_del_prefisso(STABILE) != chiave_del_prefisso(STABILE + " ")

    def test_la_chiave_non_dipende_dal_processo(self):
        """`hash()` di Python cambia a ogni avvio: due repliche dell'API
        assegnerebbero slot diversi allo stesso prompt."""
        assert chiave_del_prefisso("abc") == "ba7816bf8f01cfea414140de"


class TestKVCache:
    def test_llamacpp_riceve_cache_prompt_e_uno_slot_fisso(self):
        p = OpenAICompatibleProvider(base_url="http://localhost:8080/v1", motore="llamacpp")
        r = KVCacheStrategy("llamacpp", slot=4).applica(richiesta(), testo_stabile=STABILE)

        campi = p._campi_di_cache(r)

        assert campi["cache_prompt"] is True
        assert 0 <= campi["id_slot"] < 4

    def test_lo_stesso_prefisso_va_sempre_allo_stesso_slot(self):
        """Uno slot diverso ricalcolerebbe il prefisso da capo: il riuso ci
        sarebbe sulla carta e non nei fatti."""
        s = KVCacheStrategy("llamacpp", slot=8)
        slot = {s.applica(richiesta(d), testo_stabile=STABILE).cache.slot for d in ("a", "b", "c")}
        assert len(slot) == 1

    @pytest.mark.parametrize("motore", ["vllm", "lmstudio"])
    def test_vllm_e_lm_studio_non_ricevono_campi_in_piu(self, motore):
        """Riusano il prefisso da sé; vLLM segnala o rifiuta i campi sconosciuti."""
        p = OpenAICompatibleProvider(base_url="http://x/v1", motore=motore)
        r = KVCacheStrategy(motore).applica(richiesta(), testo_stabile=STABILE)
        assert p._campi_di_cache(r) == {}


class TestPromptCacheOpenAI:
    async def test_openai_riceve_la_chiave_del_prefisso(self):
        p = OpenAICompatibleProvider(base_url="https://api.openai.com/v1", motore="openai", default_model="gpt")
        r = PromptCacheStrategy("openai").applica(richiesta(), testo_stabile=STABILE)

        corpo = await p._build_payload(r, stream=True)

        assert corpo["prompt_cache_key"] == chiave_del_prefisso(STABILE)
        assert "cache_prompt" not in corpo

    async def test_senza_cache_nessun_campo_estraneo(self):
        p = OpenAICompatibleProvider(base_url="http://x/v1", motore="other", default_model="m")
        r = NoCacheStrategy().applica(richiesta(), testo_stabile=STABILE)

        corpo = await p._build_payload(r, stream=True)

        assert not {"prompt_cache_key", "cache_prompt", "id_slot"} & set(corpo)


# ---------------------------------------------------------------- Anthropic

def _eventi(*eventi: dict) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in eventi).encode()


def _fornitore(risposte: List[bytes], ricevute: List[dict]) -> AnthropicProvider:
    def gestisci(req: httpx.Request) -> httpx.Response:
        ricevute.append(json.loads(req.content))
        assert req.headers["x-api-key"] == "chiave-di-prova"
        assert req.headers["anthropic-version"]
        return httpx.Response(200, content=risposte.pop(0), headers={"content-type": "text/event-stream"})

    return AnthropicProvider(
        api_key="chiave-di-prova", default_model="claude-sonnet-5",
        transport=httpx.MockTransport(gestisci),
    )


RISPOSTA = _eventi(
    {"type": "message_start", "message": {"usage": {
        "input_tokens": 40, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1200,
    }}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "La vita "}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "non è breve [K1]."}},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 9}},
    {"type": "message_stop"},
)


class TestAnthropic:
    def test_il_blocco_stabile_porta_il_cache_control(self):
        """Senza, lo strato costruito apposta byte-identico si pagherebbe per
        intero a ogni risposta."""
        p = AnthropicProvider(api_key="k")
        r = PromptCacheStrategy("anthropic").applica(richiesta(), testo_stabile=STABILE)

        corpo = p.costruisci(r)

        assert corpo["system"][0] == {
            "type": "text", "text": STABILE, "cache_control": {"type": "ephemeral"},
        }

    def test_il_volatile_resta_fuori_dal_prefisso(self):
        """Memorie e passaggi cambiano a ogni turno: dentro il prefisso
        annullerebbero lo sconto."""
        corpo = AnthropicProvider(api_key="k").costruisci(
            PromptCacheStrategy("anthropic").applica(richiesta(), testo_stabile=STABILE),
        )

        assert corpo["system"][1] == {"type": "text", "text": VOLATILE}
        assert all(m["role"] in ("user", "assistant") for m in corpo["messages"])

    def test_senza_strategia_nessun_cache_control(self):
        corpo = AnthropicProvider(api_key="k").costruisci(richiesta())
        assert all("cache_control" not in b for b in corpo["system"])

    def test_i_ruoli_si_alternano_e_si_comincia_dall_utente(self):
        """Il Messages API pretende l'alternanza: due turni utente consecutivi
        si uniscono invece di far fallire la richiesta."""
        r = GenerationRequest(messages=[
            Message(role="assistant", content="Eccomi."),
            Message(role="user", content="Uno"),
            Message(role="user", content="Due"),
        ])
        messaggi = AnthropicProvider(api_key="k").costruisci(r)["messages"]

        assert messaggi[0]["role"] == "user"
        ruoli = [m["role"] for m in messaggi]
        assert all(a != b for a, b in zip(ruoli, ruoli[1:]))
        assert "Uno\n\nDue" in messaggi[-1]["content"]

    async def test_lo_streaming_restituisce_testo_e_token_da_cache(self):
        ricevute: List[dict] = []
        p = _fornitore([RISPOSTA], ricevute)

        pezzi, fine = [], None
        async for c in p.stream(PromptCacheStrategy("anthropic").applica(richiesta(), testo_stabile=STABILE)):
            if c.text:
                pezzi.append(c.text)
            if c.done:
                fine = c

        assert "".join(pezzi) == "La vita non è breve [K1]."
        assert fine.usage.cached_tokens == 1200
        assert fine.usage.prompt_tokens == 1240, "il totale in ingresso somma anche i token da cache"
        assert fine.usage.completion_tokens == 9
        assert ricevute[0]["stream"] is True
        assert ricevute[0]["max_tokens"] > 0, "il Messages API lo pretende"

    async def test_un_budget_esaurito_senza_testo_e_una_risposta_troncata(self):
        vuota = _eventi(
            {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
            {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 50}},
            {"type": "message_stop"},
        )
        p = _fornitore([vuota], [])

        with pytest.raises(TruncatedResponse):
            async for _ in p.stream(richiesta()):
                pass

    async def test_senza_chiave_l_errore_dice_cosa_configurare(self):
        from platform_core.llm.base import GenerationError

        with pytest.raises(GenerationError, match="PERSONA_ANTHROPIC_API_KEY"):
            async for _ in AnthropicProvider(api_key="").stream(richiesta()):
                pass


class TestNellaTraccia:
    def test_la_traccia_dice_quale_strategia_ha_servito_il_prefisso(self):
        from platform_core.runtime.persona_engine import EsitoTurno

        turno = EsitoTurno(modo="rag", modo_richiesto="rag")
        turno.cache = KVCacheStrategy("llamacpp", slot=2).applica(richiesta(), testo_stabile=STABILE).cache

        strategia = turno.traccia_risposta()["usage"]["strategia"]

        assert strategia["nome"] == "kv-cache:llamacpp"
        assert strategia["modo"] == "kv"
        assert strategia["chiave"] == chiave_del_prefisso(STABILE)
