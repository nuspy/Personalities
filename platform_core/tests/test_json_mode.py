"""Le tre difese per ottenere JSON da un modello.

Ognuna è nata da un guasto vero, e ognuna protegge da un fallimento che non
somiglia alla sua causa: un `400` che parla di un parametro invece che del
server, una risposta valida buttata via per le virgolette attorno, un
contenuto vuoto che sembra un errore del prompt e invece è spazio finito.
"""
from __future__ import annotations

import pytest

from platform_core.llm.base import (
    GenerationError, GenerationRequest, Message, TruncatedResponse,
)
from platform_core.llm.json_mode import (
    DIALETTI, TETTO_TOKEN, JsonNonInterpretabile, e_rifiuto_del_formato,
    estrai_json, formato_risposta, prossimo_budget, sembra_degenerata,
)
from platform_core.llm.openai_compatible import OpenAICompatibleProvider


class TestEstrazione:
    def test_json_puro(self):
        assert estrai_json('{"a": 1}') == {"a": 1}

    def test_dentro_un_recinto(self):
        """Un modello che ha appena promesso JSON lo incornicia comunque."""
        assert estrai_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_recinto_senza_linguaggio(self):
        assert estrai_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_con_prosa_attorno(self):
        testo = 'Ecco il risultato richiesto:\n\n{"a": 1}\n\nSpero sia utile.'
        assert estrai_json(testo) == {"a": 1}

    def test_una_lista(self):
        assert estrai_json("[1, 2, 3]") == [1, 2, 3]

    def test_senza_json_solleva(self):
        with pytest.raises(JsonNonInterpretabile):
            estrai_json("Mi dispiace, non posso aiutarti.")

    def test_il_messaggio_mostra_cosa_e_arrivato(self):
        """Senza, resta da indovinare se il modello abbia risposto altro o
        nulla."""
        with pytest.raises(JsonNonInterpretabile, match="Mi dispiace"):
            estrai_json("Mi dispiace, non posso aiutarti.")


class TestDialetti:
    def test_l_ordine_prova_prima_il_piu_diffuso(self):
        assert DIALETTI[0] == "json_object"
        assert DIALETTI[-1] is None, "l'ultimo tentativo è senza vincolo"

    def test_il_formato_del_dialetto(self):
        assert formato_risposta("json_object") == {"type": "json_object"}
        assert formato_risposta("json_schema")["type"] == "json_schema"
        assert formato_risposta(None) is None

    def test_lo_schema_e_permissivo(self):
        """Serve a ottenere JSON valido, non a imporre una forma: le risposte
        attese hanno chiavi diverse a seconda di chi le chiede."""
        schema = formato_risposta("json_schema")["json_schema"]
        assert schema["strict"] is False
        assert schema["schema"]["additionalProperties"] is True

    def test_riconosce_il_rifiuto_del_formato(self):
        """Distinguerlo conta: si risolve cambiando dialetto, mentre riprovare
        su un guasto di rete moltiplica solo l'attesa."""
        assert e_rifiuto_del_formato(
            GenerationError("400: 'response_format' must be 'json_schema'")
        )
        assert not e_rifiuto_del_formato(GenerationError("connection refused"))


class TestBudget:
    def test_triplica(self):
        """Raddoppiare costa un tentativo in più, e ogni tentativo è una
        generazione intera pagata."""
        assert prossimo_budget(1000) == 3000

    def test_non_supera_il_tetto(self):
        assert prossimo_budget(TETTO_TOKEN) == TETTO_TOKEN
        assert prossimo_budget(TETTO_TOKEN // 2) == TETTO_TOKEN


class TestDegenerazione:
    def test_alfabeto_poverissimo(self):
        """« .  .  .  . »: non è prosa in nessuna lingua."""
        assert sembra_degenerata(". " * 300, generati=1000)

    def test_prosa_normale_non_viene_toccata(self):
        testo = "Una frase di prosa normale, con parole diverse fra loro. " * 20
        assert not sembra_degenerata(testo, generati=1000)

    def test_una_finestra_troppo_corta_non_si_giudica(self):
        assert not sembra_degenerata("breve", generati=100)

    def test_la_ripetizione_sotto_soglia_e_legittima(self):
        """È il modo normale in cui questi modelli affinano una frase:
        interromperli lì rovinerebbe il risultato."""
        frase = "La virtù si esercita ogni giorno con pazienza. "
        assert not sembra_degenerata(frase * 10, generati=500)


class TestNegoziazione:
    """Il provider, con un trasporto controllato."""

    class ProviderProvato(OpenAICompatibleProvider):
        def __init__(self, esiti):
            super().__init__(base_url="http://finto/v1", default_model="m")
            self._esiti = list(esiti)
            self.dialetti_provati = []
            self.budget_provati = []

        async def complete(self, request):
            formato = request.extra.get("response_format")
            self.dialetti_provati.append(formato["type"] if formato else None)
            self.budget_provati.append(request.max_tokens)
            esito = self._esiti.pop(0)
            if isinstance(esito, Exception):
                raise esito
            return esito

    async def test_scende_al_dialetto_successivo(self):
        provider = self.ProviderProvato([
            GenerationError("400: response_format non supportato"),
            '{"ok": true}',
        ])

        assert await provider.complete_json(_richiesta()) == {"ok": True}
        assert provider.dialetti_provati == ["json_object", "json_schema"]

    async def test_ricorda_il_dialetto_negoziato(self):
        """Rinegoziarlo a ogni chiamata costerebbe una richiesta fallita
        ogni volta."""
        provider = self.ProviderProvato([
            GenerationError("400: response_format non supportato"),
            '{"a": 1}',
            '{"b": 2}',
        ])

        await provider.complete_json(_richiesta())
        provider.dialetti_provati.clear()
        await provider.complete_json(_richiesta())

        assert provider.dialetti_provati == ["json_schema"]

    async def test_un_errore_diverso_non_fa_cambiare_dialetto(self):
        provider = self.ProviderProvato([GenerationError("connection refused")])

        with pytest.raises(GenerationError, match="connection refused"):
            await provider.complete_json(_richiesta())

        assert provider.dialetti_provati == ["json_object"]

    async def test_alza_il_budget_quando_il_modello_lo_esaurisce(self):
        """Il caso misurato: 4094 token su 4096 spesi a ragionare, contenuto
        vuoto."""
        provider = self.ProviderProvato([
            TruncatedResponse("budget esaurito ragionando"),
            '{"ok": true}',
        ])

        assert await provider.complete_json(_richiesta(1000)) == {"ok": True}
        assert provider.budget_provati == [1000, 3000]

    async def test_si_arrende_dopo_qualche_tentativo(self):
        provider = self.ProviderProvato(
            [TruncatedResponse("ancora troncata")] * 10
        )

        with pytest.raises(GenerationError, match="non produce una risposta"):
            await provider.complete_json(_richiesta(1000))

    async def test_senza_dialetti_accettati_si_arrende(self):
        provider = self.ProviderProvato(
            [GenerationError("400: response_format")] * 3
        )

        with pytest.raises(GenerationError, match="nessuna modalità JSON"):
            await provider.complete_json(_richiesta())

        assert provider.dialetti_provati == ["json_object", "json_schema", None]


def _richiesta(max_tokens: int = 500) -> GenerationRequest:
    return GenerationRequest(
        messages=[Message(role="user", content="dammi un JSON")],
        max_tokens=max_tokens,
    )


class TestBudgetDeiModelliCheRagionano:
    """Il tetto di uscita deve coprire il ragionamento, su ogni chiamata.

    Non solo su quelle strutturate: un modello che ragiona spende lo stesso
    budget della risposta per pensarci, e un tetto calcolato sulla lunghezza
    attesa della risposta lo esaurisce prima che cominci a scrivere. Il
    risultato è testo vuoto — nessun errore — e chi chiama ricade sul proprio
    ripiego a ogni tentativo senza che nulla sembri rotto. Misurato sulla
    selezione dei passaggi con Bonsai 2 27B.
    """

    async def _tetto(self, *, ragiona: bool, chiesto: int) -> int:
        from platform_core.llm.base import GenerationRequest, Message
        from platform_core.llm.openai_compatible import OpenAICompatibleProvider

        fornitore = OpenAICompatibleProvider(
            base_url="http://127.0.0.1:9/v1", default_model="finto",
            ragiona=ragiona,
        )
        payload = await fornitore._build_payload(
            GenerationRequest(
                messages=[Message(role="user", content="ciao")],
                max_tokens=chiesto,
            ),
            stream=False,
        )
        return payload["max_tokens"]

    async def test_chi_ragiona_riceve_il_pavimento(self):
        from platform_core.llm.json_mode import BUDGET_RAGIONAMENTO

        assert await self._tetto(ragiona=True, chiesto=500) == BUDGET_RAGIONAMENTO

    async def test_chi_non_ragiona_riceve_quello_che_ha_chiesto(self):
        """Su un modello diretto il pavimento sarebbe solo un tetto più alto
        e inutile."""
        assert await self._tetto(ragiona=False, chiesto=500) == 500

    async def test_un_tetto_gia_alto_non_si_abbassa(self):
        assert await self._tetto(ragiona=True, chiesto=20_000) == 20_000
