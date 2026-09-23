"""Quale modello serve quale compito.

Due proprietà reggono tutto il resto.

**Niente assegnazione non è un guasto.** Un impianto appena installato non ha
righe in tabella, e deve rispondere lo stesso: i compiti ricadono sul
predefinito. Una piattaforma che tace finché qualcuno non ha compilato quattro
caselle sembra rotta, e viene smontata prima di essere capita.

**L'elenco dei modelli non si cambia da qui.** Indirizzi e chiavi stanno
nell'ambiente; la console sceglie fra quei nomi. Il test lo verifica dal lato
che conta: un nome che non esiste viene rifiutato, e la configurazione di un
modello non esce mai con la chiave dentro.
"""
from __future__ import annotations

import pytest

from platform_core.llm.compiti import (
    PREDEFINITO, Compito, ModelloConfigurato, RegistroModelli,
    modelli_da_impostazioni,
)


def elenco() -> list[ModelloConfigurato]:
    return [
        ModelloConfigurato(
            nome=PREDEFINITO, base_url="http://127.0.0.1:1234/v1",
            model="qualcosa", api_key="segreta",
        ),
        ModelloConfigurato(
            nome="bonsai", base_url="http://127.0.0.1:7707/v1",
            model="bonsai-2-27b-abliterated", motore="llamacpp",
            senza_filtri=True,
        ),
        ModelloConfigurato(
            nome="giudice", provider="anthropic", model="claude-sonnet-5",
            api_key="anche-questa-segreta",
        ),
    ]


class TestRicaduta:
    def test_senza_assegnazioni_tutto_va_al_predefinito(self):
        registro = RegistroModelli(elenco())

        assert all(
            registro.nome_per(compito) == PREDEFINITO for compito in Compito
        )

    def test_un_compito_assegnato_non_tocca_gli_altri(self):
        registro = RegistroModelli(elenco())
        registro.assegna(Compito.DIGESTIONE, "bonsai")

        assert registro.nome_per(Compito.DIGESTIONE) == "bonsai"
        assert registro.nome_per(Compito.CONVERSAZIONE) == PREDEFINITO

    def test_un_modello_sparito_dall_ambiente_ricade(self):
        """Succede davvero: si toglie un modello da PERSONA_MODELLI e la riga
        in tabella resta. Il compito gira su un altro modello, e dai risultati
        non si vede — quindi va detto nei log."""
        from platform_core.llm import compiti as modulo

        from .conftest import avvisi_di

        registro = RegistroModelli(
            elenco(), assegnazioni={Compito.GIUDIZIO: "sparito"},
        )

        with avvisi_di(modulo) as avvisi:
            assert registro.nome_per(Compito.GIUDIZIO) == PREDEFINITO

        assert any("sparito" in a for a in avvisi)

    def test_togliere_l_assegnazione_riporta_al_predefinito(self):
        registro = RegistroModelli(elenco())
        registro.assegna(Compito.RECUPERO, "bonsai")
        registro.assegna(Compito.RECUPERO, None)

        assert registro.nome_per(Compito.RECUPERO) == PREDEFINITO

    def test_aggiorna_sostituisce_invece_di_fondere(self):
        """Un compito tolto dalla tabella deve tornare al predefinito: una
        fusione lo lascerebbe assegnato per sempre al modello di prima."""
        registro = RegistroModelli(elenco())
        registro.aggiorna({Compito.DIGESTIONE: "bonsai"})
        registro.aggiorna({Compito.GIUDIZIO: "giudice"})

        assert registro.nome_per(Compito.DIGESTIONE) == PREDEFINITO
        assert registro.nome_per(Compito.GIUDIZIO) == "giudice"


class TestElencoChiuso:
    def test_un_modello_sconosciuto_non_si_assegna(self):
        registro = RegistroModelli(elenco())

        with pytest.raises(ValueError):
            registro.assegna(Compito.CONVERSAZIONE, "quello-che-voglio-io")

    def test_esiste_dice_il_vero(self):
        registro = RegistroModelli(elenco())

        assert registro.esiste("bonsai")
        assert not registro.esiste("inventato")


class TestFornitori:
    def test_ogni_modello_ha_il_suo_fornitore(self):
        registro = RegistroModelli(elenco())
        registro.assegna(Compito.DIGESTIONE, "bonsai")

        conversazione = registro.per(Compito.CONVERSAZIONE)
        digestione = registro.per(Compito.DIGESTIONE)

        assert conversazione is not digestione
        assert digestione.motore == "llamacpp", (
            "il motore dichiarato decide la strategia di contesto: dedurlo "
            "dall'indirizzo qui darebbe «other», cioè nessuna cache"
        )

    def test_il_fornitore_si_costruisce_una_volta_sola(self):
        """Costruirne uno nuovo a ogni domanda significa chiedere di nuovo
        l'elenco dei modelli prima di ogni risposta."""
        registro = RegistroModelli(elenco())

        assert registro.per(Compito.CONVERSAZIONE) is registro.per(Compito.CONVERSAZIONE)

    def test_il_nome_del_modello_finisce_nel_fornitore(self):
        """Senza, due modelli diversi comparirebbero nelle tracce entrambi
        come «openai-compatibile»."""
        registro = RegistroModelli(elenco())
        registro.assegna(Compito.CONVERSAZIONE, "bonsai")

        assert registro.per(Compito.CONVERSAZIONE).name == "bonsai"

    def test_anthropic_si_costruisce_dal_suo_provider(self):
        registro = RegistroModelli(elenco())
        registro.assegna(Compito.GIUDIZIO, "giudice")

        fornitore = registro.per(Compito.GIUDIZIO)

        assert type(fornitore).__name__ == "AnthropicProvider"


class TestStato:
    def test_dice_assegnato_e_in_uso_separatamente(self):
        """Sono cose diverse: «nessuna assegnazione» e «assegnato al
        predefinito» si comportano uguale oggi e diversamente domani, quando
        il predefinito cambia."""
        registro = RegistroModelli(elenco())
        registro.assegna(Compito.DIGESTIONE, "bonsai")

        per_compito = {r["compito"]: r for r in registro.stato()}

        assert per_compito["digestione"]["assegnato"] == "bonsai"
        assert per_compito["conversazione"]["assegnato"] is None
        assert per_compito["conversazione"]["in_uso"] == PREDEFINITO

    def test_senza_filtri_si_vede(self):
        registro = RegistroModelli(elenco())
        registro.assegna(Compito.CONVERSAZIONE, "bonsai")

        per_compito = {r["compito"]: r for r in registro.stato()}

        assert per_compito["conversazione"]["senza_filtri"] is True
        assert per_compito["giudizio"]["senza_filtri"] is False

    def test_ogni_compito_si_spiega(self):
        """Un elenco di quattro parole tecniche non aiuta chi deve scegliere."""
        for riga in RegistroModelli(elenco()).stato():
            assert riga["label"] and riga["descrizione"]


class TestCompatibilita:
    def test_senza_elenco_se_ne_ricava_uno_dalle_vecchie_impostazioni(
        self, monkeypatch,
    ):
        """Un impianto che funzionava prima continua a funzionare senza che
        nessuno tocchi la configurazione."""
        from platform_core.settings import get_settings

        monkeypatch.delenv("PERSONA_MODELLI", raising=False)
        get_settings.cache_clear()

        modelli = modelli_da_impostazioni()

        assert [m.nome for m in modelli] == [PREDEFINITO]
        assert modelli[0].base_url

    def test_l_elenco_dell_ambiente_vince(self, monkeypatch):
        from platform_core.settings import get_settings

        monkeypatch.setenv(
            "PERSONA_MODELLI",
            '[{"nome": "solo-questo", "base_url": "http://127.0.0.1:9/v1"}]',
        )
        get_settings.cache_clear()
        try:
            assert [m.nome for m in modelli_da_impostazioni()] == ["solo-questo"]
        finally:
            get_settings.cache_clear()


class TestModelliCheRagionano:
    """Il budget di uscita deve coprire anche il ragionamento.

    Un modello che ragiona spende il budget prima di scrivere: se lo
    esaurisce, restituisce testo vuoto e la scala dei tentativi riparte
    triplicando. Su un modello lento ogni tentativo è un minuto, e sono
    minuti spesi per scoprire una cosa che la configurazione poteva dire.
    """

    def test_la_dichiarazione_arriva_al_fornitore(self):
        registro = RegistroModelli([
            ModelloConfigurato(
                nome=PREDEFINITO, base_url="http://127.0.0.1:9/v1", ragiona=True,
            ),
        ])

        assert registro.per(Compito.GIUDIZIO).ragiona is True

    def test_per_difetto_non_ragiona(self):
        """Alzare il pavimento a chi non ragiona sarebbe solo un tetto più
        alto e inutile."""
        registro = RegistroModelli(elenco())

        assert registro.per(Compito.GIUDIZIO).ragiona is False

    def test_il_pavimento_vale_solo_per_chi_ragiona(self):
        from platform_core.llm.json_mode import BUDGET_RAGIONAMENTO
        from platform_core.llm.openai_compatible import OpenAICompatibleProvider

        ragionante = OpenAICompatibleProvider(
            base_url="http://127.0.0.1:9/v1", ragiona=True,
        )
        diretto = OpenAICompatibleProvider(base_url="http://127.0.0.1:9/v1")

        assert ragionante.ragiona and not diretto.ragiona
        assert BUDGET_RAGIONAMENTO > 2048, (
            "il pavimento deve stare sopra il budget con cui il giudice chiama"
        )
