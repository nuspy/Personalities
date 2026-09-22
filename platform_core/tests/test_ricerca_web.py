"""La ricerca online: portare nel contesto pagine che il corpus non contiene.

Il test che conta più di tutti è `test_un_risultato_fuori_lista_non_entra`.
Si può chiedere al motore di limitarsi a certi domini, e lo fa *di solito*;
«di solito» non è una garanzia, e un risultato fuori lista dentro il contesto
di una voce professionale è esattamente il caso che la lista doveva impedire.
Il filtro vero è il nostro, applicato ai risultati.

Subito dopo viene `test_una_ricerca_fallita_non_fa_fallire_la_risposta`: un
servizio esterno lento o caduto non deve diventare il punto più fragile della
piattaforma. La risposta esce con il corpus che c'è, e la traccia dice cosa è
mancato.
"""
from __future__ import annotations

import pytest

from platform_core.knowledge.ricerca_web import (
    ConfigurazioneRicerca, RicercaDisattivata, RisultatoWeb, _ammesso,
    dominio_di, normalizza_sito, passaggi_da_risultati,
)

PAGINA = (
    "Questa pagina parla di molte cose. "
    "La garanzia sui prodotti dura ventiquattro mesi dalla consegna. "
    "Il reso si richiede entro quattordici giorni. "
    "Seguono le note legali, l'informativa sui cookie e i contatti. "
) * 6


class TestDomini:
    @pytest.mark.parametrize("voce,atteso", [
        ("example.com", "example.com"),
        ("www.example.com", "example.com"),
        ("https://www.example.com/pagina?x=1", "example.com"),
        ("EXAMPLE.COM", "example.com"),
        ("  example.com  ", "example.com"),
        ("", ""),
    ])
    def test_chi_compila_la_lista_scrive_quello_che_ha_sotto_mano(
        self, voce, atteso,
    ):
        """Spesso un URL copiato dalla barra del browser. Rifiutarlo sarebbe
        pedanteria che si paga in liste sbagliate."""
        assert normalizza_sito(voce) == atteso

    def test_i_sottodomini_sono_ammessi(self):
        assert _ammesso("https://testi.seneca.it/lettere", ["seneca.it"])

    def test_un_dominio_che_somiglia_non_basta(self):
        """`falso-seneca.it` finisce per `seneca.it` come stringa. È l'errore
        che trasforma una lista in un colabrodo."""
        assert not _ammesso("https://falso-seneca.it/x", ["seneca.it"])

    def test_un_altro_dominio_e_fuori(self):
        assert not _ammesso("https://altrove.example/x", ["seneca.it"])

    def test_un_url_malformato_non_passa(self):
        assert not _ammesso("mica-un-url", ["seneca.it"])
        assert dominio_di("mica-un-url") == ""


class TestConfigurazione:
    def test_spenta_per_difetto(self):
        assert not ConfigurazioneRicerca.da_rag(None).attiva
        assert not ConfigurazioneRicerca.da_rag({}).attiva

    def test_si_legge_da_rag_config(self):
        c = ConfigurazioneRicerca.da_rag({
            "ricerca_online": {
                "attiva": True,
                "siti": ["https://www.example.com/docs", "altro.it"],
                "modo": "solo",
                "max_risultati": 5,
            }
        })

        assert c.attiva and c.solo_lista
        assert c.siti == ["example.com", "altro.it"]
        assert c.max_risultati == 5

    def test_un_modo_inventato_ricade_su_anche(self):
        """«solo» è il modo restrittivo: ricadere là per un refuso
        spegnerebbe la ricerca senza dirlo."""
        c = ConfigurazioneRicerca.da_rag({
            "ricerca_online": {"attiva": True, "modo": "soltanto"},
        })

        assert c.modo == "anche"

    def test_il_numero_di_risultati_e_limitato(self):
        c = ConfigurazioneRicerca.da_rag({
            "ricerca_online": {"attiva": True, "max_risultati": 500},
        })

        assert c.max_risultati == 10

    def test_solo_con_lista_vuota_non_si_esegue(self):
        """Il difetto che sarebbe silenzioso: «solo questi siti» con la lista
        vuota significa *nessun* sito, non *tutti*."""
        c = ConfigurazioneRicerca.da_rag({
            "ricerca_online": {"attiva": True, "modo": "solo", "siti": []},
        })

        utilizzabile, motivo = c.utilizzabile()
        assert not utilizzabile
        assert "vuota" in motivo


class TestPassaggi:
    def test_un_risultato_fuori_lista_non_entra(self):
        """Qualunque cosa il motore abbia capito della richiesta."""
        risultati = [
            RisultatoWeb(url="https://intruso.example/x", titolo="Intruso", testo=PAGINA),
            RisultatoWeb(url="https://example.com/garanzia", titolo="Garanzia", testo=PAGINA),
        ]

        passaggi = passaggi_da_risultati(
            risultati, domanda="quanto dura la garanzia?",
            siti=["example.com"], solo=True, limite=3,
        )

        assert passaggi, "il risultato ammesso doveva entrare"
        assert all(
            "intruso" not in p.corrispondenza.documento_uri for p in passaggi
        )

    def test_in_modo_anche_non_si_filtra(self):
        risultati = [
            RisultatoWeb(url="https://altrove.example/x", titolo="Altrove", testo=PAGINA),
        ]

        passaggi = passaggi_da_risultati(
            risultati, domanda="garanzia", siti=["example.com"],
            solo=False, limite=3,
        )

        assert passaggi, "in modo «anche» la lista è una preferenza, non un muro"

    def test_si_sceglie_il_pezzo_pertinente_non_il_primo(self):
        """Prendere i primi caratteri significa quasi sempre prendere il menu
        di navigazione e l'introduzione."""
        testo = (
            "Benvenuti nel nostro sito. Trovate qui notizie e contatti. " * 20
            + " La garanzia sui prodotti dura ventiquattro mesi dalla consegna. "
        )
        passaggi = passaggi_da_risultati(
            [RisultatoWeb(url="https://example.com/x", titolo="X", testo=testo)],
            domanda="quanto dura la garanzia sui prodotti?",
            limite=1,
        )

        assert passaggi
        assert "ventiquattro mesi" in passaggi[0].corrispondenza.testo

    def test_una_pagina_senza_testo_si_scarta(self):
        """Una pagina che non si lascia estrarre è da scartare, non da citare
        per il titolo."""
        passaggi = passaggi_da_risultati(
            [RisultatoWeb(url="https://example.com/x", titolo="Vuota", testo="")],
            domanda="qualunque cosa", limite=3,
        )

        assert passaggi == []

    def test_la_provenienza_dice_che_viene_dal_web(self):
        """Una fonte che qualcuno ha scelto di mettere nel corpus vale
        diversamente da una trovata da un motore di ricerca."""
        passaggi = passaggi_da_risultati(
            [RisultatoWeb(url="https://example.com/x", titolo="X", testo=PAGINA)],
            domanda="garanzia", limite=1,
        )

        assert passaggi[0].corrispondenza.provenienza == "web"

    def test_il_titolo_porta_il_dominio(self):
        """Nel prompt il modello vede questa riga e non l'URL: «secondo un
        sito» è diverso da «secondo la documentazione ufficiale»."""
        passaggi = passaggi_da_risultati(
            [RisultatoWeb(
                url="https://docs.example.com/garanzia", titolo="Garanzia",
                testo=PAGINA, pubblicato="2026-03-01T10:00:00Z",
            )],
            domanda="garanzia", limite=1,
        )

        titolo = passaggi[0].corrispondenza.documento_titolo
        assert "docs.example.com" in titolo
        assert "2026-03-01" in titolo

    def test_gli_identificativi_sono_stabili(self):
        """La stessa pagina recuperata due volte porta lo stesso id: una
        traccia di ieri si confronta con una di oggi."""
        def una():
            return passaggi_da_risultati(
                [RisultatoWeb(url="https://example.com/x", titolo="X", testo=PAGINA)],
                domanda="garanzia", limite=1,
            )[0].corrispondenza.chunk_id

        assert una() == una()

    def test_una_pagina_non_occupa_il_contesto_da_sola(self):
        passaggi = passaggi_da_risultati(
            [RisultatoWeb(url="https://example.com/x", titolo="X", testo=PAGINA * 10)],
            domanda="garanzia", limite=6,
        )

        assert len(passaggi) <= 2


class TestFornitoreSpento:
    async def test_disattivata_non_solleva(self):
        """Una personalità con la ricerca accesa su un impianto che non ce
        l'ha deve rispondere lo stesso, col corpus che ha."""
        fornitore = RicercaDisattivata()

        assert not fornitore.disponibile()
        assert await fornitore.cerca("qualunque cosa") == []

    def test_il_fornitore_predefinito_e_spento(self, monkeypatch):
        from platform_core.knowledge.ricerca_web import fornitore_ricerca
        from platform_core.settings import get_settings

        monkeypatch.delenv("PERSONA_RICERCA_PROVIDER", raising=False)
        get_settings.cache_clear()

        assert not fornitore_ricerca().disponibile()

    def test_un_fornitore_sconosciuto_non_fa_esplodere_l_avvio(self, monkeypatch):
        """Un valore sbagliato spegne la ricerca, e lo dice: spegnerla in
        silenzio lascerebbe una voce che non cerca e nessuno sa perché."""
        from platform_core.knowledge import ricerca_web
        from platform_core.settings import get_settings

        from .conftest import avvisi_di

        monkeypatch.setenv("PERSONA_RICERCA_PROVIDER", "motore-inventato")
        get_settings.cache_clear()
        try:
            with avvisi_di(ricerca_web) as avvisi:
                fornitore = ricerca_web.fornitore_ricerca()
            assert not fornitore.disponibile()
            assert any("motore-inventato" in a for a in avvisi)
        finally:
            get_settings.cache_clear()


class RicercaFinta:
    """Un fornitore che non tocca la rete."""

    nome = "finta"

    def __init__(self, risultati=(), esplode: str = "") -> None:
        self._risultati = list(risultati)
        self._esplode = esplode
        self.chiamata = None

    def disponibile(self) -> bool:
        return True

    async def cerca(self, domanda, *, siti=(), solo=False, limite=3):
        self.chiamata = {"domanda": domanda, "siti": list(siti), "solo": solo}
        if self._esplode:
            raise RuntimeError(self._esplode)
        return self._risultati


def versione(ricerca_online=None):
    from platform_core.domain.knowledge_models import PersonalityVersion

    return PersonalityVersion(
        system_prompt="Sei una voce di prova, e rispondi in italiano.",
        rag_config={"ricerca_online": ricerca_online} if ricerca_online else {},
    )


class TestNelMotore:
    """Il turno intero, senza database e senza rete."""

    async def _prepara(self, motore, v, domanda="quanto dura la garanzia?"):
        return await motore.prepara(versione=v, domanda=domanda)

    def _motore(self, ricerca):
        from platform_core.runtime.persona_engine import PersonaEngine

        class ModelloMuto:
            name = "muto"
            motore = "other"

            async def stream(self, richiesta):  # pragma: no cover - non usato
                raise AssertionError("il turno non deve generare")

        return PersonaEngine(ModelloMuto(), ricerca=ricerca)

    async def test_spenta_non_interroga_nessuno(self):
        ricerca = RicercaFinta()
        turno = await self._prepara(self._motore(ricerca), versione())

        assert ricerca.chiamata is None
        assert turno.ricerca == {}, "senza ricerca accesa la traccia resta muta"

    async def test_accesa_porta_i_passaggi_nel_contesto(self):
        ricerca = RicercaFinta([
            RisultatoWeb(url="https://example.com/garanzia", titolo="Garanzia", testo=PAGINA),
        ])
        turno = await self._prepara(
            self._motore(ricerca),
            versione({"attiva": True, "siti": ["example.com"], "modo": "solo"}),
        )

        assert ricerca.chiamata["solo"] is True
        assert ricerca.chiamata["siti"] == ["example.com"]
        assert turno.recupero is not None
        assert turno.recupero.scelti
        assert turno.recupero.scelti[0].etichetta == "K1"
        assert turno.ricerca["eseguita"] is True
        assert turno.ricerca["domini"] == ["example.com"]

    async def test_una_ricerca_fallita_non_fa_fallire_la_risposta(self):
        """Un servizio esterno caduto è un contrattempo, non un guasto della
        piattaforma: la risposta esce col corpus che c'è."""
        ricerca = RicercaFinta(esplode="connessione rifiutata")
        turno = await self._prepara(
            self._motore(ricerca), versione({"attiva": True}),
        )

        assert turno.contesto is not None, "il turno si è preparato lo stesso"
        assert turno.ricerca["eseguita"] is False
        assert "connessione rifiutata" in turno.ricerca["motivo"]

    async def test_senza_fornitore_lo_dice_invece_di_tacere(self):
        turno = await self._prepara(
            self._motore(RicercaDisattivata()), versione({"attiva": True}),
        )

        assert turno.ricerca["eseguita"] is False
        assert "PERSONA_RICERCA_PROVIDER" in turno.ricerca["motivo"]

    async def test_la_ricerca_finisce_nella_traccia(self):
        """Quando una risposta manca di un fatto recente, la prima domanda è
        se la ricerca sia avvenuta."""
        ricerca = RicercaFinta([
            RisultatoWeb(url="https://example.com/x", titolo="X", testo=PAGINA),
        ])
        turno = await self._prepara(
            self._motore(ricerca), versione({"attiva": True, "modo": "anche"}),
        )

        assert turno.traccia_risposta()["ricerca"]["fornitore"] == "finta"
        assert turno.tempi_ms["ricerca_online"] >= 0

    async def test_i_passaggi_web_non_entrano_nello_strato_stabile(self):
        """Lo strato stabile deve restare identico byte per byte, o lo sconto
        del prompt caching sparisce in silenzio."""
        ricerca = RicercaFinta([
            RisultatoWeb(url="https://example.com/x", titolo="X", testo=PAGINA),
        ])
        v = versione({"attiva": True})

        primo = await self._prepara(self._motore(ricerca), v)
        secondo = await self._prepara(self._motore(ricerca), v)

        assert primo.contesto.testo_stabile == secondo.contesto.testo_stabile
        assert "example.com" not in primo.contesto.testo_stabile
