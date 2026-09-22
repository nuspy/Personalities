"""Il modello che aiuta il recupero: riscrive la domanda, sceglie i passaggi.

Tre proprietà, e la prima vale più delle altre due messe insieme.

**Nessun aiuto fa fallire la richiesta che doveva aiutare.** Il modello del
recupero è un modello in più sul percorso della risposta: se un suo errore
diventasse un errore dell'utente, avremmo raddoppiato le cause di guasto per
migliorare il recupero.

**I testi non si riscrivono.** Il passaggio è ciò che l'utente vede nella
citazione e ciò contro cui il giudice verifica: una fonte condensata da un
modello è una fonte che nessuno ha scritto.

**Ciò che è stato scartato resta leggibile.** «Trovato e poi scartato» e «non
trovato» sono diagnosi diverse, e senza gli scartati non si distinguono.
"""
from __future__ import annotations

import pytest

from platform_core.llm.base import Message
from platform_core.runtime.recupero_assistito import (
    ConfigurazioneRecupero, RecuperoAssistito,
)


class ModelloFinto:
    """Risponde ciò che gli si dice, o esplode."""

    name = "finto"
    motore = "other"

    def __init__(self, risposta: str = "", esplode: str = "") -> None:
        self._risposta = risposta
        self._esplode = esplode
        self.richieste = []

    async def complete(self, richiesta) -> str:
        self.richieste.append(richiesta)
        if self._esplode:
            raise RuntimeError(self._esplode)
        return self._risposta


def passaggio(etichetta: str, testo: str):
    import uuid

    from platform_core.knowledge.retriever import PassaggioRecuperato
    from platform_core.knowledge.vector_store import Corrispondenza

    identificativo = uuid.uuid4()
    return PassaggioRecuperato(
        corrispondenza=Corrispondenza(
            chunk_id=identificativo, testo=testo, ordinale=0, sezione=None,
            documento_id=identificativo, documento_titolo="Documento",
            documento_uri=None, kb_id=identificativo, punteggio=1.0,
        ),
        punteggio_rrf=1.0,
        etichetta=etichetta,
    )


class TestConfigurazione:
    def test_spento_per_difetto(self):
        """Costa una chiamata per volta: acceso di default sarebbe latenza
        che nessuno ha chiesto."""
        c = ConfigurazioneRecupero.da_rag(None)

        assert not c.attivo
        assert not c.riscrivi_domanda and not c.seleziona_passaggi

    def test_si_legge_da_rag_config(self):
        c = ConfigurazioneRecupero.da_rag({
            "recupero_assistito": {
                "riscrivi_domanda": True, "seleziona_passaggi": False,
            }
        })

        assert c.attivo and c.riscrivi_domanda and not c.seleziona_passaggi


class TestRiscritturaDellaDomanda:
    async def test_rimette_dentro_il_soggetto(self):
        modello = ModelloFinto("Seneca e il valore del tempo")
        assistente = RecuperoAssistito(modello)

        domanda, motivo = await assistente.domanda_per_recupero(
            "e lui cosa ne pensava?",
            storico=[Message(role="user", content="parlami di Seneca")],
        )

        assert domanda == "Seneca e il valore del tempo"
        assert motivo == ""
        assert "parlami di Seneca" in modello.richieste[0].messages[1].content

    async def test_una_riscrittura_fallita_lascia_la_domanda_originale(self):
        """Il modello del recupero è uno in più sul percorso: un suo guasto
        non deve diventare un guasto dell'utente."""
        assistente = RecuperoAssistito(ModelloFinto(esplode="connessione persa"))

        domanda, motivo = await assistente.domanda_per_recupero("che ore sono?")

        assert domanda == "che ore sono?"
        assert "connessione persa" in motivo

    async def test_una_risposta_vuota_lascia_la_domanda_originale(self):
        assistente = RecuperoAssistito(ModelloFinto("   "))

        domanda, motivo = await assistente.domanda_per_recupero("che ore sono?")

        assert domanda == "che ore sono?"
        assert motivo

    async def test_una_riscrittura_che_divaga_viene_scartata(self):
        """Se il modello risponde invece di riscrivere, il risultato non è
        una interrogazione: cercarlo recupera peggio dell'originale."""
        assistente = RecuperoAssistito(ModelloFinto("Ecco. " + "parole " * 200))

        domanda, motivo = await assistente.domanda_per_recupero("il tempo")

        assert domanda == "il tempo"
        assert "troppo lunga" in motivo

    async def test_si_tiene_solo_la_prima_riga_senza_virgolette(self):
        assistente = RecuperoAssistito(
            ModelloFinto('"Seneca sul tempo"\nSpiegazione non richiesta.'),
        )

        domanda, _ = await assistente.domanda_per_recupero("e sul tempo?")

        assert domanda == "Seneca sul tempo"


class TestSceltaDeiPassaggi:
    async def test_tiene_quelli_indicati_nell_ordine_deciso(self):
        assistente = RecuperoAssistito(ModelloFinto("K3, K1"))
        passaggi = [
            passaggio("K1", "sul tempo"),
            passaggio("K2", "sul mercato"),
            passaggio("K3", "sulla brevità della vita"),
        ]

        tenuti, scartati, motivo = await assistente.scegli_passaggi(
            "la brevità della vita", passaggi,
        )

        assert [p.etichetta for p in tenuti] == ["K3", "K1"]
        assert [p.etichetta for p in scartati] == ["K2"]
        assert motivo == ""

    async def test_i_testi_restano_parola_per_parola(self):
        """Una fonte condensata da un modello è una fonte che nessuno ha
        scritto, e la citazione non corrisponderebbe al documento."""
        assistente = RecuperoAssistito(ModelloFinto("K1"))
        originale = "Non abbiamo poco tempo, ma ne perdiamo molto."
        passaggi = [passaggio("K1", originale), passaggio("K2", "altro")]

        tenuti, _, _ = await assistente.scegli_passaggi("il tempo", passaggi)

        assert tenuti[0].corrispondenza.testo == originale

    async def test_con_un_solo_passaggio_non_si_chiama_nessuno(self):
        modello = ModelloFinto("K1")
        assistente = RecuperoAssistito(modello)

        tenuti, _, _ = await assistente.scegli_passaggi("x", [passaggio("K1", "a")])

        assert len(tenuti) == 1
        assert modello.richieste == [], "una chiamata per confermare l'ovvio"

    async def test_una_selezione_fallita_tiene_tutto(self):
        assistente = RecuperoAssistito(ModelloFinto(esplode="il modello è giù"))
        passaggi = [passaggio("K1", "a"), passaggio("K2", "b")]

        tenuti, scartati, motivo = await assistente.scegli_passaggi("x", passaggi)

        assert len(tenuti) == 2 and scartati == []
        assert "il modello è giù" in motivo

    async def test_nessun_passaggio_utile_non_svuota_il_contesto(self):
        """Togliere ogni passaggio lascerebbe il modello senza niente a cui
        ancorarsi: è esattamente ciò che il recupero doveva impedire."""
        assistente = RecuperoAssistito(ModelloFinto("nessuno"))
        passaggi = [passaggio("K1", "a"), passaggio("K2", "b")]

        tenuti, scartati, motivo = await assistente.scegli_passaggi("x", passaggi)

        assert len(tenuti) == 2 and scartati == []
        assert "tenuti tutti" in motivo

    async def test_etichette_inventate_vengono_ignorate(self):
        assistente = RecuperoAssistito(ModelloFinto("K9, K2, K9"))
        passaggi = [passaggio("K1", "a"), passaggio("K2", "b")]

        tenuti, scartati, _ = await assistente.scegli_passaggi("x", passaggi)

        assert [p.etichetta for p in tenuti] == ["K2"]
        assert [p.etichetta for p in scartati] == ["K1"]

    async def test_al_selezionatore_si_manda_un_estratto(self):
        """Mandare quattromila caratteri per sei passaggi costerebbe più
        della risposta, e per decidere se c'entrano bastano le prime righe."""
        assistente = RecuperoAssistito(ModelloFinto("K1"))
        modello = assistente._provider
        passaggi = [passaggio("K1", "x" * 5000), passaggio("K2", "y" * 5000)]

        await assistente.scegli_passaggi("x", passaggi)

        inviato = modello.richieste[0].messages[1].content
        assert len(inviato) < 3000


class TestNelMotore:
    def _motore(self, assistente, retriever=None):
        from platform_core.runtime.persona_engine import PersonaEngine

        return PersonaEngine(ModelloFinto(), retriever=retriever, assistente=assistente)

    def _versione(self, **assistenza):
        from platform_core.domain.knowledge_models import PersonalityVersion

        return PersonalityVersion(
            system_prompt="Sei una voce di prova, e rispondi in italiano.",
            rag_config={"recupero_assistito": assistenza} if assistenza else {},
        )

    async def test_spento_non_chiama_nessuno(self):
        modello = ModelloFinto("qualcosa")
        motore = self._motore(RecuperoAssistito(modello))

        turno = await motore.prepara(versione=self._versione(), domanda="ciao")

        assert modello.richieste == []
        assert turno.recupero_assistito == {}

    async def test_la_domanda_cercata_finisce_nella_traccia(self):
        """Senza, una ricerca che non trova niente sembra colpa del corpus
        mentre è la domanda a essere stata riscritta male."""
        motore = self._motore(RecuperoAssistito(ModelloFinto("Seneca sul tempo")))

        turno = await motore.prepara(
            versione=self._versione(riscrivi_domanda=True), domanda="e sul tempo?",
        )

        traccia = turno.traccia_risposta()["recupero_assistito"]
        assert traccia["domanda_cercata"] == "Seneca sul tempo"
        assert turno.tempi_ms["riscrittura"] >= 0
