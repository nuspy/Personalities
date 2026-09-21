"""Gli strati del prompt, e il vincolo che li tiene insieme.

Il test centrale è `test_lo_strato_stabile_e_byte_identico`: protegge una
proprietà che, se si rompe, non produce nessun errore. Il prompt resta
corretto, le risposte restano buone, e l'unica traccia è una bolletta più alta
— o, su un motore locale, una latenza che cresce senza ragione apparente.
Nessuno collegherebbe il sintomo alla causa.

Gli altri test proteggono il corollario: ciò che varia non deve finire sopra
il punto di cache, per nessuna strada.
"""
from __future__ import annotations

import uuid

import pytest

from platform_core.knowledge.retriever import PassaggioRecuperato
from platform_core.knowledge.vector_store import Corrispondenza
from platform_core.runtime.context_builder import (
    ContextBuilder, StratoStabile, StratoVolatile,
)
from platform_core.runtime.persona_engine import riferimenti_citati


def passaggio(etichetta: str, testo: str, documento: str = "Lettere") -> PassaggioRecuperato:
    p = PassaggioRecuperato(
        corrispondenza=Corrispondenza(
            chunk_id=uuid.uuid4(), testo=testo, ordinale=0, sezione=None,
            documento_id=uuid.uuid4(), documento_titolo=documento,
            documento_uri=None, kb_id=uuid.uuid4(), punteggio=0.8,
        ),
        punteggio_rrf=0.03,
    )
    p.etichetta = etichetta
    return p


@pytest.fixture
def stabile() -> StratoStabile:
    return StratoStabile(
        prompt_personalita="Sei Seneca. Scrivi in periodi brevi e chiudi con una massima.",
        regole=["Non citare fatti successivi al 65 d.C.", "Dai del tu"],
    )


class TestPrefissoStabile:
    def test_lo_strato_stabile_e_byte_identico(self, stabile):
        """Due richieste alla stessa personalità producono lo stesso prefisso.

        Se questo fallisce, qualcuno ha messo nello strato 0 qualcosa che
        cambia — un orario, un identificativo, un insieme iterato senza
        ordine — e lo sconto sul contesto è sparito senza che nulla lo dica.
        """
        builder = ContextBuilder()

        primo = builder.costruisci(
            stabile=stabile,
            volatile=StratoVolatile(passaggi=[passaggio("K1", "primo passaggio")]),
            domanda="Che cos'è la virtù?",
        )
        secondo = builder.costruisci(
            stabile=stabile,
            volatile=StratoVolatile(passaggi=[passaggio("K1", "un altro passaggio")]),
            domanda="E la morte?",
        )

        assert primo.testo_stabile == secondo.testo_stabile
        assert primo.messaggi[0].content == secondo.messaggi[0].content

    def test_l_ordine_delle_regole_non_conta(self):
        """Le regole si ordinano: un insieme le itererebbe a caso.

        È il modo più subdolo di rompere il prefisso, perché il contenuto è
        identico e solo i byte differiscono.
        """
        builder = ContextBuilder()
        a = builder.costruisci(
            stabile=StratoStabile(prompt_personalita="X", regole=["beta", "alfa"]),
        )
        b = builder.costruisci(
            stabile=StratoStabile(prompt_personalita="X", regole=["alfa", "beta"]),
        )

        assert a.testo_stabile == b.testo_stabile

    def test_i_passaggi_stanno_sotto_il_punto_di_cache(self, stabile):
        """I passaggi cambiano a ogni domanda: sopra annullerebbero lo sconto."""
        builder = ContextBuilder()
        contesto = builder.costruisci(
            stabile=stabile,
            volatile=StratoVolatile(passaggi=[passaggio("K1", "testo recuperato")]),
            domanda="una domanda",
        )

        stabile_reso = contesto.messaggi[contesto.punto_di_cache].content
        assert "testo recuperato" not in stabile_reso
        assert "testo recuperato" in "\n".join(m.content for m in contesto.messaggi)

    def test_le_memorie_stanno_sotto_il_punto_di_cache(self, stabile):
        builder = ContextBuilder()
        contesto = builder.costruisci(
            stabile=stabile,
            volatile=StratoVolatile(memorie=["Vive a Vienna"]),
            domanda="dove sono?",
        )

        assert "Vienna" not in contesto.messaggi[contesto.punto_di_cache].content

    def test_i_documenti_integrali_stanno_sopra(self, stabile):
        """Il CAG vero: materiale stabile incluso per intero nel prefisso.

        Sta sopra il punto di cache proprio perché non cambia — è l'unica
        differenza che conta rispetto ai passaggi recuperati.
        """
        stabile.documenti_integrali = ["Il testo integrale di un trattato breve."]
        contesto = ContextBuilder().costruisci(stabile=stabile, domanda="x")

        assert "trattato breve" in contesto.messaggi[contesto.punto_di_cache].content


class TestRiferimenti:
    def test_i_passaggi_sono_numerati_e_attribuiti(self, stabile):
        contesto = ContextBuilder().costruisci(
            stabile=stabile,
            volatile=StratoVolatile(passaggi=[
                passaggio("K1", "la virtù si esercita", "Epistulae"),
                passaggio("K2", "il tempo è nostro", "De brevitate"),
            ]),
            domanda="?",
        )
        reso = "\n".join(m.content for m in contesto.messaggi)

        assert "[K1] (Epistulae)" in reso
        assert "[K2] (De brevitate)" in reso
        assert contesto.riferimenti_validi() == {"K1", "K2"}

    def test_la_sezione_entra_nell_attribuzione(self, stabile):
        p = passaggio("K1", "testo", "Epistulae")
        p.corrispondenza.sezione = "Lettera 47"
        contesto = ContextBuilder().costruisci(
            stabile=stabile, volatile=StratoVolatile(passaggi=[p]), domanda="?",
        )

        assert "[K1] (Epistulae — Lettera 47)" in "\n".join(
            m.content for m in contesto.messaggi
        )

    def test_l_istruzione_di_citare_e_sempre_presente(self, stabile):
        """Senza, ogni risposta cita in modo diverso e la verifica è impossibile."""
        contesto = ContextBuilder().costruisci(stabile=stabile)
        assert "[K1]" in contesto.testo_stabile

    def test_si_riconoscono_i_riferimenti_citati(self):
        testo = "La virtù si esercita [K1], e il tempo è nostro [K3]. Senza dubbio."
        assert riferimenti_citati(testo) == {"K1", "K3"}

    def test_una_citazione_inventata_si_vede_dal_confronto(self, stabile):
        """Il groundcheck deterministico della fase 2, in nuce.

        Non serve un secondo modello per accorgersi che [K9] non esiste: basta
        confrontare due insiemi.
        """
        contesto = ContextBuilder().costruisci(
            stabile=stabile,
            volatile=StratoVolatile(passaggi=[passaggio("K1", "un passaggio")]),
            domanda="?",
        )
        inventate = riferimenti_citati("Come è noto [K9].") - contesto.riferimenti_validi()

        assert inventate == {"K9"}


class TestStruttura:
    def test_lo_storico_precede_la_domanda(self, stabile):
        from platform_core.llm.base import Message

        contesto = ContextBuilder().costruisci(
            stabile=stabile,
            storico=[
                Message(role="user", content="prima domanda"),
                Message(role="assistant", content="prima risposta"),
            ],
            domanda="seconda domanda",
        )
        ruoli = [m.role for m in contesto.messaggi]

        assert ruoli[0] == "system"
        assert contesto.messaggi[-1].content == "seconda domanda"
        assert ruoli.count("user") == 2

    def test_senza_materiale_volatile_non_si_aggiunge_un_messaggio_vuoto(self, stabile):
        contesto = ContextBuilder().costruisci(stabile=stabile, domanda="?")

        assert all(m.content.strip() for m in contesto.messaggi)
        assert contesto.caratteri_volatili == 0

    def test_la_misura_degli_strati_e_riportata(self, stabile):
        """Serve a vedere quanto contesto è riutilizzabile e quanto no."""
        contesto = ContextBuilder().costruisci(
            stabile=stabile,
            volatile=StratoVolatile(passaggi=[passaggio("K1", "x" * 500)]),
            domanda="?",
        )

        assert contesto.caratteri_stabili > 0
        assert contesto.caratteri_volatili > 500
