"""Indicizzazione e recupero ibrido, su PostgreSQL vero.

L'embedder è finto ma **deterministico**: costruisce il vettore dalle parole
del testo, così due testi che condividono vocabolario finiscono vicini. Serve a
provare la meccanica — fusione, filtro per base, isolamento — senza legare la
suite alla disponibilità di un modello o alla sua qualità.

Quest'ultima non è verificabile con un'asserzione, ed è giusto che sia così:
un test che pretende «il passaggio giusto è primo» su un modello reale diventa
un test della versione del modello, e fallisce il giorno in cui qualcuno lo
aggiorna.
"""
from __future__ import annotations

import hashlib
import math
import uuid
from typing import List, Sequence

import pytest

from platform_core.domain.knowledge_models import (
    DIMENSIONI_EMBEDDING, Chunk, KnowledgeBase,
)
from platform_core.knowledge.chunker import ConfigurazioneChunking
from platform_core.knowledge.indexer import Indexer, configurazione_per_lingua
from platform_core.knowledge.retriever import K_RRF, Retriever

from .conftest import richiede_database

pytestmark = richiede_database


class EmbedderFinto:
    """Vettori deterministici ricavati dalle parole.

    Ogni parola accende una coordinata scelta dal suo hash; il vettore si
    normalizza. Due testi con parole in comune hanno somiglianza coseno alta,
    due testi senza parole in comune vicina a zero — che è quanto basta per
    provare che il recupero vettoriale ordina, filtra e fonde come deve.
    """

    modello = "finto-deterministico"
    dimensioni = DIMENSIONI_EMBEDDING

    def _vettore(self, testo: str) -> List[float]:
        v = [0.0] * self.dimensioni
        for parola in testo.lower().split():
            pulita = parola.strip(".,;:!?«»\"'()")
            if not pulita:
                continue
            h = hashlib.sha256(pulita.encode()).digest()
            v[int.from_bytes(h[:4], "big") % self.dimensioni] += 1.0
        norma = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norma for x in v]

    async def documenti(self, testi: Sequence[str]) -> List[List[float]]:
        return [self._vettore(t) for t in testi]

    async def query(self, testo: str) -> List[float]:
        return self._vettore(testo)


@pytest.fixture
def embedder() -> EmbedderFinto:
    return EmbedderFinto()


@pytest.fixture
async def base(session, utente, embedder) -> KnowledgeBase:
    kb = KnowledgeBase(
        slug=f"prova-{uuid.uuid4().hex[:8]}",
        name="Corpus di prova",
        kind="corpus",
        embed_model=embedder.modello,
        owner_id=utente.id,
    )
    session.add(kb)
    await session.flush()
    return kb


@pytest.fixture
async def indexer(session, embedder) -> Indexer:
    return Indexer(
        session, embedder, config=ConfigurazioneChunking(token_obiettivo=60),
    )


TESTO_VIRTU = (
    "La virtù non si impara a parole ma con l'esercizio quotidiano. "
    "Chi rimanda l'esercizio rimanda la vita stessa. "
    "Nessuno diventa saggio per caso, e la saggezza non arriva col tempo soltanto."
)

TESTO_TEMPO = (
    "Il tempo è l'unica cosa che ci appartiene davvero. "
    "Lo consegniamo agli altri senza contarlo, e poi ci lamentiamo che sia poco. "
    "Nessuno ti restituirà gli anni che hai dato via."
)

TESTO_MERCATO = (
    "Il prezzo del petrolio è salito del tre per cento nella giornata di ieri. "
    "Gli operatori attribuiscono il rialzo alle tensioni nel golfo. "
    "I titoli energetici hanno chiuso in forte progresso."
)


class TestIndicizzazione:
    async def test_un_documento_diventa_passaggi_con_vettori(
        self, session, indexer, base
    ):
        esito = await indexer.indicizza(
            base, titolo="Epistulae", testo=TESTO_VIRTU, lingua="it",
        )

        assert not esito.saltato
        assert esito.passaggi >= 1

        from sqlalchemy import func, select

        from platform_core.domain.knowledge_models import ChunkVector

        vettori = await session.execute(
            select(func.count()).select_from(ChunkVector).where(ChunkVector.kb_id == base.id)
        )
        assert vettori.scalar_one() == esito.passaggi, (
            "ogni passaggio deve avere il suo vettore: uno senza esiste e non si trova mai"
        )

    async def test_lo_stesso_documento_non_si_indicizza_due_volte(
        self, indexer, base
    ):
        """Altrimenti il modello legge tre volte la stessa frase credendo di
        avere tre conferme indipendenti."""
        await indexer.indicizza(base, titolo="Epistulae", testo=TESTO_VIRTU)
        secondo = await indexer.indicizza(base, titolo="Epistulae", testo=TESTO_VIRTU)

        assert secondo.saltato
        assert "già presente" in secondo.motivo

    async def test_un_embedder_diverso_viene_rifiutato(self, indexer, base, session):
        """Mescolare vettori di modelli diversi dà somiglianze prive di senso."""
        base.embed_model = "un-altro-modello"
        await session.flush()

        with pytest.raises(ValueError, match="non sarebbero confrontabili"):
            await indexer.indicizza(base, titolo="x", testo=TESTO_VIRTU)

    async def test_le_statistiche_si_aggiornano(self, indexer, base):
        await indexer.indicizza(base, titolo="Epistulae", testo=TESTO_VIRTU)

        assert base.stats["documenti"] == 1
        assert base.stats["passaggi"] >= 1

    async def test_il_tsvector_viene_popolato(self, session, indexer, base):
        """Senza, la metà lessicale del recupero non trova mai nulla."""
        await indexer.indicizza(base, titolo="Epistulae", testo=TESTO_VIRTU, lingua="it")

        from sqlalchemy import select

        righe = await session.execute(
            select(Chunk.tsv).where(Chunk.kb_id == base.id)
        )
        valori = [r[0] for r in righe]
        assert valori and all(v for v in valori)


class TestConfigurazioneLingua:
    def test_l_italiano_usa_lo_stemming_italiano(self):
        assert configurazione_per_lingua("it") == "italian"

    def test_il_latino_ricade_su_simple(self):
        """PostgreSQL non ha il latino, e lo stemming italiano su un testo
        latino produce radici inesistenti."""
        assert configurazione_per_lingua("la") == "simple"

    def test_una_lingua_ignota_ricade_su_simple(self):
        assert configurazione_per_lingua("zz") == "simple"
        assert configurazione_per_lingua(None) == "simple"


class TestRecupero:
    async def test_trova_il_passaggio_pertinente(self, session, indexer, base, embedder):
        await indexer.indicizza(base, titolo="Sulla virtù", testo=TESTO_VIRTU, lingua="it")
        await indexer.indicizza(base, titolo="Mercati", testo=TESTO_MERCATO, lingua="it")

        esito = await Retriever(session, embedder).cerca(
            "come si acquisisce la virtù con l'esercizio", [base.id], limite=1,
        )

        assert esito.scelti
        assert esito.scelti[0].corrispondenza.documento_titolo == "Sulla virtù"

    async def test_i_passaggi_scelti_sono_numerati(self, session, indexer, base, embedder):
        await indexer.indicizza(base, titolo="Sulla virtù", testo=TESTO_VIRTU)
        await indexer.indicizza(base, titolo="Sul tempo", testo=TESTO_TEMPO)

        esito = await Retriever(session, embedder).cerca(
            "la vita e il tempo", [base.id], limite=3,
        )

        assert [p.etichetta for p in esito.scelti] == [
            f"K{i}" for i in range(1, len(esito.scelti) + 1)
        ]

    async def test_gli_scartati_restano_visibili(self, session, indexer, base, embedder):
        """«Non l'ha trovato» e «l'ha trovato e scartato» sono diagnosi diverse."""
        await indexer.indicizza(base, titolo="Sulla virtù", testo=TESTO_VIRTU)
        await indexer.indicizza(base, titolo="Sul tempo", testo=TESTO_TEMPO)
        await indexer.indicizza(base, titolo="Mercati", testo=TESTO_MERCATO)

        esito = await Retriever(session, embedder).cerca(
            "virtù esercizio tempo vita", [base.id], limite=1,
        )

        assert len(esito.scelti) == 1
        assert esito.scartati, "gli scartati non compaiono nella traccia"

    async def test_una_base_non_interrogata_non_contribuisce(
        self, session, indexer, base, utente, embedder
    ):
        """Il filtro per base è ciò che impedisce a un corpus di parlare in
        una conversazione che non lo riguarda."""
        altra = KnowledgeBase(
            slug=f"altra-{uuid.uuid4().hex[:8]}", name="Altra",
            embed_model=embedder.modello, owner_id=utente.id,
        )
        session.add(altra)
        await session.flush()

        await indexer.indicizza(base, titolo="Sulla virtù", testo=TESTO_VIRTU)
        await indexer.indicizza(altra, titolo="Mercati", testo=TESTO_MERCATO)

        esito = await Retriever(session, embedder).cerca(
            "petrolio prezzo mercato", [base.id], limite=5,
        )

        titoli = {p.corrispondenza.documento_titolo for p in esito.scelti}
        assert "Mercati" not in titoli

    async def test_senza_basi_non_si_interroga_nulla(self, session, embedder):
        esito = await Retriever(session, embedder).cerca("qualunque cosa", [])
        assert esito.scelti == []

    async def test_la_ricerca_lessicale_trova_la_parola_esatta(
        self, session, indexer, base, embedder
    ):
        """Il caso in cui il vettoriale è debole: un termine raro e preciso."""
        await indexer.indicizza(
            base, titolo="Cronaca", lingua="it",
            testo="Il console Lucio Anneo ricevette la delegazione. "
                  "La seduta durò fino a sera. Nessuna decisione fu presa.",
        )
        await indexer.indicizza(base, titolo="Sul tempo", testo=TESTO_TEMPO, lingua="it")

        esito = await Retriever(session, embedder).cerca(
            "Lucio Anneo", [base.id], limite=2,
        )

        assert esito.scelti
        assert any(
            "Lucio Anneo" in p.corrispondenza.testo for p in esito.scelti
        )


class TestRicercaLessicale:
    """I due modi in cui questa metà del recupero smette di funzionare in
    silenzio, trasformando l'ibrido in un vettoriale con più codice attorno."""

    async def test_la_domanda_si_interpreta_come_i_documenti(
        self, session, indexer, base, embedder
    ):
        """Documenti stemmati e domanda no: zero corrispondenze, sempre.

        Con i passaggi indicizzati in italiano «amici» è la radice «amic», e
        una domanda interpretata senza stemming cerca la parola intera. Il
        recupero continua a funzionare grazie al vettoriale, quindi nulla
        sembra rotto.
        """
        from platform_core.knowledge.vector_store import RicercaLessicale

        await indexer.indicizza(
            base, titolo="Amicizia", lingua="it",
            testo="Gli amici veri si riconoscono nelle difficoltà. "
                  "Chi ti cerca solo nella buona sorte non è un amico. "
                  "L'amicizia vera non chiede nulla in cambio di sé.",
        )

        assert base.text_config == "italian", (
            "la configurazione dev'essere registrata sulla base, o la domanda "
            "verrebbe interpretata diversamente dai documenti"
        )

        trovati = await RicercaLessicale(session).cerca(
            "amicizia", [base.id], limite=5,
        )
        assert trovati, "la forma flessa non ha incontrato la radice indicizzata"

    async def test_una_domanda_naturale_non_pretende_tutte_le_parole(
        self, session, indexer, base, embedder
    ):
        """I termini si uniscono in OR, non in AND.

        `websearch_to_tsquery` li mette in AND: su una domanda di otto parole
        nessun passaggio le contiene tutte, e la ricerca lessicale restituisce
        sempre zero. A scremare pensa l'ordinamento per rilevanza, non il
        filtro.
        """
        from platform_core.knowledge.vector_store import RicercaLessicale

        await indexer.indicizza(base, titolo="Sulla virtù", testo=TESTO_VIRTU, lingua="it")

        trovati = await RicercaLessicale(session).cerca(
            "come si fa a diventare saggi con l'esercizio quotidiano della virtù?",
            [base.id], limite=5,
        )
        assert trovati, "una domanda in lingua naturale non trova nulla in AND"

    async def test_una_domanda_senza_parole_utili_non_esplode(
        self, session, indexer, base, embedder
    ):
        """Sola punteggiatura: `to_tsquery` darebbe errore, qui no."""
        from platform_core.knowledge.vector_store import RicercaLessicale

        await indexer.indicizza(base, titolo="Sulla virtù", testo=TESTO_VIRTU, lingua="it")

        assert await RicercaLessicale(session).cerca("?!...", [base.id]) == []

    async def test_la_configurazione_si_fissa_col_primo_documento(
        self, session, indexer, base
    ):
        await indexer.indicizza(base, titolo="Uno", testo=TESTO_VIRTU, lingua="it")
        assert base.text_config == "italian"

        # Un documento in un'altra lingua non la cambia: i passaggi già
        # scritti diventerebbero irraggiungibili.
        await indexer.indicizza(base, titolo="Due", testo=TESTO_TEMPO, lingua="en")
        assert base.text_config == "italian"


class TestFusione:
    """RRF, provato sulla funzione pura invece che sul database."""

    def _passaggi(self, retriever, vettoriali, lessicali):
        from platform_core.knowledge.vector_store import Corrispondenza

        def corr(nome: str, punteggio: float) -> Corrispondenza:
            return Corrispondenza(
                chunk_id=uuid.uuid5(uuid.NAMESPACE_OID, nome), testo=nome,
                ordinale=0, sezione=None, documento_id=uuid.uuid4(),
                documento_titolo=nome, documento_uri=None,
                kb_id=uuid.uuid4(), punteggio=punteggio,
            )

        return retriever._fondi(
            [corr(n, 0.9) for n in vettoriali],
            [corr(n, 5.0) for n in lessicali],
        )

    async def test_chi_compare_in_entrambe_le_liste_vince(self, session, embedder):
        """Il motivo per cui l'ibrido esiste: due metodi indipendenti che
        concordano valgono più di uno solo che insiste."""
        retriever = Retriever(session, embedder)
        fusi = self._passaggi(retriever, ["a", "b", "c"], ["c", "d", "e"])

        assert fusi[0].corrispondenza.testo == "c"
        assert fusi[0].trovato_da_entrambe

    async def test_il_punteggio_segue_la_formula(self, session, embedder):
        retriever = Retriever(session, embedder)
        fusi = self._passaggi(retriever, ["x"], ["x"])

        atteso = 2 * (1.0 / (K_RRF + 1))
        assert fusi[0].punteggio_rrf == pytest.approx(atteso)

    async def test_un_risultato_di_una_sola_lista_resta(self, session, embedder):
        """Non si scarta ciò che un metodo solo ha trovato: è proprio il caso
        in cui l'altro metodo era cieco."""
        retriever = Retriever(session, embedder)
        fusi = self._passaggi(retriever, ["solo-vettoriale"], ["solo-lessicale"])

        testi = {p.corrispondenza.testo for p in fusi}
        assert testi == {"solo-vettoriale", "solo-lessicale"}

    async def test_le_posizioni_di_origine_restano_leggibili(self, session, embedder):
        retriever = Retriever(session, embedder)
        fusi = self._passaggi(retriever, ["a", "b"], ["b"])
        b = next(p for p in fusi if p.corrispondenza.testo == "b")

        assert b.posizione_vettoriale == 2
        assert b.posizione_lessicale == 1
