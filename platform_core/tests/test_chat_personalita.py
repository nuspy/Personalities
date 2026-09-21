"""Conversazione con una personalità.

Tre proprietà da proteggere, tutte con la stessa radice: una risposta
dev'essere **riconducibile** a ciò che l'ha prodotta.

- la versione si fissa all'apertura della conversazione, non si risolve a ogni
  turno: altrimenti modificare una personalità cambierebbe retroattivamente il
  senso degli scambi già avvenuti;
- le fonti arrivano prima del testo, perché servono mentre si legge;
- una citazione a un riferimento inesistente viene rilevata e dichiarata.
"""
from __future__ import annotations

import json
import uuid
from typing import AsyncIterator, List

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from platform_core.api.app import create_app
from platform_core.api.deps import get_embedder, get_llm_provider
from platform_core.auth.dependencies import current_principal
from platform_core.domain.base import utcnow
from platform_core.domain.knowledge_models import (
    KnowledgeBase, Personality, PersonalityKnowledgeBase, PersonalityVersion,
)
from platform_core.domain.session import get_db_session, get_session_factory
from platform_core.knowledge.chunker import ConfigurazioneChunking
from platform_core.knowledge.indexer import Indexer
from platform_core.llm.base import StreamChunk, Usage

from .conftest import richiede_database
from .test_retrieval import TESTO_VIRTU, EmbedderFinto

pytestmark = richiede_database


class ModelloCitante:
    """Risponde citando i riferimenti che gli si dicono."""

    name = "citante"

    def __init__(self, cita: List[str] | None = None):
        self.cita = cita if cita is not None else ["K1"]
        self.richieste: List = []

    async def stream(self, request) -> AsyncIterator[StreamChunk]:
        self.richieste.append(request)
        citazioni = " ".join(f"[{c}]" for c in self.cita)
        yield StreamChunk(text=f"La virtù si esercita {citazioni}.")
        yield StreamChunk(done=True, usage=Usage(total_tokens=42))


@pytest_asyncio.fixture
async def embedder() -> EmbedderFinto:
    return EmbedderFinto()


@pytest_asyncio.fixture
async def personalita(session, utente, embedder):
    """Una personalità pubblicata con un corpus indicizzato."""
    kb = KnowledgeBase(
        slug=f"corpus-{uuid.uuid4().hex[:8]}",
        name="Corpus di prova",
        embed_model=embedder.modello,
        owner_id=utente.id,
    )
    session.add(kb)
    await session.flush()

    await Indexer(
        session, embedder, config=ConfigurazioneChunking(token_obiettivo=60),
    ).indicizza(kb, titolo="Sulla virtù", testo=TESTO_VIRTU, lingua="it")

    p = Personality(
        slug=f"saggio-{uuid.uuid4().hex[:8]}",
        display_name="Il Saggio",
        description="Una voce di prova.",
        status="published",
        owner_id=utente.id,
    )
    session.add(p)
    await session.flush()

    versione = PersonalityVersion(
        personality_id=p.id,
        version=1,
        system_prompt="Sei un saggio. Rispondi brevemente.",
        behavior_rules={"regole": ["Non inventare"]},
        rag_config={"max_chunks": 3},
        published_at=utcnow(),
    )
    session.add(versione)
    await session.flush()

    p.current_version_id = versione.id
    session.add(PersonalityKnowledgeBase(
        personality_id=p.id, kb_id=kb.id, role="voice",
    ))
    await session.flush()

    return p, versione, kb


@pytest_asyncio.fixture
async def client(session, session_factory, principal_utente, embedder):
    app = create_app()
    app.dependency_overrides[current_principal] = lambda: principal_utente

    async def sessione_del_test():
        yield session

    app.dependency_overrides[get_db_session] = sessione_del_test
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    app.dependency_overrides[get_embedder] = lambda: embedder

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


def eventi_di(testo: str) -> List[tuple]:
    risultato = []
    tipo = None
    for riga in testo.splitlines():
        if riga.startswith("event:"):
            tipo = riga[6:].strip()
        elif riga.startswith("data:"):
            risultato.append((tipo, json.loads(riga[5:])))
    return risultato


def installa(client, modello) -> None:
    client._transport.app.dependency_overrides[get_llm_provider] = lambda: modello


class TestCatalogo:
    async def test_elenca_le_pubblicate(self, client, personalita):
        p, _, _ = personalita
        elenco = (await client.get("/personalities")).json()

        assert any(voce["slug"] == p.slug for voce in elenco)

    async def test_le_bozze_non_compaiono(self, client, session, personalita):
        """`status` esiste per questo: una bozza non è ancora un'offerta."""
        p, _, _ = personalita
        p.status = "draft"
        await session.flush()

        elenco = (await client.get("/personalities")).json()
        assert not any(voce["slug"] == p.slug for voce in elenco)

    async def test_il_catalogo_non_richiede_identita(self, session):
        """È ciò che si vede prima di entrare."""
        app = create_app()

        async def sessione_del_test():
            yield session

        app.dependency_overrides[get_db_session] = sessione_del_test
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            assert (await c.get("/personalities")).status_code == 200


class TestConversazioneConPersonalita:
    async def test_le_fonti_arrivano_prima_del_testo(self, client, personalita):
        """Il margine si popola mentre la risposta scorre, non dopo."""
        p, _, _ = personalita
        installa(client, ModelloCitante())

        risposta = await client.post(
            "/chat", json={"message": "come si acquista la virtù?", "personality": p.slug},
        )
        eventi = eventi_di(risposta.text)
        tipi = [t for t, _ in eventi]

        assert "sources" in tipi
        assert tipi.index("sources") < tipi.index("token")

        fonti = next(d for t, d in eventi if t == "sources")["passaggi"]
        assert fonti and fonti[0]["etichetta"] == "K1"
        assert fonti[0]["documento"] == "Sulla virtù"

    async def test_il_prompt_contiene_la_voce_e_i_passaggi(self, client, personalita):
        p, _, _ = personalita
        modello = ModelloCitante()
        installa(client, modello)

        await client.post(
            "/chat", json={"message": "come si acquista la virtù?", "personality": p.slug},
        )

        sistema = modello.richieste[0].messages[0]
        assert sistema.role == "system"
        assert "Sei un saggio" in sistema.content
        assert "[K1]" in "\n".join(m.content for m in modello.richieste[0].messages)

    async def test_la_versione_si_fissa_alla_conversazione(
        self, client, session, personalita
    ):
        """Senza, una conversazione di tre mesi fa non è riproducibile."""
        p, versione, _ = personalita
        installa(client, ModelloCitante())

        risposta = await client.post(
            "/chat", json={"message": "una domanda", "personality": p.slug},
        )
        cid = eventi_di(risposta.text)[0][1]["conversation_id"]

        from platform_core.domain.models import Conversation

        conv = await session.get(Conversation, uuid.UUID(cid))
        assert conv.personality_version_id == versione.id
        assert conv.personality_id == p.id

    async def test_una_personalita_inesistente_da_404(self, client):
        risposta = await client.post(
            "/chat", json={"message": "ciao", "personality": "non-esiste"},
        )
        assert risposta.status_code == 404

    async def test_senza_personalita_non_si_recupera_nulla(self, client):
        """Il percorso della fase 0 resta: prova il motore senza un carattere."""
        installa(client, ModelloCitante(cita=[]))

        risposta = await client.post("/chat", json={"message": "ciao"})
        tipi = [t for t, _ in eventi_di(risposta.text)]

        assert "sources" not in tipi
        assert "token" in tipi


class TestGroundcheck:
    async def test_una_citazione_inventata_viene_dichiarata(
        self, client, personalita
    ):
        """Il livello deterministico: due insiemi a confronto, costo zero.

        Non serve un secondo modello per accorgersi che [K9] non è stato
        fornito — e questa è la sola verifica che non può sbagliare.
        """
        p, _, _ = personalita
        installa(client, ModelloCitante(cita=["K9"]))

        risposta = await client.post(
            "/chat", json={"message": "una domanda", "personality": p.slug},
        )
        finale = next(d for t, d in eventi_di(risposta.text) if t == "done")

        assert finale["riferimenti_inventati"] == ["K9"]

    async def test_una_citazione_valida_non_viene_segnalata(
        self, client, personalita
    ):
        p, _, _ = personalita
        installa(client, ModelloCitante(cita=["K1"]))

        risposta = await client.post(
            "/chat", json={"message": "una domanda", "personality": p.slug},
        )
        finale = next(d for t, d in eventi_di(risposta.text) if t == "done")

        assert finale["riferimenti_inventati"] == []

    async def test_la_traccia_registra_come_e_nata_la_risposta(
        self, client, session, personalita
    ):
        """`answer_traces` è la prova di ciò che il modello ha letto."""
        p, versione, _ = personalita
        installa(client, ModelloCitante(cita=["K1", "K7"]))

        risposta = await client.post(
            "/chat", json={"message": "come si acquista la virtù?", "personality": p.slug},
        )
        cid = eventi_di(risposta.text)[0][1]["conversation_id"]

        from sqlalchemy import select

        from platform_core.domain.knowledge_models import AnswerTrace
        from platform_core.domain.models import Message

        messaggi = (await session.execute(
            select(Message).where(Message.conversation_id == uuid.UUID(cid))
        )).scalars().all()
        assistente = next(m for m in messaggi if m.role == "assistant")

        traccia = (await session.execute(
            select(AnswerTrace).where(AnswerTrace.message_id == assistente.id)
        )).scalar_one()

        assert traccia.personality_version_id == versione.id
        assert traccia.retrieved["scelti"], "i passaggi usati non sono registrati"
        assert traccia.grounding["riferimenti_inventati"] == ["K7"]
        assert "recupero" in traccia.latency_ms
