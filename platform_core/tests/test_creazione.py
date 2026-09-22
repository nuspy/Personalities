"""Personalità e corpora creati dagli utenti.

Le promesse:

1. **il piano decide** — chi non ha il diritto riceve 403 col motivo, chi ha
   raggiunto il limite anche;
2. **ciò che si crea è privato** — fuori dal catalogo, e per chiunque altro
   inesistente: 404, non 403;
3. **il prompt si versiona** — le conversazioni già avvenute restano legate
   alla versione che le ha prodotte;
4. **si collegano solo i propri corpora**, e si carica solo nei propri.
"""
from __future__ import annotations

import uuid

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from platform_core.api.app import create_app
from platform_core.api.deps import get_embedder, get_key_value_store, get_llm_provider
from platform_core.auth.dependencies import current_principal
from platform_core.billing.plans import GestoreAbbonamenti
from platform_core.builds.queue import CodaInMemoria
from platform_core.domain.billing_models import Plan
from platform_core.domain.build_models import Build
from platform_core.domain.knowledge_models import KnowledgeBase, PersonalityVersion
from platform_core.domain.models import Conversation
from platform_core.domain.session import get_db_session, get_session_factory
from platform_core.settings import get_settings

from .conftest import richiede_database
from .test_chat_personalita import ModelloCitante, embedder, eventi_di  # noqa: F401

pytestmark = richiede_database

PROMPT = "Sei Ipazia di Alessandria. Rispondi con rigore, e dichiara ciò che non sai."


def _client(session, session_factory, principal, embedder, coda):  # noqa: F811
    app = create_app()
    app.dependency_overrides[current_principal] = lambda: principal

    async def sessione_del_test():
        yield session

    app.dependency_overrides[get_db_session] = sessione_del_test
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_key_value_store] = lambda: coda
    app.dependency_overrides[get_llm_provider] = lambda: ModelloCitante([])
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest_asyncio.fixture
async def catalogo(session):
    """Un piano gratuito senza creazioni e uno che ne consente una."""
    righe = [
        Plan(slug="free", name="Gratuito", rank=0, credits_per_period=50,
             entitlements={"categorie": ["free"]}, limits={}),
        Plan(slug="creatore", name="Creatore", rank=1, price_monthly=900, credits_per_period=500,
             entitlements={"categorie": ["free"], "corpora_propri": True, "personalita_proprie": 1},
             limits={}),
    ]
    session.add_all(righe)
    await session.flush()
    return righe


@pytest_asyncio.fixture
async def coda():
    return CodaInMemoria()


@pytest_asyncio.fixture
async def io(session, session_factory, principal_utente, embedder, utente, coda):  # noqa: F811
    async with _client(session, session_factory, principal_utente, embedder, coda) as c:
        yield c


@pytest_asyncio.fixture
async def altro(session, session_factory, principal_altro, embedder, altro_utente, coda):  # noqa: F811
    async with _client(session, session_factory, principal_altro, embedder, coda) as c:
        yield c


async def _creatore(session, utente):
    gestore = GestoreAbbonamenti(session)
    await gestore.sottoscrivi(utente.id, await gestore.piano_per_slug("creatore"))
    await session.commit()


async def _nuova(client, nome="Ipazia"):
    return await client.post("/me/personalities", json={"nome": nome, "prompt": PROMPT})


class TestDiritti:
    async def test_senza_il_diritto_e_un_403_col_motivo(self, io, catalogo, session, utente):
        gestore = GestoreAbbonamenti(session)
        await gestore.sottoscrivi(utente.id, await gestore.piano_per_slug("free"))
        await session.commit()

        r = await _nuova(io)

        assert r.status_code == 403
        assert "piano" in r.json()["detail"]

    async def test_oltre_il_limite_e_un_403(self, io, catalogo, session, utente):
        await _creatore(session, utente)
        assert (await _nuova(io)).status_code == 201

        r = await _nuova(io, "Seconda")

        assert r.status_code == 403
        assert "massimo" in r.json()["detail"]

    async def test_archiviare_libera_il_posto(self, io, catalogo, session, utente):
        await _creatore(session, utente)
        prima = (await _nuova(io)).json()
        await io.delete(f"/me/personalities/{prima['id']}")

        assert (await _nuova(io, "Seconda")).status_code == 201

    async def test_senza_catalogo_non_si_limita(self, io, session):
        """Come le quote: dove non si paga, non si contingenta."""
        assert (await _nuova(io)).status_code == 201


class TestPrivata:
    async def test_non_entra_nel_catalogo(self, io, catalogo, session, utente):
        await _creatore(session, utente)
        creata = (await _nuova(io)).json()

        catalogo_pubblico = (await io.get("/personalities")).json()

        assert creata["slug"] not in {p["slug"] for p in catalogo_pubblico}
        assert creata["slug"] in {p["slug"] for p in (await io.get("/me/creations")).json()["personalita"]}

    async def test_ci_parla_chi_l_ha_creata(self, io, catalogo, session, utente):
        await _creatore(session, utente)
        creata = (await _nuova(io)).json()

        r = await io.post("/chat", json={"message": "chi sei?", "personality": creata["slug"]})

        assert r.status_code == 200
        assert eventi_di(r.text)[0][1]["personality"] == creata["slug"]

    async def test_per_gli_altri_non_esiste(self, io, altro, catalogo, session, utente):
        await _creatore(session, utente)
        creata = (await _nuova(io)).json()

        r = await altro.post("/chat", json={"message": "chi sei?", "personality": creata["slug"]})
        modifica = await altro.put(f"/me/personalities/{creata['id']}", json={"nome": "Mia"})

        assert r.status_code == 404
        assert modifica.status_code == 404


class TestVersioni:
    async def test_cambiare_il_prompt_crea_una_versione(self, io, catalogo, session, utente):
        await _creatore(session, utente)
        creata = (await _nuova(io)).json()
        r = await io.post("/chat", json={"message": "chi sei?", "personality": creata["slug"]})
        conversazione_id = uuid.UUID(eventi_di(r.text)[0][1]["conversation_id"])

        nuovo = PROMPT + " Parla per enigmi."
        modificata = (await io.put(f"/me/personalities/{creata['id']}", json={"prompt": nuovo})).json()

        assert modificata["versione"] == 2
        conversazione = await session.get(Conversation, conversazione_id)
        await session.refresh(conversazione)
        vecchia = await session.get(PersonalityVersion, conversazione.personality_version_id)
        assert vecchia.version == 1, "la conversazione resta sulla versione che l'ha prodotta"

    async def test_lo_stesso_prompt_non_crea_versioni(self, io, catalogo, session, utente):
        await _creatore(session, utente)
        creata = (await _nuova(io)).json()

        rinominata = (await io.put(f"/me/personalities/{creata['id']}", json={"nome": "Ipazia", "prompt": PROMPT})).json()

        assert rinominata["versione"] == 1


class TestCorpora:
    async def test_si_collegano_solo_i_propri(self, io, altro, catalogo, session, utente, altro_utente, embedder):  # noqa: F811
        await _creatore(session, utente)
        creata = (await _nuova(io)).json()
        suo = KnowledgeBase(slug=f"suo-{uuid.uuid4().hex[:6]}", name="Suo", embed_model=embedder.modello,
                            owner_id=altro_utente.id)
        session.add(suo)
        await session.commit()

        r = await io.put(f"/me/personalities/{creata['id']}/corpora", json={"kb_ids": [str(suo.id)]})

        assert r.status_code == 404

    async def test_corpus_proprio_collegato_e_caricato(self, io, catalogo, session, utente, coda, tmp_path, monkeypatch):
        monkeypatch.setenv("PERSONA_UPLOAD_DIR", str(tmp_path))
        get_settings.cache_clear()
        try:
            await _creatore(session, utente)
            creata = (await _nuova(io)).json()
            corpus = (await io.post("/me/corpora", json={"nome": "Opere"})).json()

            collegata = (await io.put(
                f"/me/personalities/{creata['id']}/corpora", json={"kb_ids": [corpus["id"]]},
            )).json()
            caricato = await io.post(
                f"/me/corpora/{corpus['id']}/uploads",
                files=[("file", ("commento.txt", "Il cerchio si misura col raggio.".encode(), "text/plain"))],
            )

            assert collegata["corpora"] == [corpus["id"]]
            assert caricato.status_code == 202, caricato.text
            lavoro = (await io.get(f"/me/jobs/{caricato.json()['build_id']}")).json()
            assert lavoro["stato"] == "in_coda"
        finally:
            monkeypatch.undo()
            get_settings.cache_clear()

    async def test_il_lavoro_di_un_altro_non_si_legge(self, io, session, altro_utente):
        suo = Build(personality_id=None, kind="ingestione", status="in_coda", owner_id=altro_utente.id)
        session.add(suo)
        await session.commit()

        r = await io.get(f"/me/jobs/{suo.id}")

        assert r.status_code == 404

    async def test_senza_diritto_niente_corpus(self, io, catalogo, session, utente):
        gestore = GestoreAbbonamenti(session)
        await gestore.sottoscrivi(utente.id, await gestore.piano_per_slug("free"))
        await session.commit()

        r = await io.post("/me/corpora", json={"nome": "Opere"})

        assert r.status_code == 403
        assert not (await session.execute(
            select(KnowledgeBase).where(KnowledgeBase.owner_id == utente.id)
        )).scalars().first()
