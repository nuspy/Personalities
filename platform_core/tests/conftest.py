"""Attrezzatura comune ai test della piattaforma.

**Perché un PostgreSQL vero e non SQLite.** I modelli usano `JSONB` e, presto,
`vector`: su SQLite non esistono, e riscriverli in tipi portabili
significherebbe provare uno schema diverso da quello che andrà in produzione —
cioè non provarlo. I test che toccano il database si saltano da soli se il
database non c'è, così la suite resta utilizzabile senza Docker; quelli che
contano davvero sull'isolamento, però, sono fra questi, e un `skipped` va letto
come «non verificato», non come «a posto».

Ogni test lavora in una transazione che viene annullata alla fine: le prove si
possono eseguire in qualunque ordine e nessuna vede i dati di un'altra.
"""
from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from platform_core.auth.keycloak import Principal
from platform_core.domain.base import Base
from platform_core.domain.models import User

@pytest.fixture(autouse=True)
def niente_modelli_veri(monkeypatch):
    """Durante le prove i modelli stanno su una porta morta.

    Una prova che per sbaglio ricade sul fornitore vero — un `embedder=None`,
    un provider non sostituito — altrimenti chiamerebbe LM Studio: passerebbe
    o fallirebbe secondo che un modello sia caricato, e caricarlo occupa la
    GPU, che su questa macchina è condivisa con altri progetti. Con la porta 9
    la chiamata fallisce subito e dice dove.
    """
    from platform_core.settings import get_settings

    for variabile in ("PERSONA_LLM_BASE_URL", "PERSONA_EMBEDDING_BASE_URL", "PERSONA_TTS_BASE_URL"):
        monkeypatch.setenv(variabile, "http://127.0.0.1:9/v1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


#: Un database separato da quello di sviluppo: le prove creano e cancellano
#: righe senza riguardo, e farlo su dati veri è un incidente che capita una
#: volta sola.
URL_TEST = os.getenv(
    "PERSONA_TEST_DATABASE_URL",
    "postgresql+psycopg://persona:persona@localhost:5433/persona_test",
)


def _database_raggiungibile() -> bool:
    """Il database di test esiste ed è raggiungibile? Lo si crea se manca."""
    try:
        import psycopg
        from psycopg import sql
    except ImportError:  # pragma: no cover
        return False

    amministrazione = URL_TEST.rsplit("/", 1)[0].replace("postgresql+psycopg://", "postgresql://")
    nome = URL_TEST.rsplit("/", 1)[1]
    try:
        with psycopg.connect(f"{amministrazione}/postgres", connect_timeout=3, autocommit=True) as c:
            esiste = c.execute(
                "select 1 from pg_database where datname = %s", (nome,)
            ).fetchone()
            if not esiste:
                c.execute(sql.SQL("create database {}").format(sql.Identifier(nome)))

        # Le estensioni vanno installate **in ogni database**: lo script di
        # inizializzazione del container tocca solo quello creato all'avvio, e
        # un database di test senza `vector` fallisce alla creazione delle
        # tabelle con «type "vector" does not exist» — un errore che sembra un
        # problema dei modelli e invece è di provisioning.
        with psycopg.connect(f"{amministrazione}/{nome}", connect_timeout=3, autocommit=True) as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS vector")
            c.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        return True
    except Exception:
        return False


DATABASE_DISPONIBILE = _database_raggiungibile()

richiede_database = pytest.mark.skipif(
    not DATABASE_DISPONIBILE,
    reason=(
        "PostgreSQL non raggiungibile: `docker compose up -d postgres`. "
        "I test di isolamento fra utenti NON sono stati eseguiti."
    ),
)


def pytest_asyncio_loop_factories(config, item):
    """Su Windows il loop dev'essere quello a selettore.

    psycopg 3 non funziona in modalità asincrona sul `ProactorEventLoop`, che è
    il predefinito di Windows: senza questo, i test sul database fallirebbero
    tutti con `InterfaceError` — un errore che non nomina il driver e fa
    pensare a un problema di connessione.

    Altrove si restituisce `None`, che lascia decidere a pytest-asyncio.
    """
    import sys

    if sys.platform == "win32":
        return {"asyncio": asyncio.SelectorEventLoop}
    return None


@pytest_asyncio.fixture(scope="session")
async def engine():
    if not DATABASE_DISPONIBILE:
        pytest.skip("database non disponibile")

    motore = create_async_engine(URL_TEST, poolclass=None)
    async with motore.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield motore
    await motore.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    """Una fabbrica di sessioni su una transazione che verrà annullata.

    La transazione esterna non viene mai confermata: anche un `commit()` del
    codice sotto prova resta dentro di essa e sparisce al termine. È ciò che
    rende i test indipendenti dall'ordine senza ricreare lo schema ogni volta.

    Si espone la **fabbrica** e non solo una sessione perché il codice che
    continua a lavorare dopo la risposta — lo streaming — ne apre una propria:
    se quella finisse sul database vero invece che qui, il test smetterebbe di
    osservare ciò che voleva osservare, e passerebbe lo stesso.
    """
    async with engine.connect() as connessione:
        transazione = await connessione.begin()
        factory = async_sessionmaker(
            bind=connessione,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )
        yield factory
        await transazione.rollback()


@pytest_asyncio.fixture
async def session(session_factory) -> AsyncIterator[AsyncSession]:
    async with session_factory() as sessione:
        yield sessione


@pytest.fixture
def principal_utente() -> Principal:
    return Principal(
        subject="sub-utente-1",
        email="utente@example.com",
        display_name="Utente Di Prova",
        roles={"user"},
    )


@pytest.fixture
def principal_altro() -> Principal:
    return Principal(
        subject="sub-utente-2",
        email="altro@example.com",
        display_name="Altro Utente",
        roles={"user"},
    )


@pytest.fixture
def principal_admin() -> Principal:
    return Principal(
        subject="sub-admin",
        email="admin@example.com",
        display_name="Amministratore",
        roles={"user", "admin"},
    )


@pytest_asyncio.fixture
async def utente(session, principal_utente) -> User:
    from platform_core.domain.repositories import UserRepository

    return await UserRepository(session).ensure(principal_utente)


@pytest_asyncio.fixture
async def altro_utente(session, principal_altro) -> User:
    from platform_core.domain.repositories import UserRepository

    return await UserRepository(session).ensure(principal_altro)
