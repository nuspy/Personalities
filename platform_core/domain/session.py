"""Connessione al database e sessioni.

**Perché asincrono.** Un turno di conversazione passa la quasi totalità del suo
tempo ad aspettare: il modello che genera, l'embedder che vettorizza, il
recupero che interroga. Con un motore sincrono ogni richiesta in attesa
occuperebbe un thread del pool, e poche decine di conversazioni contemporanee
basterebbero a saturarlo mentre la CPU sta ferma.

**Perché `psycopg` e non `asyncpg`.** psycopg 3 parla entrambe le lingue con lo
stesso URL: `postgresql+psycopg://` vale per `create_async_engine` qui e per il
motore sincrono di Alembic. Un solo driver, una sola stringa di connessione in
configurazione, nessuna occasione di far divergere le due.
"""
from __future__ import annotations

import logging
from typing import Annotated, AsyncIterator, Optional

from fastapi import Depends
from sqlalchemy.ext.asyncio import (
    AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine,
)

from ..settings import get_settings

logger = logging.getLogger(__name__)

_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine() -> AsyncEngine:
    """Il motore del processo, creato una volta sola.

    `pool_pre_ping` costa un `SELECT 1` per connessione ripresa dal pool, e
    compra il caso che altrimenti si manifesta come errore casuale in
    produzione: una connessione chiusa dall'altro capo — riavvio del database,
    timeout di un proxy — che il pool crede ancora buona.
    """
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            # Sopra questa soglia la richiesta fallisce invece di restare
            # appesa: una coda infinita sul pool trasforma un rallentamento del
            # database in un blocco totale dell'API, e senza alcun errore da
            # cui accorgersene.
            pool_timeout=10,
            echo=settings.log_level == "DEBUG",
        )
        logger.info("Motore di database avviato (%s)", _safe_url(settings.database_url))
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            # Dopo il commit gli oggetti restano leggibili: senza, ogni accesso
            # a un attributo dopo la chiusura della transazione tenterebbe un
            # nuovo caricamento — su una sessione chiusa, quindi un errore.
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Dipendenza FastAPI: una sessione per richiesta.

    Il commit non è qui. Sta al chiamante decidere cosa costituisce
    un'unità di lavoro: un commit automatico a fine richiesta salverebbe anche
    le scritture parziali di un endpoint che ha poi fallito a metà.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


#: La fabbrica di sessioni, come dipendenza.
#:
#: Serve dove il lavoro continua **dopo** che l'endpoint ha restituito — un
#: flusso SSE, per esempio: la sessione della richiesta a quel punto è già
#: chiusa, e ne va aperta una nuova. Passarla come dipendenza invece di
#: prenderla dal modulo la rende sostituibile, e senza questo un test che
#: sostituisce il database si troverebbe la scrittura finale andare altrove —
#: cioè a non verificare proprio il pezzo che voleva verificare.
SessionFactory = Annotated[
    async_sessionmaker[AsyncSession], Depends(get_session_factory)
]


async def dispose_engine() -> None:
    """Chiude il pool. Da chiamare all'arresto del processo e nei test."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


def _safe_url(url: str) -> str:
    """L'URL senza la password: finisce nei log, e i log finiscono altrove."""
    if "@" not in url:
        return url
    schema, _, rest = url.partition("://")
    credentials, _, host = rest.rpartition("@")
    user = credentials.split(":", 1)[0] if credentials else ""
    return f"{schema}://{user}:***@{host}" if user else f"{schema}://{host}"
