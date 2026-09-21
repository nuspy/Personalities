"""Ambiente delle migrazioni.

Due scelte da motivare, perché entrambe sono deviazioni dal template di
Alembic.

**L'URL viene dalle impostazioni dell'applicazione**, non da `alembic.ini`. Una
migrazione applicata al database sbagliato è fra i pochi errori da cui non si
torna indietro con un `git revert`, e il modo tipico di commetterlo è avere due
stringhe di connessione che divergono.

**Il motore è sincrono**, anche se l'applicazione ne usa uno asincrono. Una
migrazione è un'operazione unica, seriale, che gira in un processo dedicato:
l'asincronia non le porterebbe nulla e obbligherebbe ogni revisione scritta a
mano a passare da `run_sync`.
"""
from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine, pool

# La radice del progetto, perché `alembic` può essere invocato da altrove.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from platform_core.domain.base import Base  # noqa: E402
from platform_core.domain import models  # noqa: E402,F401  (registra le tabelle)
from platform_core.domain import knowledge_models  # noqa: E402,F401
from platform_core.domain import build_models  # noqa: E402,F401
from platform_core.settings import get_settings  # noqa: E402

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def database_url() -> str:
    """L'URL sincrono, dedotto da quello dell'applicazione.

    psycopg 3 usa lo stesso nome di driver nei due modi, quindi la stringa è
    già buona così com'è. La conversione resta scritta per il caso in cui
    qualcuno configuri `asyncpg`: senza, `create_engine` fallirebbe con un
    errore che non nomina la causa.
    """
    url = get_settings().database_url
    return url.replace("+asyncpg", "+psycopg")


def include_object(obj, name, type_, reflected, compare_to) -> bool:
    """Tiene l'autogenerazione fuori da ciò che non le appartiene.

    Le estensioni installano tabelle e indici propri; `pg_trgm` e `vector` in
    particolare. Senza questo filtro, `--autogenerate` propone di cancellarli,
    perché non li trova fra i modelli.
    """
    if type_ == "table" and name in {"spatial_ref_sys"}:
        return False
    return True


def run_migrations_offline() -> None:
    """Genera l'SQL senza connettersi. Serve a farlo rivedere prima."""
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(database_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Senza, un `String(200)` allargato a `String(300)` non produce
            # alcuna migrazione e la differenza resta invisibile fino al primo
            # valore troncato.
            compare_type=True,
            compare_server_default=True,
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
