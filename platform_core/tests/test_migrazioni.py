"""Le migrazioni devono produrre lo schema che i modelli descrivono.

**Perché serve un test tutto suo.** La suite costruisce le tabelle con
`Base.metadata.create_all`, cioè dai modelli: è veloce e indipendente
dall'ordine, ma vuol dire che le migrazioni non vengono mai eseguite. Un
modello aggiornato senza la sua migrazione passa tutti i test e fallisce sul
database vero — che è precisamente com'è andata: `TIPI_BUILD` ha accolto
`digestione`, `ck_builds_kind` no, e la prima digestione accodata da console
è morta con una violazione di check dopo che 460 test erano passati.

**Perché i vincoli di controllo in particolare.** L'autogenerazione di Alembic
confronta tabelle, colonne, indici e chiavi esterne. **Non** confronta il
testo dei `CHECK`: quella parte va scritta a mano ogni volta, e ogni cosa che
va scritta a mano prima o poi viene dimenticata. Questo test è il posto dove
la dimenticanza si manifesta subito invece che in produzione.

Gira su un database usa e getta, perché applicare le migrazioni significa
partire da vuoto e quello della suite è già popolato dai modelli.
"""
from __future__ import annotations

import os
import re
from typing import Dict, Set

import pytest
from sqlalchemy import CheckConstraint, create_engine, text

from platform_core.domain.base import Base

# Importati perché registrino le proprie tabelle su `Base.metadata`: senza,
# il confronto girerebbe su uno schema parziale e passerebbe sempre.
from platform_core.domain import (  # noqa: F401
    billing_models, build_models, knowledge_models, memory_models, models,
)
from platform_core.tests.conftest import URL_TEST

#: Un database a parte, creato e distrutto dal test.
URL_MIGRAZIONI = os.getenv(
    "PERSONA_MIGRATIONS_DATABASE_URL",
    URL_TEST.rsplit("/", 1)[0] + "/persona_migrazioni",
)


def _crea_database_vuoto() -> bool:
    try:
        import psycopg
        from psycopg import sql
    except ImportError:  # pragma: no cover
        return False

    amministrazione = (
        URL_MIGRAZIONI.rsplit("/", 1)[0].replace("postgresql+psycopg://", "postgresql://")
    )
    nome = URL_MIGRAZIONI.rsplit("/", 1)[1]
    try:
        with psycopg.connect(
            f"{amministrazione}/postgres", connect_timeout=3, autocommit=True
        ) as c:
            c.execute(
                sql.SQL("drop database if exists {} with (force)").format(
                    sql.Identifier(nome)
                )
            )
            c.execute(sql.SQL("create database {}").format(sql.Identifier(nome)))
        with psycopg.connect(
            f"{amministrazione}/{nome}", connect_timeout=3, autocommit=True
        ) as c:
            c.execute("create extension if not exists vector")
            c.execute("create extension if not exists pg_trgm")
        return True
    except Exception:  # noqa: BLE001
        return False


def _valori_ammessi(espressione: str) -> Set[str]:
    """I letterali fra apici: è ciò che un `in (…)` sta davvero vincolando."""
    return set(re.findall(r"'([^']*)'", espressione))


@pytest.fixture(scope="module")
def schema_migrato():
    if not _crea_database_vuoto():
        pytest.skip("database non disponibile")

    from alembic import command
    from alembic.config import Config

    config = Config("alembic.ini")
    config.set_main_option(
        "sqlalchemy.url", URL_MIGRAZIONI.replace("%", "%%"),
    )
    # `env.py` legge l'URL dalle impostazioni, non dalla configurazione: la
    # variabile d'ambiente è l'unica leva che ha.
    precedente = os.environ.get("PERSONA_DATABASE_URL")
    os.environ["PERSONA_DATABASE_URL"] = URL_MIGRAZIONI
    try:
        from platform_core.settings import get_settings

        get_settings.cache_clear()
        command.upgrade(config, "head")
    finally:
        if precedente is None:
            os.environ.pop("PERSONA_DATABASE_URL", None)
        else:
            os.environ["PERSONA_DATABASE_URL"] = precedente
        from platform_core.settings import get_settings

        get_settings.cache_clear()

    motore = create_engine(URL_MIGRAZIONI)
    yield motore
    motore.dispose()


def _check_del_database(motore) -> Dict[str, str]:
    with motore.connect() as c:
        righe = c.execute(text(
            """
            select con.conname, pg_get_constraintdef(con.oid)
            from pg_constraint con
            join pg_class rel on rel.oid = con.conrelid
            join pg_namespace ns on ns.oid = rel.relnamespace
            where con.contype = 'c' and ns.nspname = 'public'
            """
        )).all()
    return {nome: definizione for nome, definizione in righe}


def _check_dei_modelli() -> Dict[str, str]:
    attesi: Dict[str, str] = {}
    for tabella in Base.metadata.tables.values():
        for vincolo in tabella.constraints:
            if isinstance(vincolo, CheckConstraint) and vincolo.name:
                attesi[vincolo.name] = str(vincolo.sqltext)
    return attesi


class TestVincoliDiControllo:
    def test_ce_ne_sono_da_confrontare(self, schema_migrato):
        """Se i modelli non ne dichiarassero nessuno, i test sotto non
        verificherebbero niente e passerebbero comunque."""
        assert _check_dei_modelli()

    def test_ogni_vincolo_dei_modelli_esiste_nel_database_migrato(
        self, schema_migrato,
    ):
        mancanti = set(_check_dei_modelli()) - set(_check_del_database(schema_migrato))
        assert not mancanti, (
            f"vincoli dichiarati nei modelli e assenti dopo le migrazioni: "
            f"{sorted(mancanti)}"
        )

    def test_ogni_vincolo_ammette_gli_stessi_valori(self, schema_migrato):
        """Il difetto vero: un `in (…)` cresciuto nei modelli e rimasto fermo
        nella migrazione. Si confrontano i letterali e non il testo, perché
        PostgreSQL riscrive la condizione e un confronto letterale
        segnalerebbe differenze inesistenti."""
        nel_database = _check_del_database(schema_migrato)

        divergenti = {}
        for nome, atteso in _check_dei_modelli().items():
            if nome not in nel_database:
                continue
            valori_attesi = _valori_ammessi(atteso)
            valori_veri = _valori_ammessi(nel_database[nome])
            if valori_attesi and valori_attesi != valori_veri:
                divergenti[nome] = (
                    sorted(valori_attesi - valori_veri),
                    sorted(valori_veri - valori_attesi),
                )

        assert not divergenti, (
            "vincoli che ammettono valori diversi da quelli dei modelli "
            f"(mancano nel database, in più nel database): {divergenti}"
        )


class TestTabelleEColonne:
    def test_nessuna_tabella_manca_dopo_le_migrazioni(self, schema_migrato):
        from sqlalchemy import inspect

        nel_database = set(inspect(schema_migrato).get_table_names())
        mancanti = set(Base.metadata.tables) - nel_database - {"alembic_version"}
        assert not mancanti, f"tabelle senza migrazione: {sorted(mancanti)}"

    def test_nessuna_colonna_manca_dopo_le_migrazioni(self, schema_migrato):
        from sqlalchemy import inspect

        ispettore = inspect(schema_migrato)
        presenti = set(ispettore.get_table_names())

        mancanti = {}
        for nome, tabella in Base.metadata.tables.items():
            if nome not in presenti:
                continue
            nel_database = {c["name"] for c in ispettore.get_columns(nome)}
            assenti = {c.name for c in tabella.columns} - nel_database
            if assenti:
                mancanti[nome] = sorted(assenti)

        assert not mancanti, f"colonne senza migrazione: {mancanti}"
