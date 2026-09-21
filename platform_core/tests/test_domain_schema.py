"""Vincoli sullo schema che nessuna revisione deve poter infrangere.

Non verificano il comportamento di una funzione: verificano che le tabelle
continuino a rispettare le regole da cui dipendono l'isolamento fra utenti e
la riproducibilità. Sono regole che si violano per distrazione — si aggiunge
una tabella e si dimentica un indice — e che non si manifestano finché i dati
non sono tanti o gli utenti più di uno.
"""
from __future__ import annotations

import pytest
from sqlalchemy import inspect

from platform_core.domain.base import Base, OwnedMixin
from platform_core.domain.models import AuditLog, Conversation, Message, User


def owned_tables():
    """Le tabelle che contengono dati appartenenti a un utente."""
    return [
        mapper.class_
        for mapper in Base.registry.mappers
        if issubclass(mapper.class_, OwnedMixin)
    ]


class TestIsolamento:
    def test_esiste_almeno_una_tabella_di_proprieta(self):
        """Se questo fallisce, i test sotto non verificherebbero nulla."""
        assert owned_tables()

    @pytest.mark.parametrize("model", owned_tables(), ids=lambda m: m.__tablename__)
    def test_ogni_tabella_di_proprieta_ha_un_indice_che_parte_dal_proprietario(
        self, model
    ):
        """Un indice che non comincia da `owner_id` non serve al filtro.

        PostgreSQL usa un indice composito solo a partire dalla sua prima
        colonna: `(last_message_at, owner_id)` sarebbe inutile per «le
        conversazioni di questo utente», e la differenza non si vede finché la
        tabella è piccola.
        """
        indici = model.__table__.indexes
        prime_colonne = {list(ix.columns)[0].name for ix in indici if len(ix.columns)}
        assert "owner_id" in prime_colonne, (
            f"{model.__tablename__} non ha indici che partano da owner_id: "
            f"trovati {sorted(prime_colonne) or 'nessuno'}"
        )

    @pytest.mark.parametrize("model", owned_tables(), ids=lambda m: m.__tablename__)
    def test_la_cancellazione_di_un_utente_si_propaga(self, model):
        """Un dato orfano di proprietario non è filtrabile da nessuno."""
        vincolo = next(
            fk for fk in model.__table__.foreign_keys
            if fk.column.table.name == "users"
        )
        assert vincolo.ondelete == "CASCADE"


class TestImmutabilita:
    def test_i_messaggi_non_hanno_data_di_modifica(self):
        """Un turno non si corregge: si aggiunge un turno nuovo."""
        assert "updated_at" not in Message.__table__.columns

    def test_il_registro_di_audit_non_ha_data_di_modifica(self):
        assert "updated_at" not in AuditLog.__table__.columns

    def test_l_autore_di_un_audit_sopravvive_alla_sua_cancellazione(self):
        """`SET NULL` e non `CASCADE`: il record resta, l'autore diventa ignoto.

        Con la propagazione, cancellare un amministratore cancellerebbe la
        prova di ciò che ha fatto — che è esattamente ciò a cui un registro di
        audit deve resistere.
        """
        vincolo = next(
            fk for fk in AuditLog.__table__.foreign_keys
            if fk.column.table.name == "users"
        )
        assert vincolo.ondelete == "SET NULL"


class TestIdentita:
    def test_l_utente_non_conserva_credenziali(self):
        """Le password stanno in Keycloak, e da nessun'altra parte."""
        sospette = {"password", "password_hash", "hashed_password", "secret", "totp"}
        assert not sospette & set(User.__table__.columns.keys())

    def test_il_soggetto_keycloak_e_unico(self):
        colonna = User.__table__.columns["keycloak_sub"]
        assert colonna.unique and not colonna.nullable

    def test_le_risorse_esposte_in_url_non_sono_indovinabili(self):
        """Id sequenziali in una URL invitano a provare quello accanto.

        Il filtro per proprietario nei repository rende comunque innocuo il
        tentativo, ma due difese sono meglio di una, e la seconda costa nulla.
        """
        for model in (Conversation, Message):
            tipo = inspect(model).primary_key[0].type
            assert tipo.python_type is __import__("uuid").UUID, (
                f"{model.__tablename__}.id dovrebbe essere un UUID"
            )
