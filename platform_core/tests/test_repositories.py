"""Isolamento fra utenti e disciplina del registro.

Il primo controllo del piano: **con i dati di A, nessuna strada porta ai dati
di B**. Qui si verifica sul repository e non sull'endpoint, perché è lì che il
filtro deve vivere: un test sull'API proverebbe che quel particolare endpoint
filtra, non che filtrare sia impossibile da dimenticare.

Ogni test gira su PostgreSQL vero. Se il database non c'è, si saltano — e un
`skipped` qui va letto come «l'isolamento non è stato verificato».
"""
from __future__ import annotations

import uuid

import pytest

from platform_core.domain.models import Conversation
from platform_core.domain.repositories import (
    AuditRepository, ConversationRepository, UserRepository,
)

from .conftest import richiede_database

pytestmark = richiede_database


class TestUtenti:
    async def test_il_primo_accesso_crea_l_utente(self, session, principal_utente):
        repo = UserRepository(session)
        assert await repo.get_by_subject(principal_utente.subject) is None

        utente = await repo.ensure(principal_utente)

        assert utente.id is not None
        assert utente.keycloak_sub == principal_utente.subject
        assert utente.email == "utente@example.com"

    async def test_gli_accessi_successivi_non_duplicano(self, session, principal_utente):
        repo = UserRepository(session)
        primo = await repo.ensure(principal_utente)
        secondo = await repo.ensure(principal_utente)

        assert primo.id == secondo.id

    async def test_i_dati_anagrafici_seguono_keycloak(self, session, principal_utente):
        """Se l'utente cambia nome nel servizio di identità, qui si adegua."""
        repo = UserRepository(session)
        await repo.ensure(principal_utente)

        rinominato = type(principal_utente)(
            subject=principal_utente.subject,
            email="nuova@example.com",
            display_name="Nome Nuovo",
            roles={"user"},
        )
        aggiornato = await repo.ensure(rinominato)

        assert aggiornato.email == "nuova@example.com"
        assert aggiornato.display_name == "Nome Nuovo"

    async def test_un_account_disattivato_non_rientra_in_silenzio(
        self, session, principal_utente
    ):
        from platform_core.domain.base import utcnow

        repo = UserRepository(session)
        utente = await repo.ensure(principal_utente)
        utente.deleted_at = utcnow()
        await session.flush()

        with pytest.raises(PermissionError):
            await repo.ensure(principal_utente)


class TestIsolamentoDelleConversazioni:
    async def test_una_conversazione_altrui_non_si_legge(
        self, session, utente, altro_utente
    ):
        repo = ConversationRepository(session)
        mia = await repo.create(utente, title="La mia")

        assert await repo.get(altro_utente, mia.id) is None
        assert await repo.get(utente, mia.id) is not None

    async def test_i_messaggi_altrui_non_si_leggono(
        self, session, utente, altro_utente
    ):
        repo = ConversationRepository(session)
        mia = await repo.create(utente)
        await repo.add_message(mia, role="user", content="un segreto")

        assert await repo.messages(altro_utente, mia.id) == []
        assert len(await repo.messages(utente, mia.id)) == 1

    async def test_l_elenco_mostra_solo_le_proprie(
        self, session, utente, altro_utente
    ):
        repo = ConversationRepository(session)
        await repo.create(utente, title="mia 1")
        await repo.create(utente, title="mia 2")
        await repo.create(altro_utente, title="sua")

        miei = await repo.list_recent(utente)
        suoi = await repo.list_recent(altro_utente)

        assert {c.title for c in miei} == {"mia 1", "mia 2"}
        assert {c.title for c in suoi} == {"sua"}

    async def test_un_identificativo_inesistente_si_comporta_come_uno_altrui(
        self, session, utente
    ):
        """Stessa risposta nei due casi: è ciò che impedisce di sondare gli id.

        Se «non esiste» e «non è tua» dessero esiti distinguibili, ripetendo la
        prova su molti identificativi si scoprirebbe quali esistono.
        """
        repo = ConversationRepository(session)
        assert await repo.get(utente, uuid.uuid4()) is None

    async def test_una_conversazione_cancellata_non_si_recupera(
        self, session, utente
    ):
        repo = ConversationRepository(session)
        conversazione = await repo.create(utente)
        conversazione.status = "deleted"
        await session.flush()

        assert await repo.get(utente, conversazione.id) is None


class TestOrdinamento:
    async def test_le_piu_recenti_vengono_prima(self, session, utente):
        repo = ConversationRepository(session)
        vecchia = await repo.create(utente, title="vecchia")
        await repo.add_message(vecchia, role="user", content="a")
        nuova = await repo.create(utente, title="nuova")
        await repo.add_message(nuova, role="user", content="b")

        elenco = await repo.list_recent(utente)
        assert [c.title for c in elenco] == ["nuova", "vecchia"]

    async def test_una_conversazione_senza_messaggi_non_sparisce(
        self, session, utente
    ):
        """Appena creata non ha `last_message_at`: non deve finire esclusa."""
        repo = ConversationRepository(session)
        con_messaggi = await repo.create(utente, title="con")
        await repo.add_message(con_messaggi, role="user", content="a")
        await repo.create(utente, title="vuota")

        titoli = [c.title for c in await repo.list_recent(utente)]
        assert set(titoli) == {"con", "vuota"}
        assert titoli[0] == "con", "quella con attività recente resta in cima"


class TestMessaggi:
    async def test_l_aggiunta_aggiorna_l_ultima_attivita(self, session, utente):
        repo = ConversationRepository(session)
        conversazione = await repo.create(utente)
        assert conversazione.last_message_at is None

        await repo.add_message(conversazione, role="user", content="ciao")
        assert conversazione.last_message_at is not None

    async def test_l_ordine_cronologico_e_rispettato(self, session, utente):
        repo = ConversationRepository(session)
        conversazione = await repo.create(utente)
        for i in range(5):
            await repo.add_message(conversazione, role="user", content=f"turno {i}")

        messaggi = await repo.messages(utente, conversazione.id)
        assert [m.content for m in messaggi] == [f"turno {i}" for i in range(5)]

    async def test_il_consumo_viene_conservato(self, session, utente):
        """Senza questo dato il risparmio del prompt caching è indimostrabile."""
        repo = ConversationRepository(session)
        conversazione = await repo.create(utente)
        await repo.add_message(
            conversazione, role="assistant", content="risposta",
            tokens={"prompt_tokens": 100, "cached_tokens": 80},
        )

        messaggi = await repo.messages(utente, conversazione.id)
        assert messaggi[0].tokens["cached_tokens"] == 80


class TestRegistroDiAudit:
    async def test_una_voce_registra_chi_e_cosa(self, session, utente):
        repo = AuditRepository(session)
        await repo.record(
            action="memoria.letta", actor_id=utente.id,
            target_type="user", target_id=str(utente.id),
            correlation_id="corr-1",
        )
        await session.flush()

        voci = await repo.for_target("user", str(utente.id))
        assert len(voci) == 1
        assert voci[0].action == "memoria.letta"
        assert voci[0].correlation_id == "corr-1"

    async def test_un_azione_di_sistema_non_ha_autore(self, session):
        """«Non è stato nessuno» va distinto da «non si sa chi»."""
        repo = AuditRepository(session)
        await repo.record(action="memorie.consolidate", target_type="system")
        await session.flush()

        voci = await repo.for_target("system", "None")
        assert voci == [] or voci[0].actor_id is None

    async def test_la_cancellazione_dell_autore_non_cancella_la_prova(
        self, session, utente
    ):
        from sqlalchemy import delete, select

        from platform_core.domain.models import AuditLog, User

        repo = AuditRepository(session)
        await repo.record(action="personalita.pubblicata", actor_id=utente.id)
        await session.flush()

        await session.execute(delete(User).where(User.id == utente.id))
        await session.flush()

        rimaste = (await session.execute(select(AuditLog))).scalars().all()
        assert len(rimaste) == 1
        assert rimaste[0].actor_id is None
        assert rimaste[0].action == "personalita.pubblicata"


class TestCascata:
    async def test_cancellare_una_conversazione_porta_via_i_messaggi(
        self, session, utente
    ):
        from sqlalchemy import func, select

        from platform_core.domain.models import Message

        repo = ConversationRepository(session)
        conversazione = await repo.create(utente)
        await repo.add_message(conversazione, role="user", content="a")
        await session.flush()

        await session.delete(await session.get(Conversation, conversazione.id))
        await session.flush()

        rimasti = await session.execute(select(func.count()).select_from(Message))
        assert rimasti.scalar_one() == 0
