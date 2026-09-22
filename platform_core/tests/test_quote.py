"""I limiti per piano, e l'estrazione delle memorie che li rispetta.

I limiti erano definiti nel catalogo e nessuno li controllava; l'estrazione
delle memorie a fine conversazione esisteva e nessuno la chiamava. Due difetti
dello stesso genere: codice scritto, provato in isolamento, mai raggiunto dal
percorso vero. Qui si prova il percorso.
"""
from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Any, List

import pytest
import pytest_asyncio
from sqlalchemy import select

from platform_core.billing.credits import RegistroCrediti
from platform_core.billing.entitlements import Diritti
from platform_core.billing.quote import ContatoreQuote, QuotaSuperata
from platform_core.domain.base import utcnow
from platform_core.domain.billing_models import Plan
from platform_core.domain.memory_models import Memory
from platform_core.domain.models import Conversation, Message
from platform_core.memory.store import MemoriaPiena, MemoryStore

from .conftest import richiede_database
from .test_billing_api import ModelloBreve, piani  # noqa: F401
from .test_chat_personalita import client, embedder, installa, personalita  # noqa: F401

pytestmark = richiede_database


def diritti(**limiti) -> Diritti:
    d = Diritti(piano="free")
    d.limiti = {"messaggi_al_giorno": 20, "conversazioni": 5, "memorie": 50, **limiti}
    return d


async def _conversazione(session, utente, *, messaggi: int = 0, quando=None, stato="active") -> Conversation:
    c = Conversation(owner_id=utente.id, title="prova", status=stato)
    session.add(c)
    await session.flush()
    for i in range(messaggi):
        session.add(Message(
            conversation_id=c.id, role="user", content=f"domanda {i}",
            created_at=quando or utcnow(),
        ))
    await session.flush()
    return c


class TestMessaggiAlGiorno:
    async def test_oltre_il_limite_si_ferma_con_il_tempo_di_attesa(self, session, utente):
        await _conversazione(session, utente, messaggi=3)

        with pytest.raises(QuotaSuperata) as exc:
            await ContatoreQuote(session).verifica_messaggio(
                utente.id, diritti(messaggi_al_giorno=3), nuova_conversazione=False,
            )

        assert exc.value.limite == "messaggi_al_giorno"
        assert exc.value.riprova_tra and 0 < exc.value.riprova_tra <= 24 * 3600

    async def test_la_finestra_e_di_24_ore_non_il_giorno_di_calendario(self, session, utente):
        """«Al giorno» a mezzanotte UTC azzererebbe il contatore all'una di
        notte a Budapest: la finestra mobile non dipende da nessun fuso."""
        await _conversazione(session, utente, messaggi=3, quando=utcnow() - timedelta(hours=25))

        await ContatoreQuote(session).verifica_messaggio(
            utente.id, diritti(messaggi_al_giorno=3), nuova_conversazione=False,
        )

    async def test_contano_solo_le_domande_non_le_risposte(self, session, utente):
        c = await _conversazione(session, utente, messaggi=1)
        session.add_all([Message(conversation_id=c.id, role="assistant", content="r", created_at=utcnow()) for _ in range(5)])
        await session.flush()

        assert await ContatoreQuote(session).messaggi_24h(utente.id) == 1

    async def test_i_messaggi_di_un_altro_utente_non_contano(self, session, utente, altro_utente):
        await _conversazione(session, altro_utente, messaggi=10)
        assert await ContatoreQuote(session).messaggi_24h(utente.id) == 0

    async def test_illimitato_significa_illimitato(self, session, utente):
        await _conversazione(session, utente, messaggi=30)
        await ContatoreQuote(session).verifica_messaggio(
            utente.id, diritti(messaggi_al_giorno=-1), nuova_conversazione=True,
        )


class TestConversazioni:
    async def test_la_nuova_oltre_il_limite_viene_rifiutata_con_la_via_d_uscita(self, session, utente):
        for _ in range(2):
            await _conversazione(session, utente)

        with pytest.raises(QuotaSuperata) as exc:
            await ContatoreQuote(session).verifica_messaggio(
                utente.id, diritti(conversazioni=2), nuova_conversazione=True,
            )

        assert exc.value.riprova_tra is None, "non si libera col tempo"
        assert "Archiviane" in exc.value.messaggio

    async def test_continuare_una_conversazione_esistente_resta_possibile(self, session, utente):
        for _ in range(2):
            await _conversazione(session, utente)
        await ContatoreQuote(session).verifica_messaggio(
            utente.id, diritti(conversazioni=2), nuova_conversazione=False,
        )

    async def test_le_archiviate_non_contano(self, session, utente):
        await _conversazione(session, utente, stato="archived")
        await _conversazione(session, utente)
        assert await ContatoreQuote(session).conversazioni_attive(utente.id) == 1


class TestMemorie:
    async def test_al_limite_la_meno_importante_lascia_il_posto(self, session, utente):
        store = MemoryStore(session)
        poco = await store.ricorda(user_id=utente.id, content="Gli piace il tè", importance=0.2)
        await store.ricorda(user_id=utente.id, content="È medico", importance=0.8)

        nuova = await store.ricorda(
            user_id=utente.id, content="Ha due figlie", importance=0.9, limite=2,
        )

        await session.refresh(poco)
        assert poco.valid_to is not None, "chiusa, non cancellata"
        assert poco.meta["chiusa"] == "limite del piano"
        assert nuova.id is not None

    async def test_una_nuova_meno_importante_di_tutte_viene_rifiutata(self, session, utente):
        """Il contrario congelerebbe la memoria di chi l'ha riempita il primo
        mese di dettagli: le cose importanti dette dopo non entrerebbero mai."""
        store = MemoryStore(session)
        await store.ricorda(user_id=utente.id, content="È medico", importance=0.8)

        with pytest.raises(MemoriaPiena):
            await store.ricorda(user_id=utente.id, content="Oggi piove", importance=0.1, limite=1)

    async def test_le_superate_e_scadute_non_contano(self, session, utente):
        store = MemoryStore(session)
        m = await store.ricorda(user_id=utente.id, content="Vive a Budapest", importance=0.5)
        await store.sostituisci(m, "Vive a Vienna")

        assert await ContatoreQuote(session).memorie_vive(utente.id) == 1


class TestDallApi:
    async def test_il_ventunesimo_messaggio_e_un_429_con_retry_after(
        self, client, personalita, piani, session, utente,  # noqa: F811
    ):
        p, _, _ = personalita
        await RegistroCrediti(session).accredita(utente.id, 100)
        await _conversazione(session, utente, messaggi=20)

        installa(client, ModelloBreve())
        risposta = await client.post("/chat", json={"message": "Ancora", "personality": p.slug})

        assert risposta.status_code == 429
        assert int(risposta.headers["Retry-After"]) > 0
        assert "24 ore" in risposta.json()["detail"]

    async def test_un_rifiuto_per_troppe_conversazioni_non_ne_crea_una(
        self, client, personalita, piani, session, utente,  # noqa: F811
    ):
        p, _, _ = personalita
        await RegistroCrediti(session).accredita(utente.id, 100)
        for _ in range(5):
            await _conversazione(session, utente)

        installa(client, ModelloBreve())
        risposta = await client.post("/chat", json={"message": "Ciao", "personality": p.slug})

        assert risposta.status_code == 403
        assert await ContatoreQuote(session).conversazioni_attive(utente.id) == 5

    async def test_archiviare_libera_il_posto(
        self, client, personalita, piani, session, utente,  # noqa: F811
    ):
        p, _, _ = personalita
        await RegistroCrediti(session).accredita(utente.id, 100)
        vecchie = [await _conversazione(session, utente) for _ in range(5)]

        archiviata = await client.post(f"/conversations/{vecchie[0].id}/archive")
        assert archiviata.status_code == 200

        installa(client, ModelloBreve())
        risposta = await client.post("/chat", json={"message": "Ciao", "personality": p.slug})
        assert risposta.status_code == 200

    async def test_non_si_archivia_la_conversazione_di_un_altro(
        self, client, session, altro_utente,  # noqa: F811
    ):
        sua = await _conversazione(session, altro_utente)
        risposta = await client.post(f"/conversations/{sua.id}/archive")
        assert risposta.status_code == 404

    async def test_senza_catalogo_non_si_limita(
        self, client, personalita, session, utente,  # noqa: F811
    ):
        """L'installazione che non addebita non contingenta nemmeno."""
        p, _, _ = personalita
        await _conversazione(session, utente, messaggi=40)

        installa(client, ModelloBreve())
        risposta = await client.post("/chat", json={"message": "Ciao", "personality": p.slug})

        assert risposta.status_code == 200

    async def test_il_conto_mostra_quanto_resta(
        self, client, piani, session, utente,  # noqa: F811
    ):
        await _conversazione(session, utente, messaggi=4)

        uso = (await client.get("/me/billing")).json()["uso"]

        assert uso["messaggi_al_giorno"]["usati"] == 4
        assert uso["conversazioni"]["usati"] == 1

    async def test_una_memoria_manuale_oltre_il_limite_e_un_409(
        self, client, piani, session, utente,  # noqa: F811
    ):
        free = (await session.execute(select(Plan).where(Plan.slug == "free"))).scalar_one()
        free.limits = {"memorie": 1}
        store = MemoryStore(session)
        await store.ricorda(user_id=utente.id, content="È medico", importance=0.9)
        await session.flush()

        risposta = await client.post("/memory", json={"content": "Oggi piove", "importance": 0.1})

        assert risposta.status_code == 409


class ModelloEstrattore:
    """Risponde come un modello che ha trovato due memorie."""

    name = "estrattore"
    motore = "other"

    def __init__(self) -> None:
        self.chiamate = 0

    async def complete_json(self, request) -> Any:
        self.chiamate += 1
        if "riassunto" in request.messages[0].content.lower():
            return {"riassunto": "Hanno parlato del tempo."}
        return {"memorie": [
            {"contenuto": "Lavora come medico a Budapest", "genere": "fatto", "importanza": 0.8},
            {"contenuto": "Preferisce risposte brevi", "genere": "preferenza", "importanza": 0.6},
        ]}


class TestEstrazioneNellaPassata:
    async def test_la_passata_estrae_dalle_conversazioni_ferme(
        self, session, session_factory, monkeypatch, utente,
    ):
        """Il codice esisteva e non lo chiamava nessuno: in produzione la
        memoria a lungo termine non si sarebbe mai riempita."""
        from platform_core.jobs.periodico import estrai_memorie

        monkeypatch.setattr("platform_core.jobs.periodico.get_session_factory", lambda: session_factory)
        c = await _conversazione(session, utente)
        c.last_message_at = utcnow() - timedelta(hours=2)
        for ruolo, testo in [
            ("user", "Sono un medico e lavoro a Budapest da dieci anni, in ospedale."),
            ("assistant", "Un mestiere che chiede misura e pazienza."),
            ("user", "Preferisco risposte brevi, ho poco tempo fra un turno e l'altro."),
            ("assistant", "Sarò breve."),
        ]:
            session.add(Message(conversation_id=c.id, role=ruolo, content=testo, created_at=utcnow()))
        await session.commit()

        esito = await estrai_memorie(provider=ModelloEstrattore(), embedder=None)

        assert esito.conversazioni_chiuse == 1
        assert esito.memorie_estratte == 2
        contenuti = {m.content for m in (await session.execute(
            select(Memory).where(Memory.user_id == utente.id)
        )).scalars()}
        assert "Lavora come medico a Budapest" in contenuti

    async def test_la_seconda_passata_non_riestrae(
        self, session, session_factory, monkeypatch, utente,
    ):
        from platform_core.jobs.periodico import estrai_memorie

        monkeypatch.setattr("platform_core.jobs.periodico.get_session_factory", lambda: session_factory)
        c = await _conversazione(session, utente)
        c.last_message_at = utcnow() - timedelta(hours=2)
        for ruolo, testo in [
            ("user", "Sono un medico e lavoro a Budapest da dieci anni, in ospedale."),
            ("assistant", "Un mestiere che chiede misura."),
        ]:
            session.add(Message(conversation_id=c.id, role=ruolo, content=testo, created_at=utcnow()))
        await session.commit()

        modello = ModelloEstrattore()
        await estrai_memorie(provider=modello, embedder=None)
        seconda = await estrai_memorie(provider=modello, embedder=None)

        assert seconda.conversazioni_chiuse == 0
