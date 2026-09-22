"""Il laboratorio: voti, esperimenti fra versioni, analisi, playground.

Le promesse:

1. **l'assegnazione è stabile** — la stessa persona ritrova la stessa
   variante, o il confronto misurerebbe il disorientamento;
2. **i pesi valgono** — una ripartizione 90/10 non esce 50/50;
3. **con pochi voti non si dichiara niente** — è il momento in cui si ferma
   un esperimento per il motivo sbagliato;
4. **un voto è di chi ha ricevuto la risposta**, e di nessun altro;
5. **il playground non lascia tracce nel prodotto** — né conversazioni, né
   crediti, né memorie — ma lascia la sua riga nel registro.
"""
from __future__ import annotations

import uuid
from collections import Counter

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from platform_core.api.deps import get_llm_provider
from platform_core.domain.base import utcnow
from platform_core.domain.knowledge_models import Personality, PersonalityVersion
from platform_core.domain.lab_models import AnswerFeedback
from platform_core.domain.models import AuditLog, Conversation, Message
from platform_core.lab.esperimenti import (
    EsperimentoNonValido, GestoreEsperimenti, Variante, assegna,
)
from platform_core.lab.statistica import confronta

from .conftest import richiede_database
from .test_admin import app_con
from .test_chat_personalita import (  # noqa: F401
    ModelloCitante, client, embedder, eventi_di, installa, personalita,
)


# ---- senza database ------------------------------------------------------


class TestStatistica:
    def test_con_pochi_voti_non_si_dichiara_niente(self):
        esito = confronta(8, 10, 2, 10)

        assert esito.p is None
        assert "insufficienti" in esito.verdetto

    def test_una_differenza_netta_e_significativa(self):
        esito = confronta(50, 100, 80, 100)

        assert esito.p < 0.05
        assert esito.verdetto == "la variante è migliore del riferimento"

    def test_una_differenza_piccola_non_lo_e(self):
        esito = confronta(50, 100, 55, 100)

        assert esito.p > 0.05
        assert esito.verdetto == "differenza non significativa"

    def test_tutti_d_accordo_non_divide_per_zero(self):
        esito = confronta(40, 40, 40, 40)

        assert esito.verdetto == "nessuna differenza"


class TestAssegnazione:
    VARIANTI = [
        Variante(uuid.uuid4(), 90, "A"),
        Variante(uuid.uuid4(), 10, "B"),
    ]

    def test_la_stessa_persona_ritrova_la_stessa_variante(self):
        esperimento = uuid.uuid4()

        scelte = {assegna(esperimento, 42, self.VARIANTI).etichetta for _ in range(20)}

        assert len(scelte) == 1

    def test_i_pesi_valgono(self):
        esperimento = uuid.uuid4()

        conteggi = Counter(
            assegna(esperimento, utente, self.VARIANTI).etichetta for utente in range(10_000)
        )

        assert 0.87 < conteggi["A"] / 10_000 < 0.93

    def test_esperimenti_diversi_mescolano_diversamente(self):
        """Con lo stesso sale per tutti, chi è in B in un esperimento sarebbe
        in B in tutti: gli stessi utenti farebbero da cavia ogni volta."""
        a, b = uuid.uuid4(), uuid.uuid4()
        pari = [Variante(uuid.uuid4(), 50, "A"), Variante(uuid.uuid4(), 50, "B")]

        uguali = sum(
            assegna(a, u, pari).etichetta == assegna(b, u, pari).etichetta
            for u in range(2_000)
        )

        assert 0.4 < uguali / 2_000 < 0.6


# ---- con database --------------------------------------------------------


@pytest_asyncio.fixture
async def due_versioni(session, personalita):  # noqa: F811
    p, v1, _ = personalita
    v2 = PersonalityVersion(
        personality_id=p.id, version=2,
        system_prompt="Sei un saggio. Rispondi con un aneddoto.",
        behavior_rules={"regole": ["Non inventare"]},
        rag_config={"max_chunks": 3},
        published_at=utcnow(),
    )
    session.add(v2)
    await session.flush()
    await session.commit()
    return p, v1, v2


@pytest_asyncio.fixture
async def admin(session, session_factory, principal_admin, embedder, utente):  # noqa: F811
    app = app_con(principal_admin, session, session_factory, embedder)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@richiede_database
class TestEsperimenti:
    async def test_servono_due_varianti_della_stessa_personalita(
        self, session, due_versioni, utente,
    ):
        p, v1, _ = due_versioni
        gestore = GestoreEsperimenti(session)

        with pytest.raises(EsperimentoNonValido):
            await gestore.crea(p, nome="uno solo", ipotesi=None,
                               varianti=[{"version_id": v1.id}], autore_id=utente.id)

        estranea = Personality(slug=f"altra-{uuid.uuid4().hex[:6]}", display_name="Altra", owner_id=utente.id)
        session.add(estranea)
        await session.flush()
        sua = PersonalityVersion(personality_id=estranea.id, version=1, system_prompt="x")
        session.add(sua)
        await session.flush()
        with pytest.raises(EsperimentoNonValido):
            await gestore.crea(p, nome="mescolato", ipotesi=None,
                               varianti=[{"version_id": v1.id}, {"version_id": sua.id}],
                               autore_id=utente.id)

    async def test_uno_solo_attivo_per_personalita(self, session, due_versioni, utente):
        p, v1, v2 = due_versioni
        gestore = GestoreEsperimenti(session)
        await gestore.crea(p, nome="primo", ipotesi=None,
                           varianti=[{"version_id": v1.id}, {"version_id": v2.id}],
                           autore_id=utente.id)

        with pytest.raises(EsperimentoNonValido):
            await gestore.crea(p, nome="secondo", ipotesi=None,
                               varianti=[{"version_id": v2.id}, {"version_id": v1.id}],
                               autore_id=utente.id)

    async def test_la_chat_serve_la_variante_e_la_registra(
        self, client, session, due_versioni, utente,  # noqa: F811
    ):
        p, v1, v2 = due_versioni
        esperimento = await GestoreEsperimenti(session).crea(
            p, nome="aneddoto", ipotesi="l'aneddoto piace di più",
            # La variante attesa si calcola con la stessa funzione della chat:
            # la prova non dipende da quale delle due tocchi a questo utente.
            varianti=[{"version_id": v1.id, "peso": 1}, {"version_id": v2.id, "peso": 100}],
            autore_id=utente.id,
        )
        await session.commit()
        attesa = assegna(esperimento.id, utente.id, [
            Variante(v1.id, 1, "A"), Variante(v2.id, 100, "B"),
        ])
        installa(client, ModelloCitante())

        r = await client.post("/chat", json={"message": "cos'è la virtù?", "personality": p.slug})
        conversazione_id = uuid.UUID(eventi_di(r.text)[0][1]["conversation_id"])

        conversazione = await session.get(Conversation, conversazione_id)
        await session.refresh(conversazione)
        assert conversazione.experiment_id == esperimento.id
        assert conversazione.personality_version_id == attesa.version_id

    async def test_i_risultati_contano_i_voti_per_variante(
        self, client, session, due_versioni, utente,  # noqa: F811
    ):
        p, v1, v2 = due_versioni
        esperimento = await GestoreEsperimenti(session).crea(
            p, nome="aneddoto", ipotesi=None,
            varianti=[{"version_id": v1.id, "peso": 1}, {"version_id": v2.id, "peso": 100}],
            autore_id=utente.id,
        )
        await session.commit()
        installa(client, ModelloCitante())

        r = await client.post("/chat", json={"message": "cos'è la virtù?", "personality": p.slug})
        salvato = next(d for t, d in eventi_di(r.text) if t == "salvato")
        await client.put(f"/messages/{salvato['message_id']}/feedback", json={"voto": 1})

        risultati = await GestoreEsperimenti(session).risultati(esperimento)

        attesa = assegna(esperimento.id, utente.id, [
            Variante(v1.id, 1, "A"), Variante(v2.id, 100, "B"),
        ])
        servita = next(v for v in risultati["varianti"] if v["version_id"] == str(attesa.version_id))
        assert servita["risposte"] == 1
        assert servita["su"] == 1
        assert servita["approvazione"] == 1.0
        assert "insufficienti" in risultati["varianti"][1]["confronto"]["verdetto"]

    async def test_concludere_pubblica_il_vincitore(self, admin, session, due_versioni, utente):
        p, v1, v2 = due_versioni
        r = await admin.post(f"/admin/personalities/{p.id}/experiments", json={
            "nome": "aneddoto", "varianti": [{"version_id": str(v1.id)}, {"version_id": str(v2.id)}],
        })
        assert r.status_code == 201, r.text

        fine = await admin.post(f"/admin/experiments/{r.json()['id']}/conclude", json={
            "vincitore": str(v2.id), "conclusione": "l'aneddoto vince", "pubblica": True,
        })

        assert fine.status_code == 200, fine.text
        assert fine.json()["stato"] == "concluso"
        await session.refresh(p)
        assert p.current_version_id == v2.id


@richiede_database
class TestVoti:
    async def _risposta(self, client, p):
        installa(client, ModelloCitante())
        r = await client.post("/chat", json={"message": "cos'è la virtù?", "personality": p.slug})
        return next(d for t, d in eventi_di(r.text) if t == "salvato")["message_id"], \
            eventi_di(r.text)[0][1]["conversation_id"]

    async def test_votare_e_rivotare_cambia_il_voto(self, client, session, personalita):  # noqa: F811
        p, _, _ = personalita
        messaggio, conversazione = await self._risposta(client, p)

        await client.put(f"/messages/{messaggio}/feedback", json={"voto": -1, "motivo": "troppo lunga"})
        await client.put(f"/messages/{messaggio}/feedback", json={"voto": 1})

        voto = await session.get(AnswerFeedback, uuid.UUID(messaggio))
        await session.refresh(voto)
        assert voto.vote == 1
        assert voto.reason is None, "il motivo del voto contrario non descrive quello favorevole"

        letti = (await client.get(f"/conversations/{conversazione}/messages")).json()
        assert next(m for m in letti if m["id"] == messaggio)["voto"] == 1

    async def test_il_voto_si_toglie(self, client, session, personalita):  # noqa: F811
        p, _, _ = personalita
        messaggio, _ = await self._risposta(client, p)
        await client.put(f"/messages/{messaggio}/feedback", json={"voto": 1})

        r = await client.delete(f"/messages/{messaggio}/feedback")

        assert r.status_code == 200
        assert await session.get(AnswerFeedback, uuid.UUID(messaggio)) is None

    async def test_non_si_vota_la_risposta_di_un_altro(
        self, client, session, personalita, altro_utente,  # noqa: F811
    ):
        altrui = Conversation(owner_id=altro_utente.id, title="sua")
        session.add(altrui)
        await session.flush()
        risposta = Message(conversation_id=altrui.id, role="assistant", content="sua", created_at=utcnow())
        session.add(risposta)
        await session.commit()

        r = await client.put(f"/messages/{risposta.id}/feedback", json={"voto": 1})

        assert r.status_code == 404

    async def test_un_voto_fuori_scala_e_rifiutato(self, client, personalita):  # noqa: F811
        p, _, _ = personalita
        messaggio, _ = await self._risposta(client, p)

        r = await client.put(f"/messages/{messaggio}/feedback", json={"voto": 5})

        assert r.status_code == 422


@richiede_database
class TestAnalisi:
    async def test_conta_risposte_e_voti_del_periodo(self, client, admin, personalita):  # noqa: F811
        p, _, _ = personalita
        installa(client, ModelloCitante())
        r = await client.post("/chat", json={"message": "cos'è la virtù?", "personality": p.slug})
        salvato = next(d for t, d in eventi_di(r.text) if t == "salvato")
        await client.put(f"/messages/{salvato['message_id']}/feedback", json={"voto": 1})

        analisi = (await admin.get("/admin/analytics?giorni=7")).json()

        assert len(analisi["al_giorno"]) == 7, "anche i giorni vuoti, o un calo sembra continuità"
        assert analisi["indicatori"]["risposte"] >= 1
        assert analisi["indicatori"]["voti"]["su"] >= 1
        assert any(x["slug"] == p.slug for x in analisi["per_personalita"])
        assert "precedente" in analisi

    async def test_solo_per_gli_amministratori(self, client):  # noqa: F811
        assert (await client.get("/admin/analytics")).status_code == 403


@richiede_database
class TestPlayground:
    async def test_prova_una_modifica_senza_toccare_il_prodotto(
        self, admin, session, personalita, utente,  # noqa: F811
    ):
        p, v1, _ = personalita
        modello = ModelloCitante()
        admin._transport.app.dependency_overrides[get_llm_provider] = lambda: modello
        conversazioni_prima = await session.scalar(select(func.count()).select_from(Conversation))

        r = await admin.post("/admin/playground", json={
            "personality_id": str(p.id),
            "system_prompt": "Sei un saggio impaziente.",
            "domanda": "cos'è la virtù?",
        })

        assert r.status_code == 200, r.text
        corpo = r.json()
        assert corpo["risposta"].startswith("La virtù")
        assert corpo["versione"]["modificata"] is True
        assert "impaziente" in corpo["prompt"][0]["content"], "il prompt mandato è quello modificato"
        assert corpo["passaggi"], "il recupero gira come in produzione"
        # La versione salvata è intatta, e nessuna conversazione è nata.
        await session.refresh(v1)
        assert "impaziente" not in v1.system_prompt
        assert await session.scalar(select(func.count()).select_from(Conversation)) == conversazioni_prima
        assert await session.scalar(
            select(func.count()).select_from(PersonalityVersion).where(PersonalityVersion.personality_id == p.id)
        ) == 1
        registro = (await session.execute(
            select(AuditLog).where(AuditLog.action == "playground.eseguito")
        )).scalars().first()
        assert registro is not None and registro.after["modificata"] is True

    async def test_senza_versione_di_partenza_e_un_409(self, admin, session, utente):
        vuota = Personality(slug=f"vuota-{uuid.uuid4().hex[:6]}", display_name="Vuota", owner_id=utente.id)
        session.add(vuota)
        await session.commit()

        r = await admin.post("/admin/playground", json={
            "personality_id": str(vuota.id), "domanda": "ciao",
        })

        assert r.status_code == 409

