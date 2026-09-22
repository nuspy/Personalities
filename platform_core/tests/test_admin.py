"""La console di amministrazione.

Due gruppi di test, con due ragioni diverse.

`TestAutorizzazione` protegge il perimetro: ogni rotta amministrativa pretende
il ruolo, e il controllo è dichiarato sul router proprio perché ripeterlo
endpoint per endpoint significa dimenticarlo su quello aggiunto di fretta. Il
test lo verifica su **tutte** le rotte, scoprendole dallo schema invece di
elencarle — così una rotta nuova è coperta il giorno in cui viene scritta.

`TestTraccia` protegge l'altra promessa: ogni operazione lascia un record. Sta
nel repository e non nel router per la stessa ragione.
"""
from __future__ import annotations

import uuid
from typing import List

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from platform_core.api.app import create_app
from platform_core.api.deps import get_embedder
from platform_core.auth.dependencies import current_principal
from platform_core.auth.keycloak import Principal
from platform_core.domain.knowledge_models import Personality
from platform_core.domain.models import AuditLog
from platform_core.domain.session import get_db_session, get_session_factory

from .conftest import richiede_database
from .test_retrieval import TESTO_MERCATO, TESTO_TEMPO, TESTO_VIRTU, EmbedderFinto

pytestmark = richiede_database


@pytest_asyncio.fixture
async def embedder() -> EmbedderFinto:
    return EmbedderFinto()


def app_con(principal: Principal, session, session_factory, embedder):
    app = create_app()
    app.dependency_overrides[current_principal] = lambda: principal

    async def sessione_del_test():
        yield session

    app.dependency_overrides[get_db_session] = sessione_del_test
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    app.dependency_overrides[get_embedder] = lambda: embedder
    return app


@pytest_asyncio.fixture
async def admin(session, session_factory, principal_admin, embedder, utente):
    """Client con ruolo di amministratore."""
    app = app_con(principal_admin, session, session_factory, embedder)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


@pytest_asyncio.fixture
async def semplice(session, session_factory, principal_utente, embedder, utente):
    """Client con un utente qualunque: non deve entrare."""
    app = app_con(principal_utente, session, session_factory, embedder)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


def rotte_amministrative() -> List[tuple]:
    """Tutte le rotte sotto `/admin`, dedotte dallo schema.

    Elencarle a mano significherebbe che una rotta nuova resta scoperta
    proprio finché nessuno si ricorda di aggiungerla qui.
    """
    spec = create_app().openapi()
    coppie = []
    for percorso, metodi in spec["paths"].items():
        if not percorso.startswith("/admin"):
            continue
        for metodo in metodi:
            if metodo in ("get", "post", "patch", "put", "delete"):
                coppie.append((metodo, percorso))
    return coppie


class TestAutorizzazione:
    def test_esistono_rotte_da_proteggere(self):
        assert len(rotte_amministrative()) >= 10

    @pytest.mark.parametrize(
        "metodo,percorso", rotte_amministrative(),
        ids=lambda v: v if isinstance(v, str) else str(v),
    )
    async def test_un_utente_senza_ruolo_non_entra(
        self, semplice, metodo, percorso
    ):
        """403 su ogni rotta, scoperta dallo schema e non elencata a mano."""
        concreto = (
            percorso
            .replace("{personality_id}", str(uuid.uuid4()))
            .replace("{version_id}", str(uuid.uuid4()))
            .replace("{kb_id}", str(uuid.uuid4()))
            .replace("{document_id}", str(uuid.uuid4()))
        )
        risposta = await semplice.request(metodo.upper(), concreto, json={})

        assert risposta.status_code == 403, (
            f"{metodo.upper()} {percorso} è raggiungibile senza il ruolo admin"
        )

    async def test_l_amministratore_entra(self, admin):
        assert (await admin.get("/admin/personalities")).status_code == 200


class TestPersonalita:
    async def test_nasce_in_bozza(self, admin):
        """Una voce senza versione non può rispondere: comparire nel catalogo
        prima significherebbe offrire qualcosa che fallisce al primo tentativo."""
        risposta = await admin.post("/admin/personalities", json={
            "slug": "prova-uno", "display_name": "Prova",
        })

        assert risposta.status_code == 201
        assert risposta.json()["status"] == "draft"

    async def test_lo_slug_duplicato_da_409(self, admin):
        await admin.post("/admin/personalities", json={
            "slug": "duplicata", "display_name": "Prima",
        })
        seconda = await admin.post("/admin/personalities", json={
            "slug": "duplicata", "display_name": "Seconda",
        })

        assert seconda.status_code == 409

    async def test_uno_slug_malformato_viene_respinto(self, admin):
        risposta = await admin.post("/admin/personalities", json={
            "slug": "Con Spazi E Maiuscole", "display_name": "X",
        })
        assert risposta.status_code == 422

    async def test_pubblicare_una_versione_rende_servibile(self, admin):
        creata = (await admin.post("/admin/personalities", json={
            "slug": "da-pubblicare", "display_name": "Da pubblicare",
        })).json()

        versione = (await admin.post(
            f"/admin/personalities/{creata['id']}/versions",
            json={"system_prompt": "Sei una voce di prova.", "pubblica": True},
        )).json()

        dettaglio = (await admin.get(
            f"/admin/personalities/{creata['id']}"
        )).json()

        assert dettaglio["status"] == "published"
        assert dettaglio["current_version_id"] == versione["id"]

    async def test_le_versioni_si_accumulano(self, admin):
        """Una versione pubblicata non si modifica: se ne crea un'altra, e la
        precedente resta con le sue conversazioni."""
        creata = (await admin.post("/admin/personalities", json={
            "slug": "molte-versioni", "display_name": "Molte",
        })).json()

        for i in range(3):
            await admin.post(
                f"/admin/personalities/{creata['id']}/versions",
                json={"system_prompt": f"Versione numero {i}."},
            )

        dettaglio = (await admin.get(f"/admin/personalities/{creata['id']}")).json()

        assert [v["version"] for v in dettaglio["versioni"]] == [3, 2, 1]

    async def test_archiviare_non_cancella(self, admin, session):
        """Le conversazioni già avvenute si riferiscono alle sue versioni."""
        creata = (await admin.post("/admin/personalities", json={
            "slug": "da-archiviare", "display_name": "Da archiviare",
        })).json()

        await admin.post(f"/admin/personalities/{creata['id']}/archive")

        rimasta = await session.get(Personality, uuid.UUID(creata["id"]))
        assert rimasta is not None
        assert rimasta.status == "archived"

    async def test_una_personalita_inesistente_da_404(self, admin):
        risposta = await admin.get(f"/admin/personalities/{uuid.uuid4()}")
        assert risposta.status_code == 404


class TestCorpora:
    async def test_collegare_una_base_inesistente_da_404(self, admin):
        creata = (await admin.post("/admin/personalities", json={
            "slug": "senza-corpus", "display_name": "Senza",
        })).json()

        risposta = await admin.post(
            f"/admin/personalities/{creata['id']}/corpora",
            json={"kb_id": str(uuid.uuid4())},
        )
        assert risposta.status_code == 404

    async def test_collegare_e_scollegare(self, admin):
        base = (await admin.post("/admin/knowledge-bases", json={
            "slug": "base-prova", "name": "Base di prova",
        })).json()
        personalita = (await admin.post("/admin/personalities", json={
            "slug": "con-corpus", "display_name": "Con corpus",
        })).json()

        await admin.post(
            f"/admin/personalities/{personalita['id']}/corpora",
            json={"kb_id": base["id"], "role": "voice"},
        )
        dopo_collegamento = (await admin.get(
            f"/admin/personalities/{personalita['id']}"
        )).json()

        await admin.delete(
            f"/admin/personalities/{personalita['id']}/corpora/{base['id']}"
        )
        dopo_scollegamento = (await admin.get(
            f"/admin/personalities/{personalita['id']}"
        )).json()

        assert len(dopo_collegamento["corpora"]) == 1
        assert dopo_collegamento["corpora"][0]["role"] == "voice"
        assert dopo_scollegamento["corpora"] == []


class TestBasiDiConoscenza:
    async def test_il_modello_di_embedding_non_si_sceglie(self, admin, embedder):
        """È quello configurato, e registrarlo sulla base impedisce di
        interrogare domani con un modello diverso da quello che ha scritto i
        vettori."""
        risposta = await admin.post("/admin/knowledge-bases", json={
            "slug": "base-embed", "name": "Base",
        })

        assert risposta.json()["embed_model"] == embedder.modello

    async def test_slug_duplicato_da_409(self, admin):
        await admin.post("/admin/knowledge-bases", json={
            "slug": "doppia", "name": "Prima",
        })
        seconda = await admin.post("/admin/knowledge-bases", json={
            "slug": "doppia", "name": "Seconda",
        })
        assert seconda.status_code == 409


class TestOsservabilitaDelRecupero:
    async def test_mostra_scelti_e_scartati(self, admin, session, embedder, utente):
        """«Non l'ha trovato» e «l'ha trovato e messo settimo» sono due difetti
        diversi, con due rimedi diversi."""
        from platform_core.domain.knowledge_models import KnowledgeBase
        from platform_core.knowledge.chunker import ConfigurazioneChunking
        from platform_core.knowledge.indexer import Indexer

        kb = KnowledgeBase(
            slug=f"prova-{uuid.uuid4().hex[:8]}", name="Prova",
            embed_model=embedder.modello, owner_id=utente.id,
        )
        session.add(kb)
        await session.flush()

        indexer = Indexer(
            session, embedder, config=ConfigurazioneChunking(token_obiettivo=40),
        )
        # Tre documenti e non uno: con un corpus di due passaggi e un limite di
        # uno, «scartati» sarebbe vuoto per mancanza di candidati invece che
        # per una scelta del recupero — e il test non proverebbe nulla.
        for titolo, testo in [
            ("Sulla virtù", TESTO_VIRTU),
            ("Sul tempo", TESTO_TEMPO),
            ("Mercati", TESTO_MERCATO),
        ]:
            await indexer.indicizza(kb, titolo=titolo, testo=testo, lingua="it")

        risposta = await admin.post("/admin/retrieval/preview", json={
            "domanda": "la virtù e l'esercizio quotidiano",
            "kb_ids": [str(kb.id)],
            "limite": 1,
        })
        corpo = risposta.json()

        assert risposta.status_code == 200
        assert len(corpo["scelti"]) == 1
        assert corpo["scartati"], "gli scartati non compaiono"

        scelto = corpo["scelti"][0]
        assert "rrf" in scelto
        assert "posizione_vettoriale" in scelto
        assert "posizione_lessicale" in scelto

    async def test_non_genera_nulla(self, admin, session, embedder, utente):
        """Con la generazione in mezzo, ogni conclusione sul recupero è
        confusa dal modello: qui non c'è fornitore da sostituire perché non
        viene interrogato."""
        from platform_core.domain.knowledge_models import KnowledgeBase

        kb = KnowledgeBase(
            slug=f"vuota-{uuid.uuid4().hex[:8]}", name="Vuota",
            embed_model=embedder.modello, owner_id=utente.id,
        )
        session.add(kb)
        await session.flush()

        risposta = await admin.post("/admin/retrieval/preview", json={
            "domanda": "qualunque cosa", "kb_ids": [str(kb.id)],
        })

        assert risposta.status_code == 200
        assert risposta.json()["scelti"] == []


class TestRealizzazione:
    async def test_rag_e_sempre_disponibile(self, admin):
        """È la ragione per cui la piattaforma resta utile su un server nudo."""
        opzioni = (await admin.get("/admin/build-options")).json()

        assert opzioni["modi"]["rag"]["available"] is True

    async def test_le_funzioni_assenti_portano_il_motivo(self, admin):
        """Un comando spento senza spiegazione sembra un difetto."""
        opzioni = (await admin.get("/admin/build-options")).json()
        lora = opzioni["modi"]["lora"]

        if not lora["available"]:
            assert lora["reason"], "disabilitato senza dire perché"


class TestTraccia:
    """Ogni operazione amministrativa lascia un record."""

    async def test_la_creazione_viene_registrata(self, admin, session):
        from sqlalchemy import select

        await admin.post("/admin/personalities", json={
            "slug": "tracciata", "display_name": "Tracciata",
        })

        voci = (await session.execute(
            select(AuditLog).where(AuditLog.action == "personalita.creata")
        )).scalars().all()

        assert voci
        assert voci[-1].after["slug"] == "tracciata"

    async def test_la_modifica_registra_prima_e_dopo(self, admin, session):
        from sqlalchemy import select

        creata = (await admin.post("/admin/personalities", json={
            "slug": "da-modificare", "display_name": "Nome vecchio",
        })).json()

        await admin.patch(
            f"/admin/personalities/{creata['id']}",
            json={"display_name": "Nome nuovo"},
        )

        voce = (await session.execute(
            select(AuditLog).where(AuditLog.action == "personalita.modificata")
        )).scalars().all()[-1]

        assert voce.before["display_name"] == "Nome vecchio"
        assert voce.after["display_name"] == "Nome nuovo"

    async def test_una_modifica_che_non_cambia_nulla_non_sporca_il_registro(
        self, admin, session
    ):
        from sqlalchemy import func, select

        creata = (await admin.post("/admin/personalities", json={
            "slug": "invariata", "display_name": "Uguale",
        })).json()

        await admin.patch(
            f"/admin/personalities/{creata['id']}",
            json={"display_name": "Uguale"},
        )

        quante = await session.scalar(
            select(func.count()).select_from(AuditLog)
            .where(AuditLog.action == "personalita.modificata")
        )
        assert quante == 0

    async def test_l_autore_e_registrato(self, admin, session, principal_admin):
        from sqlalchemy import select

        await admin.post("/admin/personalities", json={
            "slug": "con-autore", "display_name": "Con autore",
        })

        voce = (await session.execute(
            select(AuditLog).where(AuditLog.action == "personalita.creata")
        )).scalars().all()[-1]

        assert voce.actor_id is not None

    async def test_il_registro_si_legge(self, admin):
        await admin.post("/admin/personalities", json={
            "slug": "per-il-registro", "display_name": "X",
        })

        voci = (await admin.get("/admin/audit")).json()

        assert any(v["action"] == "personalita.creata" for v in voci)

    async def test_il_registro_si_filtra(self, admin):
        await admin.post("/admin/personalities", json={
            "slug": "filtrata", "display_name": "X",
        })
        await admin.post("/admin/knowledge-bases", json={
            "slug": "filtrata-kb", "name": "Y",
        })

        solo_kb = (await admin.get("/admin/audit?azione=kb.creata")).json()

        assert solo_kb
        assert all(v["action"] == "kb.creata" for v in solo_kb)


class TestTassonomia:
    async def test_i_tipi_si_creano_e_si_assegnano(self, admin):
        """Tabella e non enumerazione: un tipo nuovo non richiede un rilascio."""
        tipo = (await admin.post("/admin/types", json={
            "slug": "filosofo", "name": "Filosofo",
        })).json()
        personalita = (await admin.post("/admin/personalities", json={
            "slug": "con-tipo", "display_name": "Con tipo",
        })).json()

        aggiornata = (await admin.put(
            f"/admin/personalities/{personalita['id']}/types",
            json=[tipo["id"]],
        )).json()

        assert [t["slug"] for t in aggiornata["tipi"]] == ["filosofo"]

    async def test_una_voce_puo_avere_piu_tipi(self, admin):
        """Seneca è filosofo e drammaturgo: costringerlo a scegliere
        perderebbe metà di ciò che lo rende cercabile."""
        uno = (await admin.post("/admin/types", json={
            "slug": "poeta", "name": "Poeta",
        })).json()
        due = (await admin.post("/admin/types", json={
            "slug": "drammaturgo", "name": "Drammaturgo",
        })).json()
        personalita = (await admin.post("/admin/personalities", json={
            "slug": "poliedrica", "display_name": "Poliedrica",
        })).json()

        aggiornata = (await admin.put(
            f"/admin/personalities/{personalita['id']}/types",
            json=[uno["id"], due["id"]],
        )).json()

        assert len(aggiornata["tipi"]) == 2

    async def test_le_categorie_si_ordinano_per_rango(self, admin):
        for slug, nome, rango in [
            ("gold", "Gold", 20), ("free", "Free", 0), ("base", "Base", 10),
        ]:
            await admin.post("/admin/categories", json={
                "slug": slug, "name": nome, "rank": rango,
            })

        categorie = (await admin.get("/admin/categories")).json()

        assert [c["slug"] for c in categorie] == ["free", "base", "gold"]


class TestMotoreLocale:
    """L'API chiede, non esegue.

    È la proprietà che tiene i comandi — e la chiave SSH verso la macchina dei
    modelli — fuori dal processo che riceve le richieste pubbliche. La console
    può chiedere «accendi» o «spegni»: non può dire *cosa* eseguire.
    """

    @pytest.fixture
    def con_registro(self, monkeypatch):
        """Il supporto condiviso del test al posto di quello vero.

        Sostituito sul modulo e non come dipendenza FastAPI: l'endpoint lo
        chiama direttamente, perché non è la richiesta a deciderlo.
        """
        from platform_core.capabilities.registry import InMemoryStore

        store = InMemoryStore()
        monkeypatch.setattr(
            "platform_core.api.deps.get_key_value_store", lambda: store,
        )
        return store

    async def test_senza_worker_la_console_lo_dice(self, admin, con_registro):
        risposta = await admin.get("/admin/motore")

        assert risposta.status_code == 200
        assert risposta.json()["stato"] == "non_gestito"
        assert risposta.json()["motivo"], "uno stato senza motivo non si spiega"

    async def test_la_richiesta_arriva_al_worker(self, admin, con_registro, session):
        from platform_core.llm.accensione import CHIAVE_RICHIESTA

        risposta = await admin.post("/admin/motore", json={"azione": "accendi"})

        assert risposta.status_code == 202
        assert con_registro.get(CHIAVE_RICHIESTA) == "accendi"

    async def test_chi_accende_una_gpu_resta_scritto(
        self, admin, con_registro, session,
    ):
        """Accendere costa corrente e toglie memoria a chi altro usa quella
        macchina: è un'operazione di cui qualcuno risponde."""
        from platform_core.domain.admin_repositories import RegistroAuditRepository

        await admin.post("/admin/motore", json={"azione": "spegni"})

        voci = await RegistroAuditRepository(session).recenti(azione="motore.spegni")
        assert len(voci) == 1
        assert voci[0].target_type == "motore"

    async def test_un_azione_inventata_non_passa(self, admin, con_registro):
        risposta = await admin.post("/admin/motore", json={"azione": "riavvia-tutto"})

        assert risposta.status_code == 422
