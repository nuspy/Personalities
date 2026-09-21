"""Realizzazioni: coda, ciclo di vita, e il rifiuto che conta.

Il test centrale è `test_senza_acceleratore_si_rifiuta_subito`: se nessun
worker può eseguire, accodare significherebbe mettere l'utente davanti a un
«in attesa» che non finirà mai — e nulla nella schermata direbbe perché. Il
409 arriva subito, con il motivo che il registro delle capacità riporta.

`TestCicloDiVita` protegge l'altra proprietà: una build non torna mai
indietro, e ciò che è già partito non si annulla fingendo.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from platform_core.api.app import create_app
from platform_core.api.deps import (
    get_capability_registry, get_embedder, get_key_value_store,
)
from platform_core.auth.dependencies import current_principal
from platform_core.builds.queue import CodaBuild, CodaInMemoria, JobBuild
from platform_core.builds.repository import BuildRepository, RuntimeRepository
from platform_core.capabilities.probe import HardwareCapabilities
from platform_core.capabilities.registry import (
    CapabilityRegistry, Feature, InMemoryStore,
)
from platform_core.domain.build_models import Build
from platform_core.domain.knowledge_models import Personality
from platform_core.domain.session import get_db_session, get_session_factory
from platform_core.worker.heartbeat import CapabilityHeartbeat

from .conftest import richiede_database
from .test_retrieval import EmbedderFinto

pytestmark = richiede_database


def capacita_gpu() -> HardwareCapabilities:
    return HardwareCapabilities(
        has_cuda=True, gpu_name="Scheda di prova", vram_total_mb=32000,
        can_train_lora=True, can_full_finetune=True, can_quantize_gguf=True,
    )


@pytest.fixture
def registro_vuoto() -> CapabilityRegistry:
    """Nessun worker: la piattaforma non può costruire."""
    return CapabilityRegistry(InMemoryStore())


@pytest.fixture
def registro_con_gpu() -> CapabilityRegistry:
    registro = CapabilityRegistry(InMemoryStore())
    registro.announce("w-prova", capacita_gpu(), role="gpu")
    return registro


@pytest.fixture
def coda() -> CodaInMemoria:
    return CodaInMemoria()


@pytest_asyncio.fixture
async def personalita(session, utente) -> Personality:
    p = Personality(
        slug=f"da-costruire-{uuid.uuid4().hex[:8]}",
        display_name="Da costruire",
        status="published",
        owner_id=utente.id,
    )
    session.add(p)
    await session.flush()
    return p


def client_con(registro, coda, session, session_factory, principal):
    app = create_app()
    app.dependency_overrides[current_principal] = lambda: principal
    app.dependency_overrides[get_capability_registry] = lambda: registro
    app.dependency_overrides[get_key_value_store] = lambda: coda
    app.dependency_overrides[get_embedder] = lambda: EmbedderFinto()

    async def sessione_del_test():
        yield session

    app.dependency_overrides[get_db_session] = sessione_del_test
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    return app


@pytest_asyncio.fixture
async def senza_gpu(session, session_factory, principal_admin, registro_vuoto, coda, utente):
    app = client_con(registro_vuoto, coda, session, session_factory, principal_admin)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


@pytest_asyncio.fixture
async def con_gpu(session, session_factory, principal_admin, registro_con_gpu, coda, utente):
    app = client_con(registro_con_gpu, coda, session, session_factory, principal_admin)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


class TestRifiuto:
    async def test_senza_acceleratore_si_rifiuta_subito(
        self, senza_gpu, personalita
    ):
        """Il caso che il piano chiede di verificare.

        Accodare un lavoro che nessuno raccoglierà è peggio di rifiutarlo:
        l'utente vedrebbe «in attesa» per sempre, e nulla direbbe perché.
        """
        risposta = await senza_gpu.post(
            f"/admin/personalities/{personalita.id}/builds",
            json={"kind": "lora", "params": {}},
        )

        assert risposta.status_code == 409
        assert "acceleratore" in risposta.json()["detail"].lower()

    async def test_il_rifiuto_dice_perche(self, senza_gpu, personalita):
        risposta = await senza_gpu.post(
            f"/admin/personalities/{personalita.id}/builds",
            json={"kind": "finetune"},
        )
        detail = risposta.json()["detail"]

        assert len(detail) > 20, "un rifiuto senza spiegazione è un vicolo cieco"

    async def test_nulla_finisce_in_coda(self, senza_gpu, personalita, coda):
        await senza_gpu.post(
            f"/admin/personalities/{personalita.id}/builds",
            json={"kind": "lora"},
        )
        assert CodaBuild(coda).in_attesa() == 0

    async def test_nessuna_riga_resta_a_database(
        self, senza_gpu, personalita, session
    ):
        """Una build «in coda» che nessuno prenderà sarebbe un dato che mente."""
        from sqlalchemy import func, select

        await senza_gpu.post(
            f"/admin/personalities/{personalita.id}/builds",
            json={"kind": "lora"},
        )

        quante = await session.scalar(
            select(func.count()).select_from(Build)
            .where(Build.personality_id == personalita.id)
        )
        assert quante == 0

    async def test_con_acceleratore_si_accoda(self, con_gpu, personalita, coda):
        risposta = await con_gpu.post(
            f"/admin/personalities/{personalita.id}/builds",
            json={"kind": "lora", "params": {"base_model": "un-modello"}},
        )

        assert risposta.status_code == 202
        assert risposta.json()["status"] == "in_coda"
        assert CodaBuild(coda).in_attesa() == 1

    async def test_una_personalita_inesistente_da_404(self, con_gpu):
        risposta = await con_gpu.post(
            f"/admin/personalities/{uuid.uuid4()}/builds", json={"kind": "lora"},
        )
        assert risposta.status_code == 404

    async def test_un_tipo_sconosciuto_viene_respinto(self, con_gpu, personalita):
        risposta = await con_gpu.post(
            f"/admin/personalities/{personalita.id}/builds",
            json={"kind": "telepatia"},
        )
        assert risposta.status_code == 422


class TestCoda:
    def test_un_job_preso_resta_fra_i_presi_in_carico(self, coda):
        """Se il worker muore, il job si ritrova invece di svanire."""
        c = CodaBuild(coda)
        job = JobBuild(uuid.uuid4(), "lora", uuid.uuid4())
        c.accoda(job)

        preso = c.prendi("w-1", attesa=0)

        assert preso == job
        assert c.in_attesa() == 0
        assert c.in_carico("w-1") == [job]

    def test_completare_lo_toglie(self, coda):
        c = CodaBuild(coda)
        job = JobBuild(uuid.uuid4(), "lora", uuid.uuid4())
        c.accoda(job)
        c.prendi("w-1", attesa=0)
        c.completa("w-1", job)

        assert c.in_carico("w-1") == []

    def test_una_coda_vuota_restituisce_niente(self, coda):
        """Non è un errore: è la condizione normale, e restituirla permette al
        worker di controllare se deve fermarsi."""
        assert CodaBuild(coda).prendi("w-1", attesa=0) is None

    def test_l_ordine_e_di_arrivo(self, coda):
        c = CodaBuild(coda)
        job = [JobBuild(uuid.uuid4(), "lora", uuid.uuid4()) for _ in range(3)]
        for j in job:
            c.accoda(j)

        assert [c.prendi("w-1", attesa=0) for _ in range(3)] == job


class TestCicloDiVita:
    async def test_la_presa_in_carico_registra_il_worker(self, session, personalita):
        repo = BuildRepository(session)
        build = await repo.crea(personality_id=personalita.id, kind="lora")

        assert await repo.prendi_in_carico(build, "w-gpu-1")
        assert build.status == "in_corso"
        assert build.worker_id == "w-gpu-1"
        assert build.started_at is not None

    async def test_una_build_annullata_non_viene_presa(self, session, personalita):
        """Capita quando qualcuno annulla mentre il job stava per partire."""
        repo = BuildRepository(session)
        build = await repo.crea(personality_id=personalita.id, kind="lora")
        await repo.annulla(build)

        assert not await repo.prendi_in_carico(build, "w-gpu-1")
        assert build.status == "annullata"

    async def test_una_build_in_corso_non_si_annulla(self, session, personalita):
        """Dichiararla annullata mentre un processo gira lascerebbe una riga
        che contraddice la realtà."""
        repo = BuildRepository(session)
        build = await repo.crea(personality_id=personalita.id, kind="lora")
        await repo.prendi_in_carico(build, "w-1")

        assert not await repo.annulla(build)
        assert build.status == "in_corso"

    async def test_l_avanzamento_lascia_eventi(self, session, personalita):
        repo = BuildRepository(session)
        build = await repo.crea(personality_id=personalita.id, kind="lora")
        await repo.prendi_in_carico(build, "w-1")

        for p, m in [(10, "carico"), (50, "addestro"), (90, "salvo")]:
            await repo.avanza(build, p, m)

        eventi = await repo.eventi(build.id)
        assert [e.progress for e in eventi] == [10, 50, 90]
        assert build.progress == 90

    async def test_gli_eventi_si_riprendono_da_un_punto(self, session, personalita):
        """Chi osserva un job lungo riprende da dove era rimasto."""
        repo = BuildRepository(session)
        build = await repo.crea(personality_id=personalita.id, kind="lora")
        for i in range(5):
            await repo.avanza(build, i * 20, f"passo {i}")

        tutti = await repo.eventi(build.id)
        seguenti = await repo.eventi(build.id, dopo=tutti[1].id)

        assert len(seguenti) == 3

    async def test_la_conclusione_registra_l_artefatto(self, session, personalita):
        repo = BuildRepository(session)
        build = await repo.crea(personality_id=personalita.id, kind="lora")
        await repo.prendi_in_carico(build, "w-1")
        await repo.conclusa(build, artifact_path="/percorso/adapter")

        assert build.status == "riuscita"
        assert build.progress == 100
        assert build.artifact_path == "/percorso/adapter"
        assert build.finished_at is not None

    async def test_il_fallimento_conserva_il_motivo(self, session, personalita):
        repo = BuildRepository(session)
        build = await repo.crea(personality_id=personalita.id, kind="lora")
        await repo.prendi_in_carico(build, "w-1")
        await repo.fallita(build, "memoria video esaurita")

        assert build.status == "fallita"
        assert "memoria video" in build.error

    async def test_gli_eventi_si_potano(self, session, personalita):
        """Un addestramento ne emette centinaia: tenerli tutti gonfierebbe la
        tabella senza aggiungere nulla."""
        from platform_core.builds.repository import EVENTI_MASSIMI

        repo = BuildRepository(session)
        build = await repo.crea(personality_id=personalita.id, kind="lora")
        for i in range(EVENTI_MASSIMI + 30):
            await repo.avanza(build, i % 100, f"passo {i}")

        tolti = await repo.potatura_eventi(build)
        rimasti = await repo.eventi(build.id, limite=1000)

        assert tolti == 30
        assert len(rimasti) == EVENTI_MASSIMI
        assert "passo 229" in rimasti[-1].message, "gli ultimi devono restare"


class TestRuntime:
    async def test_senza_scelta_si_risponde_in_rag(self, session, personalita):
        assert await RuntimeRepository(session).modo_di(personalita.id) == "rag"

    async def test_una_build_non_riuscita_non_si_puo_servire(
        self, con_gpu, session, personalita
    ):
        repo = BuildRepository(session)
        build = await repo.crea(personality_id=personalita.id, kind="lora")
        await session.flush()

        risposta = await con_gpu.put(
            f"/admin/personalities/{personalita.id}/runtime",
            json={"mode": "lora", "build_id": str(build.id)},
        )

        assert risposta.status_code == 409

    async def test_una_build_riuscita_diventa_servibile(
        self, con_gpu, session, personalita
    ):
        repo = BuildRepository(session)
        build = await repo.crea(personality_id=personalita.id, kind="lora")
        await repo.conclusa(build, artifact_path="/un/percorso")
        await session.flush()

        risposta = await con_gpu.put(
            f"/admin/personalities/{personalita.id}/runtime",
            json={"mode": "lora", "build_id": str(build.id)},
        )

        assert risposta.status_code == 200
        assert risposta.json()["mode"] == "lora"

    async def test_il_modo_lora_pretende_una_build(self, con_gpu, personalita):
        risposta = await con_gpu.put(
            f"/admin/personalities/{personalita.id}/runtime",
            json={"mode": "lora"},
        )
        assert risposta.status_code == 400

    async def test_si_torna_sempre_a_rag(self, con_gpu, personalita):
        """La via di ritorno dev'essere sempre aperta: è ciò che rende la
        degradazione una scelta invece di un guasto."""
        risposta = await con_gpu.put(
            f"/admin/personalities/{personalita.id}/runtime",
            json={"mode": "rag"},
        )
        assert risposta.status_code == 200
