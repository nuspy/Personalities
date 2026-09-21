"""Test degli endpoint della fase 0.

Verificano il comportamento che regge tutto il resto: la piattaforma dichiara
cosa sa fare, e nega con un codice e un motivo comprensibili cio' che non puo'
eseguire. Un endpoint che accoda un lavoro senza avere chi lo raccolga e' il
difetto che questi test impediscono.
"""
from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from platform_core.api import deps
from platform_core.api.app import CORRELATION_HEADER, create_app
from platform_core.api.routers.capabilities import require_feature
from platform_core.capabilities.probe import HardwareCapabilities
from platform_core.capabilities.registry import (
    CapabilityRegistry, Feature, InMemoryStore,
)
from platform_core.settings import get_settings


@pytest.fixture
def registry() -> CapabilityRegistry:
    """Un registro isolato per test, al posto di Redis."""
    registry = CapabilityRegistry(InMemoryStore())
    deps.reset_dependencies()
    get_settings.cache_clear()
    return registry


@pytest.fixture
def client(registry: CapabilityRegistry) -> TestClient:
    app = create_app()
    app.dependency_overrides[deps.get_capability_registry] = lambda: registry
    return TestClient(app)


def gpu_worker() -> HardwareCapabilities:
    return HardwareCapabilities(
        has_cuda=True, gpu_name="Scheda di prova", vram_total_mb=32000,
        can_train_lora=True, can_full_finetune=True, can_quantize_gguf=True,
    )


class TestHealth:
    def test_risponde_senza_supporti(self, client):
        """La liveness non deve dipendere da database o Redis."""
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestCorrelationId:
    def test_ne_viene_assegnato_uno(self, client):
        assert client.get("/healthz").headers[CORRELATION_HEADER]

    def test_quello_del_chiamante_viene_conservato(self, client):
        """Una richiesta iniziata nel frontend resta riconoscibile."""
        response = client.get("/healthz", headers={CORRELATION_HEADER: "abc-123"})
        assert response.headers[CORRELATION_HEADER] == "abc-123"


class TestCapabilitiesEndpoint:
    def test_elenca_tutte_le_funzioni(self, client):
        """Il frontend si configura da qui: nessuna deve mancare."""
        payload = client.get("/capabilities").json()
        assert set(payload["features"]) == {f.value for f in Feature}

    def test_senza_worker_ogni_funzione_ha_un_motivo(self, client):
        for entry in client.get("/capabilities").json()["features"].values():
            if not entry["available"]:
                assert entry["reason"], f"{entry['label']} negata senza spiegazione"

    def test_un_worker_gpu_compare_nella_risposta(self, client, registry):
        registry.announce("w-gpu-1", gpu_worker(), role="gpu")
        payload = client.get("/capabilities").json()

        assert payload["features"]["build_lora"]["available"]
        assert payload["workers"][0]["gpu"] == "Scheda di prova"

    def test_la_scomparsa_del_worker_si_riflette_subito(self, client, registry):
        """Ridurre a zero le repliche deve bastare a disabilitare la funzione."""
        registry.announce("w-gpu-1", gpu_worker(), role="gpu")
        assert client.get("/capabilities").json()["features"]["build_lora"]["available"]

        registry.withdraw("w-gpu-1")
        payload = client.get("/capabilities").json()
        assert not payload["features"]["build_lora"]["available"]
        assert payload["features"]["build_lora"]["reason"]


class TestRequireFeature:
    """La regola che protegge gli endpoint di costruzione."""

    @pytest.fixture
    def guarded_client(self, registry) -> TestClient:
        app = create_app()
        app.dependency_overrides[deps.get_capability_registry] = lambda: registry

        @app.post("/builds/lora", dependencies=[Depends(require_feature(Feature.BUILD_LORA))])
        def start_build() -> dict:
            return {"started": True}

        return TestClient(app)

    def test_senza_worker_risponde_409_col_motivo(self, guarded_client):
        """409 e non 500: la richiesta e' legittima, manca chi la esegua."""
        response = guarded_client.post("/builds/lora")

        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["error"] == "feature_unavailable"
        assert detail["feature"] == "build_lora"
        assert "worker" in detail["reason"] or "GPU" in detail["reason"]

    def test_con_un_worker_gpu_la_richiesta_passa(self, guarded_client, registry):
        registry.announce("w-gpu-1", gpu_worker(), role="gpu")
        assert guarded_client.post("/builds/lora").status_code == 200

    def test_il_motivo_arriva_al_chiamante(self, guarded_client, registry):
        """Chi riceve il rifiuto deve poterlo mostrare senza reinventarlo."""
        insufficiente = HardwareCapabilities(
            has_cuda=True, gpu_name="Scheda modesta", vram_total_mb=6000,
            reasons={"can_train_lora": "servono almeno 8 GB, disponibili 6.0 GB"},
        )
        registry.announce("w-gpu-1", insufficiente, role="gpu")

        detail = guarded_client.post("/builds/lora").json()["detail"]
        assert "6.0 GB" in detail["reason"]
