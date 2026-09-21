"""Test del battito del worker e del ciclo completo con l'API.

Il comportamento protetto qui e' quello che rende la degradazione osservabile:
un worker che parte fa comparire le sue funzioni, uno che si ferma le fa
sparire. Senza, la piattaforma direbbe di saper fare cose che nessuno esegue.
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from platform_core.api import deps
from platform_core.api.app import create_app
from platform_core.capabilities.probe import HardwareCapabilities
from platform_core.capabilities.registry import (
    CapabilityRegistry, Feature, InMemoryStore,
)
from platform_core.worker.heartbeat import CapabilityHeartbeat, default_worker_id


def gpu_capabilities() -> HardwareCapabilities:
    return HardwareCapabilities(
        has_cuda=True, gpu_name="Scheda di prova", vram_total_mb=32000,
        can_train_lora=True, can_full_finetune=True, can_quantize_gguf=True,
    )


@pytest.fixture
def registry() -> CapabilityRegistry:
    deps.reset_dependencies()
    return CapabilityRegistry(InMemoryStore())


@pytest.fixture
def client(registry: CapabilityRegistry) -> TestClient:
    app = create_app()
    app.dependency_overrides[deps.get_capability_registry] = lambda: registry
    return TestClient(app)


class TestHeartbeat:
    def test_il_primo_annuncio_e_sincrono(self, registry):
        """Al ritorno da start() la piattaforma sa gia' del worker."""
        heartbeat = CapabilityHeartbeat(
            registry, worker_id="w-1", role="gpu", capabilities=gpu_capabilities()
        )
        heartbeat.start()
        try:
            assert registry.can(Feature.BUILD_LORA)[0]
        finally:
            heartbeat.stop()

    def test_l_arresto_ritira_subito_l_annuncio(self, registry):
        """Fra arresto e ritiro non deve restare una finestra di offerta."""
        heartbeat = CapabilityHeartbeat(
            registry, worker_id="w-1", role="gpu", capabilities=gpu_capabilities()
        )
        heartbeat.start()
        heartbeat.stop()

        assert not registry.can(Feature.BUILD_LORA)[0]

    def test_usabile_come_contesto(self, registry):
        with CapabilityHeartbeat(
            registry, worker_id="w-1", role="gpu", capabilities=gpu_capabilities()
        ):
            assert registry.can(Feature.BUILD_LORA)[0]
        assert not registry.can(Feature.BUILD_LORA)[0]

    def test_il_rinnovo_tiene_viva_la_voce(self):
        """La scadenza non deve far sparire un worker che continua ad annunciarsi."""
        registry = CapabilityRegistry(InMemoryStore(), ttl=2)
        heartbeat = CapabilityHeartbeat(
            registry, worker_id="w-1", role="gpu",
            capabilities=gpu_capabilities(), interval=1,
        )
        heartbeat.start()
        try:
            time.sleep(2.5)  # oltre la scadenza, ma il battito l'ha rinnovata
            assert registry.can(Feature.BUILD_LORA)[0]
        finally:
            heartbeat.stop()

    def test_un_supporto_irraggiungibile_non_ferma_il_worker(self):
        """Redis assente per un istante non deve far cadere il processo."""

        class StoreRotto(InMemoryStore):
            def set(self, *_args, **_kwargs):
                raise ConnectionError("Redis non raggiungibile")

        heartbeat = CapabilityHeartbeat(
            CapabilityRegistry(StoreRotto()), worker_id="w-1", role="gpu",
            capabilities=gpu_capabilities(),
        )
        heartbeat.start()   # non solleva
        heartbeat.stop()

    def test_identificativo_stabile(self, monkeypatch):
        """In Kubernetes il nome del pod sopravvive al riavvio del processo."""
        monkeypatch.setenv("HOSTNAME", "worker-gpu-7d4f")
        assert default_worker_id() == "worker-gpu-7d4f"


class TestWorkerApiCycle:
    """Il ciclo che la fase 0 deve dimostrare."""

    def test_avvio_e_arresto_del_worker_cambiano_la_risposta(self, client, registry):
        prima = client.get("/capabilities").json()
        assert not prima["features"]["build_lora"]["available"]
        assert prima["workers"] == []

        with CapabilityHeartbeat(
            registry, worker_id="w-gpu-1", role="gpu",
            capabilities=gpu_capabilities(),
        ):
            durante = client.get("/capabilities").json()
            assert durante["features"]["build_lora"]["available"]
            assert durante["workers"][0]["worker_id"] == "w-gpu-1"
            assert durante["workers"][0]["vram_mb"] == 32000

        dopo = client.get("/capabilities").json()
        assert not dopo["features"]["build_lora"]["available"]
        assert dopo["features"]["build_lora"]["reason"]

    def test_un_worker_cpu_non_abilita_l_addestramento(self, client, registry):
        """Il caso del server senza acceleratore: la piattaforma resta viva."""
        with CapabilityHeartbeat(
            registry, worker_id="w-cpu-1", role="cpu",
            capabilities=HardwareCapabilities(
                reasons={"can_train_lora": "nessun acceleratore CUDA visibile"}
            ),
        ):
            payload = client.get("/capabilities").json()

        assert payload["workers"][0]["role"] == "cpu"
        assert not payload["features"]["build_lora"]["available"]
