"""Test del rilevamento e del registro delle capacita'.

Il comportamento che questi test proteggono e' la **degradazione**: la
piattaforma deve funzionare anche dove manca l'acceleratore, offrendo meno
funzioni e dicendo perche'. Il difetto da impedire non e' un errore che si
vede — e' un pulsante che accoda un lavoro che nessuno raccogliera'.
"""
from __future__ import annotations

import time

import pytest

from platform_core.capabilities.probe import HardwareCapabilities, probe, reset_cache
from platform_core.capabilities.registry import (
    CapabilityRegistry, Feature, InMemoryStore, WorkerAnnouncement,
)


def gpu_worker(vram_mb: int = 32000, **overrides) -> HardwareCapabilities:
    defaults = dict(
        has_cuda=True,
        gpu_name="Scheda di prova",
        vram_total_mb=vram_mb,
        compute_capability=(12, 0),
        torch_version="2.11.0",
        can_train_lora=True,
        can_full_finetune=True,
        can_quantize_gguf=True,
    )
    defaults.update(overrides)
    return HardwareCapabilities(**defaults)


def cpu_worker() -> HardwareCapabilities:
    return HardwareCapabilities(
        reasons={"can_train_lora": "nessun acceleratore CUDA visibile da questo processo"}
    )


@pytest.fixture
def registry() -> CapabilityRegistry:
    return CapabilityRegistry(InMemoryStore())


class TestProbe:
    def test_il_rilevamento_non_solleva_mai(self):
        """Gira anche dove non c'e' nulla: e' il caso del server API."""
        assert isinstance(probe(use_cache=False), HardwareCapabilities)

    def test_ogni_capacita_negata_porta_il_motivo(self):
        caps = probe(use_cache=False)
        for name in ("can_train_lora", "can_full_finetune", "can_quantize_gguf"):
            if not getattr(caps, name):
                assert caps.reasons.get(name), f"'{name}' negata senza spiegazione"

    def test_il_risultato_e_memorizzato(self):
        reset_cache()
        assert probe() is probe()

    def test_serializzabile(self):
        import json

        json.dumps(probe(use_cache=False).as_dict())


class TestPlatformCapabilities:
    def test_senza_worker_nulla_e_disponibile(self, registry):
        caps = registry.platform_capabilities()
        for feature in Feature:
            available, reason = caps.can(feature)
            assert not available
            assert reason, f"{feature} negata senza spiegazione"

    def test_un_worker_gpu_abilita_addestramento_ed_export(self, registry):
        registry.announce("w-gpu-1", gpu_worker(), role="gpu")
        caps = registry.platform_capabilities()

        assert caps.can(Feature.BUILD_LORA)[0]
        assert caps.can(Feature.FULL_FINETUNE)[0]
        assert caps.can(Feature.GGUF_EXPORT)[0]

    def test_un_worker_cpu_non_abilita_l_addestramento(self, registry):
        registry.announce("w-cpu-1", cpu_worker(), role="cpu")
        available, reason = registry.can(Feature.BUILD_LORA)

        assert not available
        assert "CUDA" in reason or "acceleratore" in reason

    def test_la_scadenza_ritira_la_capacita(self, registry):
        """Un worker che smette di annunciarsi sparisce da solo."""
        registry_breve = CapabilityRegistry(InMemoryStore(), ttl=1)
        registry_breve.announce("w-gpu-1", gpu_worker(), role="gpu")
        assert registry_breve.can(Feature.BUILD_LORA)[0]

        time.sleep(1.1)
        available, reason = registry_breve.can(Feature.BUILD_LORA)
        assert not available
        assert "nessun worker" in reason

    def test_il_ritiro_esplicito_e_immediato(self, registry):
        """Alla chiusura ordinata non si attende la scadenza."""
        registry.announce("w-gpu-1", gpu_worker(), role="gpu")
        registry.withdraw("w-gpu-1")
        assert not registry.can(Feature.BUILD_LORA)[0]

    def test_il_motivo_specifico_prevale_sul_generico(self, registry):
        """Con worker vivi ma insufficienti, si riporta il loro motivo."""
        insufficiente = gpu_worker(
            vram_mb=8000,
            can_full_finetune=False,
            reasons={"can_full_finetune": "servono almeno 24 GB, disponibili 8.0 GB"},
        )
        registry.announce("w-gpu-1", insufficiente, role="gpu")

        available, reason = registry.can(Feature.FULL_FINETUNE)
        assert not available
        assert "24 GB" in reason, "il motivo del worker doveva essere riportato"

    def test_il_lora_resta_possibile_senza_fine_tuning(self, registry):
        """Meno memoria non significa nessuna funzione: significa meno funzioni."""
        registry.announce(
            "w-gpu-1",
            gpu_worker(vram_mb=12000, can_full_finetune=False),
            role="gpu",
        )
        assert registry.can(Feature.BUILD_LORA)[0]
        assert not registry.can(Feature.FULL_FINETUNE)[0]


class TestInferenceIsNotTraining:
    """La distinzione che impedisce di offrire build destinate a fallire.

    Un LM Studio raggiungibile sulla rete da' inferenza. Non da' addestramento:
    e' un servizio remoto, non un acceleratore del cluster. Confonderli fa
    comparire un pulsante «Costruisci LoRA» su una piattaforma che non puo'.
    """

    def test_un_motore_locale_non_abilita_l_addestramento(self, registry):
        caps = registry.platform_capabilities(local_engines=["lmstudio"])

        assert caps.can(Feature.LOCAL_INFERENCE)[0]
        assert caps.can(Feature.KV_CACHE_CAG)[0]
        assert not caps.can(Feature.BUILD_LORA)[0]

    def test_una_gpu_non_abilita_l_inferenza_locale(self, registry):
        """Il rovescio: un worker con GPU non e' un endpoint di inferenza."""
        registry.announce("w-gpu-1", gpu_worker(), role="gpu")
        caps = registry.platform_capabilities(local_engines=[])

        assert caps.can(Feature.BUILD_LORA)[0]
        assert not caps.can(Feature.LOCAL_INFERENCE)[0]


class TestSerialization:
    def test_annuncio_e_riletto_identico(self):
        original = WorkerAnnouncement(
            worker_id="w-1", role="gpu", capabilities=gpu_worker(), announced_at=123.0
        )
        restored = WorkerAnnouncement.from_json(original.to_json())

        assert restored.worker_id == original.worker_id
        assert restored.capabilities.vram_total_mb == original.capabilities.vram_total_mb
        assert restored.capabilities.compute_capability == (12, 0)

    def test_un_annuncio_corrotto_non_ferma_la_lettura(self, registry):
        """Una voce illeggibile va ignorata, non propagata come errore."""
        registry.announce("w-buono", gpu_worker(), role="gpu")
        registry._store.set("capabilities:worker:w-rotto", "{non json", ex=60)

        assert len(registry.workers()) == 1
        assert registry.can(Feature.BUILD_LORA)[0]

    def test_la_risposta_api_elenca_tutte_le_funzioni(self, registry):
        """Il frontend si configura da qui: nessuna funzione deve mancare."""
        payload = registry.platform_capabilities().as_dict()

        assert set(payload["features"]) == {f.value for f in Feature}
        for entry in payload["features"].values():
            assert entry["label"]
            assert "available" in entry
