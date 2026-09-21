"""Rilevamento delle capacita' hardware del processo corrente.

La piattaforma deve girare sia su un nodo con acceleratore — dove puo'
addestrare adapter LoRA, fare fine-tuning e servire modelli locali con
riuso della KV-cache — sia su un server che non ne ha, dove resta il solo
RAG su API remote.

Il principio che regge il modulo: **una capacita' si rileva, non si deduce**.
Assumere l'acceleratore perche' lo sviluppo gira su una macchina che ce l'ha
porta a scoprire in produzione che un pulsante accoda lavori che nessuno
raccogliera'. E ogni «no» porta con se' il motivo, perche' un'interfaccia che
disabilita senza spiegare sembra guasta.

Attenzione a una distinzione che il modulo tiene separata di proposito:
un motore di inferenza locale raggiungibile — LM Studio sulla rete — offre
*inferenza*, non *addestramento*. Sono due assi indipendenti, e confonderli
fa offrire build destinate a fallire.
"""
from __future__ import annotations

import logging
import os
import platform
import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Soglie di memoria video. Sono indicative e riferite a un modello da 7-8
# miliardi di parametri, il taglio piu' comune: un adapter LoRA a 4 bit ci sta
# in circa 10 GB, un fine-tuning completo ne chiede molti di piu'. Restano
# configurabili perche' dipendono dal modello, non dall'hardware soltanto.
MIN_VRAM_LORA_MB = int(os.getenv("PERSONA_MIN_VRAM_LORA_MB", "8000"))
MIN_VRAM_FINETUNE_MB = int(os.getenv("PERSONA_MIN_VRAM_FINETUNE_MB", "24000"))


@dataclass(frozen=True)
class HardwareCapabilities:
    """Cosa questo processo puo' fare, e perche' non puo' fare il resto."""

    has_cuda: bool = False
    gpu_name: str = ""
    vram_total_mb: int = 0
    compute_capability: Optional[Tuple[int, int]] = None
    torch_version: str = ""
    platform: str = field(default_factory=lambda: f"{platform.system()} {platform.machine()}")

    can_train_lora: bool = False
    can_full_finetune: bool = False
    can_quantize_gguf: bool = False

    #: Motori di inferenza locali raggiungibili. NON implica addestramento.
    local_engines: List[str] = field(default_factory=list)

    #: Motivo per ogni capacita' negata, rivolto a chi legge l'interfaccia.
    reasons: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "has_cuda": self.has_cuda,
            "gpu_name": self.gpu_name,
            "vram_total_mb": self.vram_total_mb,
            "compute_capability": list(self.compute_capability) if self.compute_capability else None,
            "torch_version": self.torch_version,
            "platform": self.platform,
            "can_train_lora": self.can_train_lora,
            "can_full_finetune": self.can_full_finetune,
            "can_quantize_gguf": self.can_quantize_gguf,
            "local_engines": list(self.local_engines),
            "reasons": dict(self.reasons),
        }


def probe(*, use_cache: bool = True) -> HardwareCapabilities:
    """Rileva le capacita' del processo corrente.

    Il risultato e' memorizzato: l'hardware non cambia mentre il processo
    vive, e interrogare CUDA a ogni richiesta costerebbe senza motivo.
    `use_cache=False` serve ai test e a un eventuale comando di diagnosi.
    """
    if use_cache:
        return _probe_cached()
    return _probe()


@lru_cache(maxsize=1)
def _probe_cached() -> HardwareCapabilities:
    return _probe()


def reset_cache() -> None:
    """Dimentica il rilevamento. Solo per i test."""
    _probe_cached.cache_clear()


def _probe() -> HardwareCapabilities:
    reasons: Dict[str, str] = {}

    torch_version, has_cuda, gpu_name, vram_mb, capability = _probe_torch(reasons)
    training_libs_ok = _probe_training_libraries(reasons)

    can_train_lora = has_cuda and training_libs_ok and vram_mb >= MIN_VRAM_LORA_MB
    if has_cuda and training_libs_ok and not can_train_lora:
        reasons["can_train_lora"] = (
            f"servono almeno {MIN_VRAM_LORA_MB / 1000:.0f} GB di memoria video, "
            f"disponibili {vram_mb / 1000:.1f} GB"
        )

    can_full_finetune = can_train_lora and vram_mb >= MIN_VRAM_FINETUNE_MB
    if can_train_lora and not can_full_finetune:
        reasons["can_full_finetune"] = (
            f"il fine-tuning completo richiede almeno "
            f"{MIN_VRAM_FINETUNE_MB / 1000:.0f} GB di memoria video, "
            f"disponibili {vram_mb / 1000:.1f} GB. L'adapter LoRA resta possibile"
        )

    can_quantize_gguf, gguf_reason = _probe_llama_cpp()
    if not can_quantize_gguf:
        reasons["can_quantize_gguf"] = gguf_reason

    capabilities = HardwareCapabilities(
        has_cuda=has_cuda,
        gpu_name=gpu_name,
        vram_total_mb=vram_mb,
        compute_capability=capability,
        torch_version=torch_version,
        can_train_lora=can_train_lora,
        can_full_finetune=can_full_finetune,
        can_quantize_gguf=can_quantize_gguf,
        local_engines=[],  # popolati da chi conosce i provider configurati
        reasons=reasons,
    )

    logger.info(
        "Capacita' rilevate: %s — LoRA:%s fine-tuning:%s GGUF:%s",
        gpu_name or capabilities.platform,
        can_train_lora, can_full_finetune, can_quantize_gguf,
    )
    return capabilities


def _probe_torch(
    reasons: Dict[str, str]
) -> Tuple[str, bool, str, int, Optional[Tuple[int, int]]]:
    """Interroga torch. La sua assenza e' un caso normale, non un errore.

    Il server API non ha bisogno di torch: gli servono i *metadati* delle
    capacita', che riceve dai worker. Importarlo qui e trovarlo mancante
    significa soltanto che questo processo non e' un worker di addestramento.
    """
    try:
        import torch
    except ImportError:
        reasons["can_train_lora"] = (
            "PyTorch non e' installato in questo processo: l'addestramento "
            "avviene sui worker dedicati"
        )
        return "", False, "", 0, None

    version = torch.__version__

    try:
        if not torch.cuda.is_available():
            reasons["can_train_lora"] = (
                "nessun acceleratore CUDA visibile da questo processo"
            )
            return version, False, "", 0, None

        index = torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        properties = torch.cuda.get_device_properties(index)
        vram_mb = int(properties.total_memory / (1024 * 1024))
        capability = (properties.major, properties.minor)
        return version, True, name, vram_mb, capability

    except Exception as exc:
        # Un driver assente o incompatibile solleva qui: e' un'indisponibilita'
        # da riportare, non un guasto da propagare.
        logger.warning("Interrogazione CUDA fallita: %s", exc)
        reasons["can_train_lora"] = f"CUDA presente ma non interrogabile ({exc})"
        return version, False, "", 0, None


def _probe_training_libraries(reasons: Dict[str, str]) -> bool:
    """Verifica che le librerie di addestramento siano importabili.

    Una GPU senza `peft` e `trl` non addestra nulla, e scoprirlo all'avvio
    del job invece che qui sposta soltanto l'errore piu' avanti.
    """
    missing: List[str] = []
    for module in ("peft", "trl", "datasets", "transformers"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        reasons.setdefault(
            "can_train_lora",
            "librerie di addestramento mancanti: " + ", ".join(missing),
        )
        return False
    return True


def _probe_llama_cpp() -> Tuple[bool, str]:
    """Cerca gli strumenti di llama.cpp per la conversione in GGUF.

    Lo script di conversione e' stato rinominato fra le versioni, quindi se ne
    cercano piu' nomi. La build che Unsloth Studio scarica per conto proprio
    e' fra i percorsi esaminati: chi lo ha installato non deve rifare nulla.
    """
    from pathlib import Path

    script_names = ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py")
    roots = [
        Path(p) for p in (
            os.getenv("LLAMA_CPP_PATH", ""),
            str(Path.home() / "llama.cpp"),
            str(Path.home() / ".unsloth" / "llama.cpp"),
            str(Path.cwd() / "llama.cpp"),
        ) if p
    ]

    for root in roots:
        if not root.exists():
            continue
        if any((root / name).exists() for name in script_names):
            return True, ""

    if shutil.which("llama-quantize"):
        return True, ""

    return False, (
        "strumenti llama.cpp non trovati: indicare il percorso in "
        "LLAMA_CPP_PATH per abilitare l'esportazione GGUF"
    )
