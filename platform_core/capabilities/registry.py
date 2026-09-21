"""Registro delle capacita' della piattaforma.

`probe.py` risponde a «cosa sa fare questo processo». Non e' la domanda che
conta per l'API: in un cluster il server che riceve le richieste non ha un
acceleratore e non deve averlo, perche' il lavoro pesante sta sui worker.
La domanda giusta e' **«esiste, adesso, un worker capace di addestrare?»**.

Ogni worker annuncia periodicamente le proprie capacita' con una scadenza.
Se smette di annunciarle — si e' fermato, e' stato ridotto a zero repliche,
e' caduto — la sua voce scade e la funzione torna indisponibile senza che
nessuno debba intervenire. E' lo stesso principio di un battito cardiaco:
l'assenza e' l'informazione.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Protocol, Tuple

from .probe import HardwareCapabilities

logger = logging.getLogger(__name__)

#: Prefisso delle chiavi di annuncio.
KEY_PREFIX = "capabilities:worker:"

#: Durata di un annuncio. Deve essere ampiamente superiore all'intervallo di
#: battito, o un worker vivo ma momentaneamente lento risulterebbe morto.
ANNOUNCE_TTL_SECONDS = 90
HEARTBEAT_INTERVAL_SECONDS = 30


class Feature(str, Enum):
    """Le funzioni che possono mancare a seconda di dove gira la piattaforma."""

    BUILD_LORA = "build_lora"
    FULL_FINETUNE = "full_finetune"
    GGUF_EXPORT = "gguf_export"
    KV_CACHE_CAG = "kv_cache_cag"
    LOCAL_INFERENCE = "local_inference"

    @property
    def label(self) -> str:
        return _FEATURE_LABELS[self]


_FEATURE_LABELS: Dict[Feature, str] = {
    Feature.BUILD_LORA: "Addestramento di adapter LoRA",
    Feature.FULL_FINETUNE: "Fine-tuning completo del modello",
    Feature.GGUF_EXPORT: "Esportazione in GGUF",
    Feature.KV_CACHE_CAG: "CAG con riuso della KV-cache",
    Feature.LOCAL_INFERENCE: "Inferenza su motore locale",
}

#: Motivo esposto quando nessun worker offre la funzione. E' il caso piu'
#: comune in produzione e merita un testo che dica cosa fare, non solo cosa
#: manca.
_NO_WORKER_REASON = {
    Feature.BUILD_LORA: (
        "nessun worker con acceleratore registrato: l'addestramento richiede "
        "un nodo con GPU. La personalita' resta utilizzabile in modalita' RAG"
    ),
    Feature.FULL_FINETUNE: (
        "nessun worker con memoria video sufficiente per un fine-tuning completo"
    ),
    Feature.GGUF_EXPORT: (
        "nessun worker dispone degli strumenti llama.cpp per la conversione"
    ),
    Feature.KV_CACHE_CAG: (
        "nessun motore locale disponibile: sui provider remoti il contesto "
        "stabile viene comunque scontato dal loro prompt caching"
    ),
    Feature.LOCAL_INFERENCE: (
        "nessun motore di inferenza locale raggiungibile"
    ),
}


class KeyValueStore(Protocol):
    """Il minimo di Redis che serve qui.

    Dichiararlo come protocollo tiene il registro verificabile senza un Redis
    in esecuzione, e lascia aperta la sostituzione del supporto.
    """

    def set(self, name: str, value: str, ex: Optional[int] = None) -> object: ...
    def get(self, name: str) -> Optional[bytes | str]: ...
    def scan_iter(self, match: str) -> Iterable[bytes | str]: ...
    def delete(self, *names: str) -> object: ...


@dataclass(frozen=True)
class WorkerAnnouncement:
    worker_id: str
    role: str                      # "gpu" | "cpu"
    capabilities: HardwareCapabilities
    announced_at: float

    def to_json(self) -> str:
        return json.dumps({
            "worker_id": self.worker_id,
            "role": self.role,
            "announced_at": self.announced_at,
            "capabilities": self.capabilities.as_dict(),
        })

    @staticmethod
    def from_json(raw: str | bytes) -> "WorkerAnnouncement":
        data = json.loads(raw)
        caps = data.get("capabilities", {})
        capability = caps.get("compute_capability")
        return WorkerAnnouncement(
            worker_id=data["worker_id"],
            role=data.get("role", "cpu"),
            announced_at=float(data.get("announced_at", 0)),
            capabilities=HardwareCapabilities(
                has_cuda=bool(caps.get("has_cuda")),
                gpu_name=caps.get("gpu_name", ""),
                vram_total_mb=int(caps.get("vram_total_mb", 0)),
                compute_capability=tuple(capability) if capability else None,
                torch_version=caps.get("torch_version", ""),
                platform=caps.get("platform", ""),
                can_train_lora=bool(caps.get("can_train_lora")),
                can_full_finetune=bool(caps.get("can_full_finetune")),
                can_quantize_gguf=bool(caps.get("can_quantize_gguf")),
                local_engines=list(caps.get("local_engines", [])),
                reasons=dict(caps.get("reasons", {})),
            ),
        )


@dataclass(frozen=True)
class PlatformCapabilities:
    """Cio' che la piattaforma nel suo insieme sa fare, ora."""

    features: Dict[Feature, bool] = field(default_factory=dict)
    reasons: Dict[Feature, str] = field(default_factory=dict)
    workers: List[WorkerAnnouncement] = field(default_factory=list)

    def can(self, feature: Feature) -> Tuple[bool, str]:
        available = self.features.get(feature, False)
        return available, "" if available else self.reasons.get(feature, "non disponibile")

    def as_dict(self) -> Dict[str, object]:
        return {
            "features": {
                feature.value: {
                    "available": self.features.get(feature, False),
                    "label": feature.label,
                    "reason": self.reasons.get(feature, ""),
                }
                for feature in Feature
            },
            "workers": [
                {
                    "worker_id": w.worker_id,
                    "role": w.role,
                    "gpu": w.capabilities.gpu_name,
                    "vram_mb": w.capabilities.vram_total_mb,
                }
                for w in self.workers
            ],
        }


class CapabilityRegistry:
    """Annuncia e interroga le capacita' dei worker vivi."""

    def __init__(self, store: KeyValueStore, *, ttl: int = ANNOUNCE_TTL_SECONDS):
        self._store = store
        self._ttl = ttl

    # ------------------------------------------------------------ annuncio

    def announce(
        self,
        worker_id: str,
        capabilities: HardwareCapabilities,
        *,
        role: str = "cpu",
    ) -> None:
        """Dichiara cosa questo worker sa fare, con scadenza.

        Va richiamata a intervalli piu' brevi della scadenza: e' la ripetizione
        a tenere viva la voce, non la prima chiamata.
        """
        announcement = WorkerAnnouncement(
            worker_id=worker_id,
            role=role,
            capabilities=capabilities,
            announced_at=time.time(),
        )
        self._store.set(f"{KEY_PREFIX}{worker_id}", announcement.to_json(), ex=self._ttl)

    def withdraw(self, worker_id: str) -> None:
        """Ritira l'annuncio subito, senza attendere la scadenza.

        Usata alla chiusura ordinata di un worker: evita la finestra in cui
        la piattaforma offre una funzione che nessuno raccoglierebbe piu'.
        """
        self._store.delete(f"{KEY_PREFIX}{worker_id}")

    # --------------------------------------------------------- consultazione

    def workers(self) -> List[WorkerAnnouncement]:
        found: List[WorkerAnnouncement] = []
        for key in self._store.scan_iter(f"{KEY_PREFIX}*"):
            raw = self._store.get(key if isinstance(key, str) else key.decode())
            if not raw:
                continue  # scaduta fra la scansione e la lettura
            try:
                found.append(WorkerAnnouncement.from_json(raw))
            except (ValueError, KeyError) as exc:
                logger.warning("Annuncio illeggibile ignorato (%s): %s", key, exc)
        return sorted(found, key=lambda w: w.worker_id)

    def platform_capabilities(
        self, *, local_engines: Optional[List[str]] = None
    ) -> PlatformCapabilities:
        """Unione delle capacita' dei worker vivi.

        `local_engines` arriva dal registro dei provider, non dai worker: un
        motore di inferenza locale e' un servizio raggiungibile in rete, non
        una proprieta' del cluster. Tenerli distinti e' cio' che impedisce di
        scambiare «LM Studio risponde» per «posso addestrare».
        """
        workers = self.workers()
        engines = list(local_engines or [])

        features = {
            Feature.BUILD_LORA: any(w.capabilities.can_train_lora for w in workers),
            Feature.FULL_FINETUNE: any(w.capabilities.can_full_finetune for w in workers),
            Feature.GGUF_EXPORT: any(w.capabilities.can_quantize_gguf for w in workers),
            Feature.LOCAL_INFERENCE: bool(engines),
            Feature.KV_CACHE_CAG: bool(engines),
        }

        reasons: Dict[Feature, str] = {}
        for feature, available in features.items():
            if available:
                continue
            reasons[feature] = self._explain(feature, workers)

        return PlatformCapabilities(features=features, reasons=reasons, workers=workers)

    def can(
        self, feature: Feature, *, local_engines: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """Scorciatoia per il controllo negli endpoint."""
        return self.platform_capabilities(local_engines=local_engines).can(feature)

    # ------------------------------------------------------------ spiegazioni

    @staticmethod
    def _explain(feature: Feature, workers: List[WorkerAnnouncement]) -> str:
        """Motivo dell'indisponibilita', il piu' specifico che i dati consentano.

        Con dei worker vivi ma insufficienti, il loro stesso rilevamento ha
        gia' registrato il perche' («servono 24 GB, disponibili 8»): riportarlo
        e' piu' utile di un generico «non disponibile».
        """
        if not workers:
            return _NO_WORKER_REASON[feature]

        probe_key = {
            Feature.BUILD_LORA: "can_train_lora",
            Feature.FULL_FINETUNE: "can_full_finetune",
            Feature.GGUF_EXPORT: "can_quantize_gguf",
        }.get(feature)

        if probe_key:
            specific = [
                w.capabilities.reasons[probe_key]
                for w in workers
                if probe_key in w.capabilities.reasons
            ]
            if specific:
                # Piu' worker possono negarla per ragioni diverse: si riporta
                # la piu' frequente, che e' quella che descrive il cluster.
                return max(set(specific), key=specific.count)

        return _NO_WORKER_REASON[feature]


class InMemoryStore:
    """Supporto in memoria con scadenze, per test e sviluppo senza Redis."""

    def __init__(self) -> None:
        self._data: Dict[str, Tuple[str, Optional[float]]] = {}

    def set(self, name: str, value: str, ex: Optional[int] = None) -> None:
        expires = time.time() + ex if ex else None
        self._data[name] = (value, expires)

    def get(self, name: str) -> Optional[str]:
        entry = self._data.get(name)
        if entry is None:
            return None
        value, expires = entry
        if expires is not None and time.time() > expires:
            del self._data[name]
            return None
        return value

    def scan_iter(self, match: str) -> Iterable[str]:
        prefix = match.rstrip("*")
        for name in list(self._data):
            if name.startswith(prefix) and self.get(name) is not None:
                yield name

    def delete(self, *names: str) -> None:
        for name in names:
            self._data.pop(name, None)
