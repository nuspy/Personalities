"""Scoperta, ispezione e fusione degli adapter LoRA.

Un adapter PEFT e' una cartella con `adapter_config.json` e i pesi. Prima di
usarne uno bisogna sapere **su quale modello base** e' stato addestrato: un
adapter di Llama-3 applicato a Mistral non produce un errore, produce pesi
incoerenti e un modello che parla a vanvera. Il controllo di compatibilita'
qui e' quindi esplicito e precede ogni fusione.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

ADAPTER_CONFIG_NAME = "adapter_config.json"

# Nomi dei file di pesi, in ordine di preferenza.
ADAPTER_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")


@dataclass
class AdapterInfo:
    """Metadati di un adapter trovato su disco."""

    path: Path
    name: str
    base_model: str = ""
    rank: int = 0
    alpha: int = 0
    target_modules: Tuple[str, ...] = ()
    size_mb: float = 0.0
    task_type: str = ""

    @property
    def is_valid(self) -> bool:
        return bool(self.rank) and self.has_weights

    @property
    def has_weights(self) -> bool:
        return any((self.path / name).exists() for name in ADAPTER_WEIGHT_NAMES)

    def describe(self) -> str:
        parts = [f"r={self.rank}", f"alpha={self.alpha}"]
        if self.size_mb:
            parts.append(f"{self.size_mb:.0f} MB")
        if self.base_model:
            parts.append(self.base_model.split("/")[-1])
        return " · ".join(parts)


def read_adapter(path: Path) -> Optional[AdapterInfo]:
    """Legge i metadati di un adapter. `None` se la cartella non lo e'."""
    path = Path(path)
    config_path = path / ADAPTER_CONFIG_NAME
    if not config_path.exists():
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
    except Exception as exc:
        logger.warning(f"Configurazione adapter illeggibile in {path}: {exc}")
        return None

    size_mb = 0.0
    for name in ADAPTER_WEIGHT_NAMES:
        weight_file = path / name
        if weight_file.exists():
            size_mb = weight_file.stat().st_size / 1024 / 1024
            break

    target = config.get("target_modules") or []
    if isinstance(target, str):
        target = [target]

    return AdapterInfo(
        path=path,
        name=path.name,
        base_model=str(config.get("base_model_name_or_path", "")),
        rank=int(config.get("r", 0) or 0),
        alpha=int(config.get("lora_alpha", 0) or 0),
        target_modules=tuple(sorted(str(m) for m in target)),
        size_mb=size_mb,
        task_type=str(config.get("task_type", "")),
    )


def discover_adapters(roots: Sequence[Path], max_depth: int = 4) -> List[AdapterInfo]:
    """Cerca adapter sotto le cartelle indicate.

    La ricerca e' limitata in profondita': una scansione ricorsiva senza
    limite su una cartella di progetti con checkpoint intermedi impiega
    minuti e restituisce centinaia di risultati inutili.
    """
    found: Dict[Path, AdapterInfo] = {}

    for root in roots:
        root = Path(root)
        if not root.exists():
            continue

        # `adapter_config.json` a qualunque profondita' entro il limite.
        for depth in range(max_depth + 1):
            pattern = "/".join(["*"] * depth + [ADAPTER_CONFIG_NAME])
            for config_path in root.glob(pattern):
                adapter_dir = config_path.parent
                if adapter_dir in found:
                    continue
                info = read_adapter(adapter_dir)
                if info and info.is_valid:
                    found[adapter_dir] = info

    return sorted(found.values(), key=lambda a: (a.name, str(a.path)))


def check_compatibility(
    adapters: Sequence[AdapterInfo],
    base_model: str = "",
) -> Tuple[List[str], List[str]]:
    """Verifica che gli adapter siano fondibili fra loro e col modello base.

    Restituisce `(errori, avvisi)`: gli errori impediscono la fusione, gli
    avvisi segnalano scelte che possono degradare il risultato senza
    renderlo impossibile.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not adapters:
        return errors, warnings

    # Modelli base diversi: fondere adapter di architetture diverse produce
    # pesi privi di senso, non un errore rumoroso.
    base_models = {a.base_model for a in adapters if a.base_model}
    if len(base_models) > 1:
        errors.append(
            "Gli adapter provengono da modelli base diversi ("
            + ", ".join(sorted(base_models))
            + "): non sono fondibili."
        )

    if base_model and base_models:
        adapter_base = next(iter(base_models))
        if _normalize_model_name(adapter_base) != _normalize_model_name(base_model):
            warnings.append(
                f"Il modello base indicato ('{base_model}') non coincide con quello "
                f"degli adapter ('{adapter_base}'). Procedere solo se si tratta "
                "della stessa architettura con nome diverso (es. variante quantizzata)."
            )

    # Ranghi diversi: solo la strategia SVD li gestisce.
    ranks = {a.rank for a in adapters}
    if len(ranks) > 1:
        warnings.append(
            f"Gli adapter hanno ranghi diversi ({sorted(ranks)}): "
            "usare la strategia di fusione 'svd', le altre richiedono lo stesso rango."
        )

    # Moduli bersaglio diversi: la fusione riesce ma copre solo l'intersezione.
    target_sets = {a.target_modules for a in adapters if a.target_modules}
    if len(target_sets) > 1:
        warnings.append(
            "Gli adapter agiscono su moduli diversi: la fusione interessera' "
            "solo i moduli comuni, il resto andra' perso."
        )

    for adapter in adapters:
        if not adapter.has_weights:
            errors.append(f"Adapter senza file di pesi: {adapter.path}")

    return errors, warnings


def _normalize_model_name(name: str) -> str:
    """Confronto tollerante fra nomi di modello.

    `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` e
    `meta-llama/Meta-Llama-3.1-8B-Instruct` sono lo stesso modello: la
    differenza sta nel distributore e nella quantizzazione.
    """
    cleaned = name.split("/")[-1].lower()
    for suffix in ("-bnb-4bit", "-bnb-8bit", "-4bit", "-8bit", "-gptq", "-awq", "-unsloth"):
        cleaned = cleaned.replace(suffix, "")
    return cleaned.strip("-")


def load_and_merge_adapters(
    model,
    adapters: Sequence[Any],
    strategy: str = "linear",
    merged_name: str = "merged_start",
    progress=None,
):
    """Carica piu' adapter su un modello e li fonde in uno solo.

    `adapters` sono `AdapterSource` (percorso + peso). Con un solo adapter la
    fusione si salta: caricarlo basta.

    Restituisce `(modello, nome_adapter_attivo)`.
    """
    from peft import PeftModel

    if not adapters:
        return model, ""

    def report(message: str) -> None:
        logger.info(message)
        if progress:
            progress(message)

    first = adapters[0]
    report(f"Carico l'adapter '{first.name}'...")
    model = PeftModel.from_pretrained(
        model, first.path, adapter_name=_safe_name(first.name), is_trainable=True
    )

    if len(adapters) == 1:
        return model, _safe_name(first.name)

    names = [_safe_name(first.name)]
    for source in adapters[1:]:
        name = _safe_name(source.name)
        report(f"Carico l'adapter '{source.name}'...")
        model.load_adapter(source.path, adapter_name=name, is_trainable=True)
        names.append(name)

    weights = [float(a.weight) for a in adapters]
    report(
        f"Fondo {len(names)} adapter con strategia '{strategy}' "
        f"(pesi: {', '.join(f'{w:g}' for w in weights)})..."
    )

    merge_kwargs: Dict[str, Any] = {
        "adapters": names,
        "weights": weights,
        "adapter_name": merged_name,
        "combination_type": strategy,
    }
    # ties e dare_ties richiedono una soglia di densita'; il valore usato in
    # letteratura e negli esempi PEFT e' 0.2.
    if strategy in ("ties", "dare_ties", "dare_linear", "magnitude_prune"):
        merge_kwargs["density"] = 0.2

    model.add_weighted_adapter(**merge_kwargs)
    model.set_adapter(merged_name)

    # Gli adapter di partenza non servono piu': tenerli occupa memoria e
    # rischia di far addestrare quello sbagliato.
    for name in names:
        try:
            model.delete_adapter(name)
        except Exception as exc:
            logger.debug(f"Adapter '{name}' non rimosso: {exc}")

    return model, merged_name


def _safe_name(name: str) -> str:
    """PEFT usa i nomi degli adapter come chiavi: niente separatori di percorso."""
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return cleaned or "adapter"
