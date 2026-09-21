"""Modalita' di addestramento disponibili.

Un personaggio non si costruisce sempre da zero. I casi reali sono quattro,
e differiscono per *da dove partono i pesi*:

- `LORA_NEW` — un adapter nuovo sul modello base. E' il punto di partenza.
- `LORA_CONTINUE` — si riprende un adapter esistente e lo si addestra ancora,
  su nuovo materiale o per piu' epoche. I pesi dell'adapter sono quelli gia'
  appresi, non ripartono da zero.
- `LORA_STACK` — uno o piu' adapter esistenti vengono **fusi nel modello
  base**, e sopra il risultato si addestra un adapter nuovo. Serve per
  costruire per strati: prima la lingua, poi la persona.
- `FULL_FINETUNE` — si addestrano tutti i pesi, senza adapter. Qualita'
  superiore a parita' di dati, ma richiede molta piu' VRAM.

La distinzione fra CONTINUE e STACK e' quella che conta nella pratica:
continuare modifica l'adapter esistente, impilare ne crea uno nuovo lasciando
intatti i precedenti.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TrainingMode(str, Enum):
    LORA_NEW = "lora_new"
    LORA_CONTINUE = "lora_continue"
    LORA_STACK = "lora_stack"
    FULL_FINETUNE = "full_finetune"

    @property
    def label(self) -> str:
        return _MODE_LABELS[self]

    @property
    def description(self) -> str:
        return _MODE_DESCRIPTIONS[self]

    @property
    def requires_adapters(self) -> bool:
        """Vero se la modalita' non ha senso senza adapter di partenza."""
        return self in (TrainingMode.LORA_CONTINUE, TrainingMode.LORA_STACK)

    @property
    def produces_adapter(self) -> bool:
        return self is not TrainingMode.FULL_FINETUNE


_MODE_LABELS: Dict[TrainingMode, str] = {
    TrainingMode.LORA_NEW: "Nuova LoRA",
    TrainingMode.LORA_CONTINUE: "Continua una LoRA esistente",
    TrainingMode.LORA_STACK: "Nuova LoRA sopra LoRA esistenti",
    TrainingMode.FULL_FINETUNE: "Fine-tuning completo",
}

_MODE_DESCRIPTIONS: Dict[TrainingMode, str] = {
    TrainingMode.LORA_NEW: (
        "Addestra un adapter nuovo sul modello base. Scelta predefinita per "
        "un personaggio mai addestrato."
    ),
    TrainingMode.LORA_CONTINUE: (
        "Riprende un adapter gia' addestrato e prosegue da li'. Usala per "
        "aggiungere materiale nuovo o per fare altre epoche senza ripartire "
        "da capo. L'adapter di partenza viene modificato, non duplicato: "
        "conviene salvarne una copia se serve tornare indietro."
    ),
    TrainingMode.LORA_STACK: (
        "Fonde uno o piu' adapter nel modello base e addestra un adapter "
        "nuovo sopra il risultato. Serve a costruire per strati (prima la "
        "lingua o il dominio, poi la persona). Gli adapter di partenza "
        "restano intatti."
    ),
    TrainingMode.FULL_FINETUNE: (
        "Addestra tutti i pesi del modello. Qualita' superiore a parita' di "
        "dati, ma richiede molta piu' memoria video e produce un modello "
        "intero invece di un adapter di pochi MB."
    ),
}


@dataclass
class AdapterSource:
    """Un adapter da cui partire, con il peso che avra' nella fusione."""

    path: str
    weight: float = 1.0
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            from pathlib import Path

            self.name = Path(self.path).name


# Strategie di fusione di piu' adapter, come le espone PEFT.
MERGE_STRATEGIES = {
    "linear": (
        "Media pesata dei parametri. Semplice e prevedibile: la scelta "
        "predefinita quando gli adapter hanno lo stesso rango."
    ),
    "cat": (
        "Concatena gli adapter: il rango risultante e' la somma dei ranghi. "
        "Conserva tutto, ma produce un adapter piu' grande."
    ),
    "ties": (
        "Risolve i conflitti fra adapter scartando i parametri di segno "
        "discorde. Utile quando gli adapter sono stati addestrati su compiti "
        "diversi e interferiscono."
    ),
    "dare_ties": (
        "Come ties, ma azzera prima una quota casuale di parametri. Riduce "
        "l'interferenza fra molti adapter."
    ),
    "svd": (
        "Fusione tramite decomposizione ai valori singolari. E' l'unica che "
        "gestisce adapter con ranghi diversi, al costo di piu' memoria."
    ),
}

DEFAULT_MERGE_STRATEGY = "linear"


@dataclass
class TrainingPlan:
    """Configurazione completa di un'esecuzione di addestramento."""

    mode: TrainingMode = TrainingMode.LORA_NEW
    base_model: str = "unsloth/Meta-Llama-3.1-8B-Instruct"
    adapters: List[AdapterSource] = field(default_factory=list)
    merge_strategy: str = DEFAULT_MERGE_STRATEGY
    output_name: str = "lora_adapters"
    # Sovrascrive i parametri della sezione `training` per questa esecuzione.
    overrides: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Restituisce i problemi che impedirebbero l'esecuzione."""
        problems: List[str] = []

        if self.mode.requires_adapters and not self.adapters:
            problems.append(
                f"La modalita' '{self.mode.label}' richiede almeno un adapter di partenza."
            )

        if self.mode is TrainingMode.LORA_CONTINUE and len(self.adapters) > 1:
            problems.append(
                "Si puo' continuare un solo adapter per volta. Per partire da piu' "
                "adapter usare 'Nuova LoRA sopra LoRA esistenti', che li fonde."
            )

        if self.merge_strategy not in MERGE_STRATEGIES:
            problems.append(
                f"Strategia di fusione sconosciuta: '{self.merge_strategy}'. "
                f"Valide: {', '.join(sorted(MERGE_STRATEGIES))}."
            )

        if not self.base_model.strip():
            problems.append("Nessun modello base indicato.")

        for adapter in self.adapters:
            if adapter.weight <= 0:
                problems.append(f"Peso non valido per '{adapter.name}': {adapter.weight}")

        return problems

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TrainingPlan":
        """Costruisce il piano dalla sezione `training` della configurazione."""
        training = config.get("training", {})
        raw_mode = str(training.get("mode", TrainingMode.LORA_NEW.value))

        try:
            mode = TrainingMode(raw_mode)
        except ValueError:
            mode = TrainingMode.LORA_NEW

        adapters: List[AdapterSource] = []
        for entry in training.get("start_from_adapters", []) or []:
            if isinstance(entry, str):
                adapters.append(AdapterSource(path=entry))
            elif isinstance(entry, dict) and entry.get("path"):
                adapters.append(AdapterSource(
                    path=str(entry["path"]),
                    weight=float(entry.get("weight", 1.0)),
                    name=str(entry.get("name", "")),
                ))

        return cls(
            mode=mode,
            base_model=str(training.get("base_model", cls.base_model)),
            adapters=adapters,
            merge_strategy=str(training.get("merge_strategy", DEFAULT_MERGE_STRATEGY)),
            output_name=str(training.get("output_name", "lora_adapters")),
        )
