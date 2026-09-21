"""Addestramento: LoRA nuova, ripresa, impilata o fine-tuning completo."""
from .adapter_manager import (
    AdapterInfo, check_compatibility, discover_adapters, read_adapter,
)
from .training_modes import (
    DEFAULT_MERGE_STRATEGY, MERGE_STRATEGIES, AdapterSource, TrainingMode, TrainingPlan,
)
from .training_stage import TrainingStage

__all__ = [
    "TrainingStage", "TrainingMode", "TrainingPlan", "AdapterSource",
    "AdapterInfo", "discover_adapters", "read_adapter", "check_compatibility",
    "MERGE_STRATEGIES", "DEFAULT_MERGE_STRATEGY",
]
