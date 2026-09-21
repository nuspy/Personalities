"""Caricamento configurazione con merge sui default.

Garantisce che il dizionario restituito contenga SEMPRE tutte le sezioni
attese (`persona`, `analysis`, `dataset`, `training`, `evaluation`,
`advanced`): il codice a valle puo' quindi indicizzare senza KeyError anche
quando il file di configurazione e' assente o parziale.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .paths import DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)

# Default minimi: ogni chiave letta dal codice deve comparire qui.
FALLBACK_CONFIG: Dict[str, Any] = {
    "persona": {
        "author_name": "",
        "era": "",
        "value_dict": "auto",
        "system_prompt_override": None,
    },
    "analysis": {
        "min_text_length": 100,
        "extract_formulas": True,
        "detect_special_patterns": True,
        "max_chars_per_nlp_batch": 400_000,
        "max_words_analyzed": 2_000_000,
    },
    "ingestion": {
        "target_chunk_chars": 4_000,
        "min_chunk_chars": 200,
        "ocr_fallback": True,
        "ocr_max_pages": 50,
    },
    "dataset": {
        "num_conversations": 1000,
        "conversation_types": {
            "historical_facts": 0.25,
            "philosophy_values": 0.20,
            "anachronistic": 0.20,
            "character_personality": 0.15,
            "domain_expertise": 0.15,
            "casual_conversation": 0.05,
        },
        "include_negative_examples": True,
        "negative_example_ratio": 0.2,
        "val_split": 0.1,
        "max_workers": 4,
        "max_retries": 3,
        "llm_provider": "lm_studio",
        "llm_base_url": "http://localhost:1234/v1",
        "llm_api_key": "lm-studio",
        "llm_model_name": "local-model",
    },
    "research": {
        "enabled": True,
        "max_documents": 25,
        "max_chars_per_document": 400_000,
        "sources": ["wikipedia", "wikisource", "gutenberg"],
        "user_agent": "HistoricalPersonaPipeline/0.2 (local research tool)",
    },
    "training": {
        "mode": "lora_new",
        "start_from_adapters": [],
        "merge_strategy": "linear",
        "output_name": "lora_adapters",
        "optimizer": "adamw_8bit",
        "lr_scheduler": "linear",
        "base_model": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "lora": {
            "r": 64,
            "alpha": 128,
            "dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        },
        "max_seq_length": 4096,
        "load_in_4bit": True,
        "learning_rate": 0.0002,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "num_epochs": 3,
        "max_steps": -1,
        "per_device_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "bf16": True,
        "gradient_checkpointing": True,
        "save_steps": 100,
        "save_total_limit": 3,
        "chat_template": "auto",
    },
    "conversion": {
        "llama_cpp_path": "",
        "default_format": "merged_16bit",
        "default_quantization": "q4_k_m",
        "keep_intermediate": False,
        "temperature": 0.8,
        "command_timeout": 7200,
    },
    "evaluation": {
        "style_consistency_weight": 0.30,
        "anachronism_handling_weight": 0.30,
        "personality_coherence_weight": 0.20,
        "knowledge_boundaries_weight": 0.20,
        "test_prompts_per_category": 5,
    },
    "advanced": {
        "seed": 42,
        "logging_steps": 10,
        "eval_steps": 100,
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ricorsivo: `override` vince, ma non cancella chiavi assenti."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Carica la configurazione e la fonde sui default.

    Il percorso e' derivato dal package, non dalla working directory.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    user_config: Dict[str, Any] = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.error(f"Configurazione illeggibile ({path}): {exc}. Uso i default.")
    else:
        logger.warning(f"Configurazione non trovata in {path}. Uso i default.")

    return deep_merge(FALLBACK_CONFIG, user_config)
