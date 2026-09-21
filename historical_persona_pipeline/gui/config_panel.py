"""Pannello di configurazione.

Correzioni: l'accesso diretto a `self.config['persona']` faceva crashare
l'applicazione all'avvio ogni volta che la configurazione era assente o
parziale (`load_config` restituiva `{}` se il file non veniva trovato, e il
percorso era relativo alla working directory). Ora si legge sempre con
`.get()` su una configurazione gia' fusa coi default.

L'elenco dei dizionari di valori e' letto da disco: prima era scritto a mano
con tre voci, di cui due — `bushido_code` e `victorian_values` — non
esistevano, mentre i dizionari generati dall'apposito strumento non
comparivano mai.
"""
from __future__ import annotations

from typing import Any, Dict

from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QLabel,
    QLineEdit, QSpinBox, QVBoxLayout, QWidget,
)

from ..paths import VALUE_DICT_DIR

AUTO_VALUE_DICT = "auto (scegli dall'epoca)"


class ConfigPanel(QWidget):
    def __init__(self, config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.config = config if isinstance(config, dict) else {}
        self._ensure_sections()
        self._setup_ui()

    def _ensure_sections(self) -> None:
        """Garantisce le sezioni attese, cosi' nessun accesso puo' fallire."""
        for section in ("persona", "training", "dataset", "research"):
            self.config.setdefault(section, {})
        self.config["training"].setdefault("lora", {})

    # ------------------------------------------------------------------ UI

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        persona = self.config["persona"]
        persona_group = QGroupBox("Personaggio")
        persona_layout = QFormLayout(persona_group)

        self.author_name = QLineEdit(str(persona.get("author_name", "")))
        self.author_name.setPlaceholderText("es. Gaio Giulio Cesare")
        persona_layout.addRow("Nome:", self.author_name)

        self.era = QLineEdit(str(persona.get("era", "")))
        self.era.setPlaceholderText("es. Repubblica romana, I secolo a.C.")
        persona_layout.addRow("Epoca:", self.era)

        self.value_dict = QComboBox()
        self.value_dict.addItem(AUTO_VALUE_DICT)
        self.value_dict.addItems(self._available_value_dicts())
        current = str(persona.get("value_dict", "auto"))
        index = self.value_dict.findText(current)
        self.value_dict.setCurrentIndex(index if index >= 0 else 0)
        persona_layout.addRow("Dizionario valori:", self.value_dict)

        layout.addWidget(persona_group)

        research = self.config["research"]
        research_group = QGroupBox("Ricerca online")
        research_layout = QFormLayout(research_group)

        self.research_enabled = QCheckBox("Cerca automaticamente fonti online")
        self.research_enabled.setChecked(bool(research.get("enabled", True)))
        research_layout.addRow(self.research_enabled)

        self.max_documents = QSpinBox()
        self.max_documents.setRange(1, 200)
        self.max_documents.setValue(int(research.get("max_documents", 25)))
        research_layout.addRow("Documenti max:", self.max_documents)

        self.relevance_threshold = QDoubleSpinBox()
        self.relevance_threshold.setRange(0.0, 1.0)
        self.relevance_threshold.setSingleStep(0.05)
        self.relevance_threshold.setValue(float(research.get("relevance_threshold", 0.40)))
        self.relevance_threshold.setToolTip(
            "Soglia di pertinenza: piu' alta scarta piu' omonimi, "
            "piu' bassa accetta piu' materiale di contesto."
        )
        research_layout.addRow("Soglia pertinenza:", self.relevance_threshold)

        layout.addWidget(research_group)

        dataset = self.config["dataset"]
        dataset_group = QGroupBox("Dataset")
        dataset_layout = QFormLayout(dataset_group)

        self.num_conversations = QSpinBox()
        self.num_conversations.setRange(10, 100_000)
        self.num_conversations.setValue(int(dataset.get("num_conversations", 1000)))
        dataset_layout.addRow("Conversazioni:", self.num_conversations)

        self.llm_base_url = QLineEdit(str(dataset.get("llm_base_url", "")))
        dataset_layout.addRow("Endpoint LLM:", self.llm_base_url)

        self.llm_model_name = QLineEdit(str(dataset.get("llm_model_name", "")))
        dataset_layout.addRow("Modello LLM:", self.llm_model_name)

        layout.addWidget(dataset_group)

        training = self.config["training"]
        training_group = QGroupBox("Addestramento")
        training_layout = QFormLayout(training_group)

        self.base_model = QLineEdit(
            str(training.get("base_model", "unsloth/Meta-Llama-3.1-8B-Instruct"))
        )
        training_layout.addRow("Modello base:", self.base_model)

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 100)
        self.epochs.setValue(int(training.get("num_epochs", 3)))
        training_layout.addRow("Epoche:", self.epochs)

        self.lora_rank = QSpinBox()
        self.lora_rank.setRange(4, 256)
        self.lora_rank.setValue(int(training["lora"].get("r", 64)))
        training_layout.addRow("Rango LoRA:", self.lora_rank)

        layout.addWidget(training_group)

        hint = QLabel(
            "Il profilo si costruisce senza LLM. L'endpoint serve solo per "
            "generare il dataset di addestramento."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #6c757d; font-size: 11px;")
        layout.addWidget(hint)

        layout.addStretch()

    @staticmethod
    def _available_value_dicts() -> list[str]:
        """Dizionari effettivamente presenti su disco."""
        if not VALUE_DICT_DIR.exists():
            return []
        return sorted(path.stem for path in VALUE_DICT_DIR.glob("*.json"))

    def refresh_value_dicts(self) -> None:
        """Ricarica l'elenco: serve dopo averne generato uno nuovo."""
        current = self.value_dict.currentText()
        self.value_dict.clear()
        self.value_dict.addItem(AUTO_VALUE_DICT)
        self.value_dict.addItems(self._available_value_dicts())
        index = self.value_dict.findText(current)
        self.value_dict.setCurrentIndex(index if index >= 0 else 0)

    # -------------------------------------------------------------- lettura

    def get_config(self) -> Dict[str, Any]:
        """Riporta i valori dei widget nella configurazione e la restituisce."""
        persona = self.config["persona"]
        persona["author_name"] = self.author_name.text().strip()
        persona["era"] = self.era.text().strip()
        selected = self.value_dict.currentText()
        persona["value_dict"] = "auto" if selected == AUTO_VALUE_DICT else selected

        research = self.config["research"]
        research["enabled"] = self.research_enabled.isChecked()
        research["max_documents"] = self.max_documents.value()
        research["relevance_threshold"] = self.relevance_threshold.value()

        dataset = self.config["dataset"]
        dataset["num_conversations"] = self.num_conversations.value()
        dataset["llm_base_url"] = self.llm_base_url.text().strip()
        dataset["llm_model_name"] = self.llm_model_name.text().strip()

        training = self.config["training"]
        training["base_model"] = self.base_model.text().strip()
        training["num_epochs"] = self.epochs.value()
        training["lora"]["r"] = self.lora_rank.value()

        return self.config
