from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit, QComboBox, 
    QCheckBox, QSpinBox, QDoubleSpinBox, QGroupBox, QLabel
)
from typing import Dict, Any

class ConfigPanel(QWidget):
    def __init__(self, config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.config = config
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Persona Config
        persona_group = QGroupBox("Persona Settings")
        persona_layout = QFormLayout(persona_group)
        
        self.author_name = QLineEdit(self.config['persona'].get('author_name', ''))
        persona_layout.addRow("Author Name:", self.author_name)
        
        self.era = QLineEdit(self.config['persona'].get('era', ''))
        persona_layout.addRow("Historical Era:", self.era)
        
        # We might populate this dynamically later
        self.value_dict = QComboBox()
        self.value_dict.addItems(["roman_values", "bushido_code", "victorian_values"]) 
        self.value_dict.setCurrentText(self.config['persona'].get('value_dict', 'roman_values'))
        persona_layout.addRow("Value Dictionary:", self.value_dict)
        
        layout.addWidget(persona_group)
        
        # Training Config
        training_group = QGroupBox("Training Configuration")
        training_layout = QFormLayout(training_group)
        
        self.base_model = QLineEdit(self.config['training'].get('base_model', 'unsloth/Meta-Llama-3.1-8B-Instruct'))
        training_layout.addRow("Base Model:", self.base_model)
        
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 100)
        self.epochs.setValue(self.config['training'].get('num_epochs', 3))
        training_layout.addRow("Epochs:", self.epochs)
        
        self.lora_rank = QSpinBox()
        self.lora_rank.setRange(8, 256)
        self.lora_rank.setValue(self.config['training']['lora'].get('r', 64))
        training_layout.addRow("LoRA Rank:", self.lora_rank)
        
        layout.addWidget(training_group)
        
        # Add stretch
        layout.addStretch()
        
    def get_config(self) -> Dict[str, Any]:
        """Returns the updated configuration."""
        # Update internal config object
        self.config['persona']['author_name'] = self.author_name.text()
        self.config['persona']['era'] = self.era.text()
        self.config['persona']['value_dict'] = self.value_dict.currentText()
        
        self.config['training']['base_model'] = self.base_model.text()
        self.config['training']['num_epochs'] = self.epochs.value()
        self.config['training']['lora']['r'] = self.lora_rank.value()
        
        return self.config
