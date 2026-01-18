import json
from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QTextEdit, QPushButton, QComboBox, QMessageBox, QProgressBar,
    QGroupBox, QFormLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ...pipeline.utils.llm_provider import LLMFactory
from ...pipeline.stage2_analysis.value_profile_generator import ValueProfileGenerator

class GenerationWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, config, context):
        super().__init__()
        self.config = config
        self.context = context

    def run(self):
        try:
            provider = LLMFactory.create(self.config)
            generator = ValueProfileGenerator(provider)
            result = generator.generate_profile(self.context)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class ValueGeneratorDialog(QDialog):
    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.config = config or {}
        self.setWindowTitle("Generate Dynamic Value Profile")
        self.resize(600, 700)
        self.generated_data = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # 1. Configuration Section
        config_group = QGroupBox("LLM Configuration")
        form_layout = QFormLayout(config_group)
        
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["lm_studio", "nebius", "anthropic", "openai_compatible"])
        # Set default from config if available
        current_provider = self.config.get("dataset", {}).get("llm_provider", "lm_studio")
        index = self.provider_combo.findText(current_provider)
        if index >= 0:
            self.provider_combo.setCurrentIndex(index)
            
        self.api_url = QLineEdit(self.config.get("dataset", {}).get("llm_base_url", "http://localhost:1234/v1"))
        self.api_key = QLineEdit(self.config.get("dataset", {}).get("llm_api_key", "lm-studio"))
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.model_name = QLineEdit(self.config.get("dataset", {}).get("llm_model_name", "local-model"))

        form_layout.addRow("Provider:", self.provider_combo)
        form_layout.addRow("Base URL:", self.api_url)
        form_layout.addRow("API Key:", self.api_key)
        form_layout.addRow("Model:", self.model_name)
        
        layout.addWidget(config_group)

        # 2. Input Section
        input_group = QGroupBox("Context Description")
        input_layout = QVBoxLayout(input_group)
        
        input_label = QLabel("Describe the historical or cultural context (e.g., 'Samurai Bushido 16th Century', 'Victorian Etiquette'):")
        input_layout.addWidget(input_label)
        
        self.context_input = QLineEdit()
        self.context_input.setPlaceholderText("Enter context here...")
        input_layout.addWidget(self.context_input)
        
        self.generate_btn = QPushButton("Generate Profile")
        self.generate_btn.clicked.connect(self._start_generation)
        input_layout.addWidget(self.generate_btn)
        
        layout.addWidget(input_group)

        # 3. Output Section
        output_group = QGroupBox("Generated JSON")
        output_layout = QVBoxLayout(output_group)
        
        self.result_viewer = QTextEdit()
        self.result_viewer.setReadOnly(False) # Allow manual edits
        self.result_viewer.setFontFamily("Consolas")
        output_layout.addWidget(self.result_viewer)
        
        layout.addWidget(output_group)

        # 4. Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0) # Indeterminate
        layout.addWidget(self.progress_bar)

        # 5. Dialog Buttons
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save to File")
        self.save_btn.clicked.connect(self._save_file)
        self.save_btn.setEnabled(False)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def _get_current_config(self):
        # Construct a temporary config dict for the factory
        return {
            "llm_provider": self.provider_combo.currentText(),
            "llm_base_url": self.api_url.text(),
            "llm_api_key": self.api_key.text(),
            "llm_model_name": self.model_name.text()
        }

    def _start_generation(self):
        context = self.context_input.text().strip()
        if not context:
            QMessageBox.warning(self, "Input Error", "Please provide a context description.")
            return

        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.result_viewer.clear()
        
        config = self._get_current_config()
        
        self.worker = GenerationWorker(config, context)
        self.worker.finished.connect(self._on_generation_success)
        self.worker.error.connect(self._on_generation_error)
        self.worker.start()

    def _on_generation_success(self, result):
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        
        self.generated_data = result
        formatted_json = json.dumps(result, indent=2, ensure_ascii=False)
        self.result_viewer.setText(formatted_json)

    def _on_generation_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        QMessageBox.critical(self, "Generation Error", f"An error occurred:\n{error_msg}")

    def _save_file(self):
        try:
            # Parse text from viewer in case user edited it
            content = self.result_viewer.toPlainText()
            data = json.loads(content)
            
            # Generate filename suggestion
            context_slug = self.context_input.text().strip().lower().replace(" ", "_")
            filename = f"{context_slug}_values.json"
            
            from PyQt6.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Value Profile", 
                str(Path("historical_persona_pipeline/data/value_dictionaries") / filename),
                "JSON Files (*.json)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                QMessageBox.information(self, "Success", f"Profile saved to {file_path}")
                self.accept()
                
        except json.JSONDecodeError:
            QMessageBox.warning(self, "Invalid JSON", "The content is not valid JSON. Please correct it before saving.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save file: {e}")
