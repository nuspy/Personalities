import sys
from pathlib import Path
import logging
import uuid
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QMenuBar, QMenu, QMessageBox, QStatusBar, QApplication, QPushButton
)
from PyQt6.QtGui import QAction

# Import components
from .file_selector import FileSelector
from .config_panel import ConfigPanel
from .processing_panel import ProcessingPanel
from .dialogs.value_generator_dialog import ValueGeneratorDialog

from ..pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setWindowTitle("Historical Persona Pipeline")
        self.resize(1200, 850)
        
        # Initialize project (simple folder structure for now)
        self.project_id = str(uuid.uuid4())[:8]
        self.project_dir = Path(f"historical_persona_pipeline/projects/project_{self.project_id}")
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Orchestrator
        self.orchestrator = PipelineOrchestrator(self.config, self.project_dir)
        self._connect_orchestrator_signals()
        
        self._setup_ui()
        self._create_menus()
        
    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Tabs
        self.tabs = QTabWidget()
        
        # Tab 1: Setup (Files + Config)
        setup_widget = QWidget()
        setup_layout = QVBoxLayout(setup_widget)
        
        # Split setup into horizontal: Files | Config
        top_setup = QHBoxLayout()
        self.file_selector = FileSelector()
        self.config_panel = ConfigPanel(self.config)
        
        top_setup.addWidget(self.file_selector, stretch=1)
        top_setup.addWidget(self.config_panel, stretch=1)
        setup_layout.addLayout(top_setup)
        
        # Process Button (Transition to Tab 2)
        self.btn_ingest = QPushButton("Import Files & Initialize Project")
        self.btn_ingest.setStyleSheet("padding: 10px; font-weight: bold; font-size: 14px;")
        self.btn_ingest.clicked.connect(self._start_ingestion)
        setup_layout.addWidget(self.btn_ingest)
        
        self.tabs.addTab(setup_widget, "1. Setup")
        
        # Tab 2: Processing
        self.processing_panel = ProcessingPanel()
        self.processing_panel.request_analysis.connect(self._start_analysis)
        self.processing_panel.request_dataset.connect(self._start_dataset)
        self.processing_panel.request_training.connect(self._start_training)
        
        self.tabs.addTab(self.processing_panel, "2. Processing")
        self.tabs.setTabEnabled(1, False) # Disabled until ingestion
        
        main_layout.addWidget(self.tabs)
        
        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(f"Project ID: {self.project_id}")

    def _create_menus(self):
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("&File")
        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools Menu
        tools_menu = menubar.addMenu("&Tools")
        val_gen_action = QAction("&Generate Value Profile (LLM)", self)
        val_gen_action.triggered.connect(self._open_value_generator)
        tools_menu.addAction(val_gen_action)

    def _connect_orchestrator_signals(self):
        worker = self.orchestrator.worker
        worker.progress_update.connect(self._on_progress)
        worker.stage_completed.connect(self._on_stage_completed)
        worker.error_occurred.connect(self._on_error)

    # -- Actions --

    def _start_ingestion(self):
        files = self.file_selector.get_selected_files()
        if not files:
            QMessageBox.warning(self, "No Files", "Please select at least one file.")
            return
            
        # Update config from panel before starting
        self.config.update(self.config_panel.get_config())
        
        success = self.orchestrator.start_ingestion(files)
        if success:
            self.tabs.setCurrentIndex(1)
            self.tabs.setTabEnabled(1, True)
            self.processing_panel.log(f"Starting ingestion of {len(files)} files...")
            self.processing_panel.card_ingestion.set_status("Running...", "#007bff")

    def _start_analysis(self):
        self.processing_panel.log("Starting linguistic analysis...")
        self.processing_panel.card_analysis.set_status("Running...", "#007bff")
        self.orchestrator.start_analysis()

    def _start_dataset(self):
        self.processing_panel.log("Starting dataset generation (LLM)...")
        self.processing_panel.card_dataset.set_status("Running...", "#007bff")
        self.orchestrator.start_dataset_generation()

    def _start_training(self):
        self.processing_panel.log("Starting model training (Unsloth)...")
        self.processing_panel.card_training.set_status("Running...", "#007bff")
        self.orchestrator.start_training()

    def _open_value_generator(self):
        dialog = ValueGeneratorDialog(self, self.config)
        dialog.exec()

    # -- Callbacks --

    def _on_progress(self, value, message):
        self.processing_panel.set_progress(value, message)
        self.status_bar.showMessage(message)
        if value % 10 == 0: # Avoid spamming logs
             self.processing_panel.log(f"[Progress {value}%] {message}")

    def _on_stage_completed(self, stage_name, result):
        self.processing_panel.log(f"Stage '{stage_name}' completed successfully.")
        
        if stage_name == "ingest":
            self.orchestrator.ingestion_result = result
            self.processing_panel.update_stage_status("ingest", "Completed")
            self.processing_panel.log(f"Processed {result.total_segments} text segments.")
            
        elif stage_name == "analyze":
            self.orchestrator.style_profile = result
            self.processing_panel.update_stage_status("analyze", "Completed")
            self.processing_panel.log(f"Style Profile created for: {result.author_name}")
            
        elif stage_name == "dataset":
            self.orchestrator.dataset = result
            self.processing_panel.update_stage_status("dataset", "Completed")
            self.processing_panel.log(f"Generated {result.total_conversations} conversations.")
            
        elif stage_name == "train":
            self.processing_panel.update_stage_status("train", "Completed")
            self.processing_panel.log(f"Training finished! Output: {result.get('output_path')}")
            QMessageBox.information(self, "Success", "Pipeline completed successfully!\nModel is ready.")

    def _on_error(self, message):
        self.processing_panel.log(f"ERROR: {message}")
        self.status_bar.showMessage(f"Error: {message}")
        QMessageBox.critical(self, "Pipeline Error", message)
        
        # Reset UI state if needed?
        self.processing_panel.progress_bar.setValue(0)

    def closeEvent(self, event):
        if self.orchestrator.worker.isRunning():
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "Pipeline is running. Stop and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self.orchestrator.worker.terminate() # Force kill for now
            self.orchestrator.worker.wait()
        event.accept()