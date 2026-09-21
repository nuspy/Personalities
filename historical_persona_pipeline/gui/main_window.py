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
from .conversion_panel import ConversionPanel
from .processing_panel import ProcessingPanel
from .training_panel import TrainingPanel
from .dialogs.value_generator_dialog import ValueGeneratorDialog

from ..paths import PROJECTS_DIR
from ..pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setWindowTitle("Historical Persona Pipeline")
        self.resize(1200, 850)
        
        # Percorso canonico del package: indipendente dalla cartella da cui
        # l'applicazione viene avviata.
        self.project_id = str(uuid.uuid4())[:8]
        self.project_dir = PROJECTS_DIR / f"project_{self.project_id}"
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
        
        # Azioni: ricerca automatica oppure file propri.
        actions = QHBoxLayout()

        self.btn_research = QPushButton("Cerca fonti online")
        self.btn_research.setToolTip(
            "Cerca automaticamente testi e biografie del personaggio indicato "
            "e li aggiunge all'elenco dei file."
        )
        self.btn_research.setStyleSheet("padding: 10px; font-size: 14px;")
        self.btn_research.clicked.connect(self._start_research)
        actions.addWidget(self.btn_research)

        self.btn_ingest = QPushButton("Importa e avvia")
        self.btn_ingest.setStyleSheet("padding: 10px; font-weight: bold; font-size: 14px;")
        self.btn_ingest.clicked.connect(self._start_ingestion)
        actions.addWidget(self.btn_ingest, stretch=2)

        setup_layout.addLayout(actions)
        
        self.tabs.addTab(setup_widget, "1. Setup")
        
        # Tab 2: Processing
        self.processing_panel = ProcessingPanel()
        self.processing_panel.request_analysis.connect(self._start_analysis)
        self.processing_panel.request_dataset.connect(self._start_dataset)
        self.processing_panel.request_training.connect(self._start_training)
        
        self.tabs.addTab(self.processing_panel, "2. Elaborazione")
        self.tabs.setTabEnabled(1, False)  # abilitato dopo l'ingestione

        # Tab 3: modalita' di addestramento. La scelta fra nuova LoRA,
        # ripresa, impilamento e fine-tuning completo si fa qui, prima di
        # avviare l'addestramento dal tab precedente.
        self.training_panel = TrainingPanel(self.config, self.project_dir)
        self.tabs.addTab(self.training_panel, "3. Addestramento")

        # Tab 4: conversione. Non dipende dagli altri stadi — si puo' aprire
        # il programma solo per convertire un adapter gia' esistente.
        self.conversion_panel = ConversionPanel(self.config, self.project_dir)
        self.conversion_panel.request_conversion.connect(self._start_conversion)
        self.tabs.addTab(self.conversion_panel, "4. Conversione")

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

    def _start_research(self):
        self.config.update(self.config_panel.get_config())
        author = self.config["persona"].get("author_name", "").strip()

        if not author:
            QMessageBox.warning(
                self, "Nome mancante",
                "Indicare il nome del personaggio prima di cercare online.",
            )
            return

        self.tabs.setCurrentIndex(1)
        self.tabs.setTabEnabled(1, True)
        self.processing_panel.log(f"Ricerca online di materiale su {author}...")
        self.orchestrator.start_research(author, self.config["persona"].get("era", ""))

    def _start_ingestion(self):
        files = self.file_selector.get_selected_files()
        if not files:
            QMessageBox.warning(
                self, "Nessun file",
                "Selezionare almeno un file, oppure usare 'Cerca fonti online'.",
            )
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
        plan = self.training_panel.build_plan()

        problems = plan.validate()
        if problems:
            # Meglio dirlo ora che dopo il caricamento del modello, che su un
            # 8B richiede minuti.
            QMessageBox.warning(
                self, "Configurazione incompleta",
                "\n".join(problems) + "\n\nCorreggi nel tab '3. Addestramento'.",
            )
            self.tabs.setCurrentWidget(self.training_panel)
            return

        self.processing_panel.log(f"Addestramento: {plan.mode.label}")
        if plan.adapters:
            self.processing_panel.log(
                "Adapter di partenza: "
                + ", ".join(f"{a.name} (peso {a.weight:g})" for a in plan.adapters)
            )
        self.processing_panel.card_training.set_status("In corso...", "#007bff")

        if not self.orchestrator.start_training(plan):
            self.processing_panel.log(
                "Avvio non riuscito: un'altra operazione e' in corso, "
                "oppure manca il dataset."
            )

    def _start_conversion(self, request):
        if not self.orchestrator.start_conversion(request):
            self.conversion_panel.on_error(
                "Un'altra operazione e' gia' in corso: attendere che finisca."
            )

    def _open_value_generator(self):
        dialog = ValueGeneratorDialog(self, self.config)
        dialog.exec()

    # -- Callbacks --

    def _on_progress(self, value, message):
        # Durante una conversione l'avanzamento interessa il tab conversione.
        if self.tabs.currentWidget() is self.conversion_panel:
            self.conversion_panel.set_progress(value, message)
        self.processing_panel.set_progress(value, message)
        self.status_bar.showMessage(message)
        if value % 10 == 0: # Avoid spamming logs
             self.processing_panel.log(f"[Progress {value}%] {message}")

    def _on_stage_completed(self, stage_name, result):
        self.processing_panel.log(f"Stage '{stage_name}' completed successfully.")
        
        if stage_name == "research":
            # I documenti scaricati entrano nell'elenco file e seguono da li'
            # lo stesso percorso di quelli caricati a mano.
            self.processing_panel.log(f"Scaricati {len(result)} documenti.")
            if result:
                self.file_selector.add_files(list(result))
                self.tabs.setCurrentIndex(0)
                QMessageBox.information(
                    self, "Ricerca completata",
                    f"{len(result)} documenti aggiunti all'elenco.\n"
                    "Rivedili se vuoi, poi premi 'Importa e avvia'.",
                )
            else:
                QMessageBox.warning(
                    self, "Nessun risultato",
                    "La ricerca non ha prodotto documenti pertinenti.\n"
                    "Prova a precisare nome ed epoca, oppure carica i file manualmente.",
                )

        elif stage_name == "ingest":
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
            self.processing_panel.update_stage_status("train", "Completato")
            output_path = result.get("output_path", "")
            self.processing_panel.log(f"Addestramento concluso: {output_path}")

            # Il risultato appena prodotto deve comparire subito fra gli
            # adapter disponibili, senza riavviare il programma.
            self.training_panel.refresh_adapters()
            self.conversion_panel.refresh_adapters()

            QMessageBox.information(
                self, "Addestramento completato",
                f"Modalità: {result.get('mode', '')}\n"
                f"Risultato in:\n{output_path}\n\n"
                "Ora puoi convertirlo dal tab '4. Conversione'.",
            )

        elif stage_name == "convert":
            self.conversion_panel.on_finished(result)
            self.processing_panel.log(
                f"Conversione completata: {result.get('output_path', '')}"
            )

    def _on_error(self, message):
        self.processing_panel.log(f"ERRORE: {message}")
        self.status_bar.showMessage(f"Errore: {message}")

        # Un avviso non deve interrompere il lavoro con una finestra modale:
        # gli stadi emettono error_occurred anche per segnalazioni recuperabili.
        if message.lower().startswith("avviso"):
            return

        if self.tabs.currentWidget() is self.conversion_panel:
            self.conversion_panel.on_error(message)

        QMessageBox.critical(self, "Errore", message)
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
            # `terminate()` interrompe il thread in un punto arbitrario:
            # si concede prima una chiusura ordinata.
            self.orchestrator.worker.requestInterruption()
            if not self.orchestrator.worker.wait(3000):
                self.orchestrator.worker.terminate()
                self.orchestrator.worker.wait()
        event.accept()