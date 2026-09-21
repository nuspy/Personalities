from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from .data_models import IngestionResult, CompleteStyleProfile, TrainingDataset
from .stage0_research.research_stage import ResearchStage
from .stage1_ingestion.ingestion_stage import IngestionStage
from .stage2_analysis.analysis_stage import AnalysisStage
from .stage3_dataset.dataset_stage import DatasetStage
from .stage4_training.training_stage import TrainingStage
from .stage4_training.training_modes import TrainingPlan
from .conversion.conversion_stage import ConversionStage

logger = logging.getLogger(__name__)

class PipelineWorker(QThread):
    """Background worker thread for running pipeline stages."""
    
    progress_update = pyqtSignal(int, str)  # % completion, status message
    stage_completed = pyqtSignal(str, object)  # stage_name, result_data
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, config: Dict[str, Any], project_dir: Path):
        super().__init__()
        self.config = config
        self.project_dir = project_dir
        self.current_action = None
        self.input_data = None
        
        # Initialize Stages
        self.research = ResearchStage(config, project_dir)
        self.ingestion = IngestionStage(config)
        self.analysis = AnalysisStage(config, project_dir)
        self.dataset = DatasetStage(config, project_dir)
        self.training = TrainingStage(config, project_dir)
        self.conversion = ConversionStage(config, project_dir)
        
        # Connect signals
        for stage in [self.research, self.ingestion, self.analysis,
                      self.dataset, self.training, self.conversion]:
            stage.progress_update.connect(self.progress_update)
            stage.error_occurred.connect(self.error_occurred)

    def set_action(self, action: str, input_data: Any = None):
        """Sets the action to perform next time run() is called."""
        self.current_action = action
        self.input_data = input_data

    def run(self):
        try:
            if self.current_action == "research":
                result = self.research.run(self.input_data or {})
                self.stage_completed.emit("research", result)

            elif self.current_action == "ingest":
                if not self.input_data:
                    raise ValueError("No files provided for ingestion")
                result = self.ingestion.run(self.input_data)
                self.stage_completed.emit("ingest", result)
                
            elif self.current_action == "analyze":
                if not self.input_data:
                    raise ValueError("No ingestion result provided for analysis")
                result = self.analysis.run(self.input_data)
                self.stage_completed.emit("analyze", result)
                
            elif self.current_action == "dataset":
                if not self.input_data:
                    raise ValueError("No style profile provided for dataset generation")
                result = self.dataset.run(self.input_data)
                self.stage_completed.emit("dataset", result)
                
            elif self.current_action == "train":
                if not self.input_data:
                    raise ValueError("No dataset path provided for training")
                # L'input e' (percorso_dataset, piano) oppure il solo percorso.
                if isinstance(self.input_data, tuple):
                    dataset_path, plan = self.input_data
                else:
                    dataset_path, plan = self.input_data, None
                result = self.training.run(dataset_path, plan)
                self.stage_completed.emit("train", result)

            elif self.current_action == "convert":
                if not self.input_data:
                    raise ValueError("Nessuna richiesta di conversione fornita")
                result = self.conversion.run(self.input_data)
                self.stage_completed.emit("convert", result)
                
            else:
                self.error_occurred.emit(f"Unknown action: {self.current_action}")
                
        except Exception as e:
            logger.exception("Pipeline error")
            self.error_occurred.emit(str(e))
        
        finally:
            self.finished.emit()

class PipelineOrchestrator(QObject):
    """Manages the pipeline state and worker thread."""
    
    def __init__(self, config: Dict[str, Any], project_dir: Path):
        super().__init__()
        self.worker = PipelineWorker(config, project_dir)
        self.project_dir = project_dir
        
        # State storage
        self.ingestion_result: Optional[IngestionResult] = None
        self.style_profile: Optional[CompleteStyleProfile] = None
        self.dataset: Optional[TrainingDataset] = None
        
    def start_research(self, author_name: str, era: str = "", languages=None):
        """Cerca online il materiale sul personaggio."""
        if self.worker.isRunning():
            return False
        self.worker.set_action("research", {
            "author_name": author_name,
            "era": era,
            "languages": languages,
        })
        self.worker.start()
        return True

    def start_ingestion(self, files: List[Path]):
        if self.worker.isRunning():
            return False
        self.worker.set_action("ingest", files)
        self.worker.start()
        return True

    def start_analysis(self):
        if self.worker.isRunning() or not self.ingestion_result:
            return False
        self.worker.set_action("analyze", self.ingestion_result)
        self.worker.start()
        return True

    def start_dataset_generation(self):
        if self.worker.isRunning() or not self.style_profile:
            return False
        
        # Pass both profile and ingestion result (for content extraction)
        input_data = {
            "profile": self.style_profile,
            "ingestion_result": self.ingestion_result
        }
        self.worker.set_action("dataset", input_data)
        self.worker.start()
        return True

    def start_training(self, plan: Optional[TrainingPlan] = None):
        """Avvia l'addestramento secondo il piano indicato.

        `plan` assente significa: leggilo dalla configurazione.
        """
        # Un worker gia' in esecuzione blocca sempre: avviarne un secondo sullo
        # stesso QThread corromperebbe lo stato della pipeline.
        if self.worker.isRunning():
            return False

        dataset_path = self.project_dir / "datasets" / "train_dataset.jsonl"
        if not dataset_path.exists():
            logger.error(f"Dataset non trovato: {dataset_path}")
            return False

        if plan is not None:
            problems = plan.validate()
            if problems:
                logger.error("Piano di addestramento non valido: " + " ".join(problems))
                return False

        self.worker.set_action("train", (dataset_path, plan))
        self.worker.start()
        return True

    def start_conversion(self, request: Dict[str, Any]):
        """Avvia una conversione (fusione, GGUF, pacchetto Ollama)."""
        if self.worker.isRunning():
            return False
        self.worker.set_action("convert", request)
        self.worker.start()
        return True
