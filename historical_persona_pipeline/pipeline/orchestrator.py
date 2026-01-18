from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from .data_models import IngestionResult, CompleteStyleProfile, TrainingDataset
from .stage1_ingestion.ingestion_stage import IngestionStage
from .stage2_analysis.analysis_stage import AnalysisStage
from .stage3_dataset.dataset_stage import DatasetStage
from .stage4_training.training_stage import TrainingStage

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
        self.ingestion = IngestionStage(config)
        self.analysis = AnalysisStage(config, project_dir)
        self.dataset = DatasetStage(config, project_dir)
        self.training = TrainingStage(config, project_dir)
        
        # Connect signals
        for stage in [self.ingestion, self.analysis, self.dataset, self.training]:
            stage.progress_update.connect(self.progress_update)
            stage.error_occurred.connect(self.error_occurred)

    def set_action(self, action: str, input_data: Any = None):
        """Sets the action to perform next time run() is called."""
        self.current_action = action
        self.input_data = input_data

    def run(self):
        try:
            if self.current_action == "ingest":
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
                # Input data here is expected to be the path to the jsonl file
                result = self.training.run(self.input_data)
                self.stage_completed.emit("train", result)
                
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

    def start_training(self):
        if self.worker.isRunning() or not self.dataset:
            # Check if we have a dataset on disk even if object is missing
            dataset_path = self.project_dir / "datasets" / "train_dataset.jsonl"
            if not dataset_path.exists():
                return False
        else:
            # Ideally use the dataset object to get path, but standard path is safer
            dataset_path = self.project_dir / "datasets" / "train_dataset.jsonl"
            
        self.worker.set_action("train", dataset_path)
        self.worker.start()
        return True
