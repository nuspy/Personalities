from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
from PyQt6.QtCore import QObject, pyqtSignal

class PipelineStage(QObject):
    """Abstract base class for all pipeline stages."""
    
    # Signals for GUI updates
    progress_update = pyqtSignal(int, str)  # percentage, status message
    error_occurred = pyqtSignal(str)
    stage_completed = pyqtSignal(object)  # returns stage result
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """Execute the stage logic."""
        pass
    
    def validate_config(self) -> bool:
        """Validate stage-specific configuration."""
        return True
