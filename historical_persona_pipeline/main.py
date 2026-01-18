import sys
import yaml
import logging
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from historical_persona_pipeline.gui.main_window import MainWindow

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    config_path = Path("historical_persona_pipeline/config/default_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Historical Persona Pipeline")
    
    config = load_config()
    
    window = MainWindow(config)
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
