import sys
import logging

from PyQt6.QtWidgets import QApplication

from historical_persona_pipeline.config_loader import load_config
from historical_persona_pipeline.gui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Historical Persona Pipeline")

    config = load_config()

    window = MainWindow(config)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
