from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QTextEdit, QProgressBar, QGroupBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

class StatusCard(QFrame):
    """Visual indicator for a pipeline stage status."""
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setStyleSheet("""
            StatusCard { 
                background-color: #f8f9fa; 
                border: 1px solid #dee2e6; 
                border-radius: 5px; 
            }
            QLabel#Title { font-weight: bold; color: #495057; }
            QLabel#Status { color: #6c757d; }
        """)
        
        layout = QVBoxLayout(self)
        
        self.title_lbl = QLabel(title)
        self.title_lbl.setObjectName("Title")
        layout.addWidget(self.title_lbl)
        
        self.status_lbl = QLabel("Pending")
        self.status_lbl.setObjectName("Status")
        layout.addWidget(self.status_lbl)
        
    def set_status(self, status, color=None):
        self.status_lbl.setText(status)
        if color:
            self.status_lbl.setStyleSheet(f"color: {color}; font-weight: bold;")

class ProcessingPanel(QWidget):
    # Signals to request actions from main window
    request_analysis = pyqtSignal()
    request_dataset = pyqtSignal()
    request_training = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 1. Status Overview (Cards)
        status_layout = QHBoxLayout()
        self.card_ingestion = StatusCard("1. Ingestion")
        self.card_analysis = StatusCard("2. Analysis")
        self.card_dataset = StatusCard("3. Dataset")
        self.card_training = StatusCard("4. Training")
        
        status_layout.addWidget(self.card_ingestion)
        status_layout.addWidget(self.card_analysis)
        status_layout.addWidget(self.card_dataset)
        status_layout.addWidget(self.card_training)
        
        layout.addLayout(status_layout)
        
        # 2. Controls & Logs
        controls_group = QGroupBox("Pipeline Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Action Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_analyze = QPushButton("Run Analysis")
        self.btn_analyze.clicked.connect(self.request_analysis.emit)
        self.btn_analyze.setEnabled(False) # Enabled after ingestion
        
        self.btn_dataset = QPushButton("Generate Dataset (LLM)")
        self.btn_dataset.clicked.connect(self.request_dataset.emit)
        self.btn_dataset.setEnabled(False) # Enabled after analysis
        
        self.btn_train = QPushButton("Start Training (Unsloth)")
        self.btn_train.clicked.connect(self.request_training.emit)
        self.btn_train.setStyleSheet("background-color: #d4edda; color: #155724;") # Greenish tint
        self.btn_train.setEnabled(False) # Enabled after dataset
        
        btn_layout.addWidget(self.btn_analyze)
        btn_layout.addWidget(self.btn_dataset)
        btn_layout.addWidget(self.btn_train)
        
        controls_layout.addLayout(btn_layout)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        controls_layout.addWidget(self.progress_bar)
        
        # Log Console
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFont(QFont("Consolas", 9))
        self.log_console.setStyleSheet("background-color: #212529; color: #f8f9fa;")
        controls_layout.addWidget(self.log_console)
        
        layout.addWidget(controls_group)
        
    def log(self, message: str):
        self.log_console.append(message)
        # Auto scroll
        sb = self.log_console.verticalScrollBar()
        sb.setValue(sb.maximum())
        
    def set_progress(self, value: int, message: str = ""):
        self.progress_bar.setValue(value)
        if message:
            self.progress_bar.setFormat(f"{message} (%p%)")
            
    def update_stage_status(self, stage_name: str, status: str, success: bool = True):
        color = "#28a745" if success else "#dc3545" # Green or Red
        
        if stage_name == "ingest":
            self.card_ingestion.set_status(status, color)
            self.btn_analyze.setEnabled(success)
        elif stage_name == "analyze":
            self.card_analysis.set_status(status, color)
            self.btn_dataset.setEnabled(success)
        elif stage_name == "dataset":
            self.card_dataset.set_status(status, color)
            self.btn_train.setEnabled(success)
        elif stage_name == "train":
            self.card_training.set_status(status, color)
