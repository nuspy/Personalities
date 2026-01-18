from pathlib import Path
from typing import List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QFileDialog, QGroupBox, QAbstractItemView,
    QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

class FileSelector(QWidget):
    """Widget for selecting and managing input files with drag-and-drop."""
    
    files_changed = pyqtSignal(list)
    
    SUPPORTED_EXTENSIONS = {
        '.txt': 'Text File',
        '.pdf': 'PDF Document',
        '.doc': 'Word Document (Legacy)',
        '.docx': 'Word Document',
        '.epub': 'EPUB Book',
        '.html': 'HTML Document',
        '.htm': 'HTML Document',
        '.xml': 'XML/TEI Document',
        '.tei': 'TEI Document',
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.setAcceptDrops(True)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        group = QGroupBox("Input Files")
        group_layout = QVBoxLayout(group)
        
        # Info label
        info_label = QLabel(
            "Drag and drop files here, or use the buttons below.\n"
            f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS.keys())}"
        )
        info_label.setWordWrap(True)
        group_layout.addWidget(info_label)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.file_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self._show_context_menu)
        group_layout.addWidget(self.file_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        add_files_btn = QPushButton("Add Files...")
        add_files_btn.clicked.connect(self._add_files)
        button_layout.addWidget(add_files_btn)
        
        add_folder_btn = QPushButton("Add Folder...")
        add_folder_btn.clicked.connect(self._add_folder)
        button_layout.addWidget(add_folder_btn)
        
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected)
        button_layout.addWidget(remove_btn)
        
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_all)
        button_layout.addWidget(clear_btn)
        
        group_layout.addLayout(button_layout)
        
        # Stats
        self.stats_label = QLabel("0 files selected")
        group_layout.addWidget(self.stats_label)
        
        layout.addWidget(group)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        files = []
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_file():
                files.append(path)
            elif path.is_dir():
                files.extend(self._scan_directory(path))
        
        self._add_files_to_list(files)
        event.acceptProposedAction()
    
    def _add_files(self):
        extensions = " ".join(f"*{ext}" for ext in self.SUPPORTED_EXTENSIONS.keys())
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Files", "",
            f"Supported Files ({extensions});;All Files (*.*)"
        )
        if files:
            self._add_files_to_list([Path(f) for f in files])
    
    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            files = self._scan_directory(Path(folder))
            self._add_files_to_list(files)
    
    def _scan_directory(self, path: Path) -> List[Path]:
        files = []
        for ext in self.SUPPORTED_EXTENSIONS.keys():
            files.extend(path.rglob(f"*{ext}"))
        return files
    
    def _add_files_to_list(self, files: List[Path]):
        existing = set(self._get_all_paths())
        
        for file_path in files:
            if file_path not in existing and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                item = QListWidgetItem(str(file_path))
                item.setData(Qt.ItemDataRole.UserRole, file_path)
                
                try:
                    size_mb = file_path.stat().st_size / 1024 / 1024
                    format_name = self.SUPPORTED_EXTENSIONS.get(file_path.suffix.lower(), "Unknown")
                    item.setToolTip(f"Format: {format_name}\nSize: {size_mb:.2f} MB")
                except:
                    pass
                
                self.file_list.addItem(item)
        
        self._update_stats()
        self.files_changed.emit(self.get_selected_files())
    
    def _remove_selected(self):
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))
        self._update_stats()
        self.files_changed.emit(self.get_selected_files())
    
    def _clear_all(self):
        self.file_list.clear()
        self._update_stats()
        self.files_changed.emit([])
    
    def _show_context_menu(self, position):
        menu = QMenu()
        remove_action = menu.addAction("Remove")
        remove_action.triggered.connect(self._remove_selected)
        menu.exec(self.file_list.mapToGlobal(position))
    
    def _get_all_paths(self) -> List[Path]:
        paths = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            paths.append(item.data(Qt.ItemDataRole.UserRole))
        return paths
    
    def _update_stats(self):
        count = self.file_list.count()
        try:
            total_size = sum(p.stat().st_size for p in self._get_all_paths())
            self.stats_label.setText(f"{count} files selected ({total_size / 1024 / 1024:.1f} MB total)")
        except:
            self.stats_label.setText(f"{count} files selected")
    
    def get_selected_files(self) -> List[Path]:
        return self._get_all_paths()
