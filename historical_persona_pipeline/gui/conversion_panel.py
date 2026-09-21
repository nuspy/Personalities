"""Tab di conversione: da adapter a modello utilizzabile altrove.

Un adapter LoRA da solo non si usa. Questo pannello lo trasforma in qualcosa
di eseguibile: un modello autonomo, un file GGUF per LM Studio o llama.cpp,
o un pacchetto Ollama col prompt del personaggio gia' incorporato.

La stima di spazio e' mostrata prima di iniziare, perche' la differenza fra
un `q4_k_m` e un `f16` dello stesso modello e' di decine di gigabyte e ci si
accorge del problema solo a disco pieno.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QProgressBar, QPushButton, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget,
)

from ..paths import PROJECTS_DIR
from ..pipeline.conversion.formats import (
    DEFAULT_QUANTIZATION, QUANTIZATION_ORDER, QUANTIZATIONS, ExportFormat,
    estimate_parameters_billions,
)
from ..pipeline.stage4_training.adapter_manager import (
    AdapterInfo, check_compatibility, discover_adapters, read_adapter,
)
from ..pipeline.stage4_training.training_modes import MERGE_STRATEGIES, AdapterSource

logger = logging.getLogger(__name__)


class ConversionPanel(QWidget):
    """Selezione adapter, formato di destinazione e avvio della conversione."""

    request_conversion = pyqtSignal(dict)

    def __init__(self, config: Dict[str, Any], project_dir: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self.config = config
        self.project_dir = project_dir
        self._adapters: List[AdapterInfo] = []
        self._setup_ui()
        self.refresh_adapters()
        self._on_format_changed()

    # ------------------------------------------------------------------ UI

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- adapter -------------------------------------------------------
        source_group = QGroupBox("Adapter da convertire")
        source_layout = QVBoxLayout(source_group)

        self.adapter_table = QTableWidget(0, 4)
        self.adapter_table.setHorizontalHeaderLabels(["Usa", "Adapter", "Dettagli", "Peso"])
        self.adapter_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.adapter_table.verticalHeader().setVisible(False)
        header = self.adapter_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.adapter_table.setMinimumHeight(140)
        source_layout.addWidget(self.adapter_table)

        source_buttons = QHBoxLayout()
        refresh_btn = QPushButton("Aggiorna elenco")
        refresh_btn.clicked.connect(self.refresh_adapters)
        source_buttons.addWidget(refresh_btn)

        browse_btn = QPushButton("Aggiungi cartella...")
        browse_btn.clicked.connect(self._browse_adapter)
        source_buttons.addWidget(browse_btn)
        source_buttons.addStretch()

        source_buttons.addWidget(QLabel("Fusione:"))
        self.merge_strategy = QComboBox()
        self.merge_strategy.addItems(list(MERGE_STRATEGIES))
        self.merge_strategy.setToolTip(
            "Usata solo quando si convertono piu' adapter insieme."
        )
        source_buttons.addWidget(self.merge_strategy)
        source_layout.addLayout(source_buttons)

        layout.addWidget(source_group)

        # --- destinazione --------------------------------------------------
        target_group = QGroupBox("Formato di destinazione")
        target_layout = QVBoxLayout(target_group)

        form = QFormLayout()

        self.format_combo = QComboBox()
        for export_format in ExportFormat:
            self.format_combo.addItem(export_format.label, export_format.value)
        default_format = str(
            self.config.get("conversion", {}).get("default_format", ExportFormat.MERGED_16BIT.value)
        )
        index = self.format_combo.findData(default_format)
        self.format_combo.setCurrentIndex(index if index >= 0 else 0)
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        form.addRow("Formato:", self.format_combo)

        self.quantization_combo = QComboBox()
        for name in QUANTIZATION_ORDER:
            self.quantization_combo.addItem(name, name)
        default_quant = str(
            self.config.get("conversion", {}).get("default_quantization", DEFAULT_QUANTIZATION)
        )
        quant_index = self.quantization_combo.findData(default_quant)
        self.quantization_combo.setCurrentIndex(quant_index if quant_index >= 0 else 0)
        self.quantization_combo.currentIndexChanged.connect(self._update_estimate)
        form.addRow("Quantizzazione:", self.quantization_combo)

        self.base_model = QLineEdit(
            str(self.config.get("training", {}).get("base_model", ""))
        )
        self.base_model.setToolTip(
            "Lasciare vuoto per dedurlo dall'adapter selezionato."
        )
        self.base_model.textChanged.connect(self._update_estimate)
        form.addRow("Modello base:", self.base_model)

        self.model_name = QLineEdit()
        self.model_name.setPlaceholderText("dedotto dal nome del personaggio")
        form.addRow("Nome del modello:", self.model_name)

        output_row = QHBoxLayout()
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("<progetto>/exports/…")
        output_row.addWidget(self.output_dir)
        output_btn = QPushButton("Sfoglia...")
        output_btn.clicked.connect(self._browse_output)
        output_row.addWidget(output_btn)
        form.addRow("Destinazione:", output_row)

        target_layout.addLayout(form)

        self.include_prompt = QCheckBox(
            "Incorpora il prompt di sistema del personaggio (solo Ollama)"
        )
        self.include_prompt.setChecked(True)
        target_layout.addWidget(self.include_prompt)

        self.format_description = QLabel()
        self.format_description.setWordWrap(True)
        self.format_description.setStyleSheet(
            "color: #495057; background: #f8f9fa; padding: 8px; border-radius: 4px;"
        )
        target_layout.addWidget(self.format_description)

        self.estimate_label = QLabel()
        self.estimate_label.setWordWrap(True)
        self.estimate_label.setStyleSheet("color: #6c757d; font-size: 11px;")
        target_layout.addWidget(self.estimate_label)

        layout.addWidget(target_group)

        # --- azione ---------------------------------------------------------
        self.convert_btn = QPushButton("Converti")
        self.convert_btn.setStyleSheet("padding: 10px; font-weight: bold; font-size: 14px;")
        self.convert_btn.clicked.connect(self._start_conversion)
        layout.addWidget(self.convert_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(160)
        self.log_view.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        layout.addWidget(self.log_view)

    # --------------------------------------------------------------- eventi

    def _on_format_changed(self) -> None:
        export_format = self.current_format()
        self.format_description.setText(export_format.description)

        # La quantizzazione riguarda solo i formati GGUF.
        needs_gguf = export_format.needs_gguf_toolchain
        self.quantization_combo.setEnabled(needs_gguf)
        self.include_prompt.setEnabled(export_format is ExportFormat.OLLAMA)

        self._update_estimate()

    def _update_estimate(self, *_args) -> None:
        export_format = self.current_format()

        if export_format is ExportFormat.ADAPTER_ONLY:
            self.estimate_label.setText(
                "L'adapter pesa pochi MB, ma richiede il modello base per essere usato."
            )
            return

        base = self.base_model.text().strip() or self._infer_base_model()
        if not base:
            self.estimate_label.setText("")
            return

        billions = estimate_parameters_billions(base)

        if export_format.needs_gguf_toolchain:
            quantization = QUANTIZATIONS[self.quantization_combo.currentData()]
            final_gb = quantization.estimated_size_gb(billions)
            # Durante la conversione il fp16 intermedio convive col risultato.
            peak_gb = billions * 2 + final_gb
            self.estimate_label.setText(
                f"Modello stimato {billions:g}B → file finale ~{final_gb:.1f} GB "
                f"({quantization.description}). "
                f"Durante la conversione servono ~{peak_gb:.0f} GB liberi."
            )
        elif export_format is ExportFormat.MERGED_4BIT:
            self.estimate_label.setText(
                f"Modello stimato {billions:g}B → ~{billions * 0.6:.1f} GB su disco."
            )
        else:
            self.estimate_label.setText(
                f"Modello stimato {billions:g}B → ~{billions * 2:.0f} GB su disco."
            )

    # ------------------------------------------------------------- adapter

    def refresh_adapters(self) -> None:
        previously = {str(s.path) for s in self.selected_adapters()}

        roots = [PROJECTS_DIR]
        if self.project_dir:
            roots.append(Path(self.project_dir))

        discovered = discover_adapters(roots)
        known = {str(a.path) for a in discovered}
        for adapter in self._adapters:
            if str(adapter.path) not in known and adapter.path.exists():
                discovered.append(adapter)

        self._adapters = discovered
        self._populate_table(previously)
        self._update_estimate()

    def _populate_table(self, selected: set) -> None:
        self.adapter_table.setRowCount(len(self._adapters))

        for row, adapter in enumerate(self._adapters):
            checkbox = QCheckBox()
            checkbox.setChecked(str(adapter.path) in selected)
            checkbox.stateChanged.connect(self._update_estimate)
            container = QWidget()
            container_layout = QHBoxLayout(container)
            container_layout.addWidget(checkbox)
            container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            container_layout.setContentsMargins(0, 0, 0, 0)
            self.adapter_table.setCellWidget(row, 0, container)

            name_item = QTableWidgetItem(adapter.name)
            name_item.setToolTip(str(adapter.path))
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.adapter_table.setItem(row, 1, name_item)

            detail_item = QTableWidgetItem(adapter.describe())
            detail_item.setFlags(detail_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.adapter_table.setItem(row, 2, detail_item)

            weight_spin = QDoubleSpinBox()
            weight_spin.setRange(0.05, 2.0)
            weight_spin.setSingleStep(0.05)
            weight_spin.setValue(1.0)
            self.adapter_table.setCellWidget(row, 3, weight_spin)

    def _browse_adapter(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Seleziona la cartella dell'adapter")
        if not directory:
            return

        info = read_adapter(Path(directory))
        if info is None:
            QMessageBox.warning(
                self, "Non e' un adapter",
                f"La cartella non contiene un adapter LoRA:\n{directory}",
            )
            return

        if not any(str(a.path) == str(info.path) for a in self._adapters):
            self._adapters.append(info)

        selected = {str(s.path) for s in self.selected_adapters()} | {str(info.path)}
        self._populate_table(selected)
        self._update_estimate()

    def _browse_output(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Cartella di destinazione")
        if directory:
            self.output_dir.setText(directory)

    def selected_adapters(self) -> List[AdapterSource]:
        sources: List[AdapterSource] = []
        for row in range(self.adapter_table.rowCount()):
            container = self.adapter_table.cellWidget(row, 0)
            if container is None:
                continue
            checkbox = container.findChild(QCheckBox)
            if checkbox is None or not checkbox.isChecked():
                continue

            adapter = self._adapters[row]
            weight_widget = self.adapter_table.cellWidget(row, 3)
            sources.append(AdapterSource(
                path=str(adapter.path),
                weight=weight_widget.value() if weight_widget else 1.0,
                name=adapter.name,
            ))
        return sources

    # ----------------------------------------------------------- conversione

    def _start_conversion(self) -> None:
        selected = self.selected_adapters()
        if not selected:
            QMessageBox.warning(
                self, "Nessun adapter",
                "Selezionare almeno un adapter da convertire.",
            )
            return

        infos = [a for a in self._adapters if any(str(a.path) == s.path for s in selected)]
        errors, warnings = check_compatibility(infos, self.base_model.text().strip())

        if errors:
            QMessageBox.critical(self, "Adapter incompatibili", "\n".join(errors))
            return

        if warnings:
            answer = QMessageBox.question(
                self, "Verificare prima di procedere",
                "\n\n".join(warnings) + "\n\nProcedere comunque?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.No:
                return

        export_format = self.current_format()
        request: Dict[str, Any] = {
            "adapters": [s.path for s in selected],
            "weights": [s.weight for s in selected],
            "format": export_format.value,
            "merge_strategy": self.merge_strategy.currentText(),
            "base_model": self.base_model.text().strip(),
            "model_name": self.model_name.text().strip(),
        }

        if export_format.needs_gguf_toolchain:
            request["quantization"] = self.quantization_combo.currentData()

        if self.output_dir.text().strip():
            request["output_dir"] = self.output_dir.text().strip()

        if export_format is ExportFormat.OLLAMA and not self.include_prompt.isChecked():
            request["system_prompt"] = ""

        self.set_running(True)
        self.log(f"Avvio conversione: {export_format.label}")
        self.request_conversion.emit(request)

    # -------------------------------------------------------------- stato

    def current_format(self) -> ExportFormat:
        return ExportFormat(self.format_combo.currentData())

    def _infer_base_model(self) -> str:
        for source in self.selected_adapters():
            info = read_adapter(Path(source.path))
            if info and info.base_model:
                return info.base_model
        return ""

    def set_project_dir(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir)
        self.refresh_adapters()

    def set_running(self, running: bool) -> None:
        self.convert_btn.setEnabled(not running)
        self.progress_bar.setVisible(running)
        if not running:
            self.progress_bar.setValue(0)

    def set_progress(self, value: int, message: str = "") -> None:
        self.progress_bar.setValue(value)
        if message:
            self.log(f"[{value:3d}%] {message}")

    def log(self, message: str) -> None:
        self.log_view.append(message)

    def on_finished(self, result: Dict[str, Any]) -> None:
        self.set_running(False)
        output = result.get("output_path", "")
        self.log(f"Completato: {output}")

        text = f"Conversione completata.\n\nDestinazione:\n{output}"
        if result.get("ollama_command"):
            # Il comando serve all'utente subito dopo: mostrarlo qui evita di
            # doverlo ricostruire a mano dal percorso.
            text += (
                "\n\nPer importarlo in Ollama:\n"
                f"{result['ollama_command']}"
            )
            self.log(f"Comando Ollama: {result['ollama_command']}")

        QMessageBox.information(self, "Conversione completata", text)

    def on_error(self, message: str) -> None:
        self.set_running(False)
        self.log(f"ERRORE: {message}")
