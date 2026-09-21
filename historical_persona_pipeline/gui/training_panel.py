"""Pannello di scelta della modalita' di addestramento.

Le quattro modalita' differiscono per *da dove partono i pesi*, e la
differenza non e' evidente dai soli nomi: il pannello mostra la descrizione
della modalita' selezionata e abilita la scelta degli adapter solo dove ha
senso, invece di lasciare campi attivi che verrebbero ignorati.

Gli adapter si scoprono da soli scandendo le cartelle di progetto: digitare
a mano il percorso di una cartella di pesi e' il genere di passaggio in cui
si sbaglia senza accorgersene.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from ..paths import PROJECTS_DIR
from ..pipeline.stage4_training.adapter_manager import (
    AdapterInfo, check_compatibility, discover_adapters, read_adapter,
)
from ..pipeline.stage4_training.training_modes import (
    MERGE_STRATEGIES, AdapterSource, TrainingMode, TrainingPlan,
)

logger = logging.getLogger(__name__)


class TrainingPanel(QWidget):
    """Scelta della modalita' e degli adapter di partenza."""

    plan_changed = pyqtSignal()

    def __init__(self, config: Dict[str, Any], project_dir: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self.config = config
        self.project_dir = project_dir
        self._adapters: List[AdapterInfo] = []
        self._setup_ui()
        self.refresh_adapters()
        self._on_mode_changed()

    # ------------------------------------------------------------------ UI

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        mode_group = QGroupBox("Modalita' di addestramento")
        mode_layout = QVBoxLayout(mode_group)

        form = QFormLayout()
        self.mode_combo = QComboBox()
        for mode in TrainingMode:
            self.mode_combo.addItem(mode.label, mode.value)
        current = str(self.config.get("training", {}).get("mode", TrainingMode.LORA_NEW.value))
        index = self.mode_combo.findData(current)
        self.mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Modalita':", self.mode_combo)

        self.base_model = QLineEdit(
            str(self.config.get("training", {}).get("base_model", ""))
        )
        form.addRow("Modello base:", self.base_model)

        self.output_name = QLineEdit(
            str(self.config.get("training", {}).get("output_name", "lora_adapters"))
        )
        self.output_name.setToolTip("Nome della cartella prodotta sotto <progetto>/output/")
        form.addRow("Nome risultato:", self.output_name)

        mode_layout.addLayout(form)

        self.mode_description = QLabel()
        self.mode_description.setWordWrap(True)
        self.mode_description.setStyleSheet(
            "color: #495057; background: #f8f9fa; padding: 8px; border-radius: 4px;"
        )
        mode_layout.addWidget(self.mode_description)

        layout.addWidget(mode_group)

        # --- adapter di partenza ------------------------------------------
        self.adapter_group = QGroupBox("Adapter di partenza")
        adapter_layout = QVBoxLayout(self.adapter_group)

        self.adapter_table = QTableWidget(0, 4)
        self.adapter_table.setHorizontalHeaderLabels(["Usa", "Adapter", "Dettagli", "Peso"])
        self.adapter_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.adapter_table.verticalHeader().setVisible(False)
        header = self.adapter_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.adapter_table.setMinimumHeight(150)
        adapter_layout.addWidget(self.adapter_table)

        buttons = QHBoxLayout()
        refresh_btn = QPushButton("Aggiorna elenco")
        refresh_btn.clicked.connect(self.refresh_adapters)
        buttons.addWidget(refresh_btn)

        browse_btn = QPushButton("Aggiungi cartella...")
        browse_btn.clicked.connect(self._browse_adapter)
        buttons.addWidget(browse_btn)
        buttons.addStretch()

        merge_label = QLabel("Fusione:")
        buttons.addWidget(merge_label)
        self.merge_strategy = QComboBox()
        for name in MERGE_STRATEGIES:
            self.merge_strategy.addItem(name)
        self.merge_strategy.setCurrentText(
            str(self.config.get("training", {}).get("merge_strategy", "linear"))
        )
        self.merge_strategy.currentTextChanged.connect(self._on_strategy_changed)
        buttons.addWidget(self.merge_strategy)

        adapter_layout.addLayout(buttons)

        self.strategy_hint = QLabel()
        self.strategy_hint.setWordWrap(True)
        self.strategy_hint.setStyleSheet("color: #6c757d; font-size: 11px;")
        adapter_layout.addWidget(self.strategy_hint)

        self.compatibility_label = QLabel()
        self.compatibility_label.setWordWrap(True)
        adapter_layout.addWidget(self.compatibility_label)

        layout.addWidget(self.adapter_group)
        layout.addStretch()

        self._on_strategy_changed(self.merge_strategy.currentText())

    # --------------------------------------------------------------- eventi

    def _on_mode_changed(self) -> None:
        mode = self.current_mode()
        self.mode_description.setText(mode.description)

        # Gli adapter servono solo a continue e stack: altrove i campi
        # resterebbero attivi senza avere effetto.
        needs_adapters = mode.requires_adapters
        self.adapter_group.setEnabled(needs_adapters)
        self.adapter_group.setTitle(
            "Adapter di partenza" if needs_adapters
            else "Adapter di partenza (non usati in questa modalita')"
        )

        # Si continua un adapter alla volta; impilarne serve a combinarne piu'.
        self.merge_strategy.setEnabled(mode is TrainingMode.LORA_STACK)

        self._validate_selection()
        self.plan_changed.emit()

    def _on_strategy_changed(self, name: str) -> None:
        self.strategy_hint.setText(MERGE_STRATEGIES.get(name, ""))
        self._validate_selection()

    # ------------------------------------------------------------- adapter

    def refresh_adapters(self) -> None:
        """Rilegge gli adapter disponibili conservando le scelte fatte."""
        previously_selected = {str(s.path) for s in self.selected_adapters()}
        previous_weights = {str(s.path): s.weight for s in self.selected_adapters()}

        roots = [PROJECTS_DIR]
        if self.project_dir:
            roots.append(Path(self.project_dir))

        discovered = discover_adapters(roots)

        # Gli adapter aggiunti a mano restano nell'elenco anche se fuori
        # dalle cartelle di progetto.
        known_paths = {str(a.path) for a in discovered}
        for adapter in self._adapters:
            if str(adapter.path) not in known_paths and adapter.path.exists():
                discovered.append(adapter)

        self._adapters = discovered
        self._populate_table(previously_selected, previous_weights)
        self._validate_selection()

    def _populate_table(self, selected: set, weights: Dict[str, float]) -> None:
        self.adapter_table.setRowCount(len(self._adapters))

        for row, adapter in enumerate(self._adapters):
            checkbox = QCheckBox()
            checkbox.setChecked(str(adapter.path) in selected)
            checkbox.stateChanged.connect(self._validate_selection)
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
            weight_spin.setValue(weights.get(str(adapter.path), 1.0))
            weight_spin.setToolTip(
                "Peso nella fusione: valori piu' alti danno piu' influenza a "
                "questo adapter."
            )
            self.adapter_table.setCellWidget(row, 3, weight_spin)

        if not self._adapters:
            self.compatibility_label.setText(
                "Nessun adapter trovato. Addestra prima una LoRA, oppure "
                "aggiungi una cartella con 'Aggiungi cartella...'."
            )
            self.compatibility_label.setStyleSheet("color: #6c757d;")

    def _browse_adapter(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Seleziona la cartella dell'adapter")
        if not directory:
            return

        info = read_adapter(Path(directory))
        if info is None:
            QMessageBox.warning(
                self, "Non e' un adapter",
                f"La cartella selezionata non contiene un adapter LoRA "
                f"(manca adapter_config.json):\n{directory}",
            )
            return

        if any(str(a.path) == str(info.path) for a in self._adapters):
            QMessageBox.information(self, "Gia' presente", "Questo adapter e' gia' nell'elenco.")
            return

        self._adapters.append(info)
        selected = {str(s.path) for s in self.selected_adapters()} | {str(info.path)}
        self._populate_table(selected, {})
        self._validate_selection()

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
            weight = weight_widget.value() if weight_widget else 1.0
            sources.append(AdapterSource(
                path=str(adapter.path), weight=weight, name=adapter.name
            ))
        return sources

    # ---------------------------------------------------------- validazione

    def _validate_selection(self, *_args) -> None:
        """Mostra subito i problemi, invece di scoprirli a meta' addestramento."""
        mode = self.current_mode()
        selected = self.selected_adapters()

        if not mode.requires_adapters:
            self.compatibility_label.setText("")
            return

        if not selected:
            self._show_status(
                f"Selezionare almeno un adapter: '{mode.label}' non puo' partire senza.",
                "warning",
            )
            return

        if mode is TrainingMode.LORA_CONTINUE and len(selected) > 1:
            self._show_status(
                "Si puo' continuare un solo adapter per volta. Per combinarne piu' "
                "di uno usare 'Nuova LoRA sopra LoRA esistenti'.",
                "error",
            )
            return

        infos = [a for a in self._adapters if any(str(a.path) == s.path for s in selected)]
        errors, warnings = check_compatibility(infos, self.base_model.text().strip())

        if errors:
            self._show_status(" ".join(errors), "error")
        elif warnings:
            self._show_status(" ".join(warnings), "warning")
        else:
            self._show_status(
                f"{len(selected)} adapter selezionat{'o' if len(selected) == 1 else 'i'}: "
                "compatibili.",
                "ok",
            )

    def _show_status(self, message: str, level: str) -> None:
        colors = {"ok": "#198754", "warning": "#fd7e14", "error": "#dc3545"}
        self.compatibility_label.setText(message)
        self.compatibility_label.setStyleSheet(f"color: {colors.get(level, '#6c757d')};")

    # ------------------------------------------------------------- lettura

    def current_mode(self) -> TrainingMode:
        return TrainingMode(self.mode_combo.currentData())

    def build_plan(self) -> TrainingPlan:
        mode = self.current_mode()
        return TrainingPlan(
            mode=mode,
            base_model=self.base_model.text().strip(),
            adapters=self.selected_adapters() if mode.requires_adapters else [],
            merge_strategy=self.merge_strategy.currentText(),
            output_name=self.output_name.text().strip() or "lora_adapters",
        )

    def set_project_dir(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir)
        self.refresh_adapters()
