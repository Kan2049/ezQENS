"""Minimal source-unit metadata editor without numerical conversion."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from ezqens.domain import ReducedDataset


class SourceUnitsDialog(QDialog):
    """Edit source labels only; uncertainty intentionally follows intensity."""

    units_applied = Signal(str, str)

    def __init__(self, dataset: ReducedDataset, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sourceUnitsDialog")
        self.setWindowTitle("Source Units")
        self._apply_handler: Callable[[str, str], bool] | None = None
        spectrum = dataset.spectra[0]
        self.energy_unit_combo = QComboBox()
        self.energy_unit_combo.setObjectName("energyUnitCombo")
        self.energy_unit_combo.addItems(["Not set", "µeV", "meV", "eV"])
        current_energy = spectrum.energy_unit.strip()
        selected = (
            current_energy if current_energy in {"µeV", "meV", "eV"} else "Not set"
        )
        self.energy_unit_combo.setCurrentText(selected)
        self.intensity_unit_edit = QLineEdit(spectrum.intensity_unit)
        self.intensity_unit_edit.setObjectName("intensityUnitEdit")
        note = QLabel(
            "This assigns source metadata only. Intensity uncertainty uses the "
            "same unit; values are not converted.",
        )
        note.setWordWrap(True)
        note.setProperty("secondary", True)
        form = QFormLayout()
        form.addRow("Energy", self.energy_unit_combo)
        form.addRow("Intensity / uncertainty", self.intensity_unit_edit)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply,
        )
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            self.apply
        )
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)

    def set_apply_handler(self, handler: Callable[[str, str], bool]) -> None:
        """Set the application owner of an accepted metadata change."""

        self._apply_handler = handler

    def apply(self) -> bool:
        """Emit nonempty source metadata and leave invalid input in place."""

        energy = self.energy_unit_combo.currentText()
        if energy == "Not set":
            energy = "unknown"
        intensity = self.intensity_unit_edit.text().strip()
        if not energy or not intensity:
            return False
        if self._apply_handler is not None:
            if not self._apply_handler(energy, intensity):
                return False
        else:
            self.units_applied.emit(energy, intensity)
        self.accept()
        return True
