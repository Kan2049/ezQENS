"""Manual Fit-to-all-Q controls over the validated S5 workflow seam."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import Event

from PySide6.QtCore import (
    QObject,
    QSettings,
    QSignalBlocker,
    Signal,
    Slot,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ezqens.fitting import (
    BACKGROUND_COMPONENT,
    ELASTIC_COMPONENT,
    ComponentIdentity,
    FitResult,
    ParameterMetadata,
    ParameterReference,
)
from ezqens.gui.manual_fit import manual_tie_color
from ezqens.workflow import (
    ActiveTransferableConstraints,
    CenterTieIdentity,
    ManualFitDraft,
    ManualModelState,
    MultiQConstraintTransferPolicy,
    WorkflowProject,
    active_transferable_constraints,
    fit_all_q_from_current_with_adoption,
)


def _display_float(value: float | None) -> str:
    if value is None:
        return "∞"
    return f"{float(value):.12g}"


def _bounds_text(lower: float | None, upper: float | None) -> str:
    lower_text = "−∞" if lower is None else _display_float(lower)
    upper_text = "∞" if upper is None else _display_float(upper)
    return f"[{lower_text}, {upper_text}]"


def _component_slot(model: ManualModelState, identity: ComponentIdentity) -> str:
    if identity == ELASTIC_COMPONENT:
        return "Elastic"
    if identity == BACKGROUND_COMPONENT:
        return "Background"
    return next(
        f"L{index}"
        for index, component in enumerate(model.lorentzians, start=1)
        if component.identity == identity
    )


def semantic_parameter_slot(
    model: ManualModelState,
    reference: ParameterReference,
) -> str:
    """Return a stable GUI preference slot without retaining core identities."""

    return f"{_component_slot(model, reference.component)}.{reference.family.value}"


def semantic_tie_slot(
    model: ManualModelState,
    members: tuple[ParameterReference, ...],
) -> str:
    """Describe a relationship only by its visible semantic member slots."""

    return "+".join(sorted(semantic_parameter_slot(model, item) for item in members))


class TransferDefaultsStore:
    """Small application-owned persistent selection-preference seam."""

    def __init__(self, settings: QSettings | None = None) -> None:
        self._settings = settings or QSettings("ezQENS", "ezQENS")

    def selected(self, category: str, slot: str) -> bool:
        value = self._settings.value(f"manual_fit/transfer/{category}/{slot}")
        if value is None:
            return True
        if isinstance(value, bool):
            return value
        return str(value).casefold() in {"1", "true", "yes"}

    def save(self, selections: Mapping[tuple[str, str], bool]) -> None:
        for (category, slot), selected in selections.items():
            self._settings.setValue(
                f"manual_fit/transfer/{category}/{slot}",
                selected,
            )
        self._settings.sync()


@dataclass(frozen=True)
class TransferRowWidgets:
    """Focused-test access to one rendered parameter row."""

    parameter: QLabel
    bounds: QCheckBox | None
    bounds_value: QLabel
    fixed: QCheckBox | None
    fixed_value: QLabel
    tie: QCheckBox | None


class FitAllQTransferDialog(QDialog):
    """Select only active optional constraints for one Manual Q continuation."""

    def __init__(
        self,
        model: ManualModelState,
        parent: QWidget | None = None,
        *,
        metadata: tuple[ParameterMetadata, ...],
        defaults: TransferDefaultsStore | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("fitToAllQDialog")
        self.setWindowTitle("Fit to all Q")
        self.setModal(True)
        self.model = model
        self.metadata = {item.reference: item for item in metadata}
        self.active: ActiveTransferableConstraints = active_transferable_constraints(
            model
        )
        self.defaults = defaults or TransferDefaultsStore()
        self.rows: dict[ParameterReference, TransferRowWidgets] = {}
        self.tie_checkboxes: dict[CenterTieIdentity, tuple[QCheckBox, ...]] = {}
        self._tie_members = {
            constraint.identity: constraint.members
            for constraint in self.active.center_ties
        }
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)
        heading = QLabel("Transfer active constraints")
        heading.setObjectName("fitAllQTransferTitle")
        explanation = QLabel(
            "Model composition and fitted values as sequential initial guesses "
            "are always transferred. Select the active optional constraints below."
        )
        explanation.setWordWrap(True)
        explanation.setProperty("secondary", True)
        layout.addWidget(heading)
        layout.addWidget(explanation)

        grid_widget = QWidget(self)
        grid = QGridLayout(grid_widget)
        grid.setContentsMargins(0, 4, 0, 4)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        for column, text in enumerate(("Parameter", "Bounds", "Fixed", "Tie")):
            label = QLabel(text)
            label.setProperty("secondary", True)
            grid.addWidget(label, 0, column)

        bounds = {item.reference: item for item in self.active.bounds}
        fixed = {item.reference: item for item in self.active.fixed}
        tie_for_member = {
            member: tie for tie in self.active.center_ties for member in tie.members
        }
        pending_tie_checks: dict[CenterTieIdentity, list[QCheckBox]] = {}
        for row_index, reference in enumerate(
            self.model.parameter_references(), start=1
        ):
            parameter_metadata = self.metadata[reference]
            parameter = QLabel(
                f"{_component_slot(self.model, reference.component)} "
                f"{parameter_metadata.display_label}"
            )
            grid.addWidget(parameter, row_index, 0)

            bounds_constraint = bounds.get(reference)
            bounds_checkbox: QCheckBox | None = None
            bounds_value = QLabel("—")
            if bounds_constraint is not None:
                bounds_checkbox = QCheckBox()
                slot = semantic_parameter_slot(self.model, reference)
                bounds_checkbox.setChecked(self.defaults.selected("bounds", slot))
                bounds_value.setText(
                    self._value_with_unit(
                        _bounds_text(
                            bounds_constraint.lower,
                            bounds_constraint.upper,
                        ),
                        parameter_metadata.unit,
                    )
                )
                bounds_value.setToolTip(parameter_metadata.unit)
                bounds_cell = self._check_value_cell(bounds_checkbox, bounds_value)
                grid.addWidget(bounds_cell, row_index, 1)
            else:
                grid.addWidget(bounds_value, row_index, 1)

            fixed_constraint = fixed.get(reference)
            fixed_checkbox: QCheckBox | None = None
            fixed_value = QLabel("—")
            if fixed_constraint is not None:
                fixed_checkbox = QCheckBox()
                slot = semantic_parameter_slot(self.model, reference)
                fixed_checkbox.setChecked(self.defaults.selected("fixed", slot))
                fixed_value.setText(
                    self._value_with_unit(
                        _display_float(fixed_constraint.value),
                        parameter_metadata.unit,
                    )
                )
                fixed_value.setToolTip(parameter_metadata.unit)
                fixed_cell = self._check_value_cell(fixed_checkbox, fixed_value)
                grid.addWidget(fixed_cell, row_index, 2)
            else:
                grid.addWidget(fixed_value, row_index, 2)

            tie_constraint = tie_for_member.get(reference)
            tie_checkbox: QCheckBox | None = None
            if tie_constraint is None:
                grid.addWidget(QLabel("—"), row_index, 3)
            else:
                tie_checkbox = QCheckBox()
                slot = semantic_tie_slot(self.model, tie_constraint.members)
                tie_checkbox.setChecked(self.defaults.selected("tie", slot))
                if tie_constraint.identity.group_id is not None:
                    tie_checkbox.setProperty(
                        "tieColor",
                        manual_tie_color(
                            self.model,
                            tie_constraint.identity.group_id,
                        ),
                    )
                tie_checkbox.setToolTip(
                    "Complete center-sharing relationship; members toggle together"
                )
                pending_tie_checks.setdefault(tie_constraint.identity, []).append(
                    tie_checkbox
                )
                grid.addWidget(tie_checkbox, row_index, 3)
            self.rows[reference] = TransferRowWidgets(
                parameter,
                bounds_checkbox,
                bounds_value,
                fixed_checkbox,
                fixed_value,
                tie_checkbox,
            )
        self.tie_checkboxes = {
            identity: tuple(checks) for identity, checks in pending_tie_checks.items()
        }
        for identity, checks in self.tie_checkboxes.items():
            for checkbox in checks:
                checkbox.toggled.connect(
                    lambda checked, value=identity: self._set_tie_checked(
                        value, checked
                    )
                )
        layout.addWidget(grid_widget)

        bulk = QHBoxLayout()
        self.transfer_all_button = QPushButton("Transfer all active constraints")
        self.transfer_none_button = QPushButton("Transfer none")
        self.save_defaults_button = QPushButton("Save transfer defaults")
        self.transfer_all_button.clicked.connect(lambda: self._set_all(True))
        self.transfer_none_button.clicked.connect(lambda: self._set_all(False))
        self.save_defaults_button.clicked.connect(self.save_defaults)
        bulk.addWidget(self.transfer_all_button)
        bulk.addWidget(self.transfer_none_button)
        bulk.addStretch(1)
        bulk.addWidget(self.save_defaults_button)
        layout.addLayout(bulk)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self.execute_button = buttons.addButton(
            "Fit to all Q",
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _check_value_cell(checkbox: QCheckBox, value: QLabel) -> QWidget:
        cell = QWidget()
        layout = QHBoxLayout(cell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(checkbox)
        layout.addWidget(value)
        layout.addStretch(1)
        return cell

    @staticmethod
    def _value_with_unit(value: str, unit: str) -> str:
        return value if not unit else f"{value} {unit}"

    def _set_tie_checked(self, identity: CenterTieIdentity, checked: bool) -> None:
        for checkbox in self.tie_checkboxes[identity]:
            blocker = QSignalBlocker(checkbox)
            checkbox.setChecked(checked)
            del blocker

    def _set_all(self, checked: bool) -> None:
        for row in self.rows.values():
            for checkbox in (row.bounds, row.fixed):
                if checkbox is not None:
                    checkbox.setChecked(checked)
        for identity in self.tie_checkboxes:
            self._set_tie_checked(identity, checked)

    def _preference_selections(self) -> dict[tuple[str, str], bool]:
        selections: dict[tuple[str, str], bool] = {}
        for reference, row in self.rows.items():
            slot = semantic_parameter_slot(self.model, reference)
            if row.bounds is not None:
                selections[("bounds", slot)] = row.bounds.isChecked()
            if row.fixed is not None:
                selections[("fixed", slot)] = row.fixed.isChecked()
        for identity, checks in self.tie_checkboxes.items():
            selections[
                ("tie", semantic_tie_slot(self.model, self._tie_members[identity]))
            ] = checks[0].isChecked()
        return selections

    def save_defaults(self) -> None:
        self.defaults.save(self._preference_selections())

    def transfer_policy(self) -> MultiQConstraintTransferPolicy:
        bounds = frozenset(
            reference
            for reference, row in self.rows.items()
            if row.bounds is not None and row.bounds.isChecked()
        )
        fixed = frozenset(
            reference
            for reference, row in self.rows.items()
            if row.fixed is not None and row.fixed.isChecked()
        )
        ties = frozenset(
            identity
            for identity, checks in self.tie_checkboxes.items()
            if checks[0].isChecked()
        )
        return MultiQConstraintTransferPolicy(
            self.model,
            bounds=bounds,
            fixed=fixed,
            center_ties=ties,
        )


class FitAllQWorker(QObject):
    """Run one validated Manual continuation outside the GUI thread."""

    group_started = Signal(int, int)
    progress = Signal(int, object)
    completed = Signal(int, object)
    failed = Signal(int, object)

    def __init__(
        self,
        project: WorkflowProject,
        draft: ManualFitDraft,
        anchor_group_index: int,
        current_fit: FitResult,
        transfer_policy: MultiQConstraintTransferPolicy,
        cancel_event: Event,
        generation: int,
    ) -> None:
        super().__init__()
        self._project = project
        self._draft = draft
        self._anchor_group_index = anchor_group_index
        self._current_fit = current_fit
        self._transfer_policy = transfer_policy
        self._cancel_event = cancel_event
        self._generation = generation

    @Slot()
    def run(self) -> None:
        try:
            result = fit_all_q_from_current_with_adoption(
                self._project,
                self._draft,
                anchor_group_index=self._anchor_group_index,
                current_fit=self._current_fit,
                transfer_policy=self._transfer_policy,
                cancel_requested=self._cancel_event.is_set,
                group_started_callback=lambda group: self.group_started.emit(
                    self._generation, group
                ),
                progress_callback=lambda outcome: self.progress.emit(
                    self._generation, outcome
                ),
            )
        except Exception as error:  # noqa: BLE001 - transported to the GUI thread
            self.failed.emit(self._generation, error)
        else:
            self.completed.emit(self._generation, result)
