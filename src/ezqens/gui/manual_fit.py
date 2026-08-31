"""Thin Inspector controls for the group-local Single-Q Manual Fit task."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from PySide6.QtCore import QLocale, QPoint, Qt, Signal
from PySide6.QtGui import QDoubleValidator, QFocusEvent, QValidator
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ezqens.fitting import (
    BACKGROUND_COMPONENT,
    ELASTIC_COMPONENT,
    BackgroundModel,
    ComponentIdentity,
    FitResult,
    ParameterDimension,
    ParameterFamily,
    ParameterMetadata,
    ParameterReference,
)
from ezqens.gui.icons import IconName, apply_disclosure_icon, load_icon
from ezqens.workflow import (
    ManualComponentKind,
    ManualModelState,
    ManualParameterEdit,
    ManualWorkflowReadiness,
    WorkflowDiagnostic,
    WorkflowDiagnosticCode,
)


@dataclass(frozen=True)
class _ParameterControls:
    """Widgets for one typed parameter entrance."""

    current: CompactNumberEdit
    lower: CompactNumberEdit
    upper: CompactNumberEdit
    fixed: QToolButton
    bounds_enabled: QToolButton
    chain: QToolButton | None


@dataclass(frozen=True)
class _ResultParameterWidgets:
    """Read-only widgets for one identity-mapped result parameter row."""

    name: QLabel
    value: QLabel
    uncertainty: QLabel
    details: QLabel


class ManualFitLifecycle(StrEnum):
    """Small GUI-only lifecycle for the active single-Q Manual task."""

    READY = "ready"
    FITTING = "fitting"
    CURRENT = "current"
    NEEDS_FIT = "needs_fit"


_FALLBACK_METADATA = {
    ParameterFamily.AREA: ("Area", ParameterDimension.INTEGRATED_INTENSITY),
    ParameterFamily.CENTER: ("Center", ParameterDimension.ENERGY),
    ParameterFamily.FWHM: ("FWHM", ParameterDimension.ENERGY),
    ParameterFamily.OFFSET: ("Offset", ParameterDimension.INTENSITY),
    ParameterFamily.SLOPE: ("Slope", ParameterDimension.INTENSITY_PER_ENERGY),
}


class _OptionalDoubleValidator(QDoubleValidator):
    """Accept an empty optional bound while retaining numeric validation."""

    def validate(
        self,
        input_text: str,
        position: int,
    ) -> tuple[QValidator.State, str, int]:
        if not input_text.strip():
            return QValidator.State.Acceptable, input_text, position
        return cast(
            tuple[QValidator.State, str, int],
            super().validate(input_text, position),
        )


class CompactNumberEdit(QLineEdit):
    """Compact unfocused float text with lossless untouched-value round trips."""

    def __init__(self, value: float | None, *, optional: bool = False) -> None:
        super().__init__()
        self._stored_value = None if value is None else float(value)
        self._dirty = False
        validator = (
            _OptionalDoubleValidator(self) if optional else QDoubleValidator(self)
        )
        validator.setLocale(QLocale.c())
        validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
        self.setValidator(validator)
        self.setText(_compact_float(self._stored_value))
        self.setToolTip(_round_trip_float(self._stored_value))
        self.textEdited.connect(self._mark_dirty)

    @property
    def stored_value(self) -> float | None:
        """Return the exact immutable value represented before any user edit."""

        return self._stored_value

    @property
    def user_edited(self) -> bool:
        """Return whether real user text editing occurred since the last completion."""

        return self._dirty

    def finish_user_edit(self) -> None:
        """Consume the local dirty state after one commit attempt."""

        self._dirty = False

    def precise_value(self, *, required: bool = False) -> float | None:
        """Return edited text, or the untouched exact value without reparsing."""

        if not self._dirty:
            if required and self._stored_value is None:
                raise ValueError("Current Value is required")
            return self._stored_value
        text = self.text().strip()
        if not text:
            if required:
                raise ValueError("Current Value is required")
            return None
        return float(text)

    def focusInEvent(self, event: QFocusEvent) -> None:  # noqa: N802 - Qt virtual.
        if not self._dirty:
            self.setText(_focused_float(self._stored_value))
        super().focusInEvent(event)

    def focusOutEvent(self, event: QFocusEvent) -> None:  # noqa: N802 - Qt virtual.
        super().focusOutEvent(event)
        self._dirty = False
        self.setText(_compact_float(self._stored_value))

    def _mark_dirty(self, _text: str) -> None:
        self._dirty = True


class ManualFitEditor(QFrame):
    """Compact controls that emit typed user intent; no scientific work occurs here."""

    component_interaction_requested = Signal(object)
    parameter_edit_requested = Signal(object, object)
    parameter_focused = Signal(object)
    chain_requested = Signal(object)
    new_tie_group_requested = Signal(object)
    join_tie_requested = Signal(object, str)
    component_removal_requested = Signal(object)
    apply_resolution_requested = Signal()
    run_requested = Signal()
    clear_model_requested = Signal()
    expanded_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("manualFitEditor")
        self._model: ManualModelState | None = None
        self._metadata: dict[ParameterReference, ParameterMetadata] = {}
        self._controls: dict[ParameterReference, _ParameterControls] = {}
        self._result_rows: dict[ParameterReference, _ResultParameterWidgets] = {}
        self._current_readiness: ManualWorkflowReadiness | None = None
        self._icon_colors = {
            "neutral": "#676764",
            "accent": "#496d91",
            "amber": "#a56c1d",
            "violet": "#8d819d",
        }

        self.title = QLabel("Fitting Parameters")
        self.title.setObjectName("manualFitTitle")
        self.model_label = QLabel("Model    N/A")
        self.model_label.setObjectName("manualModelSummary")
        self.add_button = QToolButton()
        self.add_button.setObjectName("manualModelAddButton")
        self.add_button.setText("+")
        self.add_button.setToolTip("Add a Manual Fit function")
        self.add_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon,
        )
        self.add_button.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        apply_disclosure_icon(
            self.add_button,
            expanded=True,
            color=self._icon_colors["neutral"],
        )
        self.add_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.add_button.setMenu(self._component_menu())
        self.clear_model_button = QToolButton()
        self.clear_model_button.setObjectName("manualModelClearButton")
        self.clear_model_button.setText("Clear")
        self.clear_model_button.setToolTip("Clear all functions from this Group")
        self.clear_model_button.setEnabled(False)
        self.clear_model_button.clicked.connect(self.clear_model_requested)
        self.status_label = QLabel()
        self.status_label.setObjectName("manualFitReadiness")
        self.status_label.setProperty("secondary", True)
        self.status_label.setWordWrap(True)
        self.run_button = QToolButton()
        self.run_button.setObjectName("manualRunFitButton")
        self.run_button.setText("Run Fit")
        self.run_button.clicked.connect(self.run_requested)
        self.diagnostics_button = QToolButton()
        self.diagnostics_button.setObjectName("manualDiagnosticsToggle")
        self.diagnostics_button.setText("Diagnostics / readiness")
        self.diagnostics_button.setCheckable(True)
        self.diagnostics_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon,
        )
        self.diagnostics_button.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        apply_disclosure_icon(
            self.diagnostics_button,
            expanded=False,
            color=self._icon_colors["neutral"],
        )
        self.diagnostics_button.toggled.connect(self._set_diagnostics_expanded)
        self.diagnostics_label = QLabel()
        self.diagnostics_label.setObjectName("manualDiagnosticsDetails")
        self.diagnostics_label.setProperty("secondary", True)
        self.diagnostics_label.setWordWrap(True)
        self.diagnostics_label.hide()
        self.apply_resolution_button = QToolButton()
        self.apply_resolution_button.setObjectName("manualApplyResolutionButton")
        self.apply_resolution_button.setText("Apply Resolution…")
        self.apply_resolution_button.clicked.connect(self.apply_resolution_requested)
        self.apply_resolution_button.hide()
        self.result_section = QFrame(self)
        self.result_section.setObjectName("manualFitResultSection")
        result_layout = QVBoxLayout(self.result_section)
        result_layout.setContentsMargins(0, 7, 0, 0)
        result_layout.setSpacing(4)
        self.result_toggle_button = QToolButton(self.result_section)
        self.result_toggle_button.setObjectName("manualFitResultToggle")
        self.result_toggle_button.setText("Fit Result")
        self.result_toggle_button.setCheckable(True)
        self.result_toggle_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon,
        )
        self.result_toggle_button.setLayoutDirection(
            Qt.LayoutDirection.RightToLeft,
        )
        apply_disclosure_icon(
            self.result_toggle_button,
            expanded=False,
            color=self._icon_colors["neutral"],
        )
        self.result_toggle_button.toggled.connect(self._set_result_expanded)
        self.result_content = QWidget(self.result_section)
        result_content_layout = QVBoxLayout(self.result_content)
        result_content_layout.setContentsMargins(0, 0, 0, 0)
        result_content_layout.setSpacing(4)
        self.result_summary_label = QLabel()
        self.result_summary_label.setObjectName("manualFitResultSummary")
        self.result_summary_label.setProperty("secondary", True)
        self.result_statistics_widget = QWidget(self.result_content)
        result_statistics_layout = QGridLayout(self.result_statistics_widget)
        result_statistics_layout.setContentsMargins(0, 0, 0, 0)
        result_statistics_layout.setHorizontalSpacing(8)
        result_statistics_layout.setVerticalSpacing(1)
        self.result_statistics: dict[str, QLabel] = {}
        for row, (key, label) in enumerate(
            (
                ("chi_square", "χ²"),
                ("reduced_chi_square", "Reduced χ²"),
                ("nominal_degrees_of_freedom", "DOF"),
                ("observations", "Points"),
            )
        ):
            title = QLabel(label)
            title.setProperty("secondary", True)
            value = QLabel("—")
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            result_statistics_layout.addWidget(title, row, 0)
            result_statistics_layout.addWidget(value, row, 1)
            self.result_statistics[key] = value
        result_statistics_layout.setColumnStretch(0, 1)
        self.result_parameters = QWidget(self.result_content)
        self._result_parameters_layout = QGridLayout(self.result_parameters)
        self._result_parameters_layout.setContentsMargins(0, 2, 0, 0)
        self._result_parameters_layout.setHorizontalSpacing(5)
        self._result_parameters_layout.setVerticalSpacing(2)
        self.result_important_label = QLabel()
        self.result_important_label.setObjectName("manualFitResultImportant")
        self.result_important_label.setWordWrap(True)
        self.result_diagnostics_button = QToolButton()
        self.result_diagnostics_button.setObjectName("manualResultDiagnosticsToggle")
        self.result_diagnostics_button.setText("Result diagnostics")
        self.result_diagnostics_button.setCheckable(True)
        self.result_diagnostics_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon,
        )
        self.result_diagnostics_button.setLayoutDirection(
            Qt.LayoutDirection.RightToLeft
        )
        apply_disclosure_icon(
            self.result_diagnostics_button,
            expanded=False,
            color=self._icon_colors["neutral"],
        )
        self.result_diagnostics_button.toggled.connect(
            self._set_result_diagnostics_expanded
        )
        self.result_diagnostics_label = QLabel()
        self.result_diagnostics_label.setObjectName("manualResultDiagnosticsDetails")
        self.result_diagnostics_label.setProperty("secondary", True)
        self.result_diagnostics_label.setWordWrap(True)
        self.result_diagnostics_label.hide()
        result_content_layout.addWidget(self.result_summary_label)
        result_content_layout.addWidget(self.result_statistics_widget)
        result_content_layout.addWidget(self.result_parameters)
        result_content_layout.addWidget(self.result_important_label)
        result_content_layout.addWidget(
            self.result_diagnostics_button,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        result_content_layout.addWidget(self.result_diagnostics_label)
        result_layout.addWidget(
            self.result_toggle_button,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        result_layout.addWidget(self.result_content)
        self.result_content.hide()
        self.result_section.hide()
        self.collapse_button = QToolButton()
        self.collapse_button.setObjectName("manualFitCollapseButton")
        self.collapse_button.setCheckable(True)
        self.collapse_button.setChecked(True)
        self.collapse_button.setToolTip("Collapse Manual Fit")
        apply_disclosure_icon(
            self.collapse_button,
            expanded=True,
            color=self._icon_colors["neutral"],
        )
        self.collapse_button.toggled.connect(self._set_manual_fit_expanded)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.title)
        header.addStretch(1)
        header.addWidget(self.collapse_button)
        composition = QHBoxLayout()
        composition.setContentsMargins(0, 0, 0, 0)
        composition.addWidget(self.model_label)
        composition.addStretch(1)
        composition.addWidget(self.add_button)
        composition.addWidget(self.clear_model_button)
        status = QHBoxLayout()
        status.setContentsMargins(0, 0, 0, 0)
        status.addWidget(self.status_label, 1)
        status.addWidget(self.run_button)

        self.sections = QWidget(self)
        self._sections_layout = QVBoxLayout(self.sections)
        self._sections_layout.setContentsMargins(0, 0, 0, 0)
        self._sections_layout.setSpacing(6)
        self.body = QWidget(self)
        self.body.setObjectName("manualFitBody")
        body_layout = QVBoxLayout(self.body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(6)
        body_layout.addLayout(composition)
        body_layout.addLayout(status)
        body_layout.addWidget(
            self.apply_resolution_button,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        body_layout.addWidget(self.sections)
        body_layout.addWidget(self.result_section)
        body_layout.addWidget(
            self.diagnostics_button,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        body_layout.addWidget(self.diagnostics_label)
        body_layout.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(header)
        layout.addWidget(self.body)

    @property
    def parameter_controls(self) -> dict[ParameterReference, _ParameterControls]:
        """Expose the compact cells for focused GUI tests and presentation refresh."""

        return dict(self._controls)

    @property
    def result_parameter_rows(
        self,
    ) -> dict[ParameterReference, _ResultParameterWidgets]:
        """Expose identity-keyed read-only result rows for focused GUI tests."""

        return dict(self._result_rows)

    def reset_result_disclosure(self) -> None:
        """Restore the default collapsed presentation for a new Manual Fit task."""

        self.result_toggle_button.setChecked(False)

    def set_expanded(self, expanded: bool) -> None:
        """Set presentation disclosure without mutating rendered Manual state."""

        self.collapse_button.setChecked(expanded)

    def populated_parameter_minimum_width(self) -> int:
        """Return the natural minimum width of the populated parameter grids."""

        if not self._controls:
            return 0
        for section in self.sections.findChildren(
            QFrame,
            options=Qt.FindChildOption.FindDirectChildrenOnly,
        ):
            section_layout = section.layout()
            if section_layout is not None:
                section_layout.activate()
        self._sections_layout.activate()
        layout = self.sections.layout()
        return 0 if layout is None else layout.minimumSize().width()

    def set_icon_colors(
        self,
        *,
        neutral: str,
        accent: str,
        amber: str,
        violet: str,
    ) -> None:
        """Apply shared theme tokens to the small lock/chain icon seam."""

        self._icon_colors = {
            "neutral": neutral,
            "accent": accent,
            "amber": amber,
            "violet": violet,
        }
        for button in self.findChildren(QToolButton):
            if button.property("compactDisclosure"):
                apply_disclosure_icon(
                    button,
                    expanded=bool(button.property("disclosureExpanded")),
                    color=neutral,
                )
        if self._model is None:
            return
        for reference, controls in self._controls.items():
            controls.fixed.setIcon(load_icon(IconName.LOCK, neutral))
            if controls.chain is None:
                continue
            tie = self._model.tie_for(reference)
            color = "neutral" if tie is None else _tie_color(self._model, tie.group_id)
            controls.chain.setIcon(load_icon(IconName.CHAIN, self._icon_colors[color]))

    def set_state(
        self,
        model: ManualModelState | None,
        metadata: tuple[ParameterMetadata, ...],
        readiness: ManualWorkflowReadiness,
        lifecycle: ManualFitLifecycle = ManualFitLifecycle.READY,
        execution_diagnostics: tuple[WorkflowDiagnostic, ...] = (),
        fit_result: FitResult | None = None,
    ) -> None:
        """Render immutable workflow state without deriving scientific values."""

        self._model = model
        self._metadata = {item.reference: item for item in metadata}
        self._current_readiness = readiness
        self._controls = {}
        self._clear_sections()
        self.model_label.setText(f"Model    {self._composition_summary(model)}")
        self.clear_model_button.setEnabled(model is not None)
        self._render_status(readiness, lifecycle, execution_diagnostics)
        self._render_fit_result(
            fit_result if lifecycle is ManualFitLifecycle.CURRENT else None,
            execution_diagnostics,
        )
        self._add_fixed_section(
            "δ",
            ELASTIC_COMPONENT,
            (ParameterFamily.AREA, ParameterFamily.CENTER),
        )
        self._add_lorentzian_section()
        self._add_fixed_section(
            "Background",
            BACKGROUND_COMPONENT,
            (ParameterFamily.OFFSET, ParameterFamily.SLOPE),
        )

    def _component_menu(self) -> QMenu:
        menu = QMenu(self)
        for text, kind in (
            ("Add Elastic", ManualComponentKind.ELASTIC),
            ("Add Lorentzian", ManualComponentKind.LORENTZIAN),
            ("Add Background", ManualComponentKind.BACKGROUND),
        ):
            action = menu.addAction(text)
            action.triggered.connect(self._component_emitter(kind))
        return menu

    def _component_emitter(self, kind: ManualComponentKind) -> Callable[..., None]:
        return lambda _checked=False: self.component_interaction_requested.emit(kind)

    def _component_removal_emitter(
        self,
        identity: ComponentIdentity,
    ) -> Callable[..., None]:
        return lambda _checked=False: self.component_removal_requested.emit(identity)

    def _new_tie_group_emitter(
        self,
        reference: ParameterReference,
    ) -> Callable[..., None]:
        return lambda _checked=False: self.new_tie_group_requested.emit(reference)

    def _join_tie_emitter(
        self,
        reference: ParameterReference,
        group_id: str,
    ) -> Callable[..., None]:
        return lambda _checked=False: self.join_tie_requested.emit(reference, group_id)

    def _chain_menu_opener(
        self,
        button: QToolButton,
        reference: ParameterReference,
    ) -> Callable[[QPoint], None]:
        return lambda position: self._show_chain_menu(button, position, reference)

    def _clear_sections(self) -> None:
        while self._sections_layout.count():
            item = self._sections_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()

    def _render_status(
        self,
        readiness: ManualWorkflowReadiness,
        lifecycle: ManualFitLifecycle,
        execution_diagnostics: tuple[WorkflowDiagnostic, ...],
    ) -> None:
        scientific_diagnostics = (
            ()
            if readiness.scientific_readiness is None
            else readiness.scientific_readiness.diagnostics
        )
        blocker_count = sum(
            item.severity.value == "error" for item in readiness.workflow_diagnostics
        ) + sum(item.severity.value == "error" for item in scientific_diagnostics)
        if lifecycle is ManualFitLifecycle.FITTING:
            status = "Fitting"
        elif not readiness.runnable:
            suffix = "blocker" if blocker_count == 1 else "blockers"
            status = f"Blocked · {blocker_count} {suffix}"
        elif lifecycle is ManualFitLifecycle.CURRENT:
            status = "Fit current"
        elif lifecycle is ManualFitLifecycle.NEEDS_FIT:
            status = "Needs fit"
        else:
            status = "Ready · Not yet fit"
        self.status_label.setText(status)
        messages = [item.message for item in readiness.workflow_diagnostics]
        messages.extend(item.message for item in scientific_diagnostics)
        messages.extend(item.message for item in execution_diagnostics)
        self.status_label.setToolTip("\n".join(messages))
        self.diagnostics_label.setText("\n".join(messages) or "No diagnostics")
        fitting = lifecycle is ManualFitLifecycle.FITTING
        self.run_button.setText("Fitting…" if fitting else "Run Fit")
        self.run_button.setEnabled(readiness.runnable and not fitting)
        resolution_missing = any(
            item.code is WorkflowDiagnosticCode.NO_APPLIED_RESOLUTION
            for item in readiness.workflow_diagnostics
        )
        self.apply_resolution_button.setVisible(resolution_missing)
        if resolution_missing:
            self.status_label.setText(
                "Blocked · Resolution required for preview and fitting"
            )

    def _render_fit_result(
        self,
        result: FitResult | None,
        execution_diagnostics: tuple[WorkflowDiagnostic, ...],
    ) -> None:
        """Render one current public FitResult without deriving scientific values."""

        self._clear_result_parameter_rows()
        if result is None or self._model is None:
            self.result_section.hide()
            self.result_content.hide()
            return
        self.result_summary_label.setText(
            f"Converged · {result.provenance.group_label} · "
            f"Q = {_result_float(result.provenance.q_value)}"
        )
        statistics = result.statistics
        self.result_statistics["chi_square"].setText(
            _result_float(statistics.chi_square)
        )
        self.result_statistics["reduced_chi_square"].setText(
            _result_float(statistics.reduced_chi_square)
        )
        self.result_statistics["nominal_degrees_of_freedom"].setText(
            str(statistics.nominal_degrees_of_freedom)
        )
        self.result_statistics["observations"].setText(str(statistics.observations))
        for column, text in enumerate(("Parameter", "Estimate", "Uncertainty", "")):
            header = QLabel(text)
            header.setProperty("secondary", True)
            self._result_parameters_layout.addWidget(header, 0, column)
        active_labels: list[str] = []
        for row_index, reference in enumerate(
            self._model.parameter_references(),
            start=1,
        ):
            estimate = result.parameter_by_reference(reference)
            name_text = self._result_reference_label(reference)
            metadata = self._metadata.get(reference, _fallback_metadata(reference))
            name = QLabel(name_text)
            value = QLabel(_result_float(estimate.value))
            uncertainty = QLabel(
                "—"
                if estimate.standard_error is None
                else _result_float(estimate.standard_error)
            )
            for label in (name, value, uncertainty):
                label.setToolTip(metadata.unit)
            details_parts: list[str] = []
            tie = self._model.tie_for(reference)
            if tie is not None:
                tie_index = self._model.parameter_ties.index(tie) + 1
                details_parts.append(f"Chain {tie_index}")
            bounds: list[str] = []
            if estimate.active_lower_bound:
                bounds.append("lower")
            if estimate.active_upper_bound:
                bounds.append("upper")
            if bounds:
                bound_text = "/".join(bounds) + " bound"
                details_parts.append(bound_text)
                active_labels.append(name_text)
            details = QLabel(" · ".join(details_parts))
            details.setProperty("secondary", True)
            details.setProperty("activeBound", bool(bounds))
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            uncertainty.setAlignment(Qt.AlignmentFlag.AlignRight)
            self._result_parameters_layout.addWidget(name, row_index, 0)
            self._result_parameters_layout.addWidget(value, row_index, 1)
            self._result_parameters_layout.addWidget(uncertainty, row_index, 2)
            self._result_parameters_layout.addWidget(details, row_index, 3)
            self._result_rows[reference] = _ResultParameterWidgets(
                name,
                value,
                uncertainty,
                details,
            )
        self._result_parameters_layout.setColumnStretch(0, 1)
        important: list[str] = []
        if not result.diagnostics.covariance_available:
            important.append("Covariance unavailable")
        if active_labels:
            important.append(f"Active bound: {', '.join(active_labels)}")
        important.extend(
            f"{item.severity.value.title()}: {item.message}"
            for item in execution_diagnostics
        )
        self.result_important_label.setText("\n".join(important))
        self.result_important_label.setVisible(bool(important))
        diagnostics = result.diagnostics
        maximum_correlation = (
            "—"
            if diagnostics.maximum_absolute_correlation is None
            else _result_float(diagnostics.maximum_absolute_correlation)
        )
        self.result_diagnostics_label.setText(
            "\n".join(
                (
                    "Optimizer: converged",
                    f"Function evaluations: {diagnostics.function_evaluations}",
                    f"Jacobian rank: {diagnostics.jacobian_rank}",
                    f"Condition number: {_result_float(diagnostics.condition_number)}",
                    "Covariance: "
                    + (
                        "available"
                        if diagnostics.covariance_available
                        else "unavailable"
                    ),
                    f"Maximum |correlation|: {maximum_correlation}",
                    f"Residual RMS: {_result_float(diagnostics.residual.rms)}",
                    "Maximum |residual|: "
                    f"{_result_float(diagnostics.residual.maximum_absolute)}",
                    f"Active-bound parameters: {len(set(active_labels))}",
                )
            )
        )
        self.result_section.show()
        self.result_content.setVisible(self.result_toggle_button.isChecked())

    def _clear_result_parameter_rows(self) -> None:
        self._result_rows = {}
        while self._result_parameters_layout.count():
            item = self._result_parameters_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()

    def _result_reference_label(self, reference: ParameterReference) -> str:
        assert self._model is not None
        if reference.component == ELASTIC_COMPONENT:
            component = "δ"
        elif reference.component == BACKGROUND_COMPONENT:
            component = "Background"
        else:
            component = next(
                f"L{index}"
                for index, item in enumerate(self._model.lorentzians, start=1)
                if item.identity == reference.component
            )
        metadata = self._metadata.get(reference, _fallback_metadata(reference))
        return f"{component} {metadata.display_label}"

    def _set_result_diagnostics_expanded(self, expanded: bool) -> None:
        apply_disclosure_icon(
            self.result_diagnostics_button,
            expanded=expanded,
            color=self._icon_colors["neutral"],
        )
        self.result_diagnostics_label.setVisible(expanded)

    def _set_manual_fit_expanded(self, expanded: bool) -> None:
        apply_disclosure_icon(
            self.collapse_button,
            expanded=expanded,
            color=self._icon_colors["neutral"],
        )
        self.collapse_button.setToolTip(
            "Collapse Manual Fit" if expanded else "Expand Manual Fit",
        )
        self.body.setVisible(expanded)
        self.expanded_changed.emit(expanded)

    def _set_result_expanded(self, expanded: bool) -> None:
        apply_disclosure_icon(
            self.result_toggle_button,
            expanded=expanded,
            color=self._icon_colors["neutral"],
        )
        self.result_content.setVisible(expanded and not self.result_section.isHidden())

    def _set_diagnostics_expanded(self, expanded: bool) -> None:
        apply_disclosure_icon(
            self.diagnostics_button,
            expanded=expanded,
            color=self._icon_colors["neutral"],
        )
        self.diagnostics_label.setVisible(expanded)

    @staticmethod
    def _composition_summary(model: ManualModelState | None) -> str:
        if model is None:
            return "N/A"
        elastic = "δ" if model.elastic_area is not None else ""
        lorentzians = f"{len(model.lorentzians)}L" if model.lorentzians else ""
        background = ""
        if model.background is BackgroundModel.CONSTANT:
            background = "B0"
        elif model.background is BackgroundModel.LINEAR:
            slope = model.b1
            background = (
                "B0"
                if slope is not None and not slope.free and slope.current_value == 0.0
                else "B1"
            )
        return f"{elastic}{lorentzians}{background}" or "N/A"

    def _add_fixed_section(
        self,
        title: str,
        identity: ComponentIdentity,
        families: tuple[ParameterFamily, ...],
    ) -> None:
        section = QFrame(self.sections)
        section.setObjectName("manualFunctionSection")
        section.setProperty("functionRole", title)
        section.setProperty("parameterColumnCount", 3)
        layout = QGridLayout(section)
        layout.setContentsMargins(0, 4, 0, 2)
        layout.setHorizontalSpacing(2)
        layout.setVerticalSpacing(2)
        header = QWidget(section)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(2)
        header_label = QLabel(title)
        header_label.setObjectName("manualFunctionTitle")
        header_layout.addWidget(header_label)
        header_layout.addStretch(1)
        present = self._has_component(identity)
        if not present:
            add = QToolButton()
            add.setObjectName(f"add{title.replace('δ', 'Elastic')}Button")
            add.setText("+")
            add.setToolTip(f"Add {title}")
            add.clicked.connect(
                self._component_emitter(self._kind_for_section(identity)),
            )
            header_layout.addWidget(add)
        else:
            remove = self._remove_button(identity, f"Remove {title}")
            header_layout.addWidget(remove)
            for column, family in enumerate(families, start=1):
                reference = self._reference(identity, family)
                layout.addWidget(self._parameter_header(reference), 1, column)
                layout.addWidget(self._parameter_cell(reference), 2, column)
        layout.addWidget(header, 0, 0, 1, 4)
        layout.setColumnMinimumWidth(0, 32)
        for column in range(1, 4):
            layout.setColumnStretch(column, 1)
        self._sections_layout.addWidget(section)

    def _add_lorentzian_section(self) -> None:
        section = QFrame(self.sections)
        section.setObjectName("manualLorentzianSection")
        section.setProperty("parameterColumnCount", 3)
        layout = QGridLayout(section)
        layout.setContentsMargins(0, 4, 0, 2)
        layout.setHorizontalSpacing(2)
        layout.setVerticalSpacing(2)
        header = QWidget(section)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(2)
        title = QLabel("Lorentzian")
        title.setObjectName("manualFunctionTitle")
        header_layout.addWidget(title)
        header_layout.addStretch(1)
        self.lorentzian_add_button = QToolButton()
        self.lorentzian_add_button.setObjectName("addLorentzianButton")
        self.lorentzian_add_button.setText("+")
        self.lorentzian_add_button.setToolTip("Add Lorentzian")
        self.lorentzian_add_button.clicked.connect(
            self._component_emitter(ManualComponentKind.LORENTZIAN),
        )
        header_layout.addWidget(self.lorentzian_add_button)
        layout.addWidget(header, 0, 0, 1, 4)
        families = (
            ParameterFamily.AREA,
            ParameterFamily.CENTER,
            ParameterFamily.FWHM,
        )
        identities = self._lorentzian_identities()
        if identities:
            for column, family in enumerate(families, start=1):
                reference = self._reference(identities[0], family)
                layout.addWidget(self._parameter_header(reference), 1, column)
        for row, identity in enumerate(identities, start=2):
            layout.addWidget(
                self._component_menu_button(identity, f"L{row - 1}"),
                row,
                0,
            )
            for column, family in enumerate(families, start=1):
                layout.addWidget(
                    self._parameter_cell(self._reference(identity, family)),
                    row,
                    column,
                )
        layout.setColumnMinimumWidth(0, 32)
        for column in range(1, 4):
            layout.setColumnStretch(column, 1)
        self._sections_layout.addWidget(section)

    def _kind_for_section(
        self,
        identity: ComponentIdentity | None,
    ) -> ManualComponentKind:
        if identity == ELASTIC_COMPONENT:
            return ManualComponentKind.ELASTIC
        if identity == BACKGROUND_COMPONENT:
            return ManualComponentKind.BACKGROUND
        return ManualComponentKind.LORENTZIAN

    def _remove_button(self, identity: ComponentIdentity, text: str) -> QToolButton:
        button = QToolButton()
        button.setFixedWidth(24)
        button.setToolTip(text)
        apply_disclosure_icon(
            button,
            expanded=True,
            color=self._icon_colors["neutral"],
        )
        menu = QMenu(button)
        action = menu.addAction(text)
        action.triggered.connect(self._component_removal_emitter(identity))
        button.setMenu(menu)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        return button

    def _component_menu_button(
        self,
        identity: ComponentIdentity,
        text: str,
    ) -> QToolButton:
        button = QToolButton()
        button.setObjectName("manualComponentRowLabel")
        button.setText(text)
        button.setFixedWidth(32)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        apply_disclosure_icon(
            button,
            expanded=True,
            color=self._icon_colors["neutral"],
        )
        menu = QMenu(button)
        action = menu.addAction("Remove Lorentzian")
        action.triggered.connect(self._component_removal_emitter(identity))
        button.setMenu(menu)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        return button

    def _parameter_header(self, reference: ParameterReference) -> QLabel:
        metadata = self._metadata.get(reference, _fallback_metadata(reference))
        label = QLabel(metadata.display_label)
        label.setObjectName("manualParameterHeader")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setToolTip(metadata.unit)
        return label

    def _parameter_cell(self, reference: ParameterReference) -> QWidget:
        assert self._model is not None
        intent = self._model.parameter_intent(reference)
        metadata = self._metadata.get(reference, _fallback_metadata(reference))
        cell = QFrame()
        cell.setObjectName("manualParameterCell")
        cell.setProperty("parameterFamily", reference.family.value)
        outer = QVBoxLayout(cell)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(1)
        first = QHBoxLayout()
        first.setContentsMargins(0, 0, 0, 0)
        first.setSpacing(1)
        current = _numeric_edit(intent.current_value)
        current.setAccessibleName(f"{metadata.display_label} Current Value")
        fixed = QToolButton()
        fixed.setIcon(load_icon(IconName.LOCK, self._icon_colors["neutral"]))
        fixed.setCheckable(True)
        fixed.setChecked(not intent.free)
        fixed.setProperty("fixedControl", True)
        fixed.setToolTip("Fixed" if not intent.free else "Free")
        fixed.setFixedWidth(20)
        chain: QToolButton | None = None
        if reference.family not in {
            ParameterFamily.OFFSET,
            ParameterFamily.SLOPE,
        }:
            chain = QToolButton()
            tie_color = "neutral"
            chain.setCheckable(True)
            chain.setProperty("chainControl", True)
            tie = self._model.tie_for(reference)
            if tie is not None:
                chain.setChecked(True)
                tie_color = _tie_color(self._model, tie.group_id)
                chain.setProperty("tieColor", tie_color)
            chain.setIcon(load_icon(IconName.CHAIN, self._icon_colors[tie_color]))
            chain.setToolTip("Leave tie group" if tie is not None else "Join Chain 1")
            chain.setFixedWidth(20)
            chain.clicked.connect(
                lambda _checked=False, value=reference: self.chain_requested.emit(
                    value
                ),
            )
            chain.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            chain.customContextMenuRequested.connect(
                self._chain_menu_opener(chain, reference),
            )
        first.addWidget(current, 1)
        first.addWidget(fixed)
        if chain is not None:
            first.addWidget(chain)
        second = QHBoxLayout()
        second.setContentsMargins(0, 0, 0, 0)
        second.setSpacing(1)
        lower = _numeric_edit(intent.user_lower_limit, optional=True)
        upper = _numeric_edit(intent.user_upper_limit, optional=True)
        bounds_enabled = QToolButton()
        bounds_enabled.setObjectName("manualBoundsEnabledButton")
        bounds_enabled.setProperty("boundsControl", True)
        bounds_enabled.setCheckable(True)
        bounds_enabled.setChecked(intent.user_bounds_enabled)
        bounds_enabled.setText("✓" if intent.user_bounds_enabled else "×")
        bounds_enabled.setToolTip(
            "User bounds enabled"
            if intent.user_bounds_enabled
            else "User bounds disabled"
        )
        bounds_enabled.setFixedWidth(20)
        lower.setAccessibleName(f"{metadata.display_label} Lower bound")
        upper.setAccessibleName(f"{metadata.display_label} Upper bound")
        lower.setProperty("boundsEnabled", intent.user_bounds_enabled)
        upper.setProperty("boundsEnabled", intent.user_bounds_enabled)
        lower.setEnabled(intent.free)
        upper.setEnabled(intent.free)
        second.addWidget(lower, 1)
        separator = QLabel("—")
        separator.setObjectName("manualBoundsSeparator")
        separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        second.addWidget(separator)
        second.addWidget(upper, 1)
        second.addWidget(bounds_enabled)
        outer.addLayout(first)
        outer.addLayout(second)
        scientific_diagnostics = (
            ()
            if self._current_readiness is None
            or self._current_readiness.scientific_readiness is None
            else self._current_readiness.scientific_readiness.diagnostics
        )
        cell.setProperty(
            "manualWarning",
            any(item.parameter == reference for item in scientific_diagnostics),
        )
        controls = _ParameterControls(
            current,
            lower,
            upper,
            fixed,
            bounds_enabled,
            chain,
        )
        self._controls[reference] = controls
        for edit in (current, lower, upper):
            edit.editingFinished.connect(
                lambda value=reference, source=edit: self._commit_parameter_text_edit(
                    value,
                    source,
                ),
            )
            edit.selectionChanged.connect(
                lambda value=reference: self.parameter_focused.emit(value),
            )
        fixed.clicked.connect(
            lambda _checked=False, value=reference: self._emit_parameter_edit(value),
        )
        bounds_enabled.clicked.connect(
            lambda checked, button=bounds_enabled: self._set_bounds_button_state(
                button,
                checked,
            ),
        )
        bounds_enabled.clicked.connect(
            lambda _checked=False, value=reference: self._emit_parameter_edit(value),
        )
        return cell

    def _commit_parameter_text_edit(
        self,
        reference: ParameterReference,
        source: CompactNumberEdit,
    ) -> None:
        """Commit exactly one real user text edit, never focus navigation alone."""

        if not source.user_edited:
            return
        try:
            self._emit_parameter_edit(reference)
        finally:
            source.finish_user_edit()

    @staticmethod
    def _set_bounds_button_state(button: QToolButton, enabled: bool) -> None:
        button.setText("✓" if enabled else "×")
        button.setToolTip("User bounds enabled" if enabled else "User bounds disabled")

    def _reference(
        self,
        identity: ComponentIdentity,
        family: ParameterFamily,
    ) -> ParameterReference:
        reference = ParameterReference(identity, family)
        if self._model is None or reference not in self._model.parameter_references():
            raise KeyError(reference)
        return reference

    def _show_chain_menu(
        self,
        button: QToolButton,
        position: QPoint,
        reference: ParameterReference,
    ) -> None:
        if self._model is None:
            return
        menu = QMenu(button)
        tie = self._model.tie_for(reference)
        if tie is not None:
            leave = menu.addAction("Leave Tie Group")
            leave.triggered.connect(
                lambda _checked=False: self.chain_requested.emit(reference),
            )
            menu.exec(button.mapToGlobal(position))
            return
        action = menu.addAction("New Tie Group")
        action.triggered.connect(self._new_tie_group_emitter(reference))
        matching = tuple(
            group
            for group in self._model.parameter_ties
            if group.members[0].family is reference.family
        )
        if matching:
            join = menu.addMenu("Join Tie Group")
            for index, group in enumerate(matching, start=1):
                item = join.addAction(f"Join Chain {index}")
                item.triggered.connect(
                    self._join_tie_emitter(reference, group.group_id),
                )
        menu.exec(button.mapToGlobal(position))

    def _emit_parameter_edit(self, reference: ParameterReference) -> None:
        controls = self._controls[reference]
        try:
            current_value = _numeric_value(controls.current, required=True)
            if current_value is None:
                return
            edit = ManualParameterEdit(
                current_value=current_value,
                user_lower_limit=_numeric_value(controls.lower),
                user_upper_limit=_numeric_value(controls.upper),
                free=not controls.fixed.isChecked(),
                user_bounds_enabled=controls.bounds_enabled.isChecked(),
            )
        except ValueError:
            return
        self.parameter_edit_requested.emit(reference, edit)

    def _has_component(self, identity: ComponentIdentity) -> bool:
        if self._model is None:
            return False
        return any(
            reference.component == identity
            for reference in self._model.parameter_references()
        )

    def _lorentzian_identities(self) -> tuple[ComponentIdentity, ...]:
        if self._model is None:
            return ()
        return tuple(item.identity for item in self._model.lorentzians)


def _fallback_metadata(reference: ParameterReference) -> ParameterMetadata:
    """Use typed presentation metadata when no Resolution allows materialization."""

    label, dimension = _FALLBACK_METADATA[reference.family]
    return ParameterMetadata(reference, label, dimension, "")


def _numeric_edit(
    value: float | None,
    *,
    optional: bool = False,
) -> CompactNumberEdit:
    edit = CompactNumberEdit(value, optional=optional)
    edit.setMinimumWidth(30)
    edit.setSizePolicy(
        QSizePolicy.Policy.Ignored,
        QSizePolicy.Policy.Fixed,
    )
    return edit


def _numeric_value(
    edit: CompactNumberEdit,
    *,
    required: bool = False,
) -> float | None:
    return edit.precise_value(required=required)


def _compact_float(value: float | None) -> str:
    return _display_float(value, decimal_places=2)


def _focused_float(value: float | None) -> str:
    return _display_float(value, decimal_places=3)


def _display_float(value: float | None, *, decimal_places: int) -> str:
    if value is None:
        return ""
    numeric = float(value)
    magnitude = abs(numeric)
    if magnitude == 0.0:
        return "0"
    decimal = format(numeric, f".{decimal_places}f").rstrip("0").rstrip(".")
    digit_positions = sum(character.isdigit() for character in decimal)
    if magnitude < 0.01 or digit_positions >= 6:
        mantissa, exponent = format(numeric, f".{decimal_places}e").lower().split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        return f"{mantissa}e{int(exponent)}"
    return decimal


def _round_trip_float(value: float | None) -> str:
    return "" if value is None else repr(float(value))


def _result_float(value: float) -> str:
    """Format a read-only result without changing its authoritative value."""

    return _display_float(float(value), decimal_places=3)


def _tie_color(model: ManualModelState, group_id: str) -> str:
    """Map immutable tie-group order to the small shared presentation palette."""

    index = next(
        index
        for index, group in enumerate(model.parameter_ties)
        if group.group_id == group_id
    )
    return ("accent", "amber", "violet")[index % 3]
