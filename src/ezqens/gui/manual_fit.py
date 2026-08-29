"""Thin Inspector controls for the group-local Single-Q Manual Fit task."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QDoubleValidator, QFocusEvent
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
    WorkflowDiagnosticCode,
)


@dataclass(frozen=True)
class _ParameterControls:
    """Widgets for one typed parameter entrance."""

    current: CompactNumberEdit
    lower: CompactNumberEdit
    upper: CompactNumberEdit
    fixed: QToolButton
    chain: QToolButton | None


_FALLBACK_METADATA = {
    ParameterFamily.AREA: ("Area", ParameterDimension.INTEGRATED_INTENSITY),
    ParameterFamily.CENTER: ("Center", ParameterDimension.ENERGY),
    ParameterFamily.FWHM: ("FWHM", ParameterDimension.ENERGY),
    ParameterFamily.OFFSET: ("Offset", ParameterDimension.INTENSITY),
    ParameterFamily.SLOPE: ("Slope", ParameterDimension.INTENSITY_PER_ENERGY),
}


class CompactNumberEdit(QLineEdit):
    """Compact unfocused float text with lossless untouched-value round trips."""

    def __init__(self, value: float | None) -> None:
        super().__init__()
        self._stored_value = None if value is None else float(value)
        self._dirty = False
        validator = QDoubleValidator(self)
        validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
        self.setValidator(validator)
        self.setText(_compact_float(self._stored_value))
        self.setToolTip(_round_trip_float(self._stored_value))
        self.textEdited.connect(self._mark_dirty)

    @property
    def stored_value(self) -> float | None:
        """Return the exact immutable value represented before any user edit."""

        return self._stored_value

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
            self.setText(_round_trip_float(self._stored_value))
        super().focusInEvent(event)

    def focusOutEvent(self, event: QFocusEvent) -> None:  # noqa: N802 - Qt virtual.
        super().focusOutEvent(event)
        if not self._dirty:
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
    close_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("manualFitEditor")
        self._model: ManualModelState | None = None
        self._metadata: dict[ParameterReference, ParameterMetadata] = {}
        self._controls: dict[ParameterReference, _ParameterControls] = {}
        self._current_readiness: ManualWorkflowReadiness | None = None
        self._icon_colors = {
            "neutral": "#676764",
            "accent": "#496d91",
            "amber": "#a56c1d",
            "violet": "#8d819d",
        }

        self.title = QLabel("Manual Fit")
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
        self.status_label = QLabel()
        self.status_label.setObjectName("manualFitReadiness")
        self.status_label.setProperty("secondary", True)
        self.status_label.setWordWrap(True)
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
        self.close_button = QToolButton()
        self.close_button.setText("Close")
        self.close_button.clicked.connect(self.close_requested)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.title)
        header.addStretch(1)
        header.addWidget(self.close_button)
        composition = QHBoxLayout()
        composition.setContentsMargins(0, 0, 0, 0)
        composition.addWidget(self.model_label)
        composition.addStretch(1)
        composition.addWidget(self.add_button)

        self.sections = QWidget(self)
        self._sections_layout = QVBoxLayout(self.sections)
        self._sections_layout.setContentsMargins(0, 0, 0, 0)
        self._sections_layout.setSpacing(6)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(header)
        layout.addLayout(composition)
        layout.addWidget(self.status_label)
        layout.addWidget(
            self.apply_resolution_button,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        layout.addWidget(self.sections)
        layout.addWidget(
            self.diagnostics_button,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        layout.addWidget(self.diagnostics_label)
        layout.addStretch(1)

    @property
    def parameter_controls(self) -> dict[ParameterReference, _ParameterControls]:
        """Expose the compact cells for focused GUI tests and presentation refresh."""

        return dict(self._controls)

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
    ) -> None:
        """Render immutable workflow state without deriving scientific values."""

        self._model = model
        self._metadata = {item.reference: item for item in metadata}
        self._current_readiness = readiness
        self._controls = {}
        self._clear_sections()
        self.model_label.setText(f"Model    {self._composition_summary(model)}")
        self._render_status(readiness)
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
                widget.deleteLater()

    def _render_status(self, readiness: ManualWorkflowReadiness) -> None:
        scientific_diagnostics = (
            ()
            if readiness.scientific_readiness is None
            else readiness.scientific_readiness.diagnostics
        )
        blocker_count = sum(
            item.severity.value == "error" for item in readiness.workflow_diagnostics
        ) + sum(item.severity.value == "error" for item in scientific_diagnostics)
        status = "Ready"
        if not readiness.runnable:
            suffix = "blocker" if blocker_count == 1 else "blockers"
            status = f"Not ready · {blocker_count} {suffix}"
        self.status_label.setText(status)
        messages = [item.message for item in readiness.workflow_diagnostics]
        messages.extend(item.message for item in scientific_diagnostics)
        self.status_label.setToolTip("\n".join(messages))
        self.diagnostics_label.setText("\n".join(messages) or "No diagnostics")
        resolution_missing = any(
            item.code is WorkflowDiagnosticCode.NO_APPLIED_RESOLUTION
            for item in readiness.workflow_diagnostics
        )
        self.apply_resolution_button.setVisible(resolution_missing)
        if resolution_missing:
            self.status_label.setText("Resolution required for preview and fitting")

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
        lower = _numeric_edit(intent.user_lower_limit)
        upper = _numeric_edit(intent.user_upper_limit)
        lower.setAccessibleName(f"{metadata.display_label} Lower bound")
        upper.setAccessibleName(f"{metadata.display_label} Upper bound")
        lower.setEnabled(intent.free)
        upper.setEnabled(intent.free)
        second.addWidget(lower, 1)
        separator = QLabel("—")
        separator.setObjectName("manualBoundsSeparator")
        separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        second.addWidget(separator)
        second.addWidget(upper, 1)
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
        controls = _ParameterControls(current, lower, upper, fixed, chain)
        self._controls[reference] = controls
        for edit in (current, lower, upper):
            edit.editingFinished.connect(
                lambda value=reference: self._emit_parameter_edit(value),
            )
            edit.selectionChanged.connect(
                lambda value=reference: self.parameter_focused.emit(value),
            )
        fixed.clicked.connect(
            lambda _checked=False, value=reference: self._emit_parameter_edit(value),
        )
        return cell

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


def _numeric_edit(value: float | None) -> CompactNumberEdit:
    edit = CompactNumberEdit(value)
    edit.setMinimumWidth(30)
    edit.setSizePolicy(
        QSizePolicy.Policy.Expanding,
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
    if value is None:
        return ""
    numeric = float(value)
    magnitude = abs(numeric)
    if magnitude == 0.0:
        return "0"
    decimal = format(numeric, ".2f").rstrip("0").rstrip(".")
    digit_positions = sum(character.isdigit() for character in decimal)
    if magnitude < 0.01 or digit_positions >= 6:
        mantissa, exponent = format(numeric, ".2e").lower().split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        return f"{mantissa}e{int(exponent)}"
    return decimal


def _round_trip_float(value: float | None) -> str:
    return "" if value is None else repr(float(value))


def _tie_color(model: ManualModelState, group_id: str) -> str:
    """Map immutable tie-group order to the small shared presentation palette."""

    index = next(
        index
        for index, group in enumerate(model.parameter_ties)
        if group.group_id == group_id
    )
    return ("accent", "amber", "violet")[index % 3]
