"""Focused GUI task for fractional-coverage-aware Q rebinning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.colors import Normalize
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ezqens.domain import (
    FractionalCoverageAvailability,
    QBins,
    QExclusionInterval,
    QRebinSpecification,
    ReducedDataset,
    fixed_width_q_bins,
    uniform_q_bins,
)
from ezqens.gui.dialogs import confirm_dialog
from ezqens.gui.q_editor import EXPLICIT_VALUES, parse_q_values
from ezqens.gui.q_editor import REGULAR_GRID as _REGULAR_GRID
from ezqens.gui.scientific_canvas import ScientificCanvas
from ezqens.gui.theme import DEFAULT_LAYOUT_TOKENS
from ezqens.preprocessing import QRebinDiagnosticCode
from ezqens.workflow import (
    ProjectDataset,
    WorkflowError,
    WorkflowProject,
    applied_resolution,
    project_dataset,
    rebin_project_sample_q,
    replace_project_dataset,
)

_MAX_VISIBLE_LABEL_PAIRS = 20
_NAVIGATION_MARKER_DIAMETER_POINTS = 6.0
_NAVIGATION_MARKER_GAP_POINTS = 3.0
REGULAR_GRID = _REGULAR_GRID
EXPLICIT_EDGES = EXPLICIT_VALUES


@dataclass(frozen=True, slots=True)
class QRebinPreview:
    """One successful immutable workflow preview ready for explicit adoption."""

    workflow: WorkflowProject
    sample: ProjectDataset
    resolution: ProjectDataset | None
    specification: QRebinSpecification
    sample_source: ReducedDataset
    resolution_source: ReducedDataset | None


def adaptive_navigation_indices(
    group_count: int,
    available_width_px: float,
    widest_label_px: float,
) -> tuple[int, ...]:
    """Return evenly sampled labels from rendered width, always keeping endpoints."""

    if group_count <= 0:
        return ()
    if group_count == 1:
        return (0,)
    slot_width = max(widest_label_px + 8.0, 1.0)
    visible_count = min(
        group_count,
        _MAX_VISIBLE_LABEL_PAIRS,
        max(2, int(available_width_px // slot_width)),
    )
    indices = np.linspace(0, group_count - 1, visible_count)
    return tuple(dict.fromkeys(int(round(value)) for value in indices))


def navigation_markers_are_separable(
    group_count: int,
    available_width_px: float,
    figure_dpi: float,
) -> bool:
    """Use discrete markers only when their rendered point size leaves a gap."""

    if group_count <= 1:
        return True
    required_spacing = (
        (_NAVIGATION_MARKER_DIAMETER_POINTS + _NAVIGATION_MARKER_GAP_POINTS)
        * figure_dpi
        / 72.0
    )
    return available_width_px / (group_count - 1) >= required_spacing


class QRebinDialog(QDialog):
    """Edit and preview one typed Q-rebin transaction without hidden mutation."""

    def __init__(
        self,
        workflow: WorkflowProject,
        sample: ProjectDataset,
        *,
        sample_name: str,
        preserved_sample: ReducedDataset,
        preserved_resolution: ReducedDataset | None,
        resolution_name: str | None,
        apply_handler: Callable[[QRebinPreview], bool],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setObjectName("qRebinDialog")
        self.setWindowTitle("Q Rebin")
        self.setModal(True)
        self.resize(820, 650)
        self.setMinimumSize(680, 560)
        self.workflow = workflow
        self.sample = project_dataset(workflow, sample)
        self.sample_name = sample_name
        self.current_resolution = applied_resolution(workflow, self.sample)
        self.preserved_sample = preserved_sample
        self.preserved_resolution = preserved_resolution
        self.resolution_name = resolution_name
        self._apply_handler = apply_handler
        self._use_preserved_source = False
        self._sample_unrebinned_confirmed = False
        self._resolution_unrebinned_confirmed = False
        self._exclusions: list[QExclusionInterval] = []
        self.preview_result: QRebinPreview | None = None
        self.current_target_group = 0
        self.visible_label_indices: tuple[int, ...] = ()
        self.discrete_markers_visible = True
        self.heatmap_source_group_count = len(preserved_sample.spectra)
        self._regular_exact_values: dict[str, float] = {}
        self._setting_regular_numeric_text = False
        self._setting_regular_group_text = False
        self._explicit_exact_edges: tuple[float, ...] | None = None
        self._setting_explicit_text = False
        self._explicit_originated_from_regular = False
        self._regular_driver = "step"
        self._visible_target_mode = REGULAR_GRID

        title = QLabel("Q Rebin")
        title.setObjectName("ezqensDialogTitle")
        self.source_label = QLabel()
        self.source_label.setObjectName("qRebinSource")
        self.source_label.setWordWrap(True)
        self.lineage_label = QLabel()
        self.lineage_label.setObjectName("qRebinLineage")
        self.lineage_label.setProperty("secondary", True)
        self.lineage_label.setWordWrap(True)
        self.sample_coverage_label = QLabel()
        self.sample_coverage_label.setObjectName("qRebinSampleCoverage")
        self.resolution_coverage_label = QLabel()
        self.resolution_coverage_label.setObjectName("qRebinResolutionCoverage")
        self.confirm_sample_button = QPushButton("Confirm unre-binned source…")
        self.confirm_sample_button.setObjectName("qRebinConfirmSample")
        self.confirm_sample_button.clicked.connect(self._prompt_confirm_sample)
        self.confirm_resolution_button = QPushButton(
            "Confirm Resolution unre-binned source…"
        )
        self.confirm_resolution_button.setObjectName("qRebinConfirmResolution")
        self.confirm_resolution_button.clicked.connect(self._prompt_confirm_resolution)
        self.use_preserved_source_button = QPushButton("Use preserved source")
        self.use_preserved_source_button.setObjectName("qRebinUsePreservedSource")
        self.use_preserved_source_button.clicked.connect(self.use_preserved_source)
        self.use_preserved_source_button.hide()

        source_layout = QGridLayout()
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.setHorizontalSpacing(DEFAULT_LAYOUT_TOKENS.row_spacing)
        source_layout.setVerticalSpacing(DEFAULT_LAYOUT_TOKENS.row_spacing)
        source_layout.addWidget(QLabel("Source"), 0, 0)
        source_layout.addWidget(self.source_label, 0, 1)
        source_layout.addWidget(QLabel("Lineage"), 1, 0)
        source_layout.addWidget(self.lineage_label, 1, 1)
        source_layout.addWidget(QLabel("Coverage"), 2, 0)
        source_layout.addWidget(self.sample_coverage_label, 2, 1)
        source_layout.addWidget(self.confirm_sample_button, 2, 2)
        source_layout.addWidget(QLabel("Resolution"), 3, 0)
        source_layout.addWidget(self.resolution_coverage_label, 3, 1)
        source_layout.addWidget(self.confirm_resolution_button, 3, 2)
        source_layout.addWidget(self.use_preserved_source_button, 4, 1)

        target_title = QLabel("Target Q grid")
        target_title.setObjectName("qRebinTargetTitle")
        self.target_mode_combo = QComboBox()
        self.target_mode_combo.setObjectName("qRebinTargetMode")
        self.target_mode_combo.addItems([REGULAR_GRID, EXPLICIT_VALUES])
        target_heading = QHBoxLayout()
        target_heading.setContentsMargins(0, 0, 0, 0)
        target_heading.setSpacing(DEFAULT_LAYOUT_TOKENS.row_spacing)
        target_heading.addWidget(target_title)
        target_heading.addWidget(self.target_mode_combo)
        target_heading.addStretch(1)

        self.regular_lower_edit = QLineEdit()
        self.regular_lower_edit.setObjectName("qRebinRegularLower")
        self.regular_lower_edit.setPlaceholderText("Lower edge")
        self.regular_upper_edit = QLineEdit()
        self.regular_upper_edit.setObjectName("qRebinRegularUpper")
        self.regular_upper_edit.setPlaceholderText("Upper edge")
        self.regular_step_edit = QLineEdit()
        self.regular_step_edit.setObjectName("qRebinRegularStep")
        self.regular_step_edit.setPlaceholderText("ΔQ")
        self.regular_group_count_edit = QLineEdit()
        self.regular_group_count_edit.setObjectName("qRebinRegularGroupCount")
        self.regular_group_count_edit.setPlaceholderText("Groups")
        self.regular_fields = QWidget()
        regular_layout = QGridLayout(self.regular_fields)
        regular_layout.setContentsMargins(0, 0, 0, 0)
        regular_layout.setHorizontalSpacing(DEFAULT_LAYOUT_TOKENS.row_spacing)
        regular_layout.setVerticalSpacing(DEFAULT_LAYOUT_TOKENS.row_spacing)
        regular_layout.addWidget(QLabel("Lower Q edge"), 0, 0)
        regular_layout.addWidget(self.regular_lower_edit, 0, 1)
        regular_layout.addWidget(QLabel("Upper Q edge"), 0, 2)
        regular_layout.addWidget(self.regular_upper_edit, 0, 3)
        regular_layout.addWidget(QLabel("Step ΔQ"), 1, 0)
        regular_layout.addWidget(self.regular_step_edit, 1, 1)
        regular_layout.addWidget(QLabel("Groups"), 1, 2)
        regular_layout.addWidget(self.regular_group_count_edit, 1, 3)
        self.regular_coverage_label = QLabel()
        self.regular_coverage_label.setObjectName("qRebinRegularCoverage")
        self.regular_coverage_label.setProperty("secondary", True)
        regular_layout.addWidget(self.regular_coverage_label, 2, 0, 1, 4)

        self.target_edges_edit = QLineEdit()
        self.target_edges_edit.setObjectName("qRebinTargetEdges")
        self.target_edges_edit.setPlaceholderText(
            "Enter explicit Q-bin edges separated by spaces or commas"
        )
        self.target_preview_label = QLabel()
        self.target_preview_label.setProperty("secondary", True)
        self.target_mode_combo.currentTextChanged.connect(self._target_mode_changed)
        for key, field in (
            ("lower", self.regular_lower_edit),
            ("upper", self.regular_upper_edit),
            ("step", self.regular_step_edit),
        ):
            field.textChanged.connect(
                lambda _text, field_key=key: self._regular_field_changed(field_key)
            )
        self.regular_group_count_edit.textChanged.connect(
            lambda _text: self._regular_field_changed("groups")
        )
        self.target_edges_edit.textChanged.connect(self._explicit_edges_changed)

        exclusion_title = QLabel("Q coverage exclusions")
        exclusion_title.setObjectName("qRebinExclusionTitle")
        exclusion_note = QLabel(
            "Continuous preprocessing intervals only; fitting masks are unchanged."
        )
        exclusion_note.setProperty("secondary", True)
        self.exclusion_lower_edit = QLineEdit()
        self.exclusion_lower_edit.setPlaceholderText("Lower Q")
        self.exclusion_upper_edit = QLineEdit()
        self.exclusion_upper_edit.setPlaceholderText("Upper Q")
        self.add_exclusion_button = QPushButton("Add interval")
        self.add_exclusion_button.clicked.connect(self._add_exclusion_from_fields)
        self.remove_exclusion_button = QPushButton("Remove selected")
        self.remove_exclusion_button.clicked.connect(self.remove_selected_exclusion)
        self.exclusion_list = QListWidget()
        self.exclusion_list.setObjectName("qRebinExclusions")
        self.exclusion_list.setMaximumHeight(74)
        exclusion_entry = QHBoxLayout()
        exclusion_entry.setContentsMargins(0, 0, 0, 0)
        exclusion_entry.setSpacing(DEFAULT_LAYOUT_TOKENS.row_spacing)
        exclusion_entry.addWidget(self.exclusion_lower_edit)
        exclusion_entry.addWidget(self.exclusion_upper_edit)
        exclusion_entry.addWidget(self.add_exclusion_button)
        exclusion_entry.addWidget(self.remove_exclusion_button)

        self.preview_canvas = ScientificCanvas()
        self.preview_canvas.setObjectName("qRebinPreviewCanvas")
        self.preview_canvas.setMinimumHeight(220)
        self.preview_canvas.mpl_connect("button_press_event", self._on_preview_press)

        self.previous_button = QToolButton()
        self.previous_button.setText("‹")
        self.previous_button.clicked.connect(self.previous_target_group)
        self.group_spinbox = QSpinBox()
        self.group_spinbox.setObjectName("qRebinGroupSpinBox")
        self.group_spinbox.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.group_spinbox.valueChanged.connect(self._group_number_changed)
        self.next_button = QToolButton()
        self.next_button.setText("›")
        self.next_button.clicked.connect(self.next_target_group)
        self.current_q_label = QLabel()
        navigation = QHBoxLayout()
        navigation.setContentsMargins(0, 0, 0, 0)
        navigation.setSpacing(DEFAULT_LAYOUT_TOKENS.row_spacing)
        navigation.addStretch(1)
        navigation.addWidget(self.previous_button)
        navigation.addWidget(QLabel("Group"))
        navigation.addWidget(self.group_spinbox)
        navigation.addWidget(self.next_button)
        navigation.addWidget(self.current_q_label)
        navigation.addStretch(1)

        self.legality_label = QLabel()
        self.legality_label.setObjectName("qRebinLegality")
        self.legality_label.setWordWrap(True)
        self.association_label = QLabel()
        self.association_label.setObjectName("qRebinAssociation")
        self.association_label.setProperty("secondary", True)
        self.association_label.setWordWrap(True)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.apply_button = QPushButton("Apply Q Rebin")
        self.apply_button.setObjectName("applyQRebinButton")
        self.apply_button.setDefault(True)
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply_rebin)
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.addStretch(1)
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.apply_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(DEFAULT_LAYOUT_TOKENS.section_spacing)
        layout.addWidget(title)
        layout.addLayout(source_layout)
        layout.addLayout(target_heading)
        layout.addWidget(self.regular_fields)
        layout.addWidget(self.target_edges_edit)
        layout.addWidget(self.target_preview_label)
        layout.addWidget(exclusion_title)
        layout.addWidget(exclusion_note)
        layout.addLayout(exclusion_entry)
        layout.addWidget(self.exclusion_list)
        layout.addWidget(self.preview_canvas, 1)
        layout.addLayout(navigation)
        layout.addWidget(self.legality_label)
        layout.addWidget(self.association_label)
        layout.addLayout(actions)

        for control in (
            self.confirm_sample_button,
            self.confirm_resolution_button,
            self.use_preserved_source_button,
            self.target_mode_combo,
            self.regular_lower_edit,
            self.regular_upper_edit,
            self.regular_step_edit,
            self.regular_group_count_edit,
            self.target_edges_edit,
            self.exclusion_lower_edit,
            self.exclusion_upper_edit,
            self.add_exclusion_button,
            self.remove_exclusion_button,
            self.previous_button,
            self.group_spinbox,
            self.next_button,
            self.cancel_button,
            self.apply_button,
        ):
            control.setMinimumHeight(DEFAULT_LAYOUT_TOKENS.control_height)

        edges = self.sample.dataset.q_bins
        if edges is not None and edges.edges is not None:
            edge_values = tuple(float(value) for value in edges.edges)
            self._set_explicit_edges(edge_values)
            self._populate_regular_grid(edges)
            mode = REGULAR_GRID if self.regular_step_edit.text() else EXPLICIT_VALUES
            self._set_target_mode(mode, capture_current=False)
        else:
            self._refresh_preview()
            self._set_target_mode(REGULAR_GRID, capture_current=False)

    @property
    def exclusions(self) -> tuple[QExclusionInterval, ...]:
        """Return the task-local preprocessing exclusions."""

        return tuple(self._exclusions)

    @property
    def using_preserved_source(self) -> bool:
        """Return whether preview legality is evaluated from the preserved source."""

        return self._use_preserved_source

    def confirm_sample_unrebinned(self, *, confirmed: bool = False) -> bool:
        """Record an explicit task-local coverage assertion for the chosen Sample."""

        if not confirmed:
            return False
        self._sample_unrebinned_confirmed = True
        self._refresh_preview()
        return True

    def confirm_resolution_unrebinned(self, *, confirmed: bool = False) -> bool:
        """Record an explicit task-local coverage assertion for the Resolution."""

        if self.current_resolution is None or not confirmed:
            return False
        self._resolution_unrebinned_confirmed = True
        self._refresh_preview()
        return True

    def _prompt_confirm_sample(self) -> None:
        source = self._selected_sample_source()
        if source.fractional_coverage is not None:
            return
        confirmed = confirm_dialog(
            self,
            "Confirm unre-binned reduced source",
            "Confirm that this reduced Sample has not previously been Q-rebinned. "
            "This establishes fractional coverage F(Q,E) = 1 for this operation.",
            accept_text="Confirm unre-binned source",
        )
        self.confirm_sample_unrebinned(confirmed=confirmed)

    def _prompt_confirm_resolution(self) -> None:
        source = self._selected_resolution_source()
        if source is None or source.fractional_coverage is not None:
            return
        confirmed = confirm_dialog(
            self,
            "Confirm unre-binned Resolution source",
            "Confirm that the associated reduced Resolution has not previously "
            "been Q-rebinned. This establishes fractional coverage F(Q,E) = 1 "
            "for this operation.",
            accept_text="Confirm unre-binned source",
        )
        self.confirm_resolution_unrebinned(confirmed=confirmed)

    def use_preserved_source(self) -> None:
        """Switch preview input without declaring current-grid refinement legal."""

        self._use_preserved_source = True
        self.use_preserved_source_button.hide()
        self._refresh_preview()

    def add_exclusion(self, lower_q: float, upper_q: float) -> bool:
        """Append one core-validated continuous preprocessing interval."""

        try:
            interval = QExclusionInterval(lower_q, upper_q)
        except ValueError as error:
            self.legality_label.setText(str(error))
            return False
        self._exclusions.append(interval)
        self.exclusion_list.addItem(f"{interval.lower_q:.6g} – {interval.upper_q:.6g}")
        self._refresh_preview()
        return True

    def _add_exclusion_from_fields(self) -> None:
        try:
            lower = float(self.exclusion_lower_edit.text())
            upper = float(self.exclusion_upper_edit.text())
        except ValueError:
            self.legality_label.setText("Q-exclusion bounds must be finite numbers")
            return
        if self.add_exclusion(lower, upper):
            self.exclusion_lower_edit.clear()
            self.exclusion_upper_edit.clear()

    def remove_selected_exclusion(self) -> None:
        """Remove only the selected task-local coverage exclusion."""

        row = self.exclusion_list.currentRow()
        if not 0 <= row < len(self._exclusions):
            return
        self._exclusions.pop(row)
        self.exclusion_list.takeItem(row)
        self._refresh_preview()

    def set_target_edges(self, edges: tuple[float, ...]) -> None:
        """Set explicit target edges through the same text-facing editor seam."""

        self._set_explicit_edges(edges, from_regular=False)
        self._set_target_mode(EXPLICIT_VALUES, capture_current=False)

    def set_regular_grid(
        self,
        lower_q: float,
        upper_q: float,
        *,
        step: float | None = None,
        group_count: int | None = None,
    ) -> None:
        """Populate one exclusive regular-grid definition for GUI/test callers."""

        if (step is None) == (group_count is None):
            raise ValueError("supply exactly one of step or group_count")
        self._set_regular_numeric_field("lower", self.regular_lower_edit, lower_q)
        self._set_regular_numeric_field("upper", self.regular_upper_edit, upper_q)
        if step is not None:
            self._regular_driver = "step"
            self._set_regular_numeric_field("step", self.regular_step_edit, step)
        else:
            assert group_count is not None
            self._regular_driver = "groups"
            self._set_regular_group_count(group_count)
        self._set_target_mode(REGULAR_GRID, capture_current=False)
        self._refresh_preview()

    @staticmethod
    def _format_edges(edges: tuple[float, ...]) -> str:
        return "  ".join(f"{value:.12g}" for value in edges)

    def _target_mode_changed(self, mode: str) -> None:
        self._set_target_mode(mode, capture_current=True)

    def _set_target_mode(self, mode: str, *, capture_current: bool) -> None:
        if mode not in {REGULAR_GRID, EXPLICIT_VALUES}:
            raise ValueError("unknown target-grid input mode")
        if capture_current and mode != self._visible_target_mode:
            if self._visible_target_mode == REGULAR_GRID:
                try:
                    q_bins = self._regular_target_q_bins()
                except ValueError:
                    pass
                else:
                    assert q_bins.edges is not None
                    self._set_explicit_edges(
                        tuple(float(value) for value in q_bins.edges),
                        from_regular=True,
                    )
            else:
                try:
                    q_bins = self._explicit_target_q_bins()
                except ValueError:
                    pass
                else:
                    if not self._explicit_originated_from_regular:
                        self._populate_regular_grid(q_bins)
        blocker = QSignalBlocker(self.target_mode_combo)
        self.target_mode_combo.setCurrentText(mode)
        del blocker
        self._visible_target_mode = mode
        regular = mode == REGULAR_GRID
        self.regular_fields.setVisible(regular)
        self.target_edges_edit.setVisible(not regular)
        self._refresh_preview()

    def _set_explicit_edges(
        self,
        edges: tuple[float, ...],
        *,
        from_regular: bool = False,
    ) -> None:
        self._explicit_exact_edges = tuple(float(value) for value in edges)
        self._explicit_originated_from_regular = from_regular
        self._setting_explicit_text = True
        try:
            self.target_edges_edit.setText(self._format_edges(edges))
        finally:
            self._setting_explicit_text = False

    def _explicit_edges_changed(self) -> None:
        if not self._setting_explicit_text:
            self._explicit_exact_edges = None
            self._explicit_originated_from_regular = False
        self._refresh_preview()

    def _set_regular_numeric_field(
        self,
        key: str,
        field: QLineEdit,
        value: float,
    ) -> None:
        self._regular_exact_values[key] = float(value)
        self._setting_regular_numeric_text = True
        try:
            field.setText(f"{value:.12g}")
        finally:
            self._setting_regular_numeric_text = False

    def _clear_regular_numeric_field(self, key: str, field: QLineEdit) -> None:
        self._regular_exact_values.pop(key, None)
        self._setting_regular_numeric_text = True
        try:
            field.clear()
        finally:
            self._setting_regular_numeric_text = False

    def _regular_field_changed(self, key: str) -> None:
        if self._setting_regular_numeric_text:
            return
        if key == "groups":
            if self._setting_regular_group_text:
                return
            self._regular_driver = "groups"
        else:
            self._regular_exact_values.pop(key, None)
            self._regular_driver = "step"
        self._refresh_preview()

    def _parse_finite_field(self, field: QLineEdit, name: str, key: str) -> float:
        if key in self._regular_exact_values:
            value = self._regular_exact_values[key]
        else:
            try:
                value = float(field.text())
            except ValueError as error:
                raise ValueError(f"{name} must be a number") from error
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    def _regular_target_q_bins(self) -> QBins:
        self.regular_coverage_label.clear()
        lower = self._parse_finite_field(
            self.regular_lower_edit,
            "Lower Q edge",
            "lower",
        )
        upper = self._parse_finite_field(
            self.regular_upper_edit,
            "Upper Q edge",
            "upper",
        )
        if self._regular_driver == "step":
            step = self._parse_finite_field(
                self.regular_step_edit,
                "Step ΔQ",
                "step",
            )
            q_bins = fixed_width_q_bins(
                lower_q_edge=lower,
                upper_q_limit=upper,
                step=step,
            )
            self._set_regular_group_count(q_bins.group_count)
        else:
            text = self.regular_group_count_edit.text().strip()
            try:
                group_count = int(text)
            except ValueError as error:
                raise ValueError(
                    "Number of groups must be a positive integer"
                ) from error
            if group_count < 1 or str(group_count) != text:
                raise ValueError("Number of groups must be a positive integer")
            q_bins = uniform_q_bins(
                lower_q_edge=lower,
                upper_q_edge=upper,
                group_count=group_count,
            )
            step = (upper - lower) / group_count
            if not np.isfinite(step) or step <= 0.0:
                raise ValueError("derived Step ΔQ must be finite and positive")
            self._set_regular_numeric_field(
                "step",
                self.regular_step_edit,
                step,
            )
        assert q_bins.edges is not None
        actual_upper = float(q_bins.edges[-1])
        if f"{actual_upper:.12g}" == f"{upper:.12g}":
            self.regular_coverage_label.clear()
        else:
            self.regular_coverage_label.setText(
                f"Requested Upper: {upper:.12g} · "
                f"Actual covered upper edge: {actual_upper:.12g}"
            )
        return q_bins

    def _set_regular_group_count(self, group_count: int) -> None:
        self._setting_regular_group_text = True
        try:
            self.regular_group_count_edit.setText(str(group_count))
        finally:
            self._setting_regular_group_text = False

    def _explicit_target_q_bins(self) -> QBins:
        values = (
            self._explicit_exact_edges
            if self._explicit_exact_edges is not None
            else tuple(
                float(value) for value in parse_q_values(self.target_edges_edit.text())
            )
        )
        return QBins.from_edges(values)

    def _populate_regular_grid(self, q_bins: QBins) -> None:
        assert q_bins.edges is not None
        edges = q_bins.edges
        lower = float(edges[0])
        upper = float(edges[-1])
        self._set_regular_numeric_field("lower", self.regular_lower_edit, lower)
        self._set_regular_numeric_field("upper", self.regular_upper_edit, upper)
        self._set_regular_group_count(q_bins.group_count)
        expected = uniform_q_bins(
            lower_q_edge=lower,
            upper_q_edge=upper,
            group_count=q_bins.group_count,
        )
        assert expected.edges is not None
        tolerance = (
            8.0
            * np.finfo(np.float64).eps
            * max(
                1.0,
                abs(lower),
                abs(upper),
            )
        )
        if np.allclose(edges, expected.edges, rtol=0.0, atol=tolerance):
            self._regular_driver = "groups"
            self._set_regular_numeric_field(
                "step",
                self.regular_step_edit,
                (upper - lower) / q_bins.group_count,
            )
        else:
            self._regular_driver = "step"
            self._clear_regular_numeric_field("step", self.regular_step_edit)

    def target_q_bins(self) -> QBins:
        """Resolve explicit target edges with the public Q-bin constructor."""

        return (
            self._regular_target_q_bins()
            if self.target_mode_combo.currentText() == REGULAR_GRID
            else self._explicit_target_q_bins()
        )

    def specification(self) -> QRebinSpecification:
        """Build the public typed operation request without local legality rules."""

        q_bins = self.target_q_bins()
        assert q_bins.edges is not None
        return QRebinSpecification(
            tuple(float(value) for value in q_bins.edges),
            tuple(self._exclusions),
        )

    def _selected_sample_source(self) -> ReducedDataset:
        return (
            self.preserved_sample
            if self._use_preserved_source
            else project_dataset(self.workflow, self.sample).dataset
        )

    def _selected_resolution_source(self) -> ReducedDataset | None:
        if self.current_resolution is None:
            return None
        if self._use_preserved_source and self.preserved_resolution is not None:
            return self.preserved_resolution
        return project_dataset(self.workflow, self.current_resolution).dataset

    @staticmethod
    def _coverage_text(dataset: ReducedDataset) -> str:
        availability = dataset.fractional_coverage_availability
        if availability is FractionalCoverageAvailability.MISSING:
            return "Missing — explicit unre-binned-source confirmation required"
        assert dataset.fractional_coverage is not None
        origin = dataset.fractional_coverage.origin.value.replace("_", " ")
        return f"Available · {origin}"

    @staticmethod
    def _grid_text(dataset: ReducedDataset) -> str:
        q_bins = dataset.q_bins
        if q_bins is None or q_bins.edges is None:
            return f"{len(dataset.spectra)} Groups · explicit Q edges unavailable"
        return (
            f"{q_bins.group_count} Groups · edges "
            f"{q_bins.edges[0]:.6g} … {q_bins.edges[-1]:.6g} Å⁻¹"
        )

    def _prepared_workflow(
        self,
    ) -> (
        tuple[
            WorkflowProject,
            ProjectDataset,
            ReducedDataset,
            ReducedDataset | None,
        ]
        | None
    ):
        workflow = self.workflow
        sample = project_dataset(workflow, self.sample)
        sample_source = self._selected_sample_source()
        if sample_source.fractional_coverage is None:
            if not self._sample_unrebinned_confirmed:
                return None
            sample_source = sample_source.confirm_unrebinned_source()
        if sample_source is not sample.dataset:
            workflow, sample = replace_project_dataset(
                workflow,
                sample,
                sample_source,
            )

        resolution = applied_resolution(workflow, sample)
        resolution_source = self._selected_resolution_source()
        if resolution is not None and resolution_source is not None:
            if resolution_source.fractional_coverage is None:
                if not self._resolution_unrebinned_confirmed:
                    return None
                resolution_source = resolution_source.confirm_unrebinned_source()
            if resolution_source is not resolution.dataset:
                workflow, _resolution = replace_project_dataset(
                    workflow,
                    resolution,
                    resolution_source,
                )
        return workflow, sample, sample_source, resolution_source

    def _refresh_preview(self) -> None:
        """Run a side-effect-free workflow preview and expose its typed outcome."""

        self.preview_result = None
        self.apply_button.setEnabled(False)
        self.use_preserved_source_button.hide()
        current_source = project_dataset(self.workflow, self.sample).dataset
        sample_source = self._selected_sample_source()
        resolution_source = self._selected_resolution_source()
        source_kind = (
            "Preserved reduced source" if self._use_preserved_source else "Current"
        )
        self.source_label.setText(
            f"{source_kind}: {self.sample_name} · {self._grid_text(sample_source)}"
        )
        if self.preserved_sample is current_source:
            self.lineage_label.setText(
                "Current grid and preserved reduced source are identical. Preview "
                "uses this reduced source, not detector-level raw data."
            )
        else:
            self.lineage_label.setText(
                f"Current grid: {self._grid_text(current_source)} · preserved "
                f"reduced source: {self._grid_text(self.preserved_sample)}"
            )
        self.sample_coverage_label.setText(self._coverage_text(sample_source))
        sample_missing = sample_source.fractional_coverage is None
        self.confirm_sample_button.setVisible(
            sample_missing and not self._sample_unrebinned_confirmed
        )
        if resolution_source is None:
            self.resolution_coverage_label.setText("No associated Resolution")
            self.confirm_resolution_button.hide()
        else:
            name = self.resolution_name or "Resolution"
            self.resolution_coverage_label.setText(
                f"{name} · {self._coverage_text(resolution_source)}"
            )
            self.confirm_resolution_button.setVisible(
                resolution_source.fractional_coverage is None
                and not self._resolution_unrebinned_confirmed
            )

        try:
            specification = self.specification()
        except ValueError as error:
            self.target_preview_label.setText("")
            self.legality_label.setText(f"Target grid not ready · {error}")
            self.association_label.setText(
                "Overall rebin legality has not been checked."
            )
            self._draw_preview(None)
            return
        target_bins = QBins.from_edges(specification.target_edges)
        self.target_preview_label.setText(
            f"{target_bins.group_count} target Groups · explicit edges "
            f"{specification.target_edges[0]:.6g} … "
            f"{specification.target_edges[-1]:.6g}"
        )
        self._draw_preview(target_bins)

        prepared = self._prepared_workflow()
        if prepared is None:
            self.legality_label.setText("Coverage confirmation required")
            self.association_label.setText(
                "Overall rebin legality has not been checked."
            )
            return
        workflow, sample, prepared_sample, prepared_resolution = prepared
        try:
            updated, rebinned_sample, rebinned_resolution = rebin_project_sample_q(
                workflow,
                sample,
                specification,
            )
        except WorkflowError as error:
            self.legality_label.setText(
                "Blocked · " + "\n".join(item.message for item in error.diagnostics)
            )
            refinement_blocked = any(
                detail.code is QRebinDiagnosticCode.TARGET_REFINES_SOURCE
                for item in error.diagnostics
                for detail in item.q_rebin_diagnostics
            )
            self.use_preserved_source_button.setVisible(
                refinement_blocked
                and not self._use_preserved_source
                and self.preserved_sample is not sample_source
            )
            self.association_label.setText(
                "No changes applied; the existing Sample/Resolution state is unchanged."
            )
            return

        self.preview_result = QRebinPreview(
            workflow=updated,
            sample=rebinned_sample,
            resolution=rebinned_resolution,
            specification=specification,
            sample_source=prepared_sample,
            resolution_source=prepared_resolution,
        )
        self.legality_label.setText("Ready · core Q-rebin legality accepted")
        if rebinned_resolution is None:
            self.association_label.setText(
                "Sample will be rebinned; no associated Resolution is applied."
            )
        else:
            self.association_label.setText(
                "Sample and associated Resolution will be rebinned transactionally."
            )
        self.apply_button.setEnabled(True)

    def apply_rebin(self) -> bool:
        """Adopt only the currently successful immutable workflow preview."""

        preview = self.preview_result
        if preview is None or not self._apply_handler(preview):
            return False
        self.accept()
        return True

    def set_target_group(self, group_index: int) -> None:
        """Navigate only the current target/rebinned Group presentation."""

        try:
            q_bins = self.target_q_bins()
        except ValueError:
            return
        if not 0 <= group_index < q_bins.group_count:
            return
        self.current_target_group = group_index
        self._sync_navigation(q_bins)
        self._draw_preview(q_bins)

    def previous_target_group(self) -> None:
        self.set_target_group(self.current_target_group - 1)

    def next_target_group(self) -> None:
        self.set_target_group(self.current_target_group + 1)

    def _group_number_changed(self, group_number: int) -> None:
        if group_number - 1 != self.current_target_group:
            self.set_target_group(group_number - 1)

    def _sync_navigation(self, q_bins: QBins) -> None:
        self.current_target_group = min(
            self.current_target_group,
            q_bins.group_count - 1,
        )
        blocker = QSignalBlocker(self.group_spinbox)
        self.group_spinbox.setRange(1, q_bins.group_count)
        self.group_spinbox.setValue(self.current_target_group + 1)
        del blocker
        self.previous_button.setEnabled(self.current_target_group > 0)
        self.next_button.setEnabled(self.current_target_group + 1 < q_bins.group_count)
        self.current_q_label.setText(
            f"Q = {q_bins.q_values[self.current_target_group]:.6g} Å⁻¹"
        )

    def _draw_preview(self, target_bins: QBins | None) -> None:
        figure = self.preview_canvas.figure
        figure.clear()
        if target_bins is None:
            self.visible_label_indices = ()
            self.preview_canvas.draw_idle()  # type: ignore[no-untyped-call]
            return
        self._sync_navigation(target_bins)
        grid = figure.add_gridspec(2, 1, height_ratios=(4.0, 1.0), hspace=0.12)
        heatmap_axes = figure.add_subplot(grid[0])
        navigator_axes = figure.add_subplot(grid[1], sharex=heatmap_axes)
        self._draw_source_heatmap(heatmap_axes, target_bins)
        self._draw_target_navigation(navigator_axes, target_bins)
        self.preview_canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _draw_source_heatmap(self, axes: Axes, target_bins: QBins) -> None:
        source = self.preserved_sample
        source_bins = source.q_bins
        if source_bins is None or source_bins.edges is None:
            axes.text(
                0.5,
                0.5,
                "Preserved source has no explicit Q-bin edges",
                transform=axes.transAxes,
                ha="center",
                va="center",
            )
            axes.set_axis_off()
            return
        finite_values = np.concatenate(
            [
                spectrum.intensity[np.isfinite(spectrum.intensity)]
                for spectrum in source.spectra
            ]
        )
        norm = None
        if finite_values.size:
            norm = Normalize(
                vmin=float(np.min(finite_values)),
                vmax=float(np.max(finite_values)),
            )
        for lower, upper, spectrum in zip(
            source_bins.edges[:-1],
            source_bins.edges[1:],
            source.spectra,
            strict=True,
        ):
            finite_energy = np.isfinite(spectrum.energy)
            if not np.any(finite_energy):
                continue
            energy = spectrum.energy[finite_energy]
            intensity = spectrum.intensity[finite_energy]
            order = np.argsort(energy, kind="stable")
            energy = energy[order]
            intensity = intensity[order]
            if energy.size > 1 and np.any(np.diff(energy) <= 0.0):
                continue
            axes.pcolormesh(
                np.asarray([lower, upper]),
                _center_cell_edges(energy),
                np.ma.masked_invalid(intensity).reshape(-1, 1),
                cmap="jet",
                norm=norm,
                shading="flat",
                edgecolors="none",
                antialiased=False,
            )
        assert target_bins.edges is not None
        for edge in target_bins.edges:
            axes.axvline(edge, color="#ffffff", linewidth=1.0, alpha=0.9)
        lower = target_bins.edges[self.current_target_group]
        upper = target_bins.edges[self.current_target_group + 1]
        axes.axvspan(
            lower,
            upper,
            facecolor="#ffffff",
            edgecolor="#314b63",
            alpha=0.18,
            linewidth=1.2,
        )
        for exclusion in self._exclusions:
            axes.axvspan(
                exclusion.lower_q,
                exclusion.upper_q,
                facecolor="#a54a4a",
                alpha=0.2,
                hatch="//",
            )
        axes.set_xlim(float(source_bins.edges[0]), float(source_bins.edges[-1]))
        axes.set_ylabel(f"Energy ({source.spectra[0].energy_unit})")
        axes.tick_params(axis="x", labelbottom=False)
        axes.set_title(
            f"Preserved reduced source · {len(source.spectra)} Groups",
            loc="left",
            fontsize=9,
        )

    def _draw_target_navigation(self, axes: Axes, target_bins: QBins) -> None:
        q_values = target_bins.q_values
        axes.plot(q_values, np.zeros_like(q_values), color="#727272", linewidth=1.0)
        available_width = max(float(axes.bbox.width), 1.0)
        widest = (
            max(
                max(len(str(target_bins.group_count)), 1),
                max(len(f"{value:.4g}") for value in q_values),
            )
            * 8.0
            * 0.62
            * axes.figure.dpi
            / 72.0
        )
        self.visible_label_indices = adaptive_navigation_indices(
            target_bins.group_count,
            available_width,
            widest,
        )
        self.discrete_markers_visible = navigation_markers_are_separable(
            target_bins.group_count,
            available_width,
            axes.figure.dpi,
        )
        if self.discrete_markers_visible:
            axes.scatter(q_values, np.zeros_like(q_values), color="#8d969d", s=14)
        axes.scatter(
            [q_values[self.current_target_group]],
            [0.0],
            color="#496d91",
            s=38,
            zorder=3,
        )
        indices = np.asarray(self.visible_label_indices, dtype=np.int64)
        axes.set_xticks(q_values[indices])
        axes.set_xticklabels(
            [str(index + 1) for index in self.visible_label_indices],
            fontsize=7,
        )
        axes.xaxis.tick_top()
        q_axis = axes.secondary_xaxis("bottom")
        q_axis.set_xticks(q_values[indices])
        q_axis.set_xticklabels(
            [f"{q_values[index]:.4g}" for index in self.visible_label_indices],
            fontsize=7,
        )
        q_axis.set_xlabel("Q (Å⁻¹)", fontsize=8)
        axes.set_ylim(-0.25, 0.25)
        axes.set_yticks([])
        for spine in axes.spines.values():
            spine.set_visible(False)

    def _on_preview_press(self, event: MouseEvent) -> None:
        if (
            event.button is not MouseButton.LEFT
            or event.inaxes is None
            or len(self.preview_canvas.figure.axes) < 2
            or event.inaxes is not self.preview_canvas.figure.axes[1]
            or event.xdata is None
        ):
            return
        try:
            q_bins = self.target_q_bins()
        except ValueError:
            return
        index = int(np.argmin(np.abs(q_bins.q_values - float(event.xdata))))
        self.set_target_group(index)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "preview_canvas"):
            try:
                target_bins = self.target_q_bins()
            except ValueError:
                return
            self._draw_preview(target_bins)


def _center_cell_edges(centers: np.ndarray) -> np.ndarray:
    """Derive display-only heatmap cell edges from ordered energy coordinates."""

    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    middle = (centers[:-1] + centers[1:]) / 2.0
    return np.concatenate(
        (
            [centers[0] - (middle[0] - centers[0])],
            middle,
            [centers[-1] + (centers[-1] - middle[-1])],
        )
    )
