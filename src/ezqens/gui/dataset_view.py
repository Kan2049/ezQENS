"""Read-only scientific view for one imported reduced dataset."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.colors import ListedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from PySide6.QtCore import QPoint, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QActionGroup, QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ezqens.domain import DiagnosticSeverity, ReducedDataset
from ezqens.fitting import (
    ComponentIdentity,
    FitResult,
    ManualModelPreview,
    ModelEvaluation,
)
from ezqens.gui.masking import BoundaryCoordinates
from ezqens.gui.q_editor import QAssignmentEditor
from ezqens.gui.scientific_canvas import (
    SCIENTIFIC_BACKGROUND,
    ScientificCanvas,
    log_display_values,
    symlog_linthresh,
    zoom_limits,
)
from ezqens.gui.theme import DEFAULT_LAYOUT_TOKENS
from ezqens.gui.workspace import (
    Q_METHOD_MIME_TYPE,
    q_method_drag_indices_from_mime_data,
)
from ezqens.preprocessing import FittingSelection
from ezqens.workflow import ManualComponentKind

Q_DISPLAY_UNIT = "Å⁻¹"
OVERVIEW_COLORMAP = "jet"
OVERVIEW_ACCENT_COLOR = "#496d91"
OVERVIEW_ACTIVE_COLOR = "#ffffff"
OVERVIEW_ACTIVE_ALPHA = 0.26
NAVIGATOR_INACTIVE_COLOR = "#8d969d"
SPECTRUM_ZOOM_DRAG_THRESHOLD_PX = 4.0
BOUNDARY_HIT_RADIUS_LOGICAL_PX = 8.0


def axis_label(quantity: str, unit: str) -> str:
    """Return a truthful axis label without decorating an unknown unit."""

    if unit.strip().casefold() == "unknown":
        return quantity
    return f"{quantity} ({unit})"


class ReducedDatasetView(QWidget):
    """Compact group navigation around a real Matplotlib scientific view."""

    group_changed = Signal(int)
    q_assignment_requested = Signal()
    units_requested = Signal()
    q_method_dropped = Signal(int, int)
    mask_edit_requested = Signal()
    mask_rectangle_requested = Signal(int, float, float, float, float, str)
    mask_lasso_requested = Signal(int, object, str)
    mask_boundary_previewed = Signal(int, object, float)
    mask_boundary_requested = Signal(int, object, float)
    mask_operation_feedback_changed = Signal(str)
    manual_fit_requested = Signal()
    manual_component_completed = Signal(object, object)
    manual_component_preview_requested = Signal(object, object)
    manual_component_cancelled = Signal()
    y_scale_changed = Signal(str)
    heatmap_scale_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("reducedDatasetView")
        self.setAcceptDrops(True)
        self.overview_canvas = ScientificCanvas()
        self.overview_canvas.setObjectName("overviewScientificCanvas")
        self.overview_canvas.setMinimumHeight(48)
        self.canvas = ScientificCanvas()
        self.q_editor = QAssignmentEditor()
        self.dataset: ReducedDataset | None = None
        self.overview_source_dataset: ReducedDataset | None = None
        self.selection: FittingSelection | None = None
        self.mask_inspection_mode = False
        self.mask_edit_available = False
        self._mask_tool: str | None = None
        self.mask_operation = "exclude"
        self.mask_operation_feedback = "Exclude"
        self._mask_points: list[tuple[float, float]] = []
        self._boundary_drag_side: str | None = None
        self._boundary_handle_artists: dict[str, Line2D] = {}
        self._boundary_handle_positions: dict[str, float] = {}
        self._mask_boundary_coordinates: tuple[BoundaryCoordinates, ...] | None = None
        self._mask_preview_rectangle: Rectangle | None = None
        self._mask_preview_line: Line2D | None = None
        self._manual_component_kind: ManualComponentKind | None = None
        self._manual_interaction_start: tuple[float, float] | None = None
        self._manual_drag_point: tuple[float, float] | None = None
        self._manual_pointer_down = False
        self._manual_preview: ManualModelPreview | None = None
        self._manual_fit_result: FitResult | None = None
        self._manual_pending_evaluation: ModelEvaluation | None = None
        self._manual_emphasized_components: frozenset[ComponentIdentity] = frozenset()
        self.current_group_index = 0
        self.overview_axes: Axes | None = None
        self.overview_q_axis: object | None = None
        self.navigator_axes: Axes | None = None
        self.spectrum_axes: Axes | None = None
        self.residual_axes: Axes | None = None
        self.standardized_residual_line: Line2D | None = None
        self.overview_meshes: tuple[QuadMesh, ...] = ()
        self.overview_x_cell_bounds: tuple[tuple[float, float], ...] = ()
        self.overview_active_highlight: Rectangle | None = None
        self.navigator_active_highlight: PathCollection | None = None
        self._overview_q_label_artists: tuple[Text, ...] = ()
        self._navigator_q_label_artists: tuple[Text, ...] = ()
        self.spectrum_q_title: Text | None = None
        self.spectrum_log_empty_message: Text | None = None
        self._spectrum_x_limits: tuple[float, float] | None = None
        self._spectrum_y_limits: tuple[float, float] | None = None
        self._spectrum_zoom_start: tuple[float, float] | None = None
        self._spectrum_zoom_start_display: tuple[float, float] | None = None
        self._spectrum_zoom_rectangle: Rectangle | None = None
        self.overview_visible = True
        self._navigation_drag_active = False
        self.y_scale = "linear"
        self.heatmap_scale = "linear"
        self.y_range_locked = True
        self._locked_y_limits: tuple[float, float] | None = None
        self._overview_surface = SCIENTIFIC_BACKGROUND
        self._overview_text = "#292928"
        self._overview_border = "#deded9"
        self._overview_accent = OVERVIEW_ACCENT_COLOR
        self._overview_masked = "#8d819d"

        self.previous_button = QToolButton()
        self.previous_button.setObjectName("previousGroupButton")
        self.previous_button.setText("‹")
        self.previous_button.setToolTip("Previous group")
        self.previous_button.clicked.connect(self.previous_group)

        self.group_navigation_label = QLabel("Group")
        self.group_navigation_label.setObjectName("groupNavigationLabel")
        self.group_navigation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.group_spinbox = QSpinBox()
        self.group_spinbox.setObjectName("groupSpinBox")
        self.group_spinbox.setRange(1, 1)
        self.group_spinbox.setKeyboardTracking(False)
        self.group_spinbox.setButtonSymbols(
            QAbstractSpinBox.ButtonSymbols.NoButtons,
        )
        self.group_spinbox.setFixedWidth(58)
        self.group_spinbox.valueChanged.connect(self._group_number_changed)

        self.current_q_label = QLabel()
        self.current_q_label.setObjectName("currentQLabel")
        self.current_q_label.setProperty("secondary", True)
        self.current_q_label.setToolTip("Double-click: edit Q assignment")
        self.current_q_label.hide()

        self.next_button = QToolButton()
        self.next_button.setObjectName("nextGroupButton")
        self.next_button.setText("›")
        self.next_button.setToolTip("Next group")
        self.next_button.clicked.connect(self.next_group)

        self.metadata_status_label = QLabel()
        self.metadata_status_label.setObjectName("datasetMetadataStatus")
        self.metadata_status_label.setProperty("muted", True)

        self.overview_toggle_button = QToolButton()
        self.overview_toggle_button.setObjectName("overviewToggleButton")
        self.overview_toggle_button.setToolTip("Show or hide the intensity overview")
        self.overview_toggle_button.clicked.connect(self._toggle_overview)

        navigation = QHBoxLayout()
        navigation.setContentsMargins(0, 0, 0, 0)
        navigation.setSpacing(5)
        navigation.addWidget(self.previous_button)
        navigation.addWidget(self.group_navigation_label)
        navigation.addWidget(self.group_spinbox)
        navigation.addWidget(self.next_button)
        navigation.addWidget(self.current_q_label)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(5)
        actions.addWidget(self.metadata_status_label)
        actions.addWidget(self.overview_toggle_button)

        self.controls_container = QWidget(self)
        self.controls_container.setObjectName("datasetControls")
        controls = QGridLayout(self.controls_container)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(5)
        controls.setColumnStretch(0, 1)
        controls.setColumnStretch(2, 1)
        controls.addLayout(navigation, 0, 1)
        controls.addLayout(actions, 0, 2, Qt.AlignmentFlag.AlignRight)
        self._navigation_layout = navigation
        self._actions_layout = actions
        self._controls_layout = controls

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.controls_container)
        self.overview_provenance_label = QLabel()
        self.overview_provenance_label.setProperty("secondary", True)
        self.overview_provenance_label.setWordWrap(True)
        self.overview_provenance_label.hide()
        layout.addWidget(self.overview_provenance_label)
        layout.addWidget(self.overview_canvas)
        layout.addWidget(self.q_editor)
        self.manual_interaction_instruction = QLabel()
        self.manual_interaction_instruction.setObjectName(
            "manualInteractionInstruction",
        )
        self.manual_interaction_instruction.setWordWrap(True)
        self.manual_interaction_instruction.hide()
        layout.addWidget(self.manual_interaction_instruction)
        self.spectrum_interaction_frame = QFrame(self)
        self.spectrum_interaction_frame.setObjectName("spectrumInteractionFrame")
        self.spectrum_interaction_frame.setProperty("manualInteractionActive", False)
        spectrum_layout = QVBoxLayout(self.spectrum_interaction_frame)
        spectrum_layout.setContentsMargins(1, 1, 1, 1)
        spectrum_layout.setSpacing(0)
        spectrum_layout.addWidget(self.canvas)
        layout.addWidget(self.spectrum_interaction_frame, 1)
        self.content_layout = layout

        self._update_overview_toggle()
        self.canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.canvas.setToolTip(
            "Double-click the Energy label to edit declared source units",
        )
        self.canvas.customContextMenuRequested.connect(self._show_spectrum_context_menu)
        self.overview_canvas.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.overview_canvas.setToolTip(
            "Click to select a Group. Double-click a Q label to edit Q assignment.",
        )
        self.overview_canvas.customContextMenuRequested.connect(
            self._show_overview_context_menu,
        )
        for widget in (
            self.previous_button,
            self.group_navigation_label,
            self.group_spinbox,
            self.next_button,
            self.current_q_label,
        ):
            widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            widget.customContextMenuRequested.connect(
                lambda position, source=widget: self._show_navigation_q_context_menu(
                    source,
                    position,
                ),
            )
        self.overview_canvas.mpl_connect("button_press_event", self._on_button_press)
        self.overview_canvas.mpl_connect(
            "button_press_event",
            self._on_overview_units_press,
        )
        self.overview_canvas.mpl_connect("motion_notify_event", self._on_mouse_motion)
        self.overview_canvas.mpl_connect(
            "button_release_event",
            self._on_button_release,
        )
        self.overview_canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_spectrum_button_press)
        self.canvas.mpl_connect("button_press_event", self._on_spectrum_q_press)
        self.canvas.mpl_connect(
            "button_press_event",
            self._on_spectrum_units_press,
        )
        self.canvas.mpl_connect("motion_notify_event", self._on_spectrum_mouse_motion)
        self.canvas.mpl_connect(
            "button_release_event",
            self._on_spectrum_button_release,
        )
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("key_press_event", self._on_spectrum_key_press)
        self.q_editor.q_center_draft_changed.connect(self._draw_q_center_draft)
        for control in (
            self.previous_button,
            self.group_spinbox,
            self.next_button,
            self.overview_toggle_button,
        ):
            control.setMinimumHeight(DEFAULT_LAYOUT_TOKENS.control_height)
        self._navigation_layout.setSpacing(DEFAULT_LAYOUT_TOKENS.row_spacing)
        self._actions_layout.setSpacing(DEFAULT_LAYOUT_TOKENS.row_spacing)
        self._controls_layout.setSpacing(DEFAULT_LAYOUT_TOKENS.row_spacing)
        self.content_layout.setSpacing(DEFAULT_LAYOUT_TOKENS.section_spacing)

    def open_dataset(
        self,
        dataset: ReducedDataset,
        *,
        overview_source: ReducedDataset | None = None,
        group_index: int = 0,
        y_scale: str = "linear",
        heatmap_scale: str = "linear",
    ) -> None:
        """Open real imported data at a validated Group without changing arrays."""

        if not 0 <= group_index < len(dataset.spectra):
            raise ValueError("group index is outside the dataset")
        if y_scale not in {"linear", "symlog", "log"}:
            raise ValueError("y scale must be 'linear', 'symlog', or 'log'")
        if heatmap_scale not in {"linear", "log"}:
            raise ValueError("heatmap scale must be 'linear' or 'log'")

        self.cancel_manual_component_interaction()
        self.dataset = dataset
        self.overview_source_dataset = overview_source
        self.selection = None
        self._mask_boundary_coordinates = None
        self._manual_preview = None
        self._manual_fit_result = None
        self._manual_pending_evaluation = None
        self.mask_inspection_mode = False
        self.current_group_index = group_index
        self._spectrum_x_limits = None
        self._spectrum_y_limits = None
        self.y_scale = y_scale
        self.heatmap_scale = heatmap_scale
        self.y_range_locked = True
        self._locked_y_limits = None
        blocker = QSignalBlocker(self.group_spinbox)
        self.group_spinbox.setRange(1, len(dataset.spectra))
        self.group_spinbox.setValue(group_index + 1)
        del blocker
        self._update_metadata_status()
        self._draw()

    def set_selection(self, selection: FittingSelection | None) -> None:
        """Set the validated effective selection used for display only."""

        if selection is not None and self.dataset is not None:
            if len(selection.dataset.spectra) != len(self.dataset.spectra):
                raise ValueError("selection does not match the open dataset")
        self.selection = selection
        if self.dataset is not None:
            self._draw()

    def set_mask_boundary_coordinates(
        self,
        coordinates: tuple[BoundaryCoordinates, ...] | None,
    ) -> None:
        """Set immutable Boundary-owned coordinates for the next Mask redraw."""

        if coordinates is not None and self.dataset is not None:
            if len(coordinates) != len(self.dataset.spectra):
                raise ValueError("boundary coordinates do not match the open dataset")
        self._mask_boundary_coordinates = coordinates

    def set_mask_inspection_mode(self, enabled: bool) -> None:
        """Reveal excluded points only while the focused Mask task is active."""

        if self.mask_inspection_mode == enabled:
            return
        self.mask_inspection_mode = enabled
        if self.dataset is not None:
            self._draw()

    def set_mask_edit_available(self, available: bool) -> None:
        """Expose Mask commands only when the application has a saved baseline."""

        self.mask_edit_available = available

    def set_mask_tool(self, tool: str | None) -> None:
        """Select one compact task-only plot editing tool."""

        if tool not in {None, "boundary", "rectangle", "lasso"}:
            raise ValueError("unknown mask editing tool")
        self._cancel_spectrum_zoom()
        self._cancel_mask_gesture()
        self._mask_tool = tool
        if tool is None:
            self.canvas.unsetCursor()

    def set_mask_operation(self, operation: str) -> None:
        """Set the persistent Rectangle/Lasso operation without touching data."""

        if operation not in {"exclude", "restore"}:
            raise ValueError("mask operation must be 'exclude' or 'restore'")
        self.mask_operation = operation
        self._set_mask_operation_feedback(operation, inverted=False)

    def open_q_editor(self) -> None:
        """Reveal the one inline Q editor directly below the overview."""

        self.q_editor.open_for_dataset(self._require_dataset())

    def clear_dataset(self) -> None:
        """Clear the scientific view when its Workspace object is removed."""

        self.cancel_manual_component_interaction()
        self._cancel_spectrum_zoom(redraw=False)
        self.dataset = None
        self.overview_source_dataset = None
        self.overview_provenance_label.clear()
        self.overview_provenance_label.hide()
        self.selection = None
        self._manual_preview = None
        self._manual_fit_result = None
        self._manual_pending_evaluation = None
        self.overview_axes = None
        self.overview_q_axis = None
        self.navigator_axes = None
        self.spectrum_axes = None
        self.residual_axes = None
        self.standardized_residual_line = None
        self.overview_meshes = ()
        self.overview_x_cell_bounds = ()
        self.overview_active_highlight = None
        self.navigator_active_highlight = None
        self._overview_q_label_artists = ()
        self._navigator_q_label_artists = ()
        self.spectrum_q_title = None
        self._spectrum_x_limits = None
        self._spectrum_y_limits = None
        self.q_editor.hide()
        self.overview_canvas.figure.clear()
        self.canvas.figure.clear()
        self.overview_canvas.draw_idle()  # type: ignore[no-untyped-call]
        self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasFormat(Q_METHOD_MIME_TYPE):
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # noqa: N802
        if event.mimeData().hasFormat(Q_METHOD_MIME_TYPE):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        if not event.mimeData().hasFormat(Q_METHOD_MIME_TYPE):
            event.ignore()
            return
        project_index, method_index = q_method_drag_indices_from_mime_data(
            event.mimeData().data(Q_METHOD_MIME_TYPE),
        )
        if project_index is None or method_index is None:
            event.ignore()
            return
        self.q_method_dropped.emit(project_index, method_index)
        event.acceptProposedAction()

    def replace_dataset(
        self,
        dataset: ReducedDataset,
        *,
        overview_source: ReducedDataset | None = None,
        allow_group_count_change: bool = False,
    ) -> None:
        """Refresh an opened dataset and its optional preserved overview source."""

        current_dataset = self._require_dataset()
        if not allow_group_count_change and len(dataset.spectra) != len(
            current_dataset.spectra
        ):
            raise ValueError("dataset replacement must preserve the number of groups")
        self.dataset = dataset
        self.overview_source_dataset = overview_source
        if self.current_group_index >= len(dataset.spectra):
            self.current_group_index = 0
        blocker = QSignalBlocker(self.group_spinbox)
        self.group_spinbox.setRange(1, len(dataset.spectra))
        self.group_spinbox.setValue(self.current_group_index + 1)
        del blocker
        self._update_metadata_status()
        self._draw()

    def set_manual_preview(
        self,
        preview: ManualModelPreview | None,
        *,
        emphasized_components: frozenset[ComponentIdentity] = frozenset(),
    ) -> None:
        """Draw only the typed workflow/core preview supplied by the task layer."""

        self._manual_preview = preview
        self._manual_pending_evaluation = None
        self._manual_emphasized_components = emphasized_components
        if self.dataset is not None:
            self._draw_spectrum_figure(self.dataset)
            self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def set_manual_fit_result(self, result: FitResult | None) -> None:
        """Show only the current workflow/core FitResult evaluation."""

        self._manual_fit_result = result
        self._manual_pending_evaluation = None
        if self.dataset is not None:
            self._draw_spectrum_figure(self.dataset)
            self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def set_manual_display_state(
        self,
        preview: ManualModelPreview | None,
        result: FitResult | None,
    ) -> None:
        """Redraw once from one coherent authoritative Manual display state."""

        self._manual_preview = preview
        self._manual_fit_result = result
        self._manual_pending_evaluation = None
        self._manual_emphasized_components = frozenset()
        if self.dataset is not None:
            self._draw_spectrum_figure(self.dataset)
            self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def set_manual_pending_preview(
        self,
        evaluation: ModelEvaluation | None,
    ) -> None:
        """Draw only a workflow-evaluated transient component proposal."""

        self._manual_pending_evaluation = evaluation
        if self.dataset is not None:
            self._draw_spectrum_figure(self.dataset)
            self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def begin_manual_component_interaction(self, kind: ManualComponentKind) -> None:
        """Enter a geometry-only component interaction without creating a model."""

        self._reset_manual_interaction_feedback()
        self._manual_component_kind = kind
        self._cancel_spectrum_zoom()
        self._cancel_mask_gesture()
        self.canvas.setCursor(Qt.CursorShape.CrossCursor)
        self.setProperty("manualInteractionActive", True)
        self.spectrum_interaction_frame.setProperty("manualInteractionActive", True)
        self._refresh_dynamic_style(self.spectrum_interaction_frame)
        self.manual_interaction_instruction.setText(
            self._manual_interaction_text(kind),
        )
        self.manual_interaction_instruction.show()
        for widget in (
            self.controls_container,
            self.overview_canvas,
            self.q_editor,
        ):
            effect = QGraphicsOpacityEffect(widget)
            effect.setOpacity(0.58)
            widget.setGraphicsEffect(effect)

    def cancel_manual_component_interaction(self) -> None:
        """Discard an incomplete geometry gesture with no workflow mutation."""

        if self._manual_component_kind is None:
            return
        self._manual_component_kind = None
        self._reset_manual_interaction_feedback()
        self.manual_component_cancelled.emit()

    @staticmethod
    def _refresh_dynamic_style(widget: QWidget) -> None:
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()

    @staticmethod
    def _manual_interaction_text(kind: ManualComponentKind) -> str:
        if kind is ManualComponentKind.ELASTIC:
            return "Add δ · Press and drag the peak position · Esc to cancel"
        if kind is ManualComponentKind.LORENTZIAN:
            return (
                "Add Lorentzian · Press peak center and drag to set width "
                "· Esc to cancel"
            )
        return "Add Background · Press and drag the baseline · Esc to cancel"

    def _reset_manual_interaction_feedback(self) -> None:
        self._manual_pending_evaluation = None
        self._manual_interaction_start = None
        self._manual_drag_point = None
        self._manual_pointer_down = False
        self.canvas.unsetCursor()
        self.setProperty("manualInteractionActive", False)
        self.spectrum_interaction_frame.setProperty("manualInteractionActive", False)
        self._refresh_dynamic_style(self.spectrum_interaction_frame)
        self.manual_interaction_instruction.hide()
        for widget in (
            self.controls_container,
            self.overview_canvas,
            self.q_editor,
        ):
            effect = widget.graphicsEffect()
            if effect is not None:
                effect.setEnabled(False)

    def set_overview_theme(
        self,
        *,
        surface: str,
        text: str,
        border: str,
        accent: str,
    ) -> None:
        """Apply application-chrome colors to overview-only navigation surfaces."""

        self._overview_surface = surface
        self._overview_text = text
        self._overview_border = border
        self._overview_accent = accent
        if self.dataset is not None:
            self._draw()

    @property
    def current_group_description(self) -> str:
        """Return the visible Group/Q context for the current scientific view."""

        return self._group_description(self._require_dataset())

    def set_current_group(self, group_index: int) -> None:
        """Select a zero-based group and synchronize controls and plots."""

        self.cancel_manual_component_interaction()
        dataset = self._require_dataset()
        if not 0 <= group_index < len(dataset.spectra):
            raise ValueError("group index is outside the dataset")
        if self.spectrum_axes is not None:
            self._spectrum_x_limits = self.spectrum_axes.get_xlim()
            if self.y_range_locked:
                self._locked_y_limits = self.spectrum_axes.get_ylim()
            else:
                self._spectrum_y_limits = None
        self.current_group_index = group_index
        # The application projects the new Group's Manual state after the signal.
        # Clear the old Group's presentation before this intermediate redraw.
        self._manual_preview = None
        self._manual_fit_result = None
        self._manual_pending_evaluation = None
        blocker = QSignalBlocker(self.group_spinbox)
        self.group_spinbox.setValue(group_index + 1)
        del blocker
        self._draw_current_group()
        self.group_changed.emit(group_index)

    def previous_group(self) -> None:
        """Move to the preceding group when one exists."""

        if self.dataset is not None and self.current_group_index > 0:
            self.set_current_group(self.current_group_index - 1)

    def next_group(self) -> None:
        """Move to the following group when one exists."""

        if self.dataset is not None and self.current_group_index + 1 < len(
            self.dataset.spectra
        ):
            self.set_current_group(self.current_group_index + 1)

    def select_group_from_overview(self, coordinate: float) -> None:
        """Select the rendered overview cell containing a real x coordinate."""

        dataset = self._require_dataset()
        coordinates = self._overview_x_coordinates(dataset)
        group_index = _overview_group_index_at_coordinate(
            coordinate,
            coordinates,
            self.overview_x_cell_bounds,
        )
        if group_index != self.current_group_index:
            self.set_current_group(group_index)

    def set_overview_visible(self, visible: bool) -> None:
        """Show or hide the overview while retaining the current discrete group."""

        if self.overview_visible == visible:
            return
        self.overview_visible = visible
        if self.dataset is not None:
            self._draw()
        else:
            self._update_overview_toggle()

    def reset_view(self) -> None:
        """Reset independent spectrum and overview axes to their data limits."""

        self._cancel_spectrum_zoom()
        self._spectrum_x_limits = None
        self._spectrum_y_limits = None
        self._locked_y_limits = None
        if self.dataset is not None:
            self._draw()
            if self.y_range_locked and self.spectrum_axes is not None:
                self._locked_y_limits = self.spectrum_axes.get_ylim()

    def _draw(self) -> None:
        dataset = self._require_dataset()
        source = self.overview_source_dataset
        self.overview_provenance_label.setVisible(source is not None)
        if source is not None:
            self.overview_provenance_label.setText(
                f"Preserved source · {len(source.spectra)} groups"
                f"   |   Current Q grouping · {len(dataset.spectra)} bins"
            )
        overview_figure = self.overview_canvas.figure
        overview_figure.clear()
        self._overview_q_label_artists = ()
        self._navigator_q_label_artists = ()
        overview_figure.set_facecolor(self._overview_surface)
        if self.overview_visible:
            self.overview_axes = overview_figure.add_subplot(111)
            self.navigator_axes = None
            self.navigator_active_highlight = None
            self.overview_axes.set_facecolor(self._overview_surface)
            self._draw_overview(
                self.overview_source_dataset or dataset,
                dataset,
                self.overview_axes,
            )
        else:
            self.overview_axes = None
            self.overview_q_axis = None
            self.overview_meshes = ()
            self.overview_x_cell_bounds = ()
            self.overview_active_highlight = None
            self.navigator_axes = overview_figure.add_subplot(111)
            self._draw_discrete_navigator(dataset, self.navigator_axes)

        self._draw_spectrum_figure(dataset)
        self._update_navigation_state(dataset)
        self._update_overview_canvas_height()
        self.overview_canvas.draw_idle()  # type: ignore[no-untyped-call]
        self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _draw_current_group(self) -> None:
        """Redraw Group-dependent artists without rebuilding the overview."""

        dataset = self._require_dataset()
        self._update_overview_group_presentation(dataset)
        self._draw_spectrum_figure(dataset)
        self._update_navigation_state(dataset)
        self.overview_canvas.draw_idle()  # type: ignore[no-untyped-call]
        self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _update_overview_group_presentation(self, dataset: ReducedDataset) -> None:
        """Move the active overview marker and refresh its sparse labels."""

        if self.overview_visible:
            if self.overview_active_highlight is None or self.overview_axes is None:
                return
            lower, upper = self.overview_x_cell_bounds[self.current_group_index]
            self.overview_active_highlight.set_x(lower)
            self.overview_active_highlight.set_width(upper - lower)
            self._update_expanded_overview_labels(dataset, self.overview_axes)
            return
        if self.navigator_active_highlight is None or self.navigator_axes is None:
            return
        coordinate = float(self.current_group_index + 1)
        self.navigator_active_highlight.set_offsets(np.asarray([[coordinate, 0.0]]))
        self._update_navigator_labels(dataset, self.navigator_axes)

    def _draw_spectrum_figure(self, dataset: ReducedDataset) -> None:
        """Rebuild only the white Spectrum/result figure role."""

        self._cancel_spectrum_zoom(redraw=False)
        spectrum_figure = self.canvas.figure
        spectrum_figure.clear()
        self.spectrum_q_title = None
        self.spectrum_log_empty_message = None
        spectrum_figure.set_facecolor(SCIENTIFIC_BACKGROUND)
        self.residual_axes = None
        self.standardized_residual_line = None
        if self._manual_fit_result is None:
            self.spectrum_axes = spectrum_figure.add_subplot(111)
        else:
            result_grid = spectrum_figure.add_gridspec(
                2,
                1,
                height_ratios=(4.0, 1.0),
                hspace=0.04,
            )
            self.spectrum_axes = spectrum_figure.add_subplot(result_grid[0])
            self.residual_axes = spectrum_figure.add_subplot(
                result_grid[1],
                sharex=self.spectrum_axes,
            )
        self.spectrum_axes.set_facecolor(SCIENTIFIC_BACKGROUND)
        self._draw_spectrum(dataset, self.spectrum_axes)
        if self.residual_axes is not None:
            self._draw_standardized_residual(
                dataset,
                self.residual_axes,
                self._manual_fit_result,
            )

    def _draw_standardized_residual(
        self,
        dataset: ReducedDataset,
        axes: Axes,
        result: FitResult | None,
    ) -> None:
        """Draw only the standardized residual supplied by the current FitResult."""

        if result is None:
            return
        axes.set_facecolor(SCIENTIFIC_BACKGROUND)
        (self.standardized_residual_line,) = axes.plot(
            result.evaluation.energy,
            result.standardized_residuals,
            color="#496d91",
            linewidth=0.9,
            marker=".",
            markersize=3.0,
            zorder=3,
        )
        axes.axhline(0.0, color="#727272", linewidth=0.7, alpha=0.7, zorder=2)
        spectrum = dataset.spectra[self.current_group_index]
        axes.set_xlabel(axis_label("Energy", spectrum.energy_unit))
        axes.set_ylabel("Std. residual")
        axes.grid(True, color="#ededed", linewidth=0.5)
        if self.spectrum_axes is not None:
            self.spectrum_axes.set_xlabel("")
            self.spectrum_axes.tick_params(axis="x", labelbottom=False)

    def _update_overview_canvas_height(self) -> None:
        """Reserve compact, DPI-aware space for the overview figure role."""

        line_count = 9 if self.overview_visible else 4
        self.overview_canvas.setFixedHeight(self.fontMetrics().height() * line_count)

    def _draw_spectrum(self, dataset: ReducedDataset, axes: Axes) -> None:
        spectrum = dataset.spectra[self.current_group_index]
        finite_coordinates = np.isfinite(spectrum.energy) & np.isfinite(
            spectrum.intensity
        )
        excluded = self._effective_excluded_mask(spectrum.energy.size)
        retained_points = finite_coordinates & ~excluded
        visible_points = retained_points
        if self.y_scale == "log":
            axes.set_yscale("log", nonpositive="mask")
            visible_points = visible_points & (spectrum.intensity > 0.0)
        _plot_visible_spectrum_segments(
            axes,
            spectrum.energy,
            spectrum.intensity,
            visible_points,
        )
        uncertainty_points = visible_points & ~spectrum.invalid_uncertainty_mask
        if np.any(uncertainty_points):
            axes.errorbar(
                spectrum.energy[uncertainty_points],
                spectrum.intensity[uncertainty_points],
                yerr=spectrum.uncertainty[uncertainty_points],
                fmt="none",
                ecolor="#727272",
                elinewidth=0.8,
                alpha=0.6,
                capsize=1.5,
            )
        if self.mask_inspection_mode:
            masked_points = finite_coordinates & excluded
            if self.y_scale == "log":
                masked_points = masked_points & (spectrum.intensity > 0.0)
            if np.any(masked_points):
                _draw_mask_inspection_spans(axes, spectrum.energy, masked_points)
                axes.scatter(
                    spectrum.energy[masked_points],
                    spectrum.intensity[masked_points],
                    color="#8d819d",
                    marker="x",
                    s=24,
                    alpha=0.52,
                    zorder=3,
                )
            self._draw_mask_boundary_handles(axes)
        self._draw_manual_preview(axes)
        axes.set_xlabel(axis_label("Energy", spectrum.energy_unit))
        axes.set_ylabel(axis_label("Intensity", spectrum.intensity_unit))
        title = axes.set_title(
            self._current_coordinate_label(dataset),
            loc="left",
            fontsize=10,
        )
        self.spectrum_q_title = title if dataset.q_bins is not None else None
        if self.y_scale == "symlog":
            axes.set_yscale(
                "symlog",
                linthresh=symlog_linthresh(spectrum.intensity[visible_points]),
            )
        elif self.y_scale == "linear":
            axes.set_yscale(self.y_scale)
        elif not self._log_display_has_positive_value(
            spectrum.intensity[retained_points]
        ):
            self.spectrum_log_empty_message = axes.text(
                0.5,
                0.5,
                "No positive values available for Log Y display",
                transform=axes.transAxes,
                ha="center",
                va="center",
                color="#676764",
                fontsize=9,
            )
        axes.grid(True, color="#e8e8e8", linewidth=0.6)
        if self._spectrum_x_limits is not None:
            axes.set_xlim(self._spectrum_x_limits)
            axes.relim()
            axes.autoscale_view(scalex=False, scaley=True)
        if self.y_range_locked and self._locked_y_limits is not None:
            axes.set_ylim(self._locked_y_limits)
        if self._spectrum_y_limits is not None:
            axes.set_ylim(self._spectrum_y_limits)

    def _draw_manual_preview(self, axes: Axes) -> None:
        """Overlay the typed core preview without evaluating a scientific model here."""

        evaluation = self._manual_display_evaluation()
        if evaluation is None:
            return
        axes.plot(
            evaluation.energy,
            self._log_display_values(evaluation.total),
            color="#314b63",
            linewidth=1.7,
            zorder=4,
        )
        for curve in evaluation.component_curves:
            emphasized = curve.component in self._manual_emphasized_components
            axes.plot(
                evaluation.energy,
                self._log_display_values(curve.values),
                color="#6d8191",
                linewidth=1.35 if emphasized else 0.8,
                alpha=1.0 if emphasized else 0.58,
                zorder=4 if emphasized else 3,
            )
        axes.plot(
            evaluation.energy,
            self._log_display_values(evaluation.background),
            color="#8a7564",
            linewidth=1.0,
            alpha=0.72,
            zorder=3,
        )

    def _manual_display_evaluation(self) -> ModelEvaluation | None:
        """Return the one evaluation currently selected by display precedence."""

        if self._manual_pending_evaluation is not None:
            return self._manual_pending_evaluation
        if self._manual_fit_result is not None:
            return self._manual_fit_result.evaluation
        if self._manual_preview is not None:
            return self._manual_preview.display_evaluation
        return None

    def _log_display_values(self, values: np.ndarray) -> np.ndarray:
        """Mask non-positive curve values only for Matplotlib Log rendering."""

        if self.y_scale != "log":
            return values
        return log_display_values(values)

    def _log_display_has_positive_value(self, measured: np.ndarray) -> bool:
        """Return whether any measured or Manual display layer can appear on Log."""

        if np.any(np.isfinite(measured) & (measured > 0.0)):
            return True
        evaluation = self._manual_display_evaluation()
        if evaluation is None:
            return False
        arrays = (
            evaluation.total,
            evaluation.background,
            *(curve.values for curve in evaluation.component_curves),
        )
        return any(np.any(np.isfinite(values) & (values > 0.0)) for values in arrays)

    def _draw_mask_boundary_handles(
        self,
        axes: Axes,
    ) -> None:
        """Render the current Group's explicit Boundary-owned coordinates."""

        self._boundary_handle_artists = {}
        self._boundary_handle_positions = {}
        if self._mask_boundary_coordinates is None:
            return
        coordinates = self._mask_boundary_coordinates[self.current_group_index]
        for side, value in (
            ("left", coordinates.left_energy),
            ("right", coordinates.right_energy),
        ):
            line = axes.axvline(
                value,
                color="#496d91",
                linewidth=1.1,
                linestyle="--",
                zorder=5,
            )
            self._boundary_handle_artists[side] = line
            self._boundary_handle_positions[side] = value

    def _effective_excluded_mask(self, point_count: int) -> np.ndarray:
        if self.selection is None:
            return np.zeros(point_count, dtype=np.bool_)
        return self.selection.excluded_mask(self.current_group_index)

    def _draw_overview(
        self,
        source_dataset: ReducedDataset,
        navigation_dataset: ReducedDataset,
        axes: Axes,
    ) -> None:
        coordinates = self._overview_x_coordinates(navigation_dataset)
        self.overview_x_cell_bounds = _overview_x_cell_bounds(
            navigation_dataset,
            coordinates,
        )
        source_coordinates = self._overview_x_coordinates(source_dataset)
        source_cell_bounds = _overview_x_cell_bounds(
            source_dataset,
            source_coordinates,
        )
        finite_intensities = np.concatenate(
            [
                spectrum.intensity[
                    np.isfinite(spectrum.energy) & np.isfinite(spectrum.intensity)
                ]
                for spectrum in source_dataset.spectra
            ]
        )
        norm = _intensity_normalization(finite_intensities, self.heatmap_scale)
        meshes: list[QuadMesh] = []
        for bounds, spectrum in zip(
            source_cell_bounds,
            source_dataset.spectra,
            strict=True,
        ):
            mesh = _draw_spectrum_overview_column(
                axes,
                bounds,
                spectrum.energy,
                spectrum.intensity,
                norm,
            )
            if mesh is not None:
                meshes.append(mesh)
        self.overview_meshes = tuple(meshes)
        if self.selection is not None and source_dataset is navigation_dataset:
            for bounds, group_index, spectrum in zip(
                self.overview_x_cell_bounds,
                range(len(navigation_dataset.spectra)),
                navigation_dataset.spectra,
                strict=True,
            ):
                _draw_mask_overlay_column(
                    axes,
                    bounds,
                    spectrum.energy,
                    self.selection.excluded_mask(group_index),
                    color=self._overview_masked,
                    alpha=0.76 if self.mask_inspection_mode else 0.42,
                )
        current_bounds = self.overview_x_cell_bounds[self.current_group_index]
        self.overview_active_highlight = axes.axvspan(
            *current_bounds,
            facecolor=(1.0, 1.0, 1.0, OVERVIEW_ACTIVE_ALPHA),
            linewidth=1.25,
            edgecolor=OVERVIEW_ACTIVE_COLOR,
            zorder=4,
        )
        spectrum = source_dataset.spectra[0]
        axes.set_ylabel(axis_label("Energy", spectrum.energy_unit), fontsize=8)
        axes.set_xlabel("")
        axes.xaxis.tick_top()
        axes.tick_params(
            axis="x",
            bottom=False,
            labelbottom=False,
            top=True,
            labeltop=True,
            labelcolor=self._overview_text,
            color=self._overview_border,
        )
        axes.tick_params(
            axis="y",
            labelsize=8,
            labelcolor=self._overview_text,
            color=self._overview_border,
        )
        axes.yaxis.label.set_color(self._overview_text)
        for spine in axes.spines.values():
            spine.set_color(self._overview_border)
        editable_q_values = self._editable_q_center_values(navigation_dataset)
        if editable_q_values is not None:
            q_axis = axes.secondary_xaxis("bottom")
            q_axis.set_xlabel(f"Q ({Q_DISPLAY_UNIT})", color=self._overview_text)
            q_axis.tick_params(
                labelcolor=self._overview_text,
                color=self._overview_border,
            )
            q_axis.spines["bottom"].set_color(self._overview_border)
            self.overview_q_axis = q_axis
            self._overview_q_label_artists = tuple(q_axis.get_xticklabels())
        else:
            self.overview_q_axis = None
            self._overview_q_label_artists = ()
        self._update_expanded_overview_labels(navigation_dataset, axes)

    def _update_expanded_overview_labels(
        self,
        dataset: ReducedDataset,
        axes: Axes,
    ) -> None:
        """Update width-aware Group/Q labels without rebuilding heatmap artists."""

        coordinates = self._overview_x_coordinates(dataset)
        label_indices = _discrete_label_indices(
            len(dataset.spectra),
            self.current_group_index,
            axes,
        )
        x_limits = axes.get_xlim()
        axes.set_xticks(coordinates[list(label_indices)])
        axes.set_xticklabels(
            [str(index + 1) for index in label_indices],
            fontsize=8,
        )
        q_axis: Any = self.overview_q_axis
        editable_q_values = self._editable_q_center_values(dataset)
        if q_axis is None or editable_q_values is None:
            self._overview_q_label_artists = ()
            axes.set_xlim(x_limits)
            return
        q_axis.set_xticks(coordinates[list(label_indices)])
        q_axis.set_xticklabels(
            [
                ""
                if editable_q_values[index] is None
                else f"{editable_q_values[index]:.4g}"
                for index in label_indices
            ],
            fontsize=8,
        )
        self._overview_q_label_artists = tuple(q_axis.get_xticklabels())
        axes.set_xlim(x_limits)

    def _draw_discrete_navigator(self, dataset: ReducedDataset, axes: Axes) -> None:
        coordinates = np.arange(1, len(dataset.spectra) + 1, dtype=np.float64)
        axes.plot(
            coordinates,
            np.zeros_like(coordinates),
            color=self._overview_border,
            linewidth=1.0,
            zorder=1,
        )
        axes.scatter(
            coordinates,
            np.zeros_like(coordinates),
            color=NAVIGATOR_INACTIVE_COLOR,
            s=16,
            zorder=2,
        )
        current_coordinate = coordinates[self.current_group_index]
        self.navigator_active_highlight = axes.scatter(
            [current_coordinate],
            [0.0],
            color=self._overview_accent,
            s=40,
            zorder=3,
        )
        axes.set_xlim(0.5, len(dataset.spectra) + 0.5)
        editable_q_values = self._editable_q_center_values(dataset)
        axes.set_ylim(-0.34 if editable_q_values is not None else -0.18, 0.18)
        axes.set_yticks([])
        axes.xaxis.tick_top()
        axes.tick_params(
            axis="x",
            length=0,
            pad=0,
            labelcolor=self._overview_text,
        )
        for spine in axes.spines.values():
            spine.set_visible(False)
        axes.set_facecolor(self._overview_surface)
        if editable_q_values is not None:
            axes.text(
                1.0,
                -0.19,
                Q_DISPLAY_UNIT,
                ha="right",
                va="top",
                fontsize=7,
                color=self._overview_text,
                transform=axes.transAxes,
            )
        self._update_navigator_labels(dataset, axes)

    def _update_navigator_labels(
        self,
        dataset: ReducedDataset,
        axes: Axes,
    ) -> None:
        """Update current-aware sparse navigator labels in place."""

        coordinates = np.arange(1, len(dataset.spectra) + 1, dtype=np.float64)
        label_indices = _discrete_label_indices(
            len(dataset.spectra),
            self.current_group_index,
            axes,
        )
        axes.set_xticks(coordinates[list(label_indices)])
        axes.set_xticklabels([str(index + 1) for index in label_indices], fontsize=7)
        for artist in self._navigator_q_label_artists:
            artist.remove()
        editable_q_values = self._editable_q_center_values(dataset)
        if editable_q_values is None:
            self._navigator_q_label_artists = ()
            return
        self._navigator_q_label_artists = tuple(
            axes.text(
                coordinates[index],
                -0.19,
                ""
                if editable_q_values[index] is None
                else f"{editable_q_values[index]:.4g}",
                ha="center",
                va="top",
                fontsize=7,
                color=self._overview_text,
            )
            for index in label_indices
        )

    def _update_navigation_state(self, dataset: ReducedDataset) -> None:
        self.previous_button.setEnabled(self.current_group_index > 0)
        self.next_button.setEnabled(self.current_group_index + 1 < len(dataset.spectra))
        blocker = QSignalBlocker(self.group_spinbox)
        self.group_spinbox.setValue(self.current_group_index + 1)
        del blocker
        self.group_spinbox.setToolTip("Click: select Group")
        if dataset.q_bins is None:
            self.current_q_label.hide()
        else:
            q_value = dataset.q_bins.q_values[self.current_group_index]
            self.current_q_label.setText(f"Q = {q_value:.6g} {Q_DISPLAY_UNIT}")
            self.current_q_label.setVisible(True)
        self._update_overview_toggle()

    def _update_metadata_status(self) -> None:
        dataset = self._require_dataset()
        statuses: list[str] = []
        if dataset.q_bins is None:
            statuses.append("Q required")
        if any(
            unit.strip().casefold() == "unknown"
            for spectrum in dataset.spectra
            for unit in (
                spectrum.energy_unit,
                spectrum.intensity_unit,
                spectrum.uncertainty_unit,
            )
        ):
            statuses.append("Units required")
        warning_count = sum(
            diagnostic.severity is DiagnosticSeverity.WARNING
            for diagnostic in dataset.diagnostics
        )
        if warning_count:
            suffix = "warning" if warning_count == 1 else "warnings"
            statuses.append(f"{warning_count} {suffix}")
        self.metadata_status_label.setText(" · ".join(statuses))
        self.metadata_status_label.setToolTip(
            "\n".join(
                f"{diagnostic.severity.value.title()}: {diagnostic.message}"
                for diagnostic in dataset.diagnostics
            )
        )

    def _group_description(self, dataset: ReducedDataset) -> str:
        description = f"Group {self.current_group_index + 1}"
        if dataset.q_bins is not None:
            q_value = dataset.q_bins.q_values[self.current_group_index]
            description += f" · Q = {q_value:.6g} {Q_DISPLAY_UNIT}"
        return description

    def _current_coordinate_label(self, dataset: ReducedDataset) -> str:
        if dataset.q_bins is None:
            return f"Group {self.current_group_index + 1}"
        q_value = dataset.q_bins.q_values[self.current_group_index]
        return f"Q = {q_value:.6g} {Q_DISPLAY_UNIT}"

    def _group_number_changed(self, group_number: int) -> None:
        if self.dataset is not None and group_number - 1 != self.current_group_index:
            self.set_current_group(group_number - 1)

    def _overview_x_coordinates(self, dataset: ReducedDataset) -> np.ndarray:
        if dataset.q_bins is not None:
            return dataset.q_bins.q_values
        return np.arange(1, len(dataset.spectra) + 1, dtype=np.float64)

    def _on_button_press(self, event: MouseEvent) -> None:
        if (
            getattr(event, "dblclick", False)
            and event.button is MouseButton.LEFT
            and self.dataset is not None
            and self.dataset.q_bins is not None
            and self._event_hits_overview_q_label(event)
        ):
            self.q_assignment_requested.emit()
            return
        if (
            event.button is MouseButton.LEFT
            and event.inaxes in (self.overview_axes, self.navigator_axes)
            and event.xdata is not None
        ):
            self._navigation_drag_active = True
            self._select_group_from_navigation_axes(event.inaxes, float(event.xdata))

    def _on_spectrum_units_press(self, event: MouseEvent) -> None:
        if (
            self._manual_component_kind is None
            and getattr(event, "dblclick", False)
            and event.button is MouseButton.LEFT
            and self.spectrum_axes is not None
            and self.spectrum_axes.xaxis.label.contains(event)[0]
        ):
            self.units_requested.emit()

    def _on_spectrum_q_press(self, event: MouseEvent) -> None:
        if (
            self._manual_component_kind is None
            and getattr(event, "dblclick", False)
            and event.button is MouseButton.LEFT
            and self.dataset is not None
            and self.dataset.q_bins is not None
            and self.spectrum_q_title is not None
            and _event_hits_text(event, self.spectrum_q_title)
        ):
            self.q_assignment_requested.emit()

    def _on_overview_units_press(self, event: MouseEvent) -> None:
        if (
            getattr(event, "dblclick", False)
            and event.button is MouseButton.LEFT
            and self.overview_axes is not None
            and self.overview_axes.yaxis.label.contains(event)[0]
        ):
            self.units_requested.emit()

    def _q_center_draft_is_editable(self) -> bool:
        return (
            not self.q_editor.isHidden()
            and self.q_editor.representation_combo.currentText() == "Q Center"
        )

    def _event_hits_overview_q_label(self, event: MouseEvent) -> bool:
        """Return whether a visible Q label, rather than navigation space, was hit."""

        if event.inaxes not in (self.overview_axes, self.navigator_axes, None):
            return False
        artists = (
            *self._overview_q_label_artists,
            *self._navigator_q_label_artists,
        )
        return any(_event_hits_text(event, artist) for artist in artists)

    def _editable_q_center_values(
        self,
        dataset: ReducedDataset,
    ) -> tuple[float | None, ...] | None:
        if self._q_center_draft_is_editable():
            return self.q_editor.q_center_slots()
        if dataset.q_bins is None:
            return None
        return tuple(float(value) for value in dataset.q_bins.q_values)

    def _nearest_navigation_group(self, axes: Axes | None, coordinate: float) -> int:
        if axes is None or self.dataset is None:
            return self.current_group_index
        if axes is self.overview_axes:
            return _overview_group_index_at_coordinate(
                coordinate,
                self._overview_x_coordinates(self.dataset),
                self.overview_x_cell_bounds,
            )
        return int(
            np.argmin(
                np.abs(
                    np.arange(1, len(self.dataset.spectra) + 1, dtype=np.float64)
                    - coordinate,
                ),
            ),
        )

    def _draw_q_center_draft(self) -> None:
        if self.dataset is not None and self._q_center_draft_is_editable():
            self._draw()

    def _on_mouse_motion(self, event: MouseEvent) -> None:
        if (
            self._navigation_drag_active
            and event.inaxes in (self.overview_axes, self.navigator_axes)
            and event.xdata is not None
        ):
            self._select_group_from_navigation_axes(event.inaxes, float(event.xdata))

    def _on_button_release(self, _event: MouseEvent) -> None:
        self._navigation_drag_active = False

    def _on_spectrum_button_press(self, event: MouseEvent) -> None:
        if self._manual_component_kind is not None:
            self._handle_manual_component_press(event)
            return
        if self._mask_tool is None:
            self._handle_spectrum_zoom_press(event)
            return
        if event.button is not MouseButton.LEFT:
            return
        if self._mask_tool == "boundary":
            if not self.mask_inspection_mode:
                return
            side = self._boundary_side_at(event)
            if side is None:
                return
            self._boundary_drag_side = side
            return
        if event.inaxes is not self.spectrum_axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._mask_points = [(float(event.xdata), float(event.ydata))]
        if self._mask_tool == "rectangle":
            self._mask_preview_rectangle = Rectangle(
                (float(event.xdata), float(event.ydata)),
                0.0,
                0.0,
                facecolor=_mask_operation_color(self._effective_mask_operation()),
                edgecolor=_mask_operation_color(self._effective_mask_operation()),
                alpha=0.18,
                linewidth=1.0,
                zorder=6,
            )
            assert self.spectrum_axes is not None
            self.spectrum_axes.add_patch(self._mask_preview_rectangle)
        else:
            assert self.spectrum_axes is not None
            line = self.spectrum_axes.plot(
                [event.xdata],
                [event.ydata],
                color=_mask_operation_color(self._effective_mask_operation()),
                linewidth=1.0,
                zorder=6,
            )
            self._mask_preview_line = line[0]
        self._set_mask_operation_feedback(
            self._effective_mask_operation(),
            inverted=self._modifier_active(),
        )
        self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _on_spectrum_mouse_motion(self, event: MouseEvent) -> None:
        if self._manual_component_kind is not None:
            self._handle_manual_component_motion(event)
            return
        if self._mask_tool is None:
            self._handle_spectrum_zoom_motion(event)
            return
        if self._boundary_drag_side is not None:
            if event.inaxes is self.spectrum_axes and event.xdata is not None:
                self._move_boundary_preview(
                    self._boundary_drag_side,
                    float(event.xdata),
                )
            return
        if not self._mask_points:
            self._set_boundary_cursor(event)
            return
        if (
            event.inaxes is not self.spectrum_axes
            or event.xdata is None
            or event.ydata is None
        ):
            return
        self._update_mask_gesture(float(event.xdata), float(event.ydata))
        self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _on_spectrum_button_release(self, event: MouseEvent) -> None:
        if self._manual_component_kind is not None:
            self._handle_manual_component_release(event)
            return
        if self._mask_tool is None:
            self._handle_spectrum_zoom_release(event)
            return
        if self._boundary_drag_side is not None:
            side = self._boundary_drag_side
            self._boundary_drag_side = None
            energy = self._boundary_handle_positions.get(side)
            if energy is not None:
                self.mask_boundary_requested.emit(
                    self.current_group_index,
                    side,
                    energy,
                )
            self._set_boundary_cursor(event)
            return
        if self._mask_tool not in {"rectangle", "lasso"} or not self._mask_points:
            self._cancel_mask_gesture()
            return
        if (
            event.inaxes is self.spectrum_axes
            and event.xdata is not None
            and event.ydata is not None
        ):
            self._update_mask_gesture(float(event.xdata), float(event.ydata))
        group_index = self.current_group_index
        operation = self._effective_mask_operation()
        self._clear_mask_preview_artists()
        if self._mask_tool == "rectangle":
            if len(self._mask_points) >= 2:
                start_x, start_y = self._mask_points[0]
                end_x, end_y = self._mask_points[-1]
                self.mask_rectangle_requested.emit(
                    group_index,
                    start_x,
                    end_x,
                    start_y,
                    end_y,
                    operation,
                )
        else:
            vertices = tuple(self._mask_points)
            if len(vertices) >= 3:
                self.mask_lasso_requested.emit(group_index, vertices, operation)
        self._mask_points = []
        self._set_mask_operation_feedback(self.mask_operation, inverted=False)

    def _on_spectrum_key_press(self, event: KeyEvent) -> None:
        if event.key == "escape":
            if self._manual_component_kind is not None:
                self.cancel_manual_component_interaction()
                return
            if self._spectrum_zoom_start is not None:
                self._cancel_spectrum_zoom()
                return
            self._cancel_mask_gesture()

    def _handle_spectrum_zoom_press(self, event: MouseEvent) -> None:
        """Begin an idle, display-only rubber-band zoom gesture."""

        if getattr(event, "dblclick", False) and self._event_hits_spectrum_label(
            event,
        ):
            return
        if (
            event.button is not MouseButton.LEFT
            or event.inaxes is not self.spectrum_axes
        ):
            return
        if getattr(event, "dblclick", False):
            self.reset_view()
            return
        if event.xdata is None or event.ydata is None:
            return
        display_point = self._spectrum_event_display_point(event)
        if display_point is None:
            return
        start = (float(event.xdata), float(event.ydata))
        self._cancel_spectrum_zoom(redraw=False)
        self._spectrum_zoom_start = start
        self._spectrum_zoom_start_display = display_point
        self._spectrum_zoom_rectangle = Rectangle(
            start,
            0.0,
            0.0,
            facecolor="#496d91",
            edgecolor="#314b63",
            alpha=0.16,
            linewidth=1.0,
            zorder=7,
        )
        assert self.spectrum_axes is not None
        self.spectrum_axes.add_patch(self._spectrum_zoom_rectangle)
        self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _event_hits_spectrum_label(self, event: MouseEvent) -> bool:
        axes = self.spectrum_axes
        if axes is None:
            return False
        if self.spectrum_q_title is not None and _event_hits_text(
            event,
            self.spectrum_q_title,
        ):
            return True
        return bool(axes.xaxis.label.contains(event)[0])

    def _handle_spectrum_zoom_motion(self, event: MouseEvent) -> None:
        """Update the visible idle zoom rectangle while the pointer is held."""

        if (
            self._spectrum_zoom_start is None
            or self._spectrum_zoom_rectangle is None
            or event.inaxes is not self.spectrum_axes
            or event.xdata is None
            or event.ydata is None
        ):
            return
        start_x, start_y = self._spectrum_zoom_start
        end_x, end_y = float(event.xdata), float(event.ydata)
        self._spectrum_zoom_rectangle.set_bounds(
            min(start_x, end_x),
            min(start_y, end_y),
            abs(end_x - start_x),
            abs(end_y - start_y),
        )
        self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _handle_spectrum_zoom_release(self, event: MouseEvent) -> None:
        """Apply normalized Spectrum view limits for a non-trivial idle drag."""

        start = self._spectrum_zoom_start
        start_display = self._spectrum_zoom_start_display
        if start is None or start_display is None:
            return
        if (
            event.inaxes is not self.spectrum_axes
            or event.xdata is None
            or event.ydata is None
        ):
            self._cancel_spectrum_zoom()
            return
        end = (float(event.xdata), float(event.ydata))
        end_display = self._spectrum_event_display_point(event)
        self._cancel_spectrum_zoom(redraw=False)
        if (
            end_display is None
            or max(
                abs(end_display[0] - start_display[0]),
                abs(end_display[1] - start_display[1]),
            )
            < SPECTRUM_ZOOM_DRAG_THRESHOLD_PX
        ):
            self.canvas.draw_idle()  # type: ignore[no-untyped-call]
            return
        x_limits = (min(start[0], end[0]), max(start[0], end[0]))
        y_limits = (min(start[1], end[1]), max(start[1], end[1]))
        if x_limits[0] == x_limits[1] or y_limits[0] == y_limits[1]:
            self.canvas.draw_idle()  # type: ignore[no-untyped-call]
            return
        axes = self.spectrum_axes
        if axes is None:
            return
        axes.set_xlim(x_limits)
        axes.set_ylim(y_limits)
        self._spectrum_x_limits = axes.get_xlim()
        self._spectrum_y_limits = axes.get_ylim()
        if self.y_range_locked:
            self._locked_y_limits = axes.get_ylim()
        self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _spectrum_event_display_point(
        self,
        event: MouseEvent,
    ) -> tuple[float, float] | None:
        event_x = getattr(event, "x", None)
        event_y = getattr(event, "y", None)
        if event_x is not None and event_y is not None:
            return float(event_x), float(event_y)
        if self.spectrum_axes is None or event.xdata is None or event.ydata is None:
            return None
        point = self.spectrum_axes.transData.transform((event.xdata, event.ydata))
        return float(point[0]), float(point[1])

    def _cancel_spectrum_zoom(self, *, redraw: bool = True) -> None:
        rectangle = self._spectrum_zoom_rectangle
        self._spectrum_zoom_rectangle = None
        self._spectrum_zoom_start = None
        self._spectrum_zoom_start_display = None
        if rectangle is not None and rectangle.axes is not None:
            rectangle.remove()
            if redraw:
                self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _handle_manual_component_press(self, event: MouseEvent) -> None:
        """Capture the first hint and request its workflow scientific preview."""

        if (
            event.button is not MouseButton.LEFT
            or event.inaxes is not self.spectrum_axes
            or event.xdata is None
            or event.ydata is None
        ):
            return
        start = (float(event.xdata), float(event.ydata))
        self._manual_pointer_down = True
        self._manual_interaction_start = start
        self._manual_drag_point = start
        self._request_manual_component_preview()

    def _handle_manual_component_motion(self, event: MouseEvent) -> None:
        """Update workflow scientific feedback while one Spectrum press is held."""

        if (
            not self._manual_pointer_down
            or event.inaxes is not self.spectrum_axes
            or event.xdata is None
            or event.ydata is None
        ):
            return
        point = (float(event.xdata), float(event.ydata))
        self._manual_drag_point = point
        self._request_manual_component_preview()

    def _request_manual_component_preview(self) -> None:
        """Forward raw gesture geometry without deriving scientific parameters."""

        kind = self._manual_component_kind
        start = self._manual_interaction_start
        point = self._manual_drag_point
        if kind is None or start is None or point is None:
            return
        if kind is ManualComponentKind.ELASTIC:
            hints: dict[str, float] | None = {
                "component_peak_center": point[0],
                "component_peak_height": point[1],
            }
        elif kind is ManualComponentKind.LORENTZIAN:
            hints = (
                None
                if point[0] == start[0]
                else {
                    "component_peak_center": start[0],
                    "component_peak_height": start[1],
                    "width_endpoint_energy": point[0],
                }
            )
        else:
            hints = {
                "first_energy": start[0],
                "first_height": start[1],
                "second_energy": point[0],
                "second_height": point[1],
            }
        self.manual_component_preview_requested.emit(kind, hints)

    def _handle_manual_component_release(self, event: MouseEvent) -> None:
        """Submit the final raw gesture hints for one pending component."""

        if not self._manual_pointer_down:
            return
        if (
            event.inaxes is not self.spectrum_axes
            or event.xdata is None
            or event.ydata is None
        ):
            self._manual_pointer_down = False
            return
        point = (float(event.xdata), float(event.ydata))
        self._manual_pointer_down = False
        self._manual_drag_point = point
        kind = self._manual_component_kind
        if kind is ManualComponentKind.ELASTIC:
            self._complete_manual_component(
                {
                    "component_peak_center": point[0],
                    "component_peak_height": point[1],
                },
            )
            return
        if kind is ManualComponentKind.LORENTZIAN:
            start = self._manual_interaction_start
            if start is None:
                return
            if point[0] == start[0]:
                self.cancel_manual_component_interaction()
                return
            self._complete_manual_component(
                {
                    "component_peak_center": start[0],
                    "component_peak_height": start[1],
                    "width_endpoint_energy": point[0],
                },
            )
            return
        if kind is not ManualComponentKind.BACKGROUND:
            return
        first = self._manual_interaction_start
        if first is None:
            return
        self._complete_manual_component(
            {
                "first_energy": first[0],
                "first_height": first[1],
                "second_energy": point[0],
                "second_height": point[1],
            },
        )

    def _complete_manual_component(self, hints: dict[str, float]) -> None:
        kind = self._manual_component_kind
        if kind is None:
            return
        self._manual_component_kind = None
        self._reset_manual_interaction_feedback()
        self.manual_component_completed.emit(kind, hints)

    def _boundary_side_at(self, event: MouseEvent) -> str | None:
        """Resolve a press near one visible handle, without midpoint inference."""

        if self.spectrum_axes is None:
            return None
        event_x = getattr(event, "x", None)
        event_y = getattr(event, "y", None)
        if event_x is not None:
            tolerance = BOUNDARY_HIT_RADIUS_LOGICAL_PX * max(
                1.0, float(self.canvas.devicePixelRatioF())
            )
            if event_y is not None and not (
                self.spectrum_axes.bbox.y0 - tolerance
                <= float(event_y)
                <= self.spectrum_axes.bbox.y1 + tolerance
            ):
                return None
            for side, x_data in self._boundary_handle_positions.items():
                handle_x = self.spectrum_axes.get_xaxis_transform().transform(
                    (x_data, 0.5)
                )[0]
                if abs(float(event_x) - handle_x) <= tolerance:
                    return side
            return None
        if event.xdata is None:
            return None
        left, right = self.spectrum_axes.get_xlim()
        tolerance = abs(right - left) * 0.02
        for side, x_data in self._boundary_handle_positions.items():
            if abs(float(event.xdata) - x_data) <= tolerance:
                return side
        return None

    def _move_boundary_preview(self, side: str, energy: float) -> None:
        """Move the visible handle and request a non-committing core preview."""

        artist = self._boundary_handle_artists.get(side)
        if artist is not None:
            artist.set_xdata([energy, energy])
        self._boundary_handle_positions[side] = energy
        self.mask_boundary_previewed.emit(self.current_group_index, side, energy)
        self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _update_mask_gesture(self, x_data: float, y_data: float) -> None:
        """Update one active Rectangle/Lasso only from valid axes coordinates."""

        operation = self._effective_mask_operation()
        self._set_mask_operation_feedback(
            operation,
            inverted=self._modifier_active(),
        )
        if self._mask_tool == "rectangle" and self._mask_preview_rectangle is not None:
            start_x, start_y = self._mask_points[0]
            self._mask_points = [(start_x, start_y), (x_data, y_data)]
            self._mask_preview_rectangle.set_bounds(
                min(start_x, x_data),
                min(start_y, y_data),
                abs(x_data - start_x),
                abs(y_data - start_y),
            )
            self._mask_preview_rectangle.set_edgecolor(_mask_operation_color(operation))
            self._mask_preview_rectangle.set_facecolor(_mask_operation_color(operation))
        elif self._mask_tool == "lasso" and self._mask_preview_line is not None:
            self._mask_points.append((x_data, y_data))
            x_values, y_values = zip(*self._mask_points, strict=True)
            self._mask_preview_line.set_data(x_values, y_values)
            self._mask_preview_line.set_color(_mask_operation_color(operation))

    def _set_boundary_cursor(self, event: MouseEvent) -> None:
        if (
            self._mask_tool == "boundary"
            and self.mask_inspection_mode
            and self._boundary_side_at(event) is not None
        ):
            self.canvas.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.canvas.unsetCursor()

    def _effective_mask_operation(self) -> str:
        return "restore" if self._modifier_active() else self.mask_operation

    @staticmethod
    def _modifier_active() -> bool:
        return bool(
            QApplication.keyboardModifiers() & Qt.KeyboardModifier.AltModifier,
        )

    def _set_mask_operation_feedback(self, operation: str, *, inverted: bool) -> None:
        text = operation.title()
        if inverted:
            text += " (Option/Alt)"
        if self.mask_operation_feedback != text:
            self.mask_operation_feedback = text
            self.mask_operation_feedback_changed.emit(text)

    def _clear_mask_preview_artists(self) -> None:
        if self._mask_preview_rectangle is not None:
            self._mask_preview_rectangle.remove()
            self._mask_preview_rectangle = None
        if self._mask_preview_line is not None:
            self._mask_preview_line.remove()
            self._mask_preview_line = None

    def _cancel_mask_gesture(self) -> None:
        self._boundary_drag_side = None
        self._mask_points = []
        self._clear_mask_preview_artists()
        self._set_mask_operation_feedback(self.mask_operation, inverted=False)
        self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _toggle_overview(self) -> None:
        self.set_overview_visible(not self.overview_visible)

    def _update_overview_toggle(self) -> None:
        action = "Hide" if self.overview_visible else "Show"
        self.overview_toggle_button.setText(f"{action} Overview")
        self.overview_toggle_button.setToolTip(f"{action} intensity overview")

    def _select_group_from_navigation_axes(
        self,
        axes: Axes | None,
        coordinate: float,
    ) -> None:
        if axes is self.overview_axes:
            self.select_group_from_overview(coordinate)
        elif axes is self.navigator_axes:
            self.select_group_from_navigator(coordinate)

    def select_group_from_navigator(self, coordinate: float) -> None:
        """Select the nearest real Group from the collapsed navigator strip."""

        dataset = self._require_dataset()
        group_index = int(
            np.argmin(
                np.abs(
                    np.arange(1, len(dataset.spectra) + 1, dtype=np.float64)
                    - coordinate,
                ),
            ),
        )
        if group_index != self.current_group_index:
            self.set_current_group(group_index)

    def set_y_scale(self, scale: str) -> bool:
        """Select a Matplotlib-only y-scale without changing source arrays."""

        if scale not in {"linear", "symlog", "log"}:
            raise ValueError("y scale must be 'linear', 'symlog', or 'log'")
        self._require_dataset()
        self.y_scale = scale
        self._spectrum_y_limits = None
        self._locked_y_limits = None
        self._draw()
        if self.y_range_locked and self.spectrum_axes is not None:
            self._locked_y_limits = self.spectrum_axes.get_ylim()
        self.y_scale_changed.emit(scale)
        return True

    def set_heatmap_scale(self, scale: str) -> bool:
        """Select an overview-only color normalization without changing data."""

        if scale not in {"linear", "log"}:
            raise ValueError("heatmap scale must be 'linear' or 'log'")
        self._require_dataset()
        self.heatmap_scale = scale
        self._draw()
        self.heatmap_scale_changed.emit(scale)
        return True

    def set_y_range_locked(self, locked: bool) -> None:
        """Preserve the current y limits across Group changes only when enabled."""

        self.y_range_locked = locked
        if locked and self.spectrum_axes is not None:
            self._locked_y_limits = self.spectrum_axes.get_ylim()
        elif not locked:
            self._locked_y_limits = None

    def _build_spectrum_context_menu(self) -> QMenu:
        menu = QMenu(self)
        self._add_q_assignment_action(menu)
        menu.addSeparator()
        reset_action = menu.addAction("Reset View")
        reset_action.triggered.connect(self.reset_view)
        scale_menu = menu.addMenu("Y Scale")
        assert scale_menu is not None
        scale_group = QActionGroup(scale_menu)
        scale_group.setExclusive(True)
        for label, scale in (
            ("Linear", "linear"),
            ("SymLog", "symlog"),
            ("Log", "log"),
        ):
            action = scale_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(self.y_scale == scale)
            action.triggered.connect(
                lambda _checked, value=scale: self.set_y_scale(value),
            )
            scale_group.addAction(action)
        menu.addSeparator()
        lock_action = menu.addAction("Lock Y Range")
        lock_action.setCheckable(True)
        lock_action.setChecked(self.y_range_locked)
        lock_action.triggered.connect(self.set_y_range_locked)
        menu.addSeparator()
        self._add_mask_edit_action(menu)
        return menu

    def _show_spectrum_context_menu(self, position: QPoint) -> None:
        if self.spectrum_axes is None:
            return
        menu = self._build_spectrum_context_menu()
        menu.exec(self.canvas.mapToGlobal(position))

    def _show_overview_context_menu(self, position: QPoint) -> None:
        axes = self.overview_axes or self.navigator_axes
        if axes is None:
            return
        menu = self._build_overview_context_menu()
        menu.exec(self.overview_canvas.mapToGlobal(position))

    def _build_overview_context_menu(self) -> QMenu:
        """Build display and workflow commands for the overview figure."""

        menu = QMenu(self)
        scale_menu = menu.addMenu("Heatmap Scale")
        assert scale_menu is not None
        scale_group = QActionGroup(scale_menu)
        scale_group.setExclusive(True)
        for label, scale in (("Linear", "linear"), ("Log", "log")):
            action = scale_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(self.heatmap_scale == scale)
            action.triggered.connect(
                lambda _checked, value=scale: self.set_heatmap_scale(value)
            )
            scale_group.addAction(action)
        menu.addSeparator()
        self._add_q_assignment_action(menu)
        self._add_mask_edit_action(menu)
        return menu

    def _show_navigation_q_context_menu(
        self,
        source: QWidget,
        position: QPoint,
    ) -> None:
        """Expose Q assignment where Group/Q navigation is already in use."""

        if self.dataset is None:
            return
        menu = QMenu(self)
        self._add_q_assignment_action(menu)
        menu.exec(source.mapToGlobal(position))

    def _add_q_assignment_action(self, menu: QMenu) -> None:
        dataset = self._require_dataset()
        action = menu.addAction(
            "Edit Q…" if dataset.q_bins is not None else "Assign Q…",
        )
        action.triggered.connect(self.q_assignment_requested.emit)

    def _add_mask_edit_action(self, menu: QMenu) -> None:
        if not self.mask_edit_available:
            return
        action = menu.addAction("Edit Mask…")
        action.triggered.connect(self.mask_edit_requested.emit)

    def _on_scroll(self, event: MouseEvent) -> None:
        if self.spectrum_axes is not None and _event_touches_axes(
            event,
            self.spectrum_axes,
        ):
            self._zoom_spectrum(event)
            return
        axes = event.inaxes
        if axes is None or axes is not self.overview_axes or event.xdata is None:
            return
        left, right = axes.get_xlim()
        scale = 0.8 if event.button == "up" else 1.25
        cursor = float(event.xdata)
        axes.set_xlim(
            cursor - (cursor - left) * scale,
            cursor + (right - cursor) * scale,
        )
        if axes is self.spectrum_axes:
            self._spectrum_x_limits = axes.get_xlim()
            self.canvas.draw_idle()  # type: ignore[no-untyped-call]
        else:
            self.overview_canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _zoom_spectrum(self, event: MouseEvent) -> None:
        """Zoom spectrum axes around the pointer without changing any data state."""

        axes = self.spectrum_axes
        if axes is None or event.x is None or event.y is None:
            return
        bbox = axes.bbox
        inside_body = bbox.contains(event.x, event.y)
        in_x_axis = bbox.x0 <= event.x <= bbox.x1 and event.y < bbox.y0
        in_y_axis = bbox.y0 <= event.y <= bbox.y1 and event.x < bbox.x0
        if not (inside_body or in_x_axis or in_y_axis):
            return
        x_cursor, y_cursor = axes.transData.inverted().transform((event.x, event.y))
        scale = 0.8 if event.button == "up" else 1.25
        if inside_body or in_x_axis:
            axes.set_xlim(*zoom_limits(axes.get_xlim(), float(x_cursor), scale))
            self._spectrum_x_limits = axes.get_xlim()
        if inside_body or in_y_axis:
            axes.set_ylim(*zoom_limits(axes.get_ylim(), float(y_cursor), scale))
            self._spectrum_y_limits = axes.get_ylim()
            if self.y_range_locked:
                self._locked_y_limits = axes.get_ylim()
        self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _require_dataset(self) -> ReducedDataset:
        if self.dataset is None:
            raise RuntimeError("no reduced dataset is open")
        return self.dataset


def _overview_x_cell_bounds(
    dataset: ReducedDataset,
    coordinates: np.ndarray,
) -> tuple[tuple[float, float], ...]:
    """Return display-only contiguous x-cell bounds around Group/Q centers."""

    if dataset.q_bins is not None and dataset.q_bins.edges is not None:
        return tuple(
            (float(lower), float(upper))
            for lower, upper in zip(
                dataset.q_bins.edges[:-1],
                dataset.q_bins.edges[1:],
                strict=True,
            )
        )
    if coordinates.size == 1:
        coordinate = float(coordinates[0])
        return ((coordinate - 0.5, coordinate + 0.5),)
    if np.all(np.diff(coordinates) > 0.0):
        edges = _center_cell_edges(coordinates)
        return tuple(
            (float(lower), float(upper))
            for lower, upper in zip(edges[:-1], edges[1:], strict=True)
        )
    return tuple(
        _individual_cell_bounds(coordinates, index) for index in range(len(coordinates))
    )


def _overview_group_index_at_coordinate(
    coordinate: float,
    representatives: np.ndarray,
    cell_bounds: tuple[tuple[float, float], ...],
) -> int:
    """Resolve a rendered overview cell before falling back to representative Q."""

    for index, (lower, upper) in enumerate(cell_bounds):
        is_last = index + 1 == len(cell_bounds)
        if lower <= coordinate < upper or (is_last and coordinate == upper):
            return index
    return int(np.argmin(np.abs(representatives - coordinate)))


def _discrete_label_indices(
    group_count: int,
    current_group_index: int,
    axes: Axes,
) -> tuple[int, ...]:
    """Thin labels only when the rendered Group labels would overlap."""

    widest_label = max(
        len(str(group_count)) * 8.0 * 0.62 * axes.figure.dpi / 72.0,
        1.0,
    )
    maximum_labels = max(2, int(axes.bbox.width // (widest_label + 8.0)))
    if group_count <= maximum_labels:
        return tuple(range(group_count))
    step = max(1, int(np.ceil((group_count - 1) / (maximum_labels - 1))))
    indices = set(range(0, group_count, step))
    indices.add(group_count - 1)
    indices.add(current_group_index)
    return tuple(sorted(indices))


def _individual_cell_bounds(
    coordinates: np.ndarray,
    index: int,
) -> tuple[float, float]:
    """Return a display-only width for unordered representative Q values."""

    coordinate = float(coordinates[index])
    distances = np.abs(coordinates - coordinate)
    positive_distances = distances[distances > 0.0]
    half_width = (
        float(np.min(positive_distances) / 2.0) if positive_distances.size else 0.5
    )
    return (coordinate - half_width, coordinate + half_width)


def _center_cell_edges(centers: np.ndarray) -> np.ndarray:
    """Derive display cell boundaries from ordered center coordinates only."""

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


class _NonpositiveUnderLogNorm(LogNorm):
    """Log-normalize positives while mapping finite nonpositives below range."""

    def __call__(
        self,
        value: Any,
        clip: bool | None = None,
    ) -> Any:
        scalar = np.isscalar(value)
        raw = np.asanyarray(value)
        normalized = np.ma.asarray(super().__call__(value, clip=clip))
        data = np.array(normalized.data, copy=True)
        mask = np.array(np.ma.getmaskarray(normalized), copy=True)
        nonpositive = np.isfinite(raw) & (raw <= 0.0)
        data[nonpositive] = -1.0
        mask[nonpositive] = False
        result = np.ma.array(data, mask=mask)
        return result[()] if scalar else result


def _intensity_normalization(
    values: np.ndarray,
    scale: str = "linear",
) -> Normalize | None:
    """Create one raw-intensity color mapping shared by every overview column."""

    if not values.size:
        return None
    if scale == "log":
        positive = values[values > 0.0]
        if not positive.size:
            return Normalize(vmin=0.0, vmax=1.0, clip=True)
        return _NonpositiveUnderLogNorm(
            vmin=float(np.min(positive)),
            vmax=float(np.max(positive)),
        )
    if scale != "linear":
        raise ValueError("heatmap scale must be 'linear' or 'log'")
    return Normalize(vmin=float(np.min(values)), vmax=float(np.max(values)))


def _draw_spectrum_overview_column(
    axes: Axes,
    x_bounds: tuple[float, float],
    energy: np.ndarray,
    intensity: np.ndarray,
    norm: Normalize | None,
) -> QuadMesh | None:
    """Draw one unresampled spectrum as adjacent display-only energy cells."""

    finite_energy = np.isfinite(energy)
    if not np.any(finite_energy):
        return None
    column_energy = energy[finite_energy]
    column_intensity = intensity[finite_energy]
    order = np.argsort(column_energy, kind="stable")
    column_energy = column_energy[order]
    column_intensity = column_intensity[order]
    if column_energy.size > 1 and np.any(np.diff(column_energy) <= 0.0):
        return None
    masked_intensity = np.ma.masked_invalid(column_intensity).reshape(-1, 1)
    return axes.pcolormesh(
        np.asarray(x_bounds),
        _center_cell_edges(column_energy),
        masked_intensity,
        cmap=OVERVIEW_COLORMAP,
        norm=norm,
        shading="flat",
        edgecolors="none",
        antialiased=False,
        snap=True,
    )


def _draw_mask_overlay_column(
    axes: Axes,
    x_bounds: tuple[float, float],
    energy: np.ndarray,
    excluded_mask: np.ndarray,
    *,
    color: str,
    alpha: float,
) -> None:
    """Overlay excluded cells without changing their source intensity values."""

    finite = np.isfinite(energy)
    if not np.any(finite):
        return
    coordinates = energy[finite]
    excluded = excluded_mask[finite]
    order = np.argsort(coordinates, kind="stable")
    coordinates = coordinates[order]
    excluded = excluded[order]
    if coordinates.size > 1 and np.any(np.diff(coordinates) <= 0.0):
        return
    overlay = np.ma.array(
        np.ones((coordinates.size, 1)),
        mask=(~excluded).reshape(-1, 1),
    )
    axes.pcolormesh(
        np.asarray(x_bounds),
        _center_cell_edges(coordinates),
        overlay,
        cmap=ListedColormap([color]),
        shading="flat",
        edgecolors="none",
        antialiased=False,
        alpha=alpha,
        zorder=4,
    )


def _plot_visible_spectrum_segments(
    axes: Axes,
    energy: np.ndarray,
    intensity: np.ndarray,
    visible: np.ndarray,
) -> None:
    """Plot every contiguous retained run without bridging across exclusions."""

    indices = np.flatnonzero(visible)
    if not indices.size:
        axes.plot([], [], color="#496d91", linewidth=1.0, marker="o", markersize=3.0)
        return
    starts = np.r_[0, np.flatnonzero(np.diff(indices) != 1) + 1]
    stops = np.r_[starts[1:], indices.size]
    for start, stop in zip(starts, stops, strict=True):
        run = indices[start:stop]
        axes.plot(
            energy[run],
            intensity[run],
            color="#496d91",
            linewidth=1.0,
            marker="o",
            markersize=3.0,
        )


def _draw_mask_inspection_spans(
    axes: Axes,
    energy: np.ndarray,
    masked: np.ndarray,
) -> None:
    """Add restrained task-only range cues behind individually inspectable points."""

    indices = np.flatnonzero(masked)
    if not indices.size:
        return
    starts = np.r_[0, np.flatnonzero(np.diff(indices) != 1) + 1]
    stops = np.r_[starts[1:], indices.size]
    for start, stop in zip(starts, stops, strict=True):
        run = indices[start:stop]
        axes.axvspan(
            energy[run[0]],
            energy[run[-1]],
            facecolor="#8d819d",
            alpha=0.12,
            zorder=1,
        )


def _mask_operation_color(operation: str) -> str:
    """Keep provisional mask intent visible without giving it data meaning."""

    return "#47785a" if operation == "restore" else "#a54a4a"


def _event_touches_axes(event: MouseEvent, axes: Axes) -> bool:
    """Include the data body plus the adjacent x/y axis interaction bands."""

    if event.x is None or event.y is None:
        return event.inaxes is axes
    bbox = axes.bbox
    return bool(
        bbox.contains(event.x, event.y)
        or (bbox.x0 <= event.x <= bbox.x1 and event.y < bbox.y0)
        or (bbox.y0 <= event.y <= bbox.y1 and event.x < bbox.x0)
    )


def _event_hits_text(event: MouseEvent, artist: Text) -> bool:
    """Treat a small band around a rendered label as its practical hit target."""

    if event.x is None or event.y is None:
        return False
    try:
        bbox = artist.get_window_extent()
    except RuntimeError:
        return False
    return bool(bbox.padded(5.0).contains(event.x, event.y))
