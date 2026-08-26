"""Read-only scientific view for one imported reduced dataset."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent
from matplotlib.collections import QuadMesh
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from PySide6.QtCore import QPoint, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QActionGroup, QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
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
from ezqens.gui.dialogs import show_message_dialog
from ezqens.gui.q_editor import QAssignmentEditor
from ezqens.gui.scientific_canvas import SCIENTIFIC_BACKGROUND, ScientificCanvas
from ezqens.gui.theme import DEFAULT_LAYOUT_TOKENS
from ezqens.gui.workspace import (
    Q_METHOD_MIME_TYPE,
    q_method_drag_indices_from_mime_data,
)
from ezqens.preprocessing import FittingSelection

Q_DISPLAY_UNIT = "Å⁻¹"
OVERVIEW_COLORMAP = "jet"
OVERVIEW_ACTIVE_COLOR = "#496d91"
NAVIGATOR_INACTIVE_COLOR = "#8d969d"


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
        self._mask_preview_rectangle: Rectangle | None = None
        self._mask_preview_line: Line2D | None = None
        self.current_group_index = 0
        self.overview_axes: Axes | None = None
        self.overview_q_axis: object | None = None
        self.navigator_axes: Axes | None = None
        self.spectrum_axes: Axes | None = None
        self.overview_meshes: tuple[QuadMesh, ...] = ()
        self.overview_x_cell_bounds: tuple[tuple[float, float], ...] = ()
        self.overview_active_highlight: Rectangle | None = None
        self._overview_q_label_artists: tuple[Text, ...] = ()
        self._navigator_q_label_artists: tuple[Text, ...] = ()
        self.spectrum_q_title: Text | None = None
        self._spectrum_x_limits: tuple[float, float] | None = None
        self.overview_visible = True
        self._navigation_drag_active = False
        self.y_scale = "linear"
        self.y_range_locked = False
        self._locked_y_limits: tuple[float, float] | None = None
        self._overview_surface = SCIENTIFIC_BACKGROUND
        self._overview_text = "#292928"
        self._overview_border = "#deded9"
        self._overview_accent = OVERVIEW_ACTIVE_COLOR
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

        controls = QGridLayout()
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
        layout.addLayout(controls)
        layout.addWidget(self.overview_canvas)
        layout.addWidget(self.q_editor)
        layout.addWidget(self.canvas, 1)
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

    def open_dataset(self, dataset: ReducedDataset) -> None:
        """Open real imported data at Group 1 without changing its arrays."""

        self.dataset = dataset
        self.selection = None
        self.mask_inspection_mode = False
        self.current_group_index = 0
        self._spectrum_x_limits = None
        self.y_scale = "linear"
        self.y_range_locked = False
        self._locked_y_limits = None
        blocker = QSignalBlocker(self.group_spinbox)
        self.group_spinbox.setRange(1, len(dataset.spectra))
        self.group_spinbox.setValue(1)
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

        self.dataset = None
        self.selection = None
        self.overview_axes = None
        self.overview_q_axis = None
        self.navigator_axes = None
        self.spectrum_axes = None
        self.overview_meshes = ()
        self.overview_x_cell_bounds = ()
        self.overview_active_highlight = None
        self._overview_q_label_artists = ()
        self._navigator_q_label_artists = ()
        self.spectrum_q_title = None
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

    def replace_dataset(self, dataset: ReducedDataset) -> None:
        """Refresh an opened dataset after a role-only Workspace change."""

        current_dataset = self._require_dataset()
        if len(dataset.spectra) != len(current_dataset.spectra):
            raise ValueError("role change must preserve the number of groups")
        self.dataset = dataset
        self._update_metadata_status()
        self._draw()

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

        dataset = self._require_dataset()
        if not 0 <= group_index < len(dataset.spectra):
            raise ValueError("group index is outside the dataset")
        if self.spectrum_axes is not None:
            self._spectrum_x_limits = self.spectrum_axes.get_xlim()
        self.current_group_index = group_index
        blocker = QSignalBlocker(self.group_spinbox)
        self.group_spinbox.setValue(group_index + 1)
        del blocker
        self._draw()
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

        self._spectrum_x_limits = None
        if self.dataset is not None:
            self._draw()

    def _draw(self) -> None:
        dataset = self._require_dataset()
        overview_figure = self.overview_canvas.figure
        overview_figure.clear()
        self._overview_q_label_artists = ()
        self._navigator_q_label_artists = ()
        overview_figure.set_facecolor(self._overview_surface)
        if self.overview_visible:
            self.overview_axes = overview_figure.add_subplot(111)
            self.navigator_axes = None
            self.overview_axes.set_facecolor(self._overview_surface)
            self._draw_overview(dataset, self.overview_axes)
        else:
            self.overview_axes = None
            self.overview_q_axis = None
            self.overview_meshes = ()
            self.overview_x_cell_bounds = ()
            self.overview_active_highlight = None
            self.navigator_axes = overview_figure.add_subplot(111)
            self._draw_discrete_navigator(dataset, self.navigator_axes)

        spectrum_figure = self.canvas.figure
        spectrum_figure.clear()
        self.spectrum_q_title = None
        spectrum_figure.set_facecolor(SCIENTIFIC_BACKGROUND)
        self.spectrum_axes = spectrum_figure.add_subplot(111)
        self.spectrum_axes.set_facecolor(SCIENTIFIC_BACKGROUND)
        self._draw_spectrum(dataset, self.spectrum_axes)
        self._update_navigation_state(dataset)
        self._update_overview_canvas_height()
        self.overview_canvas.draw_idle()  # type: ignore[no-untyped-call]
        self.canvas.draw_idle()  # type: ignore[no-untyped-call]

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
        visible_points = finite_coordinates & ~excluded
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
            self._draw_mask_boundary_handles(axes, spectrum.energy, visible_points)
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
                linthresh=_symlog_linthresh(spectrum.intensity[visible_points]),
            )
        else:
            axes.set_yscale(self.y_scale)
        axes.grid(True, color="#e8e8e8", linewidth=0.6)
        if self._spectrum_x_limits is not None:
            axes.set_xlim(self._spectrum_x_limits)
            axes.relim()
            axes.autoscale_view(scalex=False, scaley=True)
        if self.y_range_locked and self._locked_y_limits is not None:
            axes.set_ylim(self._locked_y_limits)

    def _draw_mask_boundary_handles(
        self,
        axes: Axes,
        energy: np.ndarray,
        visible_points: np.ndarray,
    ) -> None:
        """Render the effective left/right selection edges as draggable handles."""

        retained_energy = energy[visible_points]
        self._boundary_handle_artists = {}
        self._boundary_handle_positions = {}
        if not retained_energy.size:
            return
        for side, value in (
            ("left", float(np.min(retained_energy))),
            ("right", float(np.max(retained_energy))),
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

    def _draw_overview(self, dataset: ReducedDataset, axes: Axes) -> None:
        coordinates = self._overview_x_coordinates(dataset)
        self.overview_x_cell_bounds = _overview_x_cell_bounds(dataset, coordinates)
        finite_intensities = np.concatenate(
            [
                spectrum.intensity[
                    np.isfinite(spectrum.energy) & np.isfinite(spectrum.intensity)
                ]
                for spectrum in dataset.spectra
            ]
        )
        norm = _intensity_normalization(finite_intensities)
        meshes: list[QuadMesh] = []
        for bounds, spectrum in zip(
            self.overview_x_cell_bounds,
            dataset.spectra,
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
        if self.selection is not None:
            for bounds, group_index, spectrum in zip(
                self.overview_x_cell_bounds,
                range(len(dataset.spectra)),
                dataset.spectra,
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
            facecolor=self._overview_accent,
            alpha=0.18,
            linewidth=0.8,
            edgecolor=self._overview_accent,
            zorder=3,
        )
        spectrum = dataset.spectra[self.current_group_index]
        axes.set_ylabel(axis_label("Energy", spectrum.energy_unit), fontsize=8)
        axes.set_xlabel("")
        label_indices = _discrete_label_indices(
            len(dataset.spectra),
            self.current_group_index,
            axes,
        )
        axes.set_xticks(coordinates[list(label_indices)])
        axes.set_xticklabels(
            [str(index + 1) for index in label_indices],
            fontsize=8,
        )
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
        editable_q_values = self._editable_q_center_values(dataset)
        if editable_q_values is not None:
            q_axis = axes.secondary_xaxis("bottom")
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

    def _draw_discrete_navigator(self, dataset: ReducedDataset, axes: Axes) -> None:
        coordinates = np.arange(1, len(dataset.spectra) + 1, dtype=np.float64)
        label_indices = _discrete_label_indices(
            len(dataset.spectra),
            self.current_group_index,
            axes,
        )
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
        axes.scatter(
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
        axes.set_xticks(coordinates[list(label_indices)])
        axes.set_xticklabels([str(index + 1) for index in label_indices], fontsize=7)
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
        q_label_artists: list[Text] = []
        if editable_q_values is not None:
            for index in label_indices:
                q_label_artists.append(
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
                )
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
        self._navigator_q_label_artists = tuple(q_label_artists)

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
            getattr(event, "dblclick", False)
            and event.button is MouseButton.LEFT
            and self.spectrum_axes is not None
            and self.spectrum_axes.xaxis.label.contains(event)[0]
        ):
            self.units_requested.emit()

    def _on_spectrum_q_press(self, event: MouseEvent) -> None:
        if (
            getattr(event, "dblclick", False)
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
        if (
            self._mask_tool is None
            or event.button is not MouseButton.LEFT
            or event.inaxes is not self.spectrum_axes
        ):
            return
        if self._mask_tool == "boundary":
            if event.xdata is None:
                return
            side = self._boundary_side_at(event)
            if side is None:
                return
            self._boundary_drag_side = side
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
        if self._mask_tool is None:
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
        if self._boundary_drag_side is not None:
            side = self._boundary_drag_side
            self._boundary_drag_side = None
            if event.inaxes is self.spectrum_axes and event.xdata is not None:
                self.mask_boundary_requested.emit(
                    self.current_group_index,
                    side,
                    float(event.xdata),
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
            self._cancel_mask_gesture()

    def _boundary_side_at(self, event: MouseEvent) -> str | None:
        """Resolve a press near one visible handle, without midpoint inference."""

        if self.spectrum_axes is None or event.xdata is None:
            return None
        if event.x is not None:
            for side, x_data in self._boundary_handle_positions.items():
                handle_x = self.spectrum_axes.transData.transform((x_data, 0.0))[0]
                if abs(event.x - handle_x) <= 8.0:
                    return side
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
            and event.inaxes is self.spectrum_axes
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
        dataset = self._require_dataset()
        spectrum = dataset.spectra[self.current_group_index]
        effective = ~self._effective_excluded_mask(spectrum.energy.size)
        displayed = spectrum.intensity[np.isfinite(spectrum.intensity) & effective]
        if scale == "log" and (not displayed.size or np.any(displayed <= 0.0)):
            show_message_dialog(
                self,
                "Log Y Scale Unavailable",
                "Log display requires positive visible intensity values. Choose "
                "SymLog to inspect negative, zero, and positive values.",
            )
            return False
        self.y_scale = scale
        self._locked_y_limits = None
        self._draw()
        if self.y_range_locked and self.spectrum_axes is not None:
            self._locked_y_limits = self.spectrum_axes.get_ylim()
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
        canvas_position = (position.x(), self.canvas.height() - position.y())
        if not self.spectrum_axes.bbox.contains(*canvas_position):
            return
        menu = self._build_spectrum_context_menu()
        menu.exec(self.canvas.mapToGlobal(position))

    def _show_overview_context_menu(self, position: QPoint) -> None:
        axes = self.overview_axes or self.navigator_axes
        if axes is None:
            return
        canvas_position = (
            position.x(),
            self.overview_canvas.height() - position.y(),
        )
        if not axes.bbox.contains(*canvas_position):
            return
        menu = QMenu(self)
        self._add_q_assignment_action(menu)
        self._add_mask_edit_action(menu)
        menu.exec(self.overview_canvas.mapToGlobal(position))

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
            axes.set_xlim(*_zoom_limits(axes.get_xlim(), float(x_cursor), scale))
            self._spectrum_x_limits = axes.get_xlim()
        if inside_body or in_y_axis:
            axes.set_ylim(*_zoom_limits(axes.get_ylim(), float(y_cursor), scale))
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


def _intensity_normalization(values: np.ndarray) -> Normalize | None:
    """Create one raw-intensity color mapping shared by every overview column."""

    if not values.size:
        return None
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


def _symlog_linthresh(values: np.ndarray) -> float:
    """Choose a display-only central linear band from the current plotted data."""

    finite = np.abs(values[np.isfinite(values)])
    if not finite.size:
        return 1.0
    return max(float(np.max(finite)) * 0.01, np.finfo(np.float64).tiny)


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


def _zoom_limits(
    limits: tuple[float, float],
    cursor: float,
    scale: float,
) -> tuple[float, float]:
    """Scale a view around its pointer without assigning analytical meaning."""

    lower, upper = limits
    return (
        cursor - (cursor - lower) * scale,
        cursor + (upper - cursor) * scale,
    )
