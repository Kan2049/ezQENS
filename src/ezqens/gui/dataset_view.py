"""Read-only scientific view for one imported reduced dataset."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.collections import QuadMesh
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from PySide6.QtCore import QPoint, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ezqens.domain import DiagnosticSeverity, ReducedDataset
from ezqens.gui.scientific_canvas import SCIENTIFIC_BACKGROUND, ScientificCanvas

Q_DISPLAY_UNIT = "Å⁻¹"
OVERVIEW_COLORMAP = "cividis"
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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("reducedDatasetView")
        self.overview_canvas = ScientificCanvas()
        self.overview_canvas.setObjectName("overviewScientificCanvas")
        self.overview_canvas.setMinimumHeight(48)
        self.canvas = ScientificCanvas()
        self.dataset: ReducedDataset | None = None
        self.current_group_index = 0
        self.overview_axes: Axes | None = None
        self.overview_q_axis: object | None = None
        self.navigator_axes: Axes | None = None
        self.spectrum_axes: Axes | None = None
        self.overview_meshes: tuple[QuadMesh, ...] = ()
        self.overview_x_cell_bounds: tuple[tuple[float, float], ...] = ()
        self.overview_active_highlight: Rectangle | None = None
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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(controls)
        layout.addWidget(self.overview_canvas)
        layout.addWidget(self.canvas, 1)

        self._update_overview_toggle()
        self.canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._show_spectrum_context_menu)
        self.overview_canvas.mpl_connect("button_press_event", self._on_button_press)
        self.overview_canvas.mpl_connect("motion_notify_event", self._on_mouse_motion)
        self.overview_canvas.mpl_connect(
            "button_release_event",
            self._on_button_release,
        )
        self.overview_canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def open_dataset(self, dataset: ReducedDataset) -> None:
        """Open real imported data at Group 1 without changing its arrays."""

        self.dataset = dataset
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
        axes.plot(
            spectrum.energy,
            spectrum.intensity,
            color="#496d91",
            linewidth=1.0,
            marker="o",
            markersize=3.0,
        )
        uncertainty_points = finite_coordinates & ~spectrum.invalid_uncertainty_mask
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
        axes.set_xlabel(axis_label("Energy", spectrum.energy_unit))
        axes.set_ylabel(axis_label("Intensity", spectrum.intensity_unit))
        axes.set_title(self._current_coordinate_label(dataset), loc="left", fontsize=10)
        axes.set_yscale(self.y_scale)
        axes.grid(True, color="#e8e8e8", linewidth=0.6)
        if self._spectrum_x_limits is not None:
            axes.set_xlim(self._spectrum_x_limits)
            axes.relim()
            axes.autoscale_view(scalex=False, scaley=True)
        if self.y_range_locked and self._locked_y_limits is not None:
            axes.set_ylim(self._locked_y_limits)

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
        if dataset.q_bins is not None:
            q_axis = axes.secondary_xaxis("bottom")
            q_axis.set_xticks(coordinates[list(label_indices)])
            q_axis.set_xticklabels(
                [f"{coordinates[index]:.4g}" for index in label_indices],
                fontsize=8,
            )
            q_axis.set_xlabel(f"Q ({Q_DISPLAY_UNIT})", color=self._overview_text)
            q_axis.tick_params(
                labelcolor=self._overview_text,
                color=self._overview_border,
            )
            q_axis.spines["bottom"].set_color(self._overview_border)
            self.overview_q_axis = q_axis
        else:
            self.overview_q_axis = None

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
        axes.set_ylim(-0.34 if dataset.q_bins is not None else -0.18, 0.18)
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
        if dataset.q_bins is not None:
            for index in label_indices:
                axes.text(
                    coordinates[index],
                    -0.19,
                    f"{dataset.q_bins.q_values[index]:.4g}",
                    ha="center",
                    va="top",
                    fontsize=7,
                    color=self._overview_text,
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

    def _update_navigation_state(self, dataset: ReducedDataset) -> None:
        self.previous_button.setEnabled(self.current_group_index > 0)
        self.next_button.setEnabled(self.current_group_index + 1 < len(dataset.spectra))
        blocker = QSignalBlocker(self.group_spinbox)
        self.group_spinbox.setValue(self.current_group_index + 1)
        del blocker
        self.group_spinbox.setToolTip(self._group_description(dataset))
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
            event.button is MouseButton.LEFT
            and event.inaxes in (self.overview_axes, self.navigator_axes)
            and event.xdata is not None
        ):
            self._navigation_drag_active = True
            self._select_group_from_navigation_axes(event.inaxes, float(event.xdata))

    def _on_mouse_motion(self, event: MouseEvent) -> None:
        if (
            self._navigation_drag_active
            and event.inaxes in (self.overview_axes, self.navigator_axes)
            and event.xdata is not None
        ):
            self._select_group_from_navigation_axes(event.inaxes, float(event.xdata))

    def _on_button_release(self, _event: MouseEvent) -> None:
        self._navigation_drag_active = False

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

        if scale not in {"linear", "log"}:
            raise ValueError("y scale must be 'linear' or 'log'")
        dataset = self._require_dataset()
        if scale == "log" and not np.any(
            np.isfinite(dataset.spectra[self.current_group_index].intensity)
            & (dataset.spectra[self.current_group_index].intensity > 0.0)
        ):
            QMessageBox.information(
                self,
                "Log Y Scale Unavailable",
                "The current spectrum has no positive intensity values for "
                "log display.",
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
        reset_action = menu.addAction("Reset View")
        reset_action.triggered.connect(self.reset_view)
        scale_menu = menu.addMenu("Y Scale")
        assert scale_menu is not None
        scale_group = QActionGroup(scale_menu)
        scale_group.setExclusive(True)
        for label, scale in (("Linear", "linear"), ("Log", "log")):
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
        return menu

    def _show_spectrum_context_menu(self, position: QPoint) -> None:
        if self.spectrum_axes is None:
            return
        canvas_position = (position.x(), self.canvas.height() - position.y())
        if not self.spectrum_axes.bbox.contains(*canvas_position):
            return
        menu = self._build_spectrum_context_menu()
        menu.exec(self.canvas.mapToGlobal(position))

    def _on_scroll(self, event: MouseEvent) -> None:
        axes = event.inaxes
        if (
            axes is None
            or axes not in (self.spectrum_axes, self.overview_axes)
            or event.xdata is None
        ):
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
