"""Focused Single-Q AutoFit candidate inspection dialog."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.patches import Rectangle
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QActionGroup, QFont
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ezqens.fitting import CandidateFitResult, FitResult
from ezqens.gui.scientific_canvas import (
    SCIENTIFIC_BACKGROUND,
    ScientificCanvas,
    log_display_values,
    symlog_linthresh,
    zoom_limits,
)
from ezqens.workflow import SingleQAutoFitOutcome

_ZOOM_DRAG_THRESHOLD_PX = 5.0


class AutoFitCandidateDialog(QDialog):
    """Inspect authoritative AutoFit evidence and explicitly choose one result."""

    def __init__(
        self,
        outcome: SingleQAutoFitOutcome,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("autoFitDialog")
        self.setWindowTitle("AutoFit")
        self.setModal(True)
        self.resize(920, 600)
        self.setMinimumSize(720, 480)
        self.outcome = outcome
        self.accepted_candidate: CandidateFitResult | None = None
        self._candidates = outcome.recommendation.candidate_results
        self.y_scale = "linear"
        self.spectrum_axes: Axes | None = None
        self.residual_axes: Axes | None = None
        self._x_limits: tuple[float, float] | None = None
        self._y_limits: tuple[float, float] | None = None
        self._zoom_start: tuple[float, float] | None = None
        self._zoom_start_display: tuple[float, float] | None = None
        self._zoom_rectangle: Rectangle | None = None

        title = QLabel("AutoFit Candidates")
        title.setObjectName("ezqensDialogTitle")
        recommendation = outcome.recommendation
        summary = QLabel(
            " · ".join(
                (
                    recommendation.primary_family_support.value.replace("_", " "),
                    recommendation.primary_residual_adequacy.value.replace("_", " "),
                ),
            ).capitalize(),
        )
        summary.setObjectName("autoFitRecommendationSummary")
        summary.setProperty("secondary", True)
        summary.setWordWrap(True)

        self.candidate_list = QListWidget()
        self.candidate_list.setObjectName("autoFitCandidateList")
        for row, candidate in enumerate(self._candidates):
            item = QListWidgetItem(self._candidate_text(candidate))
            item.setData(Qt.ItemDataRole.UserRole, row)
            if candidate is recommendation.most_recommended:
                font = QFont(item.font())
                font.setBold(True)
                item.setFont(font)
                item.setData(Qt.ItemDataRole.AccessibleDescriptionRole, "Recommended")
            self.candidate_list.addItem(item)

        reason = QLabel(recommendation.additional_complexity.reason)
        reason.setObjectName("autoFitRecommendationReason")
        reason.setProperty("secondary", True)
        reason.setWordWrap(True)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_layout.addWidget(title)
        left_layout.addWidget(summary)
        left_layout.addWidget(self.candidate_list, 1)
        left_layout.addWidget(reason)

        self.preview_title = QLabel()
        self.preview_title.setObjectName("autoFitPreviewTitle")
        self.preview_title.setProperty("secondary", True)
        self.preview_canvas = ScientificCanvas()
        self.preview_canvas.setObjectName("autoFitPreviewCanvas")
        self.preview_canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.preview_canvas.customContextMenuRequested.connect(
            self._show_preview_context_menu
        )
        self.preview_canvas.mpl_connect("scroll_event", self._on_scroll)
        self.preview_canvas.mpl_connect("button_press_event", self._on_button_press)
        self.preview_canvas.mpl_connect("motion_notify_event", self._on_mouse_motion)
        self.preview_canvas.mpl_connect("button_release_event", self._on_button_release)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        right_layout.addWidget(self.preview_title)
        right_layout.addWidget(self.preview_canvas, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("autoFitCandidateSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 600])

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.use_candidate_button = QPushButton("Use Candidate")
        self.use_candidate_button.setObjectName("autoFitUseCandidateButton")
        self.use_candidate_button.setDefault(True)
        self.use_candidate_button.clicked.connect(self._use_focused_candidate)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(6)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.use_candidate_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addWidget(splitter, 1)
        layout.addLayout(buttons)

        self.candidate_list.currentRowChanged.connect(self._focus_candidate)
        initial_row = self._initial_row()
        self.candidate_list.setCurrentRow(initial_row)

    @property
    def focused_candidate(self) -> CandidateFitResult | None:
        """Return the previewed candidate without adopting it."""

        row = self.candidate_list.currentRow()
        return self._candidates[row] if 0 <= row < len(self._candidates) else None

    def reject(self) -> None:
        """Cancel without retaining an adoption choice."""

        self.accepted_candidate = None
        super().reject()

    def _initial_row(self) -> int:
        recommended = self.outcome.recommendation.most_recommended
        if recommended is not None:
            return next(
                index
                for index, candidate in enumerate(self._candidates)
                if candidate is recommended
            )
        return next(
            (
                index
                for index, candidate in enumerate(self._candidates)
                if candidate.success
            ),
            0,
        )

    def _candidate_text(self, candidate: CandidateFitResult) -> str:
        recommended = candidate is self.outcome.recommendation.most_recommended
        suffix = "  ·  Recommended" if recommended else ""
        if candidate.fit is None:
            return f"{candidate.candidate.name}{suffix}\nUnavailable"
        statistics = candidate.fit.statistics
        return (
            f"{candidate.candidate.name}{suffix}\n"
            f"AICc {statistics.aicc:.4g}  ·  BIC {statistics.bic:.4g}  ·  "
            f"reduced χ² {statistics.reduced_chi_square:.4g}"
        )

    def _focus_candidate(self, row: int) -> None:
        self._capture_view_state()
        candidate = self._candidates[row] if 0 <= row < len(self._candidates) else None
        self.use_candidate_button.setEnabled(
            candidate is not None and candidate.success,
        )
        self._draw_candidate(candidate)

    def set_y_scale(self, scale: str) -> bool:
        """Change only the candidate spectrum's Matplotlib presentation scale."""

        if scale not in {"linear", "symlog", "log"}:
            raise ValueError(
                "AutoFit preview y scale must be 'linear', 'symlog', or 'log'"
            )
        if scale == self.y_scale:
            return False
        self._capture_view_state()
        self.y_scale = scale
        self._y_limits = None
        self._draw_candidate(self.focused_candidate)
        return True

    def reset_view(self) -> None:
        """Restore the full current-candidate plot extent without refitting."""

        self._x_limits = None
        self._y_limits = None
        self._cancel_zoom(redraw=False)
        self._draw_candidate(self.focused_candidate)

    def _build_preview_context_menu(self) -> QMenu:
        menu = QMenu(self)
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
                lambda _checked, value=scale: self.set_y_scale(value)
            )
            scale_group.addAction(action)
        menu.addSeparator()
        reset_action = menu.addAction("Reset View")
        reset_action.triggered.connect(self.reset_view)
        return menu

    def _show_preview_context_menu(self, position: QPoint) -> None:
        if self.spectrum_axes is None:
            return
        menu = self._build_preview_context_menu()
        menu.exec(self.preview_canvas.mapToGlobal(position))

    def _draw_candidate(self, candidate: CandidateFitResult | None) -> None:
        figure = self.preview_canvas.figure
        self._cancel_zoom(redraw=False)
        figure.clear()
        self.spectrum_axes = None
        self.residual_axes = None
        figure.set_facecolor(SCIENTIFIC_BACKGROUND)
        if candidate is None:
            self.preview_title.setText("No candidate selected")
            self.preview_canvas.draw_idle()  # type: ignore[no-untyped-call]
            return
        fit = candidate.fit
        self.preview_title.setText(candidate.candidate.name)
        if fit is None:
            axes = figure.add_subplot(111)
            axes.set_facecolor(SCIENTIFIC_BACKGROUND)
            axes.text(
                0.5,
                0.5,
                candidate.error_message or "Candidate fit unavailable",
                transform=axes.transAxes,
                ha="center",
                va="center",
                wrap=True,
            )
            axes.set_axis_off()
            self.preview_canvas.draw_idle()  # type: ignore[no-untyped-call]
            return

        grid = figure.add_gridspec(2, 1, height_ratios=(4.0, 1.0), hspace=0.06)
        spectrum_axes = figure.add_subplot(grid[0])
        residual_axes = figure.add_subplot(grid[1], sharex=spectrum_axes)
        for axes in (spectrum_axes, residual_axes):
            axes.set_facecolor(SCIENTIFIC_BACKGROUND)
            axes.tick_params(colors="#292928")
            for spine in axes.spines.values():
                spine.set_color("#727272")

        measured_points = np.isfinite(self.outcome.measured_energy) & np.isfinite(
            self.outcome.measured_intensity
        )
        if self.y_scale == "log":
            spectrum_axes.set_yscale("log", nonpositive="mask")
            measured_points &= self.outcome.measured_intensity > 0.0
        elif self.y_scale == "symlog":
            spectrum_axes.set_yscale(
                "symlog",
                linthresh=symlog_linthresh(self.outcome.measured_intensity),
            )
        spectrum_axes.errorbar(
            self.outcome.measured_energy[measured_points],
            self.outcome.measured_intensity[measured_points],
            yerr=self.outcome.measured_uncertainty[measured_points],
            fmt=".",
            color="#292928",
            ecolor="#727272",
            elinewidth=0.8,
            capsize=1.5,
            label="Measured",
            zorder=3,
        )
        spectrum_axes.plot(
            fit.evaluation.energy,
            self._spectrum_display_values(fit.evaluation.total),
            color="#314b63",
            linewidth=1.7,
            label="Total fit",
            zorder=4,
        )
        for index, curve in enumerate(fit.evaluation.component_curves, start=1):
            spectrum_axes.plot(
                fit.evaluation.energy,
                self._spectrum_display_values(curve.values),
                linewidth=0.9,
                alpha=0.72,
                label=f"Component {index}",
                zorder=2,
            )
        spectrum_axes.set_ylabel("Intensity")
        spectrum_axes.grid(True, color="#e8e8e8", linewidth=0.6)
        spectrum_axes.legend(loc="best", fontsize=8)
        spectrum_axes.tick_params(axis="x", labelbottom=False)
        if self.y_scale == "log" and not self._has_positive_spectrum_value(fit):
            spectrum_axes.text(
                0.5,
                0.5,
                "No positive values available for Log Y display",
                transform=spectrum_axes.transAxes,
                ha="center",
                va="center",
                color="#676764",
                fontsize=9,
            )

        residual_axes.plot(
            fit.evaluation.energy,
            fit.standardized_residuals,
            color="#496d91",
            linewidth=0.9,
            marker=".",
            markersize=3.0,
        )
        residual_axes.axhline(0.0, color="#727272", linewidth=0.7, alpha=0.7)
        residual_axes.set_xlabel(f"Energy ({fit.provenance.energy_unit})")
        residual_axes.set_ylabel("Std. residual")
        residual_axes.grid(True, color="#ededed", linewidth=0.5)
        self.spectrum_axes = spectrum_axes
        self.residual_axes = residual_axes
        if self._x_limits is not None:
            spectrum_axes.set_xlim(self._x_limits)
        if self._y_limits is not None and (
            self.y_scale != "log" or min(self._y_limits) > 0.0
        ):
            spectrum_axes.set_ylim(self._y_limits)
        self.preview_canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _capture_view_state(self) -> None:
        if self.spectrum_axes is None:
            return
        x_limits = self.spectrum_axes.get_xlim()
        y_limits = self.spectrum_axes.get_ylim()
        self._x_limits = (float(x_limits[0]), float(x_limits[1]))
        self._y_limits = (float(y_limits[0]), float(y_limits[1]))

    def _on_scroll(self, event: MouseEvent) -> None:
        axes = self.spectrum_axes
        residual_axes = self.residual_axes
        if axes is None or residual_axes is None or event.x is None or event.y is None:
            return
        touched = event.inaxes
        if touched not in (axes, residual_axes):
            return
        cursor_axes = touched
        x_cursor, y_cursor = cursor_axes.transData.inverted().transform(
            (event.x, event.y)
        )
        scale = 0.8 if event.button == "up" else 1.25
        axes.set_xlim(*zoom_limits(axes.get_xlim(), float(x_cursor), scale))
        if touched is axes:
            axes.set_ylim(*zoom_limits(axes.get_ylim(), float(y_cursor), scale))
        self._capture_view_state()
        self.preview_canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _on_button_press(self, event: MouseEvent) -> None:
        if (
            event.button is not MouseButton.LEFT
            or event.inaxes is not self.spectrum_axes
            or event.xdata is None
            or event.ydata is None
        ):
            return
        if getattr(event, "dblclick", False):
            self.reset_view()
            return
        display_point = self._event_display_point(event)
        if display_point is None:
            return
        start = (float(event.xdata), float(event.ydata))
        self._cancel_zoom(redraw=False)
        self._zoom_start = start
        self._zoom_start_display = display_point
        self._zoom_rectangle = Rectangle(
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
        self.spectrum_axes.add_patch(self._zoom_rectangle)
        self.preview_canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _on_mouse_motion(self, event: MouseEvent) -> None:
        if (
            self._zoom_start is None
            or self._zoom_rectangle is None
            or event.inaxes is not self.spectrum_axes
            or event.xdata is None
            or event.ydata is None
        ):
            return
        start_x, start_y = self._zoom_start
        end_x, end_y = float(event.xdata), float(event.ydata)
        self._zoom_rectangle.set_bounds(
            min(start_x, end_x),
            min(start_y, end_y),
            abs(end_x - start_x),
            abs(end_y - start_y),
        )
        self.preview_canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _on_button_release(self, event: MouseEvent) -> None:
        start = self._zoom_start
        start_display = self._zoom_start_display
        if start is None or start_display is None:
            return
        if (
            event.inaxes is not self.spectrum_axes
            or event.xdata is None
            or event.ydata is None
        ):
            self._cancel_zoom()
            return
        end = (float(event.xdata), float(event.ydata))
        end_display = self._event_display_point(event)
        self._cancel_zoom(redraw=False)
        if (
            end_display is None
            or max(
                abs(end_display[0] - start_display[0]),
                abs(end_display[1] - start_display[1]),
            )
            < _ZOOM_DRAG_THRESHOLD_PX
        ):
            self.preview_canvas.draw_idle()  # type: ignore[no-untyped-call]
            return
        x_limits = (min(start[0], end[0]), max(start[0], end[0]))
        y_limits = (min(start[1], end[1]), max(start[1], end[1]))
        if x_limits[0] == x_limits[1] or y_limits[0] == y_limits[1]:
            self.preview_canvas.draw_idle()  # type: ignore[no-untyped-call]
            return
        axes = self.spectrum_axes
        if axes is None:
            return
        axes.set_xlim(x_limits)
        axes.set_ylim(y_limits)
        self._capture_view_state()
        self.preview_canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _event_display_point(self, event: MouseEvent) -> tuple[float, float] | None:
        if event.x is not None and event.y is not None:
            return float(event.x), float(event.y)
        if self.spectrum_axes is None or event.xdata is None or event.ydata is None:
            return None
        point = self.spectrum_axes.transData.transform((event.xdata, event.ydata))
        return float(point[0]), float(point[1])

    def _cancel_zoom(self, *, redraw: bool = True) -> None:
        rectangle = self._zoom_rectangle
        self._zoom_rectangle = None
        self._zoom_start = None
        self._zoom_start_display = None
        if rectangle is not None and rectangle.axes is not None:
            rectangle.remove()
            if redraw:
                self.preview_canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _spectrum_display_values(self, values: np.ndarray) -> np.ndarray:
        return log_display_values(values) if self.y_scale == "log" else values

    def _has_positive_spectrum_value(self, fit: FitResult) -> bool:
        arrays = (
            self.outcome.measured_intensity,
            fit.evaluation.total,
            *(curve.values for curve in fit.evaluation.component_curves),
        )
        return any(np.any(np.isfinite(values) & (values > 0.0)) for values in arrays)

    def _use_focused_candidate(self) -> None:
        candidate = self.focused_candidate
        if candidate is None or not candidate.success:
            return
        self.accepted_candidate = candidate
        self.accept()
