"""Read-only Parameter Viewer; all scientific values come from workflow."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QActionGroup, QCloseEvent, QKeyEvent
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ezqens.derived import DerivedUnifiedQENSResult, DerivedValueStatus, EISFResult
from ezqens.fitting import (
    BACKGROUND_COMPONENT,
    ELASTIC_COMPONENT,
    ComponentIdentity,
    ParameterFamily,
    ParameterMetadata,
)
from ezqens.gui.result_presentation import SERIES_ALIAS_NOTE, series_color
from ezqens.gui.scientific_canvas import (
    SCIENTIFIC_BACKGROUND_COLOR,
    SCIENTIFIC_ELASTIC_COLOR,
    SCIENTIFIC_RESIDUAL_COLOR,
    SCIENTIFIC_TOTAL_FIT_COLOR,
    SCIENTIFIC_UNCERTAINTY_COLOR,
    ScientificCanvas,
    clamped_axes_data_point,
    zoom_limits,
)
from ezqens.gui.theme import Q_NAVIGATION_SELECTION
from ezqens.workflow import IdentityMark, ResultDisplayStatus


class ParameterViewer(QDialog):
    """One owned view with explicit panel scales and retained user limits."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("parameterViewer")
        self.setModal(False)
        self.resize(640, 760)
        self.panel_buttons: dict[str, QToolButton] = {}
        self.controls = QHBoxLayout()
        for panel, label in (("fwhm", "FWHM"), ("tau", "τ"), ("eisf", "EISF")):
            button = QToolButton()
            button.setText(label)
            button.setProperty("viewToggle", True)
            button.setCheckable(True)
            button.setChecked(True)
            button.clicked.connect(self._panels_changed)
            self.panel_buttons[panel] = button
            self.controls.addWidget(button)
        self.add_button = QToolButton()
        self.add_button.setText("+")
        self.add_button.setToolTip("Add a fitted parameter panel")
        self.add_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.add_menu = QMenu(self.add_button)
        self.add_menu.aboutToShow.connect(self._populate_add_menu)
        self.add_button.setMenu(self.add_menu)
        self.controls.addWidget(self.add_button)
        self.controls.addStretch()
        self.alias_note = QLabel(SERIES_ALIAS_NOTE)
        self.alias_note.setProperty("secondary", True)
        self.message = QLabel("No Result open")
        self.canvas = ScientificCanvas()
        self._panel_minimum_width = self.canvas.minimumWidth()
        layout = QVBoxLayout(self)
        layout.addLayout(self.controls)
        layout.addWidget(self.alias_note)
        layout.addWidget(self.message)
        layout.addWidget(self.canvas, 1)
        self.derived: DerivedUnifiedQENSResult | None = None
        self.slots: dict[ComponentIdentity, int] = {}
        self.statuses: tuple[ResultDisplayStatus | None, ...] = ()
        self.current_group = 0
        self.axes: dict[str, Axes] = {}
        self.markers: dict[str, Line2D] = {}
        self.hits: list[tuple[Axes, int, float, float, str]] = []
        self.unavailable: list[tuple[Axes, int, float, str]] = []
        self.scales = {"fwhm": "linear", "tau": "log", "eisf": "linear"}
        # Window-only choices use display names, never identities from an old
        # Result. Each refresh indexes the active Result's typed metadata anew.
        self._parameter_items: dict[str, dict[int, ParameterMetadata]] = {}
        self._x_limits: tuple[float, float] | None = None
        self._y_limits: dict[str, tuple[float, float]] = {}
        self._zoom_panel: str | None = None
        self._zoom_start: tuple[float, float] | None = None
        self._zoom_display: tuple[float, float] | None = None
        self._zoom_rectangle: Rectangle | None = None
        self.canvas.mpl_connect("button_press_event", self._press)
        self.canvas.mpl_connect("button_release_event", self._release)
        self.canvas.mpl_connect("motion_notify_event", self._hover)
        self.canvas.mpl_connect("scroll_event", self._scroll)
        self.canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._show_menu)
        self.setWindowTitle("Parameter Viewer — No Result open")

    def _panels_changed(self) -> None:
        self._draw()

    def _component_name(self, identity: ComponentIdentity) -> str:
        if identity == ELASTIC_COMPONENT:
            return "δ"
        if identity == BACKGROUND_COMPONENT:
            return "Background"
        return f"L{self.slots[identity] + 1}"

    def _parameter_color(self, metadata: ParameterMetadata) -> str:
        if metadata.component == ELASTIC_COMPONENT:
            return SCIENTIFIC_ELASTIC_COLOR
        if metadata.component == BACKGROUND_COMPONENT:
            return SCIENTIFIC_BACKGROUND_COLOR
        return series_color(self.slots[metadata.component])

    def _populate_add_menu(self) -> None:
        self.add_menu.clear()
        for name in self._parameter_items:
            if name not in self.panel_buttons:
                self.add_menu.addAction(name).triggered.connect(
                    lambda _checked=False, value=name: self.add_panel(value)
                )
        self.add_button.setEnabled(bool(self.add_menu.actions()))

    def add_panel(self, name: str) -> None:
        if name not in self._parameter_items or name in self.panel_buttons:
            return
        button = QToolButton()
        button.setText(name)
        button.setProperty("viewToggle", True)
        button.setCheckable(True)
        button.setChecked(True)
        button.clicked.connect(self._panels_changed)
        button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        button.customContextMenuRequested.connect(
            lambda position: self._toggle_menu(name).exec(button.mapToGlobal(position))
        )
        self.panel_buttons[name] = button
        self.scales[name] = "linear"
        self.controls.insertWidget(self.controls.indexOf(self.add_button), button)
        self._populate_add_menu()
        self._draw()

    def _toggle_menu(self, name: str) -> QMenu:
        menu = QMenu(self)
        menu.addAction("Remove").triggered.connect(lambda: self.remove_panel(name))
        return menu

    def remove_panel(self, name: str) -> None:
        if name in {"fwhm", "tau", "eisf"} or name not in self.panel_buttons:
            return
        self._capture_limits()
        button = self.panel_buttons.pop(name)
        self.controls.removeWidget(button)
        button.deleteLater()
        self.scales.pop(name)
        self._y_limits.pop(name, None)
        self._populate_add_menu()
        self._draw()

    def set_result(
        self,
        derived: DerivedUnifiedQENSResult | None,
        slots: dict[ComponentIdentity, int],
        statuses: tuple[ResultDisplayStatus | None, ...],
        group: int,
        title: str,
        parameter_metadata: tuple[tuple[ParameterMetadata, ...], ...] = (),
    ) -> None:
        self.derived, self.slots, self.statuses = derived, slots, statuses
        self.current_group = group
        self.set_title(title)
        self._parameter_items = {}
        for index, items in enumerate(parameter_metadata):
            for item in items:
                if item.family is ParameterFamily.FWHM:
                    continue
                name = f"{self._component_name(item.component)} {item.display_label}"
                self._parameter_items.setdefault(name, {})[index] = item
        for name, button in self.panel_buttons.items():
            if name not in {"fwhm", "tau", "eisf"}:
                available = name in self._parameter_items
                button.setEnabled(available)
                button.setToolTip("" if available else "Not in this Result")
        self._populate_add_menu()
        self._draw()

    def set_title(self, title: str) -> None:
        self.setWindowTitle(
            f"Parameter Viewer — {title}"
            if self.derived is not None
            else "Parameter Viewer — No Result open"
        )

    def set_current_group(self, group: int) -> None:
        """Only move artists: no derivation, axes creation or autoscaling."""
        self.current_group = group
        if self.derived is not None:
            q = self.derived.points[group].q_value
            for marker in self.markers.values():
                marker.set_xdata([q, q])
            self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _capture_limits(self) -> None:
        for panel, axes in self.axes.items():
            if panel not in self.panel_buttons:
                continue
            if not axes.get_autoscalex_on():
                self._x_limits = axes.get_xlim()
            if not axes.get_autoscaley_on():
                self._y_limits[panel] = axes.get_ylim()

    @staticmethod
    def _valid_limits(limits: tuple[float, float], scale: str) -> bool:
        return bool(
            np.all(np.isfinite(limits))
            and limits[0] != limits[1]
            and (scale != "log" or min(limits) > 0)
        )

    def _draw(self) -> None:
        self._capture_limits()
        self._cancel_zoom()
        self.canvas.figure.clear()
        self.axes, self.markers = {}, {}
        self.hits, self.unavailable = [], []
        derived = self.derived
        panels = [
            name
            for name, button in self.panel_buttons.items()
            if button.isChecked() and button.isEnabled()
        ]
        self.message.setText(
            "No Result open" if derived is None else "No panel selected"
        )
        self.message.setVisible(derived is None or not panels)
        self.canvas.setVisible(derived is not None and bool(panels))
        if derived is None or not panels:
            self.canvas.draw_idle()  # type: ignore[no-untyped-call]
            return
        # Stable physical-Q permutation includes the gaps. Stored Result topology
        # and the original group indices used for navigation remain untouched.
        order = np.argsort(derived.q_bins.q_values, kind="stable")
        q_values = derived.q_bins.q_values[order]
        columns = 1 if len(panels) <= 3 else 2 if len(panels) <= 6 else 3
        rows = (len(panels) + columns - 1) // columns
        self.canvas.setMinimumWidth(self._panel_minimum_width * columns)
        grid = self.canvas.figure.add_gridspec(rows, columns, hspace=0.06)
        shared = None
        for index, panel in enumerate(panels):
            axes = self.canvas.figure.add_subplot(
                grid[index // columns, index % columns], sharex=shared
            )
            shared = axes
            self.axes[panel] = axes
            metadata = (
                next(iter(self._parameter_items[panel].values()))
                if panel in self._parameter_items
                else None
            )
            axes.set_ylabel(
                f"{panel} ({metadata.unit})"
                if metadata is not None
                else {"fwhm": "FWHM (meV)", "tau": "τ (ps)", "eisf": "EISF"}[panel]
            )
            if metadata is not None:
                axes.yaxis.label.set_color(self._parameter_color(metadata))
            axes.set_yscale(self.scales[panel])
            if index + columns >= len(panels):
                axes.set_xlabel("Q (Å⁻¹)")
            else:
                axes.tick_params(labelbottom=False)
            entries = (
                [(None, panel, self._parameter_color(metadata))]
                if metadata is not None
                else [(None, "EISF", SCIENTIFIC_ELASTIC_COLOR)]
                if panel == "eisf"
                else [
                    (identity, f"L{slot + 1}", series_color(slot))
                    for identity, slot in self.slots.items()
                ]
            )
            reasons: dict[int, list[str]] = {}
            if not entries:
                reasons = {
                    int(index): [f"No Lorentzian series · {self.statuses[int(index)]}"]
                    for index in order
                }
            explanations: dict[str, str] = {}
            for identity, label, color in entries:
                values = (
                    None
                    if metadata is not None
                    else derived.eisf_series
                    if identity is None
                    else derived.lorentzian_series(identity)
                )
                y = np.full(len(order), np.nan)
                for display_index, group_raw in enumerate(order):
                    group = int(group_raw)
                    item = None if values is None else values[group]
                    parameter = (
                        self._parameter_items[panel].get(group)
                        if metadata is not None
                        else None
                    )
                    point = derived.points[group]
                    status = self.statuses[group]
                    if (
                        (parameter is None if metadata is not None else item is None)
                        or status is not ResultDisplayStatus.FITTED_CURRENT
                        or (
                            item is not None
                            and item.validity is not DerivedValueStatus.AVAILABLE
                        )
                    ):
                        reason = f"{label}: {status}"
                        if status is ResultDisplayStatus.FITTED_CURRENT:
                            reason += f" · {point.derivation_status.value}"
                        if point.source_outcome.error_message:
                            reason += f"\n{point.source_outcome.error_message}"
                        reason += "".join(
                            f"\n{diagnostic.message}"
                            for diagnostic in point.source_outcome.diagnostics
                        )
                        reason += "".join(f"\n{w.message}" for w in point.warnings)
                        if (
                            item is None
                            and status is ResultDisplayStatus.FITTED_CURRENT
                        ):
                            reason += " · series unavailable at this Q"
                        if item is not None:
                            reason += "".join(f"\n{w.message}" for w in item.warnings)
                        reasons.setdefault(group, []).append(reason)
                        continue
                    notes = ""
                    value: float | None
                    if parameter is not None:
                        fit = derived.source_result.point(group).fit_result
                        assert fit is not None
                        estimate = fit.parameter_by_reference(parameter.reference)
                        value, error = estimate.value, estimate.standard_error
                        uncertainty = (
                            "Fixed parameter"
                            if not estimate.free
                            else "Fit covariance unavailable"
                            if not fit.diagnostics.covariance_available
                            else "No standard error supplied"
                        )
                        if len(estimate.references) > 1:
                            notes = "\nTied parameter (shared estimate)"
                    elif isinstance(item, EISFResult):
                        value, error = item.value, item.standard_error
                        uncertainty = item.uncertainty_status.value
                    elif panel == "tau":
                        assert item is not None and not isinstance(item, EISFResult)
                        value, error = item.tau_ps, item.tau_standard_error_ps
                        uncertainty = item.uncertainty_status.value
                    else:
                        assert item is not None and not isinstance(item, EISFResult)
                        value, error = item.fwhm_mev, item.fwhm_standard_error_mev
                        uncertainty = item.uncertainty_status.value
                    if value is None:
                        continue
                    y[display_index] = value
                    warnings = tuple(
                        dict.fromkeys(
                            w.message
                            for w in (
                                *point.warnings,
                                *(item.warnings if item is not None else ()),
                            )
                        )
                    )
                    marks = derived.source_result.point(group).identity_marks
                    marker = "o"
                    paired, swapped = (
                        IdentityMark.PAIRED_BY_WIDTH in marks,
                        IdentityMark.USER_SWAP in marks,
                    )
                    if paired and swapped:
                        marker = "*"
                        explanations["Paired by width + user swap"] = marker
                    elif swapped:
                        marker = "^"
                        explanations["User swap"] = marker
                    elif paired:
                        marker = "s"
                        explanations["Paired by width"] = marker
                    text = f"{label}: {value:.6g}" + (
                        f" ± {error:.6g}"
                        if error is not None
                        else "\nUncertainty unavailable: " + uncertainty
                    )
                    if parameter is not None:
                        text += f"\n{parameter.unit}"
                    text += notes
                    text += f"\nQ = {point.q_value:.6g} · {status}"
                    text += "".join(
                        f"\n{mark.value.replace('_', ' ')}" for mark in marks
                    )
                    text += "".join(f"\n{warning}" for warning in warnings)
                    # Caps and hollow markers expose small available errors;
                    # no scientific error sizes or values are inflated.
                    axes.errorbar(
                        [point.q_value],
                        [value],
                        yerr=None if error is None else [error],
                        color=color,
                        marker=marker,
                        markerfacecolor="white",
                        markersize=4,
                        linestyle="none",
                        capsize=4,
                        capthick=1,
                        barsabove=True,
                        zorder=4,
                    )
                    if warnings:
                        axes.plot(
                            [point.q_value],
                            [value],
                            marker="D",
                            markerfacecolor="none",
                            color=SCIENTIFIC_UNCERTAINTY_COLOR,
                            markersize=8,
                            linestyle="none",
                            zorder=5,
                        )
                        explanations["Derived warning"] = "D"
                    if error is None:
                        axes.plot(
                            [point.q_value],
                            [value],
                            marker="x",
                            color=SCIENTIFIC_UNCERTAINTY_COLOR,
                            markersize=5,
                            linestyle="none",
                            zorder=6,
                        )
                        explanations["Uncertainty unavailable"] = "x"
                    self.hits.append((axes, group, point.q_value, value, text))
                axes.plot(q_values, y, color=color, label=label)
            for group in order:
                index = int(group)
                if index not in reasons:
                    continue
                q = derived.points[index].q_value
                axes.plot(
                    [q],
                    [0.025],
                    transform=axes.get_xaxis_transform(),
                    marker="|",
                    color=SCIENTIFIC_UNCERTAINTY_COLOR,
                    markersize=7,
                    linestyle="none",
                    clip_on=False,
                )
                self.unavailable.append((axes, index, q, "\n".join(reasons[index])))
            if reasons:
                explanations["Unavailable series at Q"] = "|"
            handles, labels = axes.get_legend_handles_labels()
            for label, marker in explanations.items():
                handles.append(
                    Line2D(
                        [],
                        [],
                        marker=marker,
                        color=SCIENTIFIC_UNCERTAINTY_COLOR,
                        markerfacecolor="none",
                        linestyle="none",
                    )
                )
                labels.append(label)
            if handles:
                axes.legend(handles, labels, fontsize="small")
            limits = axes.get_xlim()
            self.markers[panel] = axes.axvline(
                derived.points[self.current_group].q_value,
                color=Q_NAVIGATION_SELECTION.outline,
                linewidth=Q_NAVIGATION_SELECTION.default_linewidth,
            )
            axes.set_xlim(limits, auto=True)
            if panel in self._y_limits:
                limits = self._y_limits[panel]
                if self._valid_limits(limits, self.scales[panel]):
                    axes.set_ylim(limits)
                else:
                    self._y_limits.pop(panel)
        if (
            self._x_limits is not None
            and shared is not None
            and self._valid_limits(self._x_limits, "linear")
        ):
            shared.set_xlim(self._x_limits)
        self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _hit(self, event: MouseEvent) -> tuple[int, str] | None:
        best = None
        distance = 9.0 * self.canvas.devicePixelRatioF()
        for axes, index, q, value, text in self.hits:
            if event.inaxes is not axes:
                continue
            x, y = axes.transData.transform((q, value))
            delta = float(np.hypot(x - event.x, y - event.y))
            if delta < distance:
                best, distance = (index, text), delta
        return best

    def _press(self, event: MouseEvent) -> None:
        panel = next(
            (name for name, axes in self.axes.items() if axes is event.inaxes), None
        )
        if event.button is not MouseButton.LEFT or panel is None:
            return
        if event.dblclick:
            self.reset_view(panel)
            return
        if event.xdata is None or event.ydata is None:
            return
        self._cancel_zoom()
        self._zoom_panel = panel
        self._zoom_start = (float(event.xdata), float(event.ydata))
        self._zoom_display = (float(event.x), float(event.y))
        self._zoom_rectangle = Rectangle(
            self._zoom_start,
            0,
            0,
            facecolor=SCIENTIFIC_RESIDUAL_COLOR,
            edgecolor=SCIENTIFIC_TOTAL_FIT_COLOR,
            alpha=0.16,
            linewidth=1,
            zorder=7,
        )
        self.axes[panel].add_patch(self._zoom_rectangle)
        self.canvas.begin_pointer_capture()
        self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _release(self, event: MouseEvent) -> None:
        panel, start, display = self._zoom_panel, self._zoom_start, self._zoom_display
        if panel is None or start is None or display is None:
            return
        axes = self.axes[panel]
        end, pixels = clamped_axes_data_point(axes, (float(event.x), float(event.y)))
        self._cancel_zoom()
        if (
            max(abs(pixels[0] - display[0]), abs(pixels[1] - display[1]))
            >= 4 * self.canvas.devicePixelRatioF()
        ):
            if end[0] != start[0] and end[1] != start[1]:
                axes.set_xlim(min(start[0], end[0]), max(start[0], end[0]))
                axes.set_ylim(min(start[1], end[1]), max(start[1], end[1]))
        self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _cancel_zoom(self) -> None:
        if self._zoom_rectangle is not None and self._zoom_rectangle.axes is not None:
            self._zoom_rectangle.remove()
        self._zoom_panel, self._zoom_start, self._zoom_display, self._zoom_rectangle = (
            None,
            None,
            None,
            None,
        )
        self.canvas.end_pointer_capture()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._cancel_zoom()
        super().closeEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape and self._zoom_panel is not None:
            self._cancel_zoom()
            self.canvas.draw_idle()  # type: ignore[no-untyped-call]
            event.accept()
            return
        super().keyPressEvent(event)

    def _hover(self, event: MouseEvent) -> None:
        if (
            self._zoom_panel is not None
            and self._zoom_start is not None
            and self._zoom_rectangle is not None
        ):
            end, _pixels = clamped_axes_data_point(
                self.axes[self._zoom_panel], (float(event.x), float(event.y))
            )
            start = self._zoom_start
            self._zoom_rectangle.set_bounds(
                min(start[0], end[0]),
                min(start[1], end[1]),
                abs(end[0] - start[0]),
                abs(end[1] - start[1]),
            )
            self.canvas.draw_idle()  # type: ignore[no-untyped-call]
            return
        hit = self._hit(event)
        if hit is not None:
            self.canvas.setToolTip(hit[1])
            return
        for axes, _index, q, reason in self.unavailable:
            x, y = axes.get_xaxis_transform().transform((q, 0.025))
            if np.hypot(x - event.x, y - event.y) < 9 * self.canvas.devicePixelRatioF():
                self.canvas.setToolTip(reason)
                return
        self.canvas.setToolTip("")

    def _scroll(self, event: MouseEvent) -> None:
        if (
            event.inaxes not in self.axes.values()
            or event.xdata is None
            or event.ydata is None
        ):
            return
        axes = event.inaxes
        assert isinstance(axes, Axes)
        factor = 0.8 if event.step > 0 else 1.25
        axes.set_xlim(zoom_limits(axes.get_xlim(), event.xdata, factor))
        # Display-coordinate zoom works for both Linear and Log Y.
        lower, upper = axes.transData.transform(
            [(event.xdata, value) for value in axes.get_ylim()]
        )[:, 1]
        pixels = zoom_limits((float(lower), float(upper)), event.y, factor)
        limits = axes.transData.inverted().transform(
            [(event.x, value) for value in pixels]
        )[:, 1]
        axes.set_ylim(float(limits[0]), float(limits[1]))
        self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def build_context_menu(self, panel: str) -> QMenu:
        menu = QMenu(self)
        group = QActionGroup(menu)
        group.setExclusive(True)
        for label, scale in (("Linear", "linear"), ("Log", "log")):
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(self.scales[panel] == scale)
            group.addAction(action)
            action.triggered.connect(
                lambda _checked=False, value=scale: self.set_scale(panel, value)
            )
        menu.addSeparator()
        menu.addAction("Reset View").triggered.connect(lambda: self.reset_view(panel))
        if panel in self._parameter_items:
            menu.addSeparator()
            menu.addAction("Remove").triggered.connect(lambda: self.remove_panel(panel))
        return menu

    def _show_menu(self, position: QPoint) -> None:
        if not self.axes:
            return
        # Logical widget coordinates -> figure fractions. Margins belong to the
        # nearest panel too; no mixed Qt/Matplotlib pixel bbox gating.
        x = position.x() / max(1, self.canvas.width())
        y = 1 - position.y() / max(1, self.canvas.height())
        panel = min(
            self.axes,
            key=lambda key: (
                (
                    self.axes[key].get_position().x0
                    + self.axes[key].get_position().width / 2
                    - x
                )
                ** 2
                + (
                    self.axes[key].get_position().y0
                    + self.axes[key].get_position().height / 2
                    - y
                )
                ** 2
            ),
        )
        self.build_context_menu(panel).exec(self.canvas.mapToGlobal(position))

    def set_scale(self, panel: str, scale: str) -> None:
        if scale not in {"linear", "log"}:
            raise ValueError("Parameter Viewer supports Linear and Log")
        self._capture_limits()
        self.scales[panel] = scale
        if panel in self.axes:
            axes = self.axes[panel]
            axes.set_yscale(scale)
            limits = self._y_limits.get(panel)
            if limits is not None and self._valid_limits(limits, scale):
                axes.set_ylim(limits)
            else:
                self._y_limits.pop(panel, None)
                axes.autoscale(enable=True, axis="y")
            self.canvas.draw_idle()  # type: ignore[no-untyped-call]

    def reset_view(self, panel: str) -> None:
        self._x_limits = None
        self._y_limits.pop(panel, None)
        if panel in self.axes:
            for axes in self.axes.values():
                axes.set_autoscalex_on(True)
            self.axes[panel].autoscale(enable=True, axis="both")
            self.canvas.draw_idle()  # type: ignore[no-untyped-call]
