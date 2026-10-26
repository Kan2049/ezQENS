"""Real workflow Result projections into the non-modal Parameter Viewer view."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QContextMenuEvent, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QMenu, QToolButton
from test_gui_autofit_result_lifecycle import LifecycleProblem, _seed_single
from test_gui_autofit_result_lifecycle import lifecycle_source as lifecycle_source
from test_gui_result_review_fixes import Problem
from test_gui_result_review_fixes import problem as problem
from test_gui_unified_result import _respond
from test_gui_unified_result import state as state
from test_manual_workflow import make_dataset

from ezqens.domain import QBins, ReducedDataset, SpectrumRole
from ezqens.fitting import (
    ELASTIC_COMPONENT,
    BackgroundModel,
    ManualParameterIntent,
    ParameterFamily,
    ParameterReference,
)
from ezqens.gui.q_dependence import ParameterViewer
from ezqens.gui.result_presentation import (
    SERIES_ALIAS_NOTE,
    lorentzian_slots,
    series_color,
)
from ezqens.gui.scientific_canvas import (
    SCIENTIFIC_BACKGROUND_COLOR,
    SCIENTIFIC_ELASTIC_COLOR,
    SCIENTIFIC_LORENTZIAN_COLORS,
    SCIENTIFIC_MEASURED_COLOR,
    SCIENTIFIC_TOTAL_FIT_COLOR,
    SCIENTIFIC_UNCERTAINTY_COLOR,
)
from ezqens.gui.theme import Q_NAVIGATION_SELECTION
from ezqens.workflow import (
    ManualLorentzianState,
    ManualModelState,
    ManualParameterEdit,
    MultiQConstraintTransferPolicy,
    ResultDisplayStatus,
    adopt_single_q_auto_fit_candidate,
    create_parameter_tie,
    derive_result_in_context,
    fit_all_q_from_current_with_adoption,
    set_current_result,
)


@pytest.fixture
def fitted(
    problem: Problem,
    qt_slot_errors: list[BaseException],
    request: pytest.FixtureRequest,
) -> Iterator[Problem]:
    window, previous_project, _sample, _session = problem
    project = window.workspace.new_project()
    datasets = []
    for dataset in previous_project.datasets:
        spectra = (
            *dataset.dataset.spectra,
            replace(dataset.dataset.spectra[0], group_index=2, group_label="Group 3"),
        )
        current = window.workspace.add_dataset(
            project,
            ReducedDataset(
                role=dataset.dataset.role,
                spectra=spectra,
                q_bins=QBins.from_q_values(getattr(request, "param", (0.4, 0.6, 0.8))),
            ),
        )
        datasets.append(current)
    sample = next(item for item in datasets if item.dataset.role is SpectrumRole.SAMPLE)
    resolution = next(
        item for item in datasets if item.dataset.role is SpectrumRole.RESOLUTION
    )
    window.open_dataset(project, sample)
    assert window.apply_resolution_for_sample(
        project, sample, resolution, replace_confirmed=True
    )
    window.show_manual_fit(project, sample)
    session = window._active_manual_session
    assert session is not None
    model = ManualModelState(
        elastic_area=ManualParameterIntent(0.3, free=False),
        energy_shift=ManualParameterIntent(0.0, free=False),
        lorentzians=tuple(
            ManualLorentzianState(
                area=ManualParameterIntent(0.2, free=False),
                center=ManualParameterIntent(0.0, free=False),
                fwhm=ManualParameterIntent(width, free=False),
            )
            for width in (0.4, 0.8)
        ),
        background=BackgroundModel.LINEAR,
        b0=ManualParameterIntent(0.5),
        b1=ManualParameterIntent(0.0, free=False),
    )
    for index in range(3):
        window._manual_draft = session.draft.with_model(index, model)
        window.dataset_view.set_current_group(index)
        window._run_manual_fit()
        assert window._manual_fit_result is not None
    window.dataset_view.set_current_group(0)
    window.show()
    yield window, project, sample, session
    assert not qt_slot_errors


def _view(state: Problem) -> ParameterViewer:
    QTest.mouseClick(state[0].parameter_viewer_button, Qt.MouseButton.LeftButton)
    view = state[0].parameter_viewer
    assert view is not None
    QApplication.processEvents()
    view.canvas.draw()  # type: ignore[no-untyped-call]
    return view


def _add_panel(view: ParameterViewer, name: str) -> None:
    """Use the actual + popup, catching any exceptions through qt_slot_errors."""

    def choose() -> None:
        menu = QApplication.activePopupWidget()
        assert isinstance(menu, QMenu)
        action = next(action for action in menu.actions() if action.text() == name)
        QTest.mouseClick(
            menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center()
        )

    QTimer.singleShot(0, choose)
    QTest.mouseClick(view.add_button, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    view.canvas.draw()  # type: ignore[no-untyped-call]
    assert name in view.axes and view.panel_buttons[name].isChecked()


def _remove_panel(view: ParameterViewer, name: str, *, toggle: bool) -> None:
    def choose() -> None:
        menu = QApplication.activePopupWidget()
        assert isinstance(menu, QMenu)
        action = next(action for action in menu.actions() if action.text() == "Remove")
        QTest.mouseClick(
            menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center()
        )

    widget = view.panel_buttons[name] if toggle else view.canvas
    if toggle:
        position = widget.rect().center()
    else:
        box = view.axes[name].get_position()
        position = QPoint(
            round((box.x0 + box.width / 2) * view.canvas.width()),
            round((1 - box.y0 - box.height / 2) * view.canvas.height()),
        )
    QTimer.singleShot(0, choose)
    QApplication.sendEvent(
        widget,
        QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse, position, widget.mapToGlobal(position)
        ),
    )
    QApplication.processEvents()
    assert name not in view.panel_buttons and name not in view.axes


def _grid_columns(view: ParameterViewer) -> set[int]:
    columns = set()
    for axes in view.axes.values():
        spec = axes.get_subplotspec()
        assert spec is not None
        columns.add(spec.get_gridspec().ncols)
    return columns


def test_window_buttons_panels_metric_and_close_are_view_only(
    fitted: Problem, tmp_path: Path
) -> None:
    window, _project, _sample, session = fitted
    before, draft = session.result_workspace, session.draft
    view = _view(fitted)
    assert not view.isModal() and view.isVisible()
    assert view.grab().save(str(tmp_path / "q-dependence.png"))
    assert window.grab().save(str(tmp_path / "main-window.png"))
    QTest.mouseClick(window.parameter_viewer_button, Qt.MouseButton.LeftButton)
    assert window.findChildren(ParameterViewer) == [view]
    assert list(view.axes) == ["fwhm", "tau", "eisf"]
    assert not hasattr(window.dataset_view, "fwhm_button")
    assert not hasattr(window.dataset_view, "eisf_button")
    assert not view.findChildren(QComboBox)
    for name in ("fwhm", "tau"):
        QTest.mouseClick(view.panel_buttons[name], Qt.MouseButton.LeftButton)
    assert list(view.axes) == ["eisf"]
    QTest.mouseClick(view.panel_buttons["eisf"], Qt.MouseButton.LeftButton)
    assert not view.axes and view.message.text() == "No panel selected"
    assert view.message.isVisible()
    QTest.mouseClick(view.panel_buttons["eisf"], Qt.MouseButton.LeftButton)
    QTest.mouseClick(view.panel_buttons["tau"], Qt.MouseButton.LeftButton)
    QTest.mouseClick(view.panel_buttons["fwhm"], Qt.MouseButton.LeftButton)
    assert view.axes["tau"].get_ylabel() == "τ (ps)"
    assert (
        view.axes["fwhm"]
        .get_shared_x_axes()
        .joined(view.axes["fwhm"], view.axes["eisf"])
    )
    menu = view.build_context_menu("fwhm")
    next(action for action in menu.actions() if action.text() == "Log").trigger()
    assert view.axes["fwhm"].get_yscale() == "log"
    view.close()
    assert session.result_workspace is before and session.draft is draft
    QTest.mouseClick(window.parameter_viewer_button, Qt.MouseButton.LeftButton)
    assert window.parameter_viewer is view and view.isVisible()


def test_click_does_not_navigate_marker_follows_main_and_needs_fit_gap(
    fitted: Problem,
) -> None:
    window, _project, _sample, session = fitted
    view = _view(fitted)
    before = session.result_workspace
    axes, index, q, value, text = next(hit for hit in view.hits if hit[1] == 2)
    assert "fitted_current" in text
    x, y = axes.transData.transform((q, value))
    event = MouseEvent("button_press_event", view.canvas, x, y, button=MouseButton.LEFT)
    view.canvas.callbacks.process("button_press_event", event)
    view.canvas.callbacks.process(
        "button_release_event",
        MouseEvent("button_release_event", view.canvas, x, y, button=MouseButton.LEFT),
    )
    assert window.dataset_view.current_group_index == view.current_group == 0
    assert session.result_workspace is before
    window.dataset_view.set_current_group(index)
    for ax in view.axes.values():
        np.testing.assert_array_equal(ax.lines[-1].get_xdata(), [q, q])
    window.dataset_view.set_current_group(1)
    model = session.draft.setup(1).model
    assert model is not None
    ref = ParameterReference(model.lorentzians[0].identity, ParameterFamily.FWHM)
    window._update_manual_parameter(
        ref, ManualParameterEdit(0.45, None, None, False, False)
    )
    assert view.statuses[1] is ResultDisplayStatus.NEEDS_FIT
    assert all(hit[1] != 1 for hit in view.hits)
    for line in view.axes["fwhm"].lines:
        if line.get_label() in {"L1", "L2"}:
            assert np.isnan(np.asarray(line.get_ydata())[1])
    assert (
        len([line for line in view.axes["eisf"].lines if line.get_label() == "EISF"])
        == 1
    )


def test_swap_uses_workflow_and_updates_all_identity_projections(
    fitted: Problem,
) -> None:
    window, project, sample, session = fitted
    _respond(("Save Result As", "Before"))
    window._save_manual_result_as()
    entry = session.result_workspace.saved_results[-1]
    original = session.result_workspace.current_result
    assert original is not None
    model = session.draft.setup(1).model
    assert model is not None
    first, second = (item.identity for item in model.lorentzians)
    window.dataset_view.set_current_group(1)
    view = _view(fitted)
    button = next(
        button
        for button in window.manual_fit_editor.findChildren(QToolButton)
        if button.objectName() == "manualComponentRowLabel"
        and button.text() == "L1"
        and button.isVisibleTo(window.manual_fit_editor)
    )
    menu = button.menu()
    assert menu is not None
    action = next(
        action for action in menu.actions() if action.text() == "Swap series with L2"
    )
    assert action.isEnabled()
    assert button.palette().color(QPalette.ColorRole.ButtonText).name() == series_color(
        0
    )
    action.trigger()
    QApplication.processEvents()
    current = session.result_workspace.current_result
    assert current is not None and current is not original
    assert session.result_workspace.saved_results[0].modified_copy is not None
    assert "position swap" in next(hit[4] for hit in view.hits if hit[1] == 1)
    before_fit, after_fit = original.point(1).fit_result, current.point(1).fit_result
    assert before_fit is not None and after_fit is not None
    np.testing.assert_array_equal(
        before_fit.evaluation.component_curve(first),
        after_fit.evaluation.component_curve(first),
    )
    assert lorentzian_slots(current, group_index=1) == view.slots
    for ref, widgets in window.manual_fit_editor._result_rows.items():
        if ref.component in view.slots:
            assert widgets.name.text().startswith(f"L{view.slots[ref.component] + 1} ")
            assert widgets.name.palette().color(
                QPalette.ColorRole.WindowText
            ).name() == series_color(view.slots[ref.component])
    assert window.dataset_view.spectrum_axes is not None
    for line in window.dataset_view.spectrum_axes.lines:
        label = str(line.get_label())
        if label in {"L1", "L2"}:
            assert line.get_color() == series_color(int(label[1:]) - 1)
    window._revert_saved_result(
        project, sample, session.result_workspace.saved_results[0]
    )
    assert session.result_workspace.current_result is entry.snapshot.result
    assert (
        view.derived is not None and view.derived.source_result is entry.snapshot.result
    )


def test_series_stop_only_l2_and_single_lorentzian_swap_absent(fitted: Problem) -> None:
    window, _project, _sample, session = fitted
    model = session.draft.setup(2).model
    assert model is not None
    window.dataset_view.set_current_group(0)
    window._manual_draft = session.draft.with_model(
        0, replace(model, lorentzians=model.lorentzians[:1])
    )
    window._run_manual_fit()
    window.dataset_view.set_current_group(2)
    window._manual_draft = session.draft.with_model(
        2, replace(model, lorentzians=model.lorentzians[1:])
    )
    window._run_manual_fit()
    view = _view(fitted)
    assert view.derived is not None
    assert view.derived.lorentzian_series(1)[2] is not None
    assert view.derived.lorentzian_series(2)[0] is None
    assert view.derived.lorentzian_series(2)[2] is None
    buttons = [
        b
        for b in window.manual_fit_editor.findChildren(QToolButton)
        if b.objectName() == "manualComponentRowLabel"
        and b.isVisibleTo(window.manual_fit_editor)
    ]
    assert [b.text() for b in buttons] == ["L1"]
    menu = buttons[0].menu()
    assert menu is not None and not any("Swap" in a.text() for a in menu.actions())
    assert window.dataset_view.spectrum_axes is not None
    assert {line.get_label() for line in window.dataset_view.spectrum_axes.lines} >= {
        "L1"
    }


def test_follow_run_fresh_load_dataset_switch_and_excluded(fitted: Problem) -> None:
    window, project, sample, session = fitted
    view = _view(fitted)
    _respond(("Save Result As", "Stored"))
    window._save_manual_result_as()
    entry = session.result_workspace.saved_results[-1]
    window._fresh_workspace_dataset(project, sample)
    assert view.derived is None and view.message.text() == "No Result open"
    window._open_saved_result(project, sample, entry)
    assert "Stored" in view.windowTitle()
    resolution = next(item for item in project.datasets if item is not sample)
    window.open_dataset(project, resolution)
    assert view.derived is None
    window.open_dataset(project, sample)
    assert view.derived is not None
    model = session.draft.setup(0).model
    assert session.result_workspace.current_result is not None
    fit = session.result_workspace.current_result.point(0).fit_result
    assert model is not None and fit is not None
    outcome = fit_all_q_from_current_with_adoption(
        window._workflow_project_for(project),
        session.draft,
        anchor_group_index=0,
        current_fit=fit,
        transfer_policy=MultiQConstraintTransferPolicy(model),
        fit_excluded_groups=(1,),
    )
    assert outcome.snapshot is not None
    session.result_workspace = set_current_result(
        session.result_workspace, outcome.snapshot
    )
    assert outcome.adopted_draft is not None
    session._template = outcome.adopted_draft
    window._refresh_manual_fit()
    assert view.statuses[1] is ResultDisplayStatus.EXCLUDED
    assert all(hit[1] != 1 for hit in view.hits)
    # Run the real GUI worker and verify the same view follows its adoption.
    window._start_fit_all_q(session, fit, MultiQConstraintTransferPolicy(model))
    deadline = time.monotonic() + 15
    while window._fit_all_q_thread is not None and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.005)
    assert window._fit_all_q_thread is None
    assert (
        view.derived is not None
        and view.derived.source_result is session.result_workspace.current_result
    )


def test_failed_outcomes_are_gaps_and_unfitted_swap_is_disabled(
    fitted: Problem,
) -> None:
    window, project, sample, session = fitted
    model = session.draft.setup(0).model
    result = session.result_workspace.current_result
    assert model is not None and result is not None
    fit = result.point(0).fit_result
    assert fit is not None
    outcome = fit_all_q_from_current_with_adoption(
        window._workflow_project_for(project),
        session.draft,
        anchor_group_index=0,
        current_fit=fit,
        transfer_policy=MultiQConstraintTransferPolicy(model),
        max_nfev=1,
    )
    assert outcome.snapshot is not None and outcome.adopted_draft is not None
    session.result_workspace = set_current_result(
        session.result_workspace, outcome.snapshot
    )
    session._template = outcome.adopted_draft
    window._refresh_manual_fit()
    view = _view(fitted)
    assert ResultDisplayStatus.FAILED in view.statuses
    for index, status in enumerate(view.statuses):
        if status is ResultDisplayStatus.FAILED:
            assert all(hit[1] != index for hit in view.hits)
    _respond(("Replace Untitled", "Discard"))
    window._fresh_workspace_dataset(project, sample)
    window._manual_draft = session.draft.with_model(0, model)
    window._refresh_manual_fit()
    QApplication.processEvents()
    actions = [
        action
        for button in window.manual_fit_editor.findChildren(QToolButton)
        if button.objectName() == "manualComponentRowLabel"
        and button.isVisibleTo(window.manual_fit_editor)
        and button.menu() is not None
        for action in button.menu().actions()
        if "Swap series" in action.text()
    ]
    assert len(actions) == 2 and all(not action.isEnabled() for action in actions)
    assert view.derived is None


def test_unfitted_qs_are_gaps_and_tooltip_retains_derived_warnings(
    fitted: Problem,
) -> None:
    window, project, sample, session = fitted
    model = session.draft.setup(0).model
    assert model is not None
    _respond(("Replace Untitled", "Discard"))
    window._fresh_workspace_dataset(project, sample)
    window._manual_draft = session.draft.with_model(0, model)
    window._run_manual_fit()
    view = _view(fitted)
    assert view.statuses[1:] == (
        ResultDisplayStatus.NOT_FITTED,
        ResultDisplayStatus.NOT_FITTED,
    )
    assert {hit[1] for hit in view.hits} == {0}
    assert view.derived is not None
    axes, _group, q, value, text = view.hits[0]
    warnings = view.derived.point(0).warnings
    assert warnings and all(w.message in text for w in warnings)
    x, y = axes.transData.transform((q, value))
    view.canvas.callbacks.process(
        "motion_notify_event", MouseEvent("motion_notify_event", view.canvas, x, y)
    )
    assert view.canvas.toolTip() and "Q =" in view.canvas.toolTip()
    for line in axes.lines:
        if line.get_label() in {"L1", "L2"}:
            assert np.all(np.isnan(np.asarray(line.get_ydata())[1:]))


def test_window_title_follows_active_result_and_dataset_rename(fitted: Problem) -> None:
    window, project, sample, session = fitted
    view = _view(fitted)
    _respond(("Save Result As", "Before"))
    window._save_manual_result_as()
    entry = session.result_workspace.saved_results[-1]
    _respond(("Rename Result", "After"))
    window._rename_saved_result(project, sample, entry)
    assert view.windowTitle() == f"Parameter Viewer — {sample.name} · After"
    window.workspace.rename_dataset(project, sample, "Renamed Sample")
    assert view.windowTitle() == "Parameter Viewer — Renamed Sample · After"


@pytest.mark.parametrize("fitted", [(0.4, 0.8, 0.6)], indirect=True)
def test_nonmonotonic_gaps_keep_original_hit_indices_in_every_panel(
    fitted: Problem,
) -> None:
    window, project, sample, session = fitted
    model = session.draft.setup(0).model
    assert model is not None
    _respond(("Replace Untitled", "Discard"))
    window._fresh_workspace_dataset(project, sample)
    for group in (0, 1):
        window._manual_draft = session.draft.with_model(group, model)
        window.dataset_view.set_current_group(group)
        window._run_manual_fit()
    before = session.result_workspace
    view = _view(fitted)
    for _panel, axes in view.axes.items():
        lines = [
            line for line in axes.lines if line.get_label() in {"L1", "L2", "EISF"}
        ]
        assert lines
        for line in lines:
            np.testing.assert_array_equal(line.get_xdata(), (0.4, 0.6, 0.8))
            assert np.isnan(np.asarray(line.get_ydata())[1])
        assert any(
            index == 2 and q == 0.6
            for ax, index, q, _reason in view.unavailable
            if ax is axes
        )
        hit = next(hit for hit in view.hits if hit[0] is axes and hit[1] == 1)
        x, y = axes.transData.transform((hit[2], hit[3]))
        view.canvas.callbacks.process(
            "button_press_event",
            MouseEvent(
                "button_press_event", view.canvas, x, y, button=MouseButton.LEFT
            ),
        )
        view.canvas.callbacks.process(
            "button_release_event",
            MouseEvent(
                "button_release_event", view.canvas, x, y, button=MouseButton.LEFT
            ),
        )
        assert window.dataset_view.current_group_index == 1  # unchanged by the click
    assert session.result_workspace is before
    assert view.derived is not None
    np.testing.assert_array_equal(view.derived.q_bins.q_values, (0.4, 0.8, 0.6))


def test_navigation_is_marker_only_and_closed_view_does_not_compute(
    fitted: Problem, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ezqens.gui.main_window as module

    window, project, sample, session = fitted
    view = _view(fitted)
    axes = dict(view.axes)
    for name, ax in axes.items():
        ax.set_ylim((0.2, 1.1) if name != "tau" else (0.5, 5))
    axes["fwhm"].set_xlim(0.35, 0.85)
    limits = {name: (ax.get_xlim(), ax.get_ylim()) for name, ax in axes.items()}
    calls = []
    original = derive_result_in_context

    def derive(*args: object, **kwargs: object) -> object:
        calls.append(True)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "derive_result_in_context", derive)
    before = session.result_workspace
    for group in (1, 2, 0):
        window.dataset_view.set_current_group(group)
        assert view.current_group == group
        assert view.axes == axes
        for name, ax in axes.items():
            assert (ax.get_xlim(), ax.get_ylim()) == limits[name]
    window.show_parameter_viewer()
    assert not calls and session.result_workspace is before
    view.close()
    window.dataset_view.set_current_group(2)
    assert not calls
    with monkeypatch.context() as patch:

        def unexpected(*args: object) -> None:
            raise AssertionError("closed viewer must not resolve contexts")

        patch.setattr(window, "_manual_context", unexpected)
        window._refresh_parameter_viewer()
    window.show_parameter_viewer()
    assert len(calls) == 1 and view.current_group == 2
    for name, ax in view.axes.items():
        assert (ax.get_xlim(), ax.get_ylim()) == limits[name]

    # A committed scientific-context replacement, not merely navigation, must
    # refresh derivation while leaving the user's display limits intact.
    spectra = sample.dataset.spectra
    window.workspace._replace_dataset(
        project,
        sample,
        replace(
            sample.dataset,
            spectra=(
                replace(spectra[0], intensity=spectra[0].intensity + 0.01),
                *spectra[1:],
            ),
        ),
    )
    assert len(calls) > 1
    context_calls = len(calls)
    assert view.statuses[0] is ResultDisplayStatus.NEEDS_FIT
    for name, ax in view.axes.items():
        assert (ax.get_xlim(), ax.get_ylim()) == limits[name]
    model = session.draft.setup(2).model
    assert model is not None
    ref = ParameterReference(model.lorentzians[0].identity, ParameterFamily.FWHM)
    window._update_manual_parameter(
        ref, ManualParameterEdit(0.45, None, None, False, False)
    )
    assert len(calls) == context_calls + 1
    assert view.statuses[2] is ResultDisplayStatus.NEEDS_FIT
    assert all(hit[1] != 2 for hit in view.hits)
    for name, ax in view.axes.items():
        assert (ax.get_xlim(), ax.get_ylim()) == limits[name]


def test_worker_navigation_moves_existing_markers(fitted: Problem) -> None:
    window, _project, _sample, session = fitted
    view = _view(fitted)
    result = session.result_workspace.current_result
    model = session.draft.setup(0).model
    assert result is not None and model is not None
    fit = result.point(0).fit_result
    assert fit is not None
    seen = []

    def changed(group: int) -> None:
        if window._active_fit_all_q_generation is not None:
            seen.append(group)
            assert view.current_group == group
            assert view.derived is not None
            for marker in view.markers.values():
                np.testing.assert_array_equal(
                    marker.get_xdata(), [view.derived.points[group].q_value] * 2
                )

    window.dataset_view.group_changed.connect(changed)
    window._start_fit_all_q(session, fit, MultiQConstraintTransferPolicy(model))
    deadline = time.monotonic() + 15
    while window._fit_all_q_thread is not None and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.005)
    assert window._fit_all_q_thread is None and seen


def test_context_menu_scales_and_reset_preserve_other_panel_views(
    fitted: Problem,
) -> None:
    view = _view(fitted)
    axes = view.axes["tau"]
    axes.set_ylim(0.5, 10)
    view.axes["eisf"].set_ylim(-0.5, 1.5)

    def choose_log() -> None:
        menu = QApplication.activePopupWidget()
        assert isinstance(menu, QMenu)
        assert [a.text() for a in menu.actions() if not a.isSeparator()] == [
            "Linear",
            "Log",
            "Reset View",
        ]
        action = next(a for a in menu.actions() if a.text() == "Log")
        QTest.mouseClick(
            menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center()
        )

    QTimer.singleShot(0, choose_log)
    box = axes.get_position()
    position = QPoint(1, round((1 - (box.y0 + box.height / 2)) * view.canvas.height()))
    QApplication.sendEvent(
        view.canvas,
        QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse, position, view.canvas.mapToGlobal(position)
        ),
    )
    assert view.scales["tau"] == "log" and axes.get_ylim() == (0.5, 10)
    assert view.axes["fwhm"].get_yscale() == "linear"
    assert view.axes["eisf"].get_ylim() == (-0.5, 1.5)
    model = fitted[3].draft.setup(0).model
    assert model is not None
    fitted[0]._swap_result_series(
        model.lorentzians[0].identity, model.lorentzians[1].identity
    )
    axes = view.axes["tau"]
    assert axes.get_yscale() == "log" and axes.get_ylim() == (0.5, 10)
    assert view.axes["eisf"].get_ylim() == (-0.5, 1.5)
    menu = view.build_context_menu("tau")
    next(a for a in menu.actions() if a.text() == "Reset View").trigger()
    assert axes.get_autoscaley_on()
    assert view.axes["eisf"].get_ylim() == (-0.5, 1.5)


def test_marks_warning_ticks_and_alias_are_visible(fitted: Problem) -> None:
    window, _project, _sample, session = fitted
    view = _view(fitted)
    assert view.alias_note.text() == SERIES_ALIAS_NOTE and view.alias_note.isVisible()
    model = session.draft.setup(1).model
    assert model is not None
    window.dataset_view.set_current_group(1)
    window._swap_result_series(
        model.lorentzians[0].identity, model.lorentzians[1].identity
    )
    for axes in view.axes.values():
        legend = axes.get_legend()
        assert legend is not None
        labels = [label.get_text() for label in legend.get_texts()]
        assert "User swap" in labels and "Derived warning" in labels
        assert "Uncertainty unavailable" in labels
    for button in window.manual_fit_editor.findChildren(QToolButton):
        if button.objectName() == "manualComponentRowLabel":
            assert button.toolTip() == SERIES_ALIAS_NOTE
    for ref, widgets in window.manual_fit_editor._result_rows.items():
        if ref.component in view.slots:
            assert SERIES_ALIAS_NOTE in widgets.name.toolTip()
    ref = ParameterReference(model.lorentzians[0].identity, ParameterFamily.FWHM)
    window._update_manual_parameter(
        ref, ManualParameterEdit(0.5, None, None, False, False)
    )
    view.canvas.draw()  # type: ignore[no-untyped-call]
    for _panel, axes in view.axes.items():
        tick = next(
            tick for tick in view.unavailable if tick[0] is axes and tick[1] == 1
        )
        x, y = axes.get_xaxis_transform().transform((tick[2], 0.025))
        view.canvas.callbacks.process(
            "motion_notify_event", MouseEvent("motion_notify_event", view.canvas, x, y)
        )
        assert "needs_fit" in view.canvas.toolTip()


def test_real_available_uncertainty_matches_info_and_capped_bars(
    problem: Problem, qt_slot_errors: list[BaseException]
) -> None:
    window, _old_project, _old_sample, _old_session = problem
    project = window.workspace.new_project()
    sample = window.workspace.add_dataset(
        project, make_dataset(SpectrumRole.SAMPLE, (0.5,))
    )
    resolution = window.workspace.add_dataset(
        project, make_dataset(SpectrumRole.RESOLUTION, (0.5,))
    )
    window.open_dataset(project, sample)
    assert window.apply_resolution_for_sample(
        project, sample, resolution, replace_confirmed=True
    )
    window.show_manual_fit(project, sample)
    session = window._active_manual_session
    assert session is not None
    model = ManualModelState(
        energy_shift=ManualParameterIntent(0.0, free=False),
        lorentzians=(
            ManualLorentzianState(
                area=ManualParameterIntent(0.05),
                center=ManualParameterIntent(0.0, free=False),
                fwhm=ManualParameterIntent(0.2),
            ),
        ),
        background=BackgroundModel.LINEAR,
        b0=ManualParameterIntent(1.0),
        b1=ManualParameterIntent(0.0, free=False),
    )
    window._manual_draft = session.draft.with_model(0, model)
    window._run_manual_fit()
    fit = window._manual_fit_result
    assert fit is not None
    window.show()
    view = _view((window, project, sample, session))
    reference = ParameterReference(model.lorentzians[0].identity, ParameterFamily.FWHM)
    estimate = fit.parameter_by_reference(reference)
    assert estimate.standard_error is not None and estimate.standard_error > 0
    widgets = window.manual_fit_editor._result_rows[reference]
    assert widgets.uncertainty.text() != "—"
    assert float(widgets.uncertainty.text()) == pytest.approx(
        estimate.standard_error, abs=0.00051
    )
    assert view.derived is not None
    quantity = view.derived.lorentzian_series(1)[0]
    assert quantity is not None
    assert quantity.fwhm_standard_error_mev == estimate.standard_error
    for panel, value, error in (
        ("fwhm", quantity.fwhm_mev, quantity.fwhm_standard_error_mev),
        ("tau", quantity.tau_ps, quantity.tau_standard_error_ps),
    ):
        assert value is not None and error is not None
        containers = [
            container
            for container in view.axes[panel].containers
            if isinstance(container, ErrorbarContainer)
        ]
        assert len(containers) == 1 and containers[0].has_yerr
        marker, caps, bars = containers[0].lines
        assert marker is not None and marker.get_markerfacecolor() == "white"
        assert len(caps) == 2 and all(cap.get_markersize() > 0 for cap in caps)
        np.testing.assert_allclose(
            bars[0].get_segments()[0][:, 1], (value - error, value + error)
        )
    assert not qt_slot_errors


def test_adoption_has_no_obsolete_width_pairing_marks(
    state: LifecycleProblem, qt_slot_errors: list[BaseException]
) -> None:
    _seed_single(state)
    adopted = adopt_single_q_auto_fit_candidate(
        state.workflow,
        state.session.draft,
        state.evaluation,
        state.candidates[1],
    )
    assert adopted.snapshot is not None and adopted.adopted_draft is not None
    state.session.result_workspace = set_current_result(
        state.session.result_workspace, adopted.snapshot, draft=adopted.adopted_draft
    )
    state.window._refresh_manual_fit()
    state.window.show_parameter_viewer()
    view = state.window.parameter_viewer
    assert view is not None
    for axes in view.axes.values():
        legend = axes.get_legend()
        assert legend is not None and "Paired by width" not in [
            text.get_text() for text in legend.get_texts()
        ]
    assert "paired by width" not in view.hits[0][4]
    assert not qt_slot_errors


def test_result_without_lorentzians_has_unavailable_ticks_not_values(
    problem: Problem, qt_slot_errors: list[BaseException]
) -> None:
    window, _project, _sample, _session = problem
    window._run_manual_fit()
    window.show()
    view = _view(problem)
    assert not view.hits
    for axes in view.axes.values():
        assert len([tick for tick in view.unavailable if tick[0] is axes]) == 2
    assert not qt_slot_errors


def test_parameter_menu_grid_remove_and_no_duplicates(
    fitted: Problem, tmp_path: Path, qt_slot_errors: list[BaseException]
) -> None:
    view = _view(fitted)
    assert view.axes["tau"].get_yscale() == "log"
    names = [action.text() for action in view.add_menu.actions()]
    assert names == [
        "δ Area",
        "δ Center",
        "L1 Area",
        "L1 Center",
        "L2 Area",
        "L2 Center",
        "Background Offset",
        "Background Slope",
    ]
    assert len(names) == len(set(names)) and not any("FWHM" in name for name in names)
    assert _grid_columns(view) == {1}
    _add_panel(view, "δ Area")
    assert _grid_columns(view) == {2}
    view.add_panel("δ Area")  # even repeated programmatic requests are idempotent
    assert list(view.panel_buttons).count("δ Area") == 1
    assert "δ Area" not in [action.text() for action in view.add_menu.actions()]
    for name in ("L1 Area", "L1 Center", "Background Offset"):
        _add_panel(view, name)
    assert list(view.axes) == [
        "fwhm",
        "tau",
        "eisf",
        "δ Area",
        "L1 Area",
        "L1 Center",
        "Background Offset",
    ]
    assert _grid_columns(view) == {3}
    assert all(
        view.axes["fwhm"].get_shared_x_axes().joined(view.axes["fwhm"], axes)
        for axes in view.axes.values()
    )
    assert view.controls.indexOf(view.add_button) == len(view.panel_buttons)
    assert view.grab().save(str(tmp_path / "parameter-viewer-seven-panels.png"))
    _remove_panel(view, "L1 Area", toggle=True)
    _remove_panel(view, "Background Offset", toggle=False)
    assert "L1 Area" in [action.text() for action in view.add_menu.actions()]
    assert "Background Offset" in [action.text() for action in view.add_menu.actions()]
    assert not qt_slot_errors


def test_added_panel_auto_hides_and_returns_across_real_saved_results(
    fitted: Problem, qt_slot_errors: list[BaseException]
) -> None:
    window, project, sample, session = fitted
    _respond(("Save Result As", "Two Lorentzians"))
    window._save_manual_result_as()
    original = session.result_workspace.saved_results[-1]
    model = session.draft.setup(0).model
    assert model is not None
    view = _view(fitted)
    _add_panel(view, "L2 Area")
    toggle = view.panel_buttons["L2 Area"]
    window._fresh_workspace_dataset(project, sample)
    for group in range(3):
        window._manual_draft = session.draft.with_model(
            group, replace(model, lorentzians=model.lorentzians[:1])
        )
        window.dataset_view.set_current_group(group)
        window._run_manual_fit()
        assert window._manual_fit_result is not None
    assert not toggle.isEnabled() and toggle.isChecked()
    assert toggle.toolTip() == "Not in this Result" and toggle.isVisible()
    assert "L2 Area" not in view.axes
    assert not any(
        action.text().startswith("L2 ") for action in view.add_menu.actions()
    )
    _respond(("Save Result As", "One Lorentzian"))
    window._save_manual_result_as()
    single = session.result_workspace.saved_results[-1]
    window._open_saved_result(project, sample, original)
    assert view.panel_buttons["L2 Area"] is toggle
    assert toggle.isEnabled() and toggle.isChecked() and not toggle.toolTip()
    assert "L2 Area" in view.axes
    assert original.snapshot is session.result_workspace.current
    window._open_saved_result(project, sample, single)
    assert "L2 Area" not in view.axes and not toggle.isEnabled()
    assert not qt_slot_errors


@pytest.mark.parametrize("fitted", [(0.4, 0.8, 0.6)], indirect=True)
def test_added_parameters_read_exact_estimates_errors_units_and_sorted_gaps(
    fitted: Problem, qt_slot_errors: list[BaseException]
) -> None:
    window, _project, sample, session = fitted
    view = _view(fitted)
    for name in ("δ Area", "L1 Area", "Background Offset", "Background Slope"):
        _add_panel(view, name)
        axes = view.axes[name]
        metadata = view._parameter_items[name]
        expected = []
        for group in (0, 2, 1):
            fit = session.result_workspace.current_result.point(group).fit_result  # type: ignore[union-attr]
            assert fit is not None
            item = metadata[group]
            estimate = fit.parameter_by_reference(item.reference)
            expected.append(estimate.value)
            hit = next(hit for hit in view.hits if hit[0] is axes and hit[1] == group)
            assert hit[3] == estimate.value and item.unit in hit[4]
            errorbars = [c for c in axes.containers if isinstance(c, ErrorbarContainer)]
            bar = errorbars[(0, 2, 1).index(group)]
            if estimate.standard_error is None:
                assert not bar.has_yerr and "Fixed parameter" in hit[4]
            else:
                assert bar.has_yerr
                np.testing.assert_array_equal(
                    bar.lines[2][0].get_segments()[0][:, 1],
                    (
                        estimate.value - estimate.standard_error,
                        estimate.value + estimate.standard_error,
                    ),
                )
        line = next(line for line in axes.lines if line.get_label() == name)
        np.testing.assert_array_equal(line.get_xdata(), (0.4, 0.6, 0.8))
        np.testing.assert_array_equal(line.get_ydata(), expected)
        assert axes.get_ylabel() == f"{name} ({metadata[0].unit})"
    before = session.result_workspace.current_result
    window.dataset_view.set_current_group(2)
    model = session.draft.setup(2).model
    assert model is not None
    window._update_manual_parameter(
        ParameterReference(model.lorentzians[0].identity, ParameterFamily.AREA),
        ManualParameterEdit(0.21, None, None, False, False),
    )
    for name in ("δ Area", "L1 Area", "Background Offset", "Background Slope"):
        axes = view.axes[name]
        line = next(line for line in axes.lines if line.get_label() == name)
        assert np.isnan(np.asarray(line.get_ydata())[1])
        assert not any(hit[0] is axes and hit[1] == 2 for hit in view.hits)
        assert any(tick[0] is axes and tick[1] == 2 for tick in view.unavailable)
        legend = axes.get_legend()
        assert legend is not None
        assert "Unavailable series at Q" in [
            text.get_text() for text in legend.get_texts()
        ]
    assert session.result_workspace.current_result is before
    np.testing.assert_array_equal(sample.dataset.q_bins.q_values, (0.4, 0.8, 0.6))  # type: ignore[union-attr]
    assert not qt_slot_errors


def test_shared_center_estimate_is_listed_per_function_and_fixed_points_marked(
    fitted: Problem, qt_slot_errors: list[BaseException]
) -> None:
    window, _project, _sample, session = fitted
    model = session.draft.setup(0).model
    assert model is not None
    refs = (
        ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER),
        ParameterReference(model.lorentzians[0].identity, ParameterFamily.CENTER),
    )
    window._manual_draft = create_parameter_tie(
        session.draft, 0, refs, source_member=refs[0], tie_group_id="shared-centers"
    )
    window._run_manual_fit()
    fit = window._manual_fit_result
    assert fit is not None and fit.parameter_by_reference(
        refs[0]
    ) is fit.parameter_by_reference(refs[1])
    view = _view(fitted)
    for name in ("δ Center", "L1 Center"):
        _add_panel(view, name)
        hit = next(
            hit for hit in view.hits if hit[0] is view.axes[name] and hit[1] == 0
        )
        assert hit[3] == fit.parameter_by_reference(refs[0]).value
        assert "Tied parameter (shared estimate)" in hit[4]
        assert "Uncertainty unavailable: Fixed parameter" in hit[4]
        assert any(line.get_marker() == "x" for line in view.axes[name].lines)
    assert not qt_slot_errors


def test_semantic_series_and_annotation_colors_are_distinct_with_third_lorentzian(
    fitted: Problem, qt_slot_errors: list[BaseException]
) -> None:
    window, _project, _sample, session = fitted
    model = session.draft.setup(0).model
    assert model is not None
    third = ManualLorentzianState(
        area=ManualParameterIntent(0.1, free=False),
        center=ManualParameterIntent(0.0, free=False),
        fwhm=ManualParameterIntent(1.2, free=False),
    )
    window._manual_draft = session.draft.with_model(
        0, replace(model, lorentzians=(*model.lorentzians, third))
    )
    window._run_manual_fit()
    assert window._manual_fit_result is not None
    view = _view(fitted)
    for name in ("δ Area", "Background Offset", "L3 Area"):
        _add_panel(view, name)
    for panel, color in (
        ("eisf", SCIENTIFIC_ELASTIC_COLOR),
        ("δ Area", SCIENTIFIC_ELASTIC_COLOR),
        ("Background Offset", SCIENTIFIC_BACKGROUND_COLOR),
        ("L3 Area", series_color(2)),
    ):
        name = "EISF" if panel == "eisf" else panel
        assert (
            next(
                line for line in view.axes[panel].lines if line.get_label() == name
            ).get_color()
            == color
        )
    colors = [
        line.get_color()
        for line in view.axes["fwhm"].lines
        if line.get_label() in {"L1", "L2", "L3"}
    ]
    assert colors == list(SCIENTIFIC_LORENTZIAN_COLORS[:3]) and len(set(colors)) == 3
    assert SCIENTIFIC_UNCERTAINTY_COLOR not in {
        *SCIENTIFIC_LORENTZIAN_COLORS,
        SCIENTIFIC_ELASTIC_COLOR,
        SCIENTIFIC_BACKGROUND_COLOR,
        SCIENTIFIC_MEASURED_COLOR,
        SCIENTIFIC_TOTAL_FIT_COLOR,
        Q_NAVIGATION_SELECTION.outline,
    }
    for axes in view.axes.values():
        for line in axes.lines:
            if line.get_marker() in {"D", "x"}:
                assert line.get_color() == SCIENTIFIC_UNCERTAINTY_COLOR
        legend = axes.get_legend()
        assert legend is not None
        for label, handle in zip(
            legend.get_texts(), legend.legend_handles, strict=True
        ):
            if label.get_text() in {"Derived warning", "Uncertainty unavailable"}:
                assert isinstance(handle, Line2D)
                assert handle.get_color() == SCIENTIFIC_UNCERTAINTY_COLOR
    QApplication.processEvents()
    button = next(
        button
        for button in window.manual_fit_editor.findChildren(QToolButton)
        if button.text() == "L3" and button.isVisibleTo(window.manual_fit_editor)
    )
    assert button.palette().color(QPalette.ColorRole.ButtonText).name() == series_color(
        2
    )
    assert window.dataset_view.spectrum_axes is not None
    assert next(
        line
        for line in window.dataset_view.spectrum_axes.lines
        if line.get_label() == "L3"
    ).get_color() == series_color(2)
    assert not qt_slot_errors


def test_box_zoom_links_q_only_target_y_and_owns_release_outside(
    fitted: Problem, qt_slot_errors: list[BaseException]
) -> None:
    window, _project, _sample, session = fitted
    view = _view(fitted)
    _add_panel(view, "L1 Area")
    before = session.result_workspace
    other_y = {
        name: axes.get_ylim() for name, axes in view.axes.items() if name != "tau"
    }
    axes = view.axes["tau"]
    ratio = view.canvas.devicePixelRatioF()
    bbox = axes.bbox
    start = QPoint(
        round((bbox.x0 + bbox.width * 0.25) / ratio),
        round(view.canvas.height() - (bbox.y0 + bbox.height * 0.25) / ratio),
    )
    end = QPoint(
        round((bbox.x0 + bbox.width * 0.75) / ratio),
        round(view.canvas.height() - (bbox.y0 + bbox.height * 0.75) / ratio),
    )
    expected = axes.transData.inverted().transform(
        [
            (start.x() * ratio, (view.canvas.height() - start.y()) * ratio),
            (end.x() * ratio, (view.canvas.height() - end.y()) * ratio),
        ]
    )
    QTest.mousePress(view.canvas, Qt.MouseButton.LeftButton, pos=start)
    assert view.canvas.pointer_capture_active
    QTest.mouseMove(view.canvas, end)
    QTest.mouseRelease(view.canvas, Qt.MouseButton.LeftButton, pos=end)
    assert not view.canvas.pointer_capture_active
    for name, ax in view.axes.items():
        np.testing.assert_allclose(ax.get_xlim(), sorted(expected[:, 0]))
        if name != "tau":
            assert ax.get_ylim() == other_y[name]
    np.testing.assert_allclose(axes.get_ylim(), sorted(expected[:, 1]))
    assert (
        window.dataset_view.current_group_index == 0
        and session.result_workspace is before
    )
    view.canvas.draw()
    start = QPoint(
        round((axes.bbox.x0 + axes.bbox.width * 0.5) / ratio),
        round(view.canvas.height() - (axes.bbox.y0 + axes.bbox.height * 0.5) / ratio),
    )
    QTest.mousePress(view.canvas, Qt.MouseButton.LeftButton, pos=start)
    outside = QPoint(view.canvas.width() + 30, -20)
    QTest.mouseMove(view.canvas, outside)
    QTest.mouseRelease(view.canvas, Qt.MouseButton.LeftButton, pos=outside)
    assert not view.canvas.pointer_capture_active and view._zoom_rectangle is None
    assert window.dataset_view.current_group_index == 0
    window.dataset_view.set_current_group(1)
    assert view.current_group == 1
    for marker in view.markers.values():
        np.testing.assert_array_equal(marker.get_xdata(), (0.6, 0.6))
    assert not qt_slot_errors


def test_removing_last_added_panel_preserves_shared_q_zoom(
    fitted: Problem, qt_slot_errors: list[BaseException]
) -> None:
    view = _view(fitted)
    for name in ("fwhm", "tau", "eisf"):
        QTest.mouseClick(view.panel_buttons[name], Qt.MouseButton.LeftButton)
    _add_panel(view, "L1 Area")
    assert list(view.axes) == ["L1 Area"]
    axes = view.axes["L1 Area"]
    ratio = view.canvas.devicePixelRatioF()
    box = axes.bbox
    start = QPoint(
        round((box.x0 + box.width * 0.25) / ratio),
        round(view.canvas.height() - (box.y0 + box.height * 0.25) / ratio),
    )
    end = QPoint(
        round((box.x0 + box.width * 0.75) / ratio),
        round(view.canvas.height() - (box.y0 + box.height * 0.75) / ratio),
    )
    original = axes.get_xlim()
    QTest.mousePress(view.canvas, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view.canvas, end)
    QTest.mouseRelease(view.canvas, Qt.MouseButton.LeftButton, pos=end)
    zoomed = axes.get_xlim()
    assert zoomed[0] > original[0] and zoomed[1] < original[1]
    assert view._x_limits is None  # the zoom has not yet been captured by a redraw
    _remove_panel(view, "L1 Area", toggle=True)
    assert not view.axes and view.message.text() == "No panel selected"
    QTest.mouseClick(view.panel_buttons["fwhm"], Qt.MouseButton.LeftButton)
    assert view.axes["fwhm"].get_xlim() == zoomed
    assert not qt_slot_errors
