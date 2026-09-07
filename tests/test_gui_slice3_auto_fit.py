"""Focused GUI boundaries for Single-Q AutoFit candidate adoption."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.container import ErrorbarContainer
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMenu, QPushButton, QToolButton

import ezqens.gui.main_window as main_window_module
from ezqens.domain import QBins, ReducedDataset, Spectrum, SpectrumRole
from ezqens.fitting import (
    BACKGROUND_COMPONENT,
    ELASTIC_COMPONENT,
    BackgroundModel,
    CandidateFitResult,
    ManualParameterIntent,
    ParameterFamily,
    ParameterReference,
)
from ezqens.gui import MainWindow, create_application
from ezqens.gui.auto_fit import AutoFitCandidateDialog
from ezqens.gui.manual_fit import ManualFitLifecycle
from ezqens.workflow import (
    ManualModelState,
    ManualParameterEdit,
    commit_fitting_selection,
    run_single_q_auto_fit,
)


@pytest.fixture(scope="module")
def application() -> Iterator[QApplication]:
    app = create_application(["ezqens-autofit-gui-tests"])
    yield app
    app.closeAllWindows()


def _autofit_sample(group_count: int = 2) -> ReducedDataset:
    energy = np.linspace(-0.5, 0.5, 101)
    resolution = np.exp(-0.5 * np.square(energy / 0.04))
    normalized = resolution / np.trapezoid(resolution, energy)
    return ReducedDataset(
        role=SpectrumRole.SAMPLE,
        spectra=tuple(
            Spectrum(
                role=SpectrumRole.SAMPLE,
                group_index=index,
                group_label=f"Group {index + 1}",
                energy=energy,
                intensity=(2.0 + 0.1 * index) * normalized,
                uncertainty=np.full(energy.size, 0.02),
                energy_unit="meV",
                intensity_unit="arb. unit",
                uncertainty_unit="arb. unit",
            )
            for index in range(group_count)
        ),
        q_bins=QBins.from_q_values(
            tuple(0.4 + 0.2 * index for index in range(group_count)),
        ),
    )


def _resolution(sample: ReducedDataset) -> ReducedDataset:
    return ReducedDataset(
        role=SpectrumRole.RESOLUTION,
        spectra=tuple(
            Spectrum(
                role=SpectrumRole.RESOLUTION,
                group_index=spectrum.group_index,
                group_label=spectrum.group_label,
                energy=spectrum.energy,
                intensity=spectrum.intensity,
                uncertainty=spectrum.uncertainty,
                energy_unit=spectrum.energy_unit,
                intensity_unit=spectrum.intensity_unit,
                uncertainty_unit=spectrum.uncertainty_unit,
            )
            for spectrum in sample.spectra
        ),
        q_bins=sample.q_bins,
    )


def test_autofit_dialog_preview_and_explicit_adoption_are_group_local(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    first = window.workspace.add_dataset(project, _autofit_sample())
    resolution = window.workspace.add_dataset(project, _resolution(first.dataset))
    assert window.open_dataset(project, first)
    assert not window.auto_fit_action.isEnabled()
    assert not window.auto_fit_button.isEnabled()
    assert window.apply_resolution_for_sample(
        project,
        first,
        resolution,
        replace_confirmed=True,
    )
    assert window.auto_fit_action.isEnabled()
    assert window.auto_fit_button.isEnabled()
    window.show_manual_fit(project, first)
    first_session = window._active_manual_session
    assert first_session is not None

    second = window.workspace.add_dataset(project, _autofit_sample())
    assert window.apply_resolution_for_sample(
        project,
        second,
        resolution,
        replace_confirmed=True,
    )
    window.show_manual_fit(project, second)
    second_session = window._active_manual_session
    assert second_session is not None
    second_session.draft = second_session.draft.with_model(
        0,
        ManualModelState(
            background=BackgroundModel.CONSTANT,
            b0=ManualParameterIntent(0.2),
        ),
    )

    assert window.open_dataset(project, first)
    first_session = window._active_manual_session
    assert first_session is not None
    workflow = window._workflow_project_for(project)
    outcome = run_single_q_auto_fit(workflow, first_session.draft, 0)
    recommendation = outcome.recommendation.most_recommended
    assert recommendation is not None
    successful = tuple(
        candidate
        for candidate in outcome.recommendation.candidate_results
        if candidate.success
    )
    assert len(successful) > 1
    chosen = next(
        candidate
        for candidate in successful
        if candidate.candidate.lorentzian_count == 2
        and candidate.candidate.background is BackgroundModel.CONSTANT
    )
    assert chosen.fit is not None

    retained_result = chosen.fit
    first_session.draft = first_session.draft.with_model(
        1,
        ManualModelState(
            background=BackgroundModel.CONSTANT,
            b0=ManualParameterIntent(0.3),
        ),
    )
    first_group_one = first_session.execution_state(1)
    first_group_one.lifecycle = ManualFitLifecycle.CURRENT
    first_group_one.fit_result = retained_result
    second_group_zero = second_session.execution_state(0)
    second_group_zero.lifecycle = ManualFitLifecycle.CURRENT
    second_group_zero.fit_result = retained_result
    second_draft = second_session.draft

    dialog = AutoFitCandidateDialog(outcome, window)
    recommended_row = next(
        index
        for index, candidate in enumerate(
            outcome.recommendation.candidate_results,
        )
        if candidate is recommendation
    )
    recommended_item = dialog.candidate_list.item(recommended_row)
    assert recommended_item is not None
    assert "Recommended" in recommended_item.text()
    assert (
        recommended_item.data(Qt.ItemDataRole.AccessibleDescriptionRole)
        == "Recommended"
    )
    chosen_row = next(
        index
        for index, candidate in enumerate(
            outcome.recommendation.candidate_results,
        )
        if candidate is chosen
    )
    before_focus = first_session.draft
    dialog.candidate_list.setCurrentRow(chosen_row)
    application.processEvents()
    assert dialog.focused_candidate is chosen
    assert first_session.draft is before_focus
    assert len(dialog.preview_canvas.figure.axes) == 2
    spectrum_axes, residual_axes = dialog.preview_canvas.figure.axes
    total_line = next(
        line for line in spectrum_axes.lines if line.get_label() == "Total fit"
    )
    np.testing.assert_array_equal(total_line.get_ydata(), chosen.fit.evaluation.total)
    np.testing.assert_array_equal(
        residual_axes.lines[0].get_ydata(),
        chosen.fit.standardized_residuals,
    )
    measured_snapshot = outcome.measured_intensity.copy()
    evaluation_snapshot = chosen.fit.evaluation.total.copy()
    fit_identity = chosen.fit
    visible_scale_buttons = {
        button.text()
        for button in (
            *dialog.findChildren(QPushButton),
            *dialog.findChildren(QToolButton),
        )
        if button.isVisible()
    }
    assert not {"Linear", "Log"} & visible_scale_buttons
    preview_menu = dialog._build_preview_context_menu()
    scale_menu = preview_menu.actions()[0].menu()
    assert isinstance(scale_menu, QMenu)
    assert [action.text() for action in scale_menu.actions()] == ["Linear", "Log"]
    scale_menu.actions()[1].trigger()
    application.processEvents()
    spectrum_axes, residual_axes = dialog.preview_canvas.figure.axes
    assert spectrum_axes.get_yscale() == "log"
    assert residual_axes.get_yscale() == "linear"
    assert dialog.y_scale == "log"

    initial_x_limits = spectrum_axes.get_xlim()
    initial_y_limits = spectrum_axes.get_ylim()
    cursor_x, cursor_y = spectrum_axes.transData.transform((0.0, 1.0))
    dialog._on_scroll(
        cast(
            MouseEvent,
            SimpleNamespace(
                inaxes=spectrum_axes,
                x=cursor_x,
                y=cursor_y,
                button="up",
            ),
        )
    )
    zoomed_x_limits = spectrum_axes.get_xlim()
    zoomed_y_limits = spectrum_axes.get_ylim()
    assert np.ptp(zoomed_x_limits) < np.ptp(initial_x_limits)
    assert np.ptp(zoomed_y_limits) < np.ptp(initial_y_limits)
    comparison_row = next(
        index
        for index, candidate in enumerate(outcome.recommendation.candidate_results)
        if candidate.success and index != chosen_row
    )
    dialog.candidate_list.setCurrentRow(comparison_row)
    application.processEvents()
    assert dialog.y_scale == "log"
    assert dialog.spectrum_axes is not None
    assert dialog.residual_axes is not None
    assert dialog.spectrum_axes.get_xlim() == pytest.approx(zoomed_x_limits)
    assert dialog.spectrum_axes.get_ylim() == pytest.approx(zoomed_y_limits)
    assert dialog.residual_axes.get_xlim() == pytest.approx(zoomed_x_limits)
    assert dialog.residual_axes.get_yscale() == "linear"

    axes = dialog.spectrum_axes
    start = (-0.2, max(float(axes.get_ylim()[0]), 0.01))
    end = (0.2, float(axes.get_ylim()[1]) * 0.9)
    start_display = axes.transData.transform(start)
    end_display = axes.transData.transform(end)
    dialog._on_button_press(
        cast(
            MouseEvent,
            SimpleNamespace(
                button=MouseButton.LEFT,
                inaxes=axes,
                xdata=start[0],
                ydata=start[1],
                x=start_display[0],
                y=start_display[1],
                dblclick=False,
            ),
        )
    )
    dialog._on_mouse_motion(
        cast(
            MouseEvent,
            SimpleNamespace(
                inaxes=axes,
                xdata=end[0],
                ydata=end[1],
            ),
        )
    )
    dialog._on_button_release(
        cast(
            MouseEvent,
            SimpleNamespace(
                inaxes=axes,
                xdata=end[0],
                ydata=end[1],
                x=end_display[0],
                y=end_display[1],
            ),
        )
    )
    rectangle_x_limits = axes.get_xlim()
    assert rectangle_x_limits == pytest.approx((-0.2, 0.2))
    dialog.candidate_list.setCurrentRow(chosen_row)
    application.processEvents()
    assert dialog.spectrum_axes is not None
    assert dialog.spectrum_axes.get_xlim() == pytest.approx(rectangle_x_limits)
    dialog.reset_view()
    assert dialog.spectrum_axes is not None
    assert np.ptp(dialog.spectrum_axes.get_xlim()) > np.ptp(rectangle_x_limits)

    assert dialog.set_y_scale("linear")
    spectrum_axes, residual_axes = dialog.preview_canvas.figure.axes
    assert spectrum_axes.get_yscale() == "linear"
    assert residual_axes.get_yscale() == "linear"
    assert chosen.fit is fit_identity
    np.testing.assert_array_equal(outcome.measured_intensity, measured_snapshot)
    np.testing.assert_array_equal(chosen.fit.evaluation.total, evaluation_snapshot)

    nonpositive_measured = measured_snapshot.copy()
    nonpositive_measured[:2] = (-1.0, 0.0)
    nonpositive_outcome = replace(
        outcome,
        measured_intensity=nonpositive_measured,
    )
    nonpositive_dialog = AutoFitCandidateDialog(nonpositive_outcome, window)
    nonpositive_dialog.candidate_list.setCurrentRow(chosen_row)
    assert nonpositive_dialog.set_y_scale("log")
    spectrum_axes, residual_axes = nonpositive_dialog.preview_canvas.figure.axes
    measured_container = next(
        container
        for container in spectrum_axes.containers
        if container.get_label() == "Measured"
    )
    measured_line = cast(ErrorbarContainer, measured_container).lines[0]
    assert np.all(np.asarray(measured_line.get_ydata()) > 0.0)
    assert spectrum_axes.get_yscale() == "log"
    assert residual_axes.get_yscale() == "linear"
    np.testing.assert_array_equal(
        nonpositive_outcome.measured_intensity,
        nonpositive_measured,
    )
    assert chosen.fit is fit_identity
    nonpositive_dialog.close()
    dialog.reject()
    assert dialog.accepted_candidate is None
    assert first_session.draft is before_focus

    adoption_dialog = AutoFitCandidateDialog(outcome, window)
    adoption_dialog.candidate_list.setCurrentRow(chosen_row)
    adoption_dialog.use_candidate_button.click()
    assert adoption_dialog.accepted_candidate is chosen

    class FakeDialog:
        selected: CandidateFitResult | None = None

        def __init__(self, _outcome: object, _parent: object) -> None:
            self.accepted_candidate = self.selected

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(main_window_module, "run_single_q_auto_fit", lambda *_: outcome)
    monkeypatch.setattr(main_window_module, "AutoFitCandidateDialog", FakeDialog)

    before_cancel = first_session.draft
    window.show_auto_fit(project, first)
    assert first_session.draft is before_cancel
    assert first_session.execution_state(1).fit_result is retained_result
    assert second_session.draft is second_draft
    assert second_session.execution_state(0).fit_result is retained_result

    FakeDialog.selected = chosen
    changed_selection = outcome.scientific_context.selection.with_group_range(
        0,
        lower_energy=-0.4,
        upper_energy=0.4,
    )
    window._workflow_projects[project] = commit_fitting_selection(
        workflow,
        window._workflow_dataset(workflow, first),
        changed_selection,
    )
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        main_window_module,
        "show_message_dialog",
        lambda _parent, title, message: messages.append((title, message)),
    )
    stale_draft = first_session.draft
    stale_group_zero = first_session.execution_state(0)
    stale_lifecycle = stale_group_zero.lifecycle
    stale_result = stale_group_zero.fit_result
    window.show_auto_fit(project, first)
    assert messages and messages[-1][0] == "AutoFit"
    assert first_session.draft is stale_draft
    assert first_session.execution_state(0).lifecycle is stale_lifecycle
    assert first_session.execution_state(0).fit_result is stale_result
    assert first_session.execution_state(1).fit_result is retained_result
    assert second_session.draft is second_draft
    assert second_session.execution_state(0).fit_result is retained_result

    window._workflow_projects[project] = workflow
    window.show_auto_fit(project, first)
    assert window._active_manual_session is first_session
    assert window._manual_fit_result is chosen.fit
    assert window.dataset_view._manual_fit_result is chosen.fit
    assert window.manual_fit_editor.title.text() == "Fitting Parameters"
    adopted_model = first_session.draft.setup(0).model
    assert adopted_model is not None
    assert set(window.manual_fit_editor.parameter_controls) == set(
        adopted_model.parameter_references()
    )
    slope_reference = ParameterReference(
        BACKGROUND_COMPONENT,
        ParameterFamily.SLOPE,
    )
    assert slope_reference in window.manual_fit_editor.parameter_controls
    assert adopted_model.b1 == ManualParameterIntent(0.0, free=False)
    assert window.manual_fit_editor.model_label.text().endswith("B0")
    assert adopted_model.energy_shift is None
    assert len(adopted_model.center_groups) == 1
    center_references = tuple(
        reference
        for reference in adopted_model.parameter_references()
        if reference.family is ParameterFamily.CENTER
    )
    assert (
        adopted_model.center_group_members(adopted_model.center_groups[0].group_id)
        == center_references
    )
    for reference in center_references:
        chain = window.manual_fit_editor.parameter_controls[reference].chain
        assert chain is not None
        assert chain.isChecked()
        assert chain.property("tieColor") == "accent"
    assert first_session.execution_state(1).fit_result is retained_result
    assert first_session.execution_state(1).lifecycle is ManualFitLifecycle.CURRENT
    assert second_session.draft is second_draft
    assert second_session.execution_state(0).fit_result is retained_result

    detached_reference = center_references[-1]
    detached_chain = window.manual_fit_editor.parameter_controls[
        detached_reference
    ].chain
    assert detached_chain is not None
    detached_chain.click()
    detached_model = first_session.draft.setup(0).model
    assert detached_model is not None
    assert detached_model.center_group_for(detached_reference) is None
    assert first_session.execution_state(0).lifecycle is ManualFitLifecycle.NEEDS_FIT
    assert first_session.execution_state(0).fit_result is None
    retie_chain = window.manual_fit_editor.parameter_controls[detached_reference].chain
    assert retie_chain is not None
    retie_chain.click()
    retied_model = first_session.draft.setup(0).model
    assert retied_model is not None
    assert retied_model.center_group_for(detached_reference) is not None
    assert first_session.execution_state(1).fit_result is retained_result
    assert second_session.execution_state(0).fit_result is retained_result

    area_reference = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA)
    area = adopted_model.parameter_intent(area_reference)
    window._update_manual_parameter(
        area_reference,
        ManualParameterEdit(
            current_value=area.current_value + 0.01,
            user_lower_limit=area.user_lower_limit,
            user_upper_limit=area.user_upper_limit,
            free=area.free,
            user_bounds_enabled=area.user_bounds_enabled,
        ),
    )
    assert first_session.execution_state(0).lifecycle is ManualFitLifecycle.NEEDS_FIT
    assert first_session.execution_state(0).fit_result is None
    assert first_session.execution_state(1).fit_result is retained_result
    assert second_session.execution_state(0).fit_result is retained_result

    controls = [
        *window.findChildren(QPushButton),
        *window.findChildren(QToolButton),
    ]
    control_text = {control.text() for control in controls}
    assert not {"MultiFit", "Fit All Groups", "FWHM(Q)", "EISF(Q)"} & control_text
    assert window.manual_fit_button.text() == "Fitting Parameters"
    assert window.manual_fit_action.text() == "Fitting Parameters…"
    assert window.auto_fit_button.text() == "AutoFit…"
    window.close()
