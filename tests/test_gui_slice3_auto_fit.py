"""Focused GUI boundaries for Single-Q AutoFit candidate adoption."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from typing import cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.container import ErrorbarContainer
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QToolButton,
)

import ezqens.gui.auto_fit as auto_fit_module
import ezqens.gui.main_window as main_window_module
from ezqens.batch import (
    MultiQBranchResult,
    MultiQExecutionStatus,
    MultiQFitOutcome,
    MultiQFitStatus,
)
from ezqens.domain import QBins, ReducedDataset, Spectrum, SpectrumRole
from ezqens.fitting import (
    BACKGROUND_COMPONENT,
    ELASTIC_COMPONENT,
    AutoFitRecommendation,
    BackgroundModel,
    CandidateFitResult,
    ComponentFamily,
    ManualParameterIntent,
    ParameterFamily,
    ParameterReference,
    StandardModelCandidate,
)
from ezqens.gui import MainWindow, create_application
from ezqens.gui.auto_fit import AutoFitCandidateDialog
from ezqens.gui.main_window import ManualFitSession
from ezqens.gui.manual_fit import ManualFitLifecycle
from ezqens.gui.scientific_canvas import (
    SCIENTIFIC_BACKGROUND_COLOR,
    SCIENTIFIC_ELASTIC_COLOR,
    SCIENTIFIC_LORENTZIAN_COLORS,
    SCIENTIFIC_RESIDUAL_COLOR,
    SCIENTIFIC_TOTAL_FIT_COLOR,
)
from ezqens.gui.workspace import DatasetState, ProjectState
from ezqens.workflow import (
    AutoFitCandidateBranchResult,
    FittingWorkspaceState,
    ManualFitDraft,
    ManualModelState,
    ManualParameterEdit,
    SelectedAutoFitMultiQResult,
    SingleQAutoFitOutcome,
    WorkflowProject,
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


AutoFitGuiProblem = tuple[
    MainWindow,
    ProjectState,
    DatasetState,
    WorkflowProject,
    ManualFitSession,
    SingleQAutoFitOutcome,
    tuple[CandidateFitResult, ...],
]


@pytest.fixture(scope="module")
def auto_fit_gui_problem(application: QApplication) -> Iterator[AutoFitGuiProblem]:
    window = MainWindow()
    project = window.workspace.new_project()
    sample = window.workspace.add_dataset(project, _autofit_sample(10))
    resolution = window.workspace.add_dataset(project, _resolution(sample.dataset))
    assert window.open_dataset(project, sample)
    assert window.apply_resolution_for_sample(
        project,
        sample,
        resolution,
        replace_confirmed=True,
    )
    window.show_manual_fit(project, sample)
    session = window._active_manual_session
    assert session is not None
    workflow = window._workflow_project_for(project)
    outcome = run_single_q_auto_fit(workflow, session.draft, 0)
    successful = tuple(
        candidate
        for candidate in outcome.recommendation.candidate_results
        if candidate.success
    )
    assert len(successful) >= 4
    yield window, project, sample, workflow, session, outcome, successful
    window.close()


def _wait_until(
    application: QApplication,
    predicate: Callable[[], bool],
    *,
    timeout: float = 3.0,
) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.005)
    application.processEvents()
    assert predicate()


def _candidate_branch(
    evidence: CandidateFitResult,
    group_count: int,
    *,
    status: MultiQExecutionStatus = MultiQExecutionStatus.COMPLETED,
    failed_group: int | None = None,
    processed_groups: int = 1,
) -> AutoFitCandidateBranchResult:
    assert evidence.fit is not None
    outcomes: list[MultiQFitOutcome] = []
    for group_index in range(group_count):
        if (
            status is MultiQExecutionStatus.CANCELLED
            and group_index >= processed_groups
        ):
            outcomes.append(MultiQFitOutcome(group_index, MultiQFitStatus.NOT_RUN))
        elif failed_group == group_index:
            outcomes.append(
                MultiQFitOutcome(
                    group_index,
                    MultiQFitStatus.FAILED,
                    error_type="RuntimeError",
                    error_message="synthetic execution failure",
                )
            )
        else:
            outcomes.append(
                MultiQFitOutcome(
                    group_index,
                    MultiQFitStatus.SUCCESS,
                    evidence.fit,
                )
            )
    branch = MultiQBranchResult(0, status, tuple(outcomes))
    return AutoFitCandidateBranchResult(evidence, branch)


def _select_only(
    dialog: AutoFitCandidateDialog,
    selected: tuple[CandidateFitResult, ...],
) -> None:
    keys = {item.candidate for item in selected}
    for row, candidate in enumerate(dialog.outcome.recommendation.candidate_results):
        dialog.set_candidate_checked(row, candidate.candidate in keys)


def _axes_do_not_overlap(axes: tuple[Axes, ...]) -> bool:
    positions = [item.get_position() for item in axes]
    for index, first in enumerate(positions):
        for second in positions[index + 1 :]:
            separated = (
                first.x1 <= second.x0
                or second.x1 <= first.x0
                or first.y1 <= second.y0
                or second.y1 <= first.y0
            )
            if not separated:
                return False
    return True


def test_candidate_checks_are_eligible_ordered_and_independent_from_focus(
    application: QApplication,
    auto_fit_gui_problem: AutoFitGuiProblem,
) -> None:
    window, _project, _sample, _workflow, _session, outcome, successful = (
        auto_fit_gui_problem
    )
    failed = replace(
        successful[0],
        fit=None,
        error_type="FittingError",
        error_message="candidate unavailable",
    )
    candidates = list(outcome.recommendation.candidate_results)
    failed_row = candidates.index(successful[0])
    candidates[failed_row] = failed
    fallback = successful[1]
    recommendation = cast(
        AutoFitRecommendation,
        SimpleNamespace(
            candidate_results=tuple(candidates),
            most_recommended=fallback,
            best_supported_candidate=fallback,
            primary_family_support=(outcome.recommendation.primary_family_support),
            primary_residual_adequacy=(
                outcome.recommendation.primary_residual_adequacy
            ),
            additional_complexity=outcome.recommendation.additional_complexity,
            transition_assessments=outcome.recommendation.transition_assessments,
            interpretation_limitations=(
                outcome.recommendation.interpretation_limitations
            ),
            scientific_warnings=outcome.recommendation.scientific_warnings,
        ),
    )
    dialog = AutoFitCandidateDialog(
        replace(outcome, recommendation=recommendation),
        window,
    )
    assert len(dialog.findChildren(type(dialog.candidate_checkbox(0)))) == len(
        candidates
    )
    assert not dialog.candidate_checkbox(failed_row).isEnabled()
    assert not dialog.set_candidate_checked(failed_row, True)
    assert not dialog.candidate_checkbox(failed_row).isChecked()

    first_row = candidates.index(fallback)
    second = successful[2]
    second_row = candidates.index(second)
    fallback_item = dialog.candidate_list.item(first_row)
    assert fallback_item is not None
    assert fallback_item.text() == ""
    assert fallback_item.data(Qt.ItemDataRole.DisplayRole) is None
    fallback_widget = dialog._candidate_rows[first_row]
    assert fallback_widget.recommendation_label.text() == "Most Recommended"
    assert "Best Supported" in fallback_widget.accessibleDescription()
    visible_text = [
        label.text() for label in fallback_widget.findChildren(QLabel) if label.text()
    ]
    assert visible_text.count(fallback.candidate.name) == 1
    assert not any(
        metric in text for text in visible_text for metric in ("AICc", "BIC")
    )
    assert fallback.fit is not None
    assert fallback_widget.reduced_chi_square_label.text() == (
        f"χ²/ν {fallback.fit.statistics.reduced_chi_square:.4g}"
    )
    assert "χ²r" not in fallback_widget.reduced_chi_square_label.text()
    row_layout = fallback_widget.layout()
    assert isinstance(row_layout, QHBoxLayout)
    assert fallback_widget.status_dot.property("executionState") == "orange"
    assert fallback_widget.status_dot.toolTip() == "Anchor ready"
    failed_widget = dialog._candidate_rows[failed_row]
    assert failed_widget.status_dot.property("executionState") == "red"
    assert failed_widget.status_dot.toolTip() == "Anchor unavailable"
    dialog.candidate_list.setCurrentRow(first_row)
    active_before_check = dialog.focused_candidate
    dialog.candidate_checkbox(second_row).click()
    application.processEvents()
    assert dialog.focused_candidate is active_before_check
    assert dialog.candidate_checkbox(second_row).isChecked()

    checked = dialog.checked_candidates
    expected = tuple(
        candidate
        for candidate in candidates
        if candidate.success
        and candidate.candidate in {fallback.candidate, second.candidate}
    )
    assert checked == expected
    assert set(dialog.spectrum_axes_by_candidate) == {
        item.candidate for item in checked
    }
    assert set(dialog.residual_axes_by_candidate) == {
        item.candidate for item in checked
    }
    for candidate in checked:
        spectrum_axes = dialog.spectrum_axes_by_candidate[candidate.candidate]
        residual_axes = dialog.residual_axes_by_candidate[candidate.candidate]
        assert spectrum_axes.get_title(loc="left") == candidate.candidate.name
        assert "Anchor ready" not in spectrum_axes.get_title(loc="left")
        assert "Total fit" in {line.get_label() for line in spectrum_axes.lines}
        assert "Std. residual" in {line.get_label() for line in residual_axes.lines}
        assert (
            sum(
                container.get_label() == "Measured"
                for container in spectrum_axes.containers
            )
            == 1
        )

    assert dialog.set_y_scale("symlog")
    assert all(
        axes.get_yscale() == "symlog"
        for axes in dialog.spectrum_axes_by_candidate.values()
    )
    assert all(
        axes.get_yscale() == "linear"
        for axes in dialog.residual_axes_by_candidate.values()
    )
    dialog.reject()


@pytest.mark.parametrize("checked_count", (1, 2, 4, 9))
def test_checked_candidates_use_responsive_independent_panels(
    application: QApplication,
    auto_fit_gui_problem: AutoFitGuiProblem,
    checked_count: int,
) -> None:
    window, _project, _sample, _workflow, _session, outcome, successful = (
        auto_fit_gui_problem
    )
    assert len(successful) >= 9
    selected = successful[:checked_count]
    dialog = AutoFitCandidateDialog(outcome, window)
    _select_only(dialog, selected)
    dialog.resize(1400, 820)
    dialog.show()
    application.processEvents()

    expected = {item.candidate for item in selected}
    assert set(dialog.spectrum_axes_by_candidate) == expected
    assert set(dialog.residual_axes_by_candidate) == expected
    assert len(dialog.preview_canvas.figure.axes) == checked_count * 2
    assert dialog.preview_title.text() == f"Group 1 / 10 · {checked_count} selected"
    assert dialog._comparison_columns <= 3
    wide_columns = dialog._comparison_columns
    assert dialog.preview_canvas.minimumWidth() >= (dialog._comparison_columns * 320)
    assert _axes_do_not_overlap(tuple(dialog.preview_canvas.figure.axes))
    assert not dialog.preview_canvas.grab().isNull()
    assert not dialog.grab().isNull()

    for candidate in selected:
        assert candidate.fit is not None
        spectrum_axes = dialog.spectrum_axes_by_candidate[candidate.candidate]
        residual_axes = dialog.residual_axes_by_candidate[candidate.candidate]
        spectrum_labels = {line.get_label() for line in spectrum_axes.lines}
        expected_components = {
            dialog._component_style(candidate.candidate, curve.component)[0]
            for curve in candidate.fit.evaluation.component_curves
        }
        assert {"Total fit", *expected_components} <= spectrum_labels
        assert "Std. residual" in {line.get_label() for line in residual_axes.lines}

    if checked_count > 1:
        before = tuple(dialog.spectrum_axes_by_candidate)
        focus_row = tuple(outcome.recommendation.candidate_results).index(selected[-1])
        dialog.candidate_list.setCurrentRow(focus_row)
        application.processEvents()
        assert tuple(dialog.spectrum_axes_by_candidate) == before
        assert len(dialog.preview_canvas.figure.axes) == checked_count * 2

    if checked_count >= 4:
        dialog.resize(800, 820)
        application.processEvents()
        assert dialog._comparison_columns <= wide_columns
        assert set(dialog.spectrum_axes_by_candidate) == expected
        assert _axes_do_not_overlap(tuple(dialog.preview_canvas.figure.axes))

    if checked_count == 9:
        assert dialog.comparison_scroll_area.verticalScrollBar().maximum() > 0
        rows = dialog._candidate_rows
        assert all(
            dialog.candidate_list.item(index).text() == ""
            for index in range(dialog.candidate_list.count())
        )
        assert all(
            dialog.candidate_list.item(index).sizeHint() == rows[index].item_size_hint()
            for index in range(dialog.candidate_list.count())
        )
        assert all(
            dialog.candidate_list.item(index).sizeHint().width() == 0
            for index in range(dialog.candidate_list.count())
        )
        for row in rows:
            layout = row.layout()
            assert isinstance(layout, QHBoxLayout)
            assert row.status_dot.size().width() == 8
            assert row.status_dot.size().height() == 8
            visible_text = [
                label.text() for label in row.findChildren(QLabel) if label.text()
            ]
            assert not any(
                metric in text for text in visible_text for metric in ("AICc", "BIC")
            )
            assert row.reduced_chi_square_label.text().startswith("χ²/ν ")
            assert "χ²r" not in row.reduced_chi_square_label.text()
            assert not any(
                status in visible_text
                for status in ("Not run", "Done", "Complete", "Anchor ready")
            )
        item_rects = [
            dialog.candidate_list.visualItemRect(dialog.candidate_list.item(index))
            for index in range(dialog.candidate_list.count())
        ]
        assert all(
            rows[index].geometry() == item_rects[index]
            for index in range(dialog.candidate_list.count())
        )
        assert all(
            first.bottom() + 1 == second.top()
            for first, second in zip(item_rects, item_rects[1:], strict=False)
        )
        assert item_rects[-1].bottom() - item_rects[0].top() + 1 == sum(
            rect.height() for rect in item_rects
        )
        assert len({row.name_label.geometry().left() for row in rows}) == 1
        assert len({row.name_label.width() for row in rows}) == 1
        assert (
            len({row.reduced_chi_square_label.geometry().left() for row in rows}) == 1
        )
        assert len({row.reduced_chi_square_label.width() for row in rows}) == 1
        recommendation_rows = [row for row in rows if row.recommendation_label.text()]
        assert (
            len(
                {
                    row.recommendation_label.geometry().right()
                    for row in recommendation_rows
                }
            )
            == 1
        )
        recommended_row = next(
            row for row in rows if "Recommended" in row.recommendation_label.text()
        )
        assert recommended_row.recommendation_label.text()
        assert recommended_row.status_dot.toolTip()

    dialog.close()


def test_candidate_panels_reuse_function_semantic_colors_and_component_identity(
    auto_fit_gui_problem: AutoFitGuiProblem,
) -> None:
    window, _project, _sample, _workflow, _session, outcome, successful = (
        auto_fit_gui_problem
    )
    two_lorentzian = tuple(
        candidate
        for candidate in successful
        if candidate.candidate.lorentzian_count == 2
        and candidate.candidate.background is not BackgroundModel.NONE
    )
    assert len(two_lorentzian) >= 2
    selected = two_lorentzian[:2]
    dialog = AutoFitCandidateDialog(outcome, window)
    _select_only(dialog, selected)

    for candidate in selected:
        assert candidate.fit is not None
        axes = dialog.spectrum_axes_by_candidate[candidate.candidate]
        colors = {str(line.get_label()): line.get_color() for line in axes.lines}
        assert colors["Total fit"] == SCIENTIFIC_TOTAL_FIT_COLOR
        assert colors["Elastic"] == SCIENTIFIC_ELASTIC_COLOR
        assert colors["L1"] == SCIENTIFIC_LORENTZIAN_COLORS[0]
        assert colors["L2"] == SCIENTIFIC_LORENTZIAN_COLORS[1]
        assert colors["Background"] == SCIENTIFIC_BACKGROUND_COLOR
        residual_axes = dialog.residual_axes_by_candidate[candidate.candidate]
        residual = next(
            line for line in residual_axes.lines if line.get_label() == "Std. residual"
        )
        assert residual.get_color() == SCIENTIFIC_RESIDUAL_COLOR

    evidence = selected[0]
    assert evidence.fit is not None
    anchor_identities = tuple(
        component.identity for component in evidence.fit.configuration.lorentzians
    )
    identity_a, identity_b = anchor_identities
    assert dialog._component_style(evidence.candidate, identity_a) == (
        "L1",
        SCIENTIFIC_LORENTZIAN_COLORS[0],
    )
    assert dialog._component_style(evidence.candidate, identity_b) == (
        "L2",
        SCIENTIFIC_LORENTZIAN_COLORS[1],
    )

    fitted_model = evidence.fit.fitted_model
    assert fitted_model is not None
    fitted_by_identity = {
        component.identity: component for component in fitted_model.lorentzians
    }
    narrow_fwhm, wide_fwhm = sorted(
        (component.fwhm for component in fitted_model.lorentzians),
        key=lambda configuration: configuration.initial_value,
    )
    anchor_a = replace(fitted_by_identity[identity_a], fwhm=narrow_fwhm)
    anchor_b = replace(fitted_by_identity[identity_b], fwhm=wide_fwhm)
    target_a = replace(fitted_by_identity[identity_a], fwhm=wide_fwhm)
    target_b = replace(fitted_by_identity[identity_b], fwhm=narrow_fwhm)
    anchor_fit = replace(
        evidence.fit,
        fitted_model=replace(fitted_model, lorentzians=(anchor_a, anchor_b)),
    )
    target_curves = tuple(
        sorted(
            evidence.fit.evaluation.component_curves,
            key=lambda curve: (
                curve.component.family is not ComponentFamily.ELASTIC,
                0
                if curve.component == identity_b
                else 1
                if curve.component == identity_a
                else 2,
            ),
        )
    )
    target_fit = replace(
        evidence.fit,
        fitted_model=replace(fitted_model, lorentzians=(target_b, target_a)),
        evaluation=replace(
            evidence.fit.evaluation,
            component_curves=target_curves,
        ),
    )
    anchor_fitted_model = anchor_fit.fitted_model
    target_fitted_model = target_fit.fitted_model
    assert anchor_fitted_model is not None
    assert target_fitted_model is not None
    assert tuple(
        component.identity for component in anchor_fitted_model.lorentzians
    ) == (identity_a, identity_b)
    assert tuple(
        component.identity for component in target_fitted_model.lorentzians
    ) == (identity_b, identity_a)
    assert anchor_a.fwhm.initial_value < anchor_b.fwhm.initial_value
    assert target_a.fwhm.initial_value > target_b.fwhm.initial_value

    branch_evidence = replace(evidence, fit=anchor_fit)
    candidate_index = dialog._candidates.index(evidence)
    candidates = list(dialog._candidates)
    candidates[candidate_index] = branch_evidence
    dialog._candidates = tuple(candidates)
    outcomes = tuple(
        MultiQFitOutcome(
            group_index,
            MultiQFitStatus.SUCCESS,
            target_fit if group_index == 1 else anchor_fit,
        )
        for group_index in range(10)
    )
    branch = AutoFitCandidateBranchResult(
        branch_evidence,
        MultiQBranchResult(0, MultiQExecutionStatus.COMPLETED, outcomes),
    )
    dialog._executed_branches[evidence.candidate] = branch
    _select_only(dialog, (branch_evidence,))

    def assert_component_styles(group_index: int) -> None:
        assert dialog.set_current_group(group_index) or dialog.current_group_index == 0
        axes = dialog.spectrum_axes_by_candidate[evidence.candidate]
        fit = anchor_fit if group_index == 0 else target_fit
        curves = {
            curve.component: curve.values
            for curve in fit.evaluation.component_curves
            if curve.component.family is ComponentFamily.LORENTZIAN
        }
        for identity, label, color in (
            (identity_a, "L1", SCIENTIFIC_LORENTZIAN_COLORS[0]),
            (identity_b, "L2", SCIENTIFIC_LORENTZIAN_COLORS[1]),
        ):
            line = next(line for line in axes.lines if line.get_label() == label)
            assert line.get_color() == color
            np.testing.assert_array_equal(line.get_ydata(), curves[identity])

    assert_component_styles(0)
    assert_component_styles(1)
    assert_component_styles(0)

    one_lorentzian = next(
        candidate
        for candidate in successful
        if candidate.candidate.lorentzian_count == 1
    )
    assert one_lorentzian.fit is not None
    one_identity = one_lorentzian.fit.configuration.lorentzians[0].identity
    assert dialog._component_style(one_lorentzian.candidate, one_identity) == (
        "L1",
        SCIENTIFIC_LORENTZIAN_COLORS[0],
    )

    other_branch = selected[1]
    assert other_branch.fit is not None
    other_identity = other_branch.fit.configuration.lorentzians[0].identity
    assert dialog._component_style(other_branch.candidate, other_identity) == (
        "L1",
        SCIENTIFIC_LORENTZIAN_COLORS[0],
    )
    assert (
        dialog._lorentzian_styles_by_candidate[evidence.candidate]
        is not dialog._lorentzian_styles_by_candidate[other_branch.candidate]
    )
    assert dialog._lorentzian_styles_by_candidate[evidence.candidate] == (
        dialog._anchor_lorentzian_styles(evidence)
    )
    assert dialog._lorentzian_styles_by_candidate[other_branch.candidate] == (
        dialog._anchor_lorentzian_styles(other_branch)
    )
    dialog.close()


def test_candidate_status_coverage_and_checked_group_counts_use_cached_results(
    auto_fit_gui_problem: AutoFitGuiProblem,
) -> None:
    window, _project, _sample, workflow, session, outcome, successful = (
        auto_fit_gui_problem
    )
    first, second = successful[:2]
    dialog = AutoFitCandidateDialog(
        outcome,
        window,
        project=workflow,
        draft=session.draft,
    )
    _select_only(dialog, (first, second))
    assert dialog.progress_label.text() == "Calculated groups: 2 / 20"

    partial = _candidate_branch(
        first,
        10,
        status=MultiQExecutionStatus.CANCELLED,
        processed_groups=3,
    )
    dialog._executed_branches[first.candidate] = partial
    dialog._update_candidate_rows()
    dialog._sync_controls()
    first_row = tuple(outcome.recommendation.candidate_results).index(first)
    assert (
        dialog._candidate_rows[first_row].status_dot.property("executionState")
        == "orange"
    )
    assert dialog.progress_label.text() == "Calculated groups: 4 / 20"

    second_row = tuple(outcome.recommendation.candidate_results).index(second)
    assert dialog.set_candidate_checked(second_row, False)
    assert dialog.progress_label.text() == "Calculated groups: 3 / 10"
    assert dialog.set_candidate_checked(second_row, True)
    assert dialog.progress_label.text() == "Calculated groups: 4 / 20"
    dialog.close()


def test_active_candidate_note_uses_only_candidate_specific_core_evidence(
    application: QApplication,
    auto_fit_gui_problem: AutoFitGuiProblem,
) -> None:
    window, _project, _sample, _workflow, _session, outcome, successful = (
        auto_fit_gui_problem
    )
    target, best_supported, neutral = successful[:3]
    transition_reason = "Candidate-specific transition evidence."
    limitation_message = "Candidate-specific interpretation limitation."
    warning_message = "Candidate-specific scientific warning."
    recommendation = cast(
        AutoFitRecommendation,
        SimpleNamespace(
            candidate_results=outcome.recommendation.candidate_results,
            most_recommended=target,
            best_supported_candidate=best_supported,
            primary_family_support=outcome.recommendation.primary_family_support,
            primary_residual_adequacy=(
                outcome.recommendation.primary_residual_adequacy
            ),
            additional_complexity=SimpleNamespace(
                proposed_candidate=None,
                reason="Global family-level note must not be substituted.",
            ),
            transition_assessments=(
                SimpleNamespace(
                    proposed_candidate=target,
                    reason=transition_reason,
                ),
            ),
            interpretation_limitations=(
                SimpleNamespace(
                    candidate=best_supported,
                    message=limitation_message,
                ),
            ),
            scientific_warnings=(
                SimpleNamespace(candidate=best_supported, message=warning_message),
            ),
        ),
    )
    dialog = AutoFitCandidateDialog(
        replace(outcome, recommendation=recommendation),
        window,
    )
    target_row = tuple(outcome.recommendation.candidate_results).index(target)
    best_supported_row = tuple(outcome.recommendation.candidate_results).index(
        best_supported
    )
    neutral_row = tuple(outcome.recommendation.candidate_results).index(neutral)
    assert (
        dialog._candidate_rows[target_row].recommendation_label.text()
        == "Most Recommended"
    )
    assert (
        dialog._candidate_rows[best_supported_row].recommendation_label.text()
        == "Best Supported"
    )

    dialog.candidate_list.setCurrentRow(target_row)
    application.processEvents()
    note = dialog.candidate_note_label.text()
    assert transition_reason in note
    assert limitation_message not in note
    assert warning_message not in note
    assert "Global family-level" not in note

    dialog.candidate_list.setCurrentRow(best_supported_row)
    application.processEvents()
    note = dialog.candidate_note_label.text()
    assert limitation_message in note
    assert warning_message in note
    assert transition_reason not in note
    assert "Global family-level" not in note

    dialog.candidate_list.setCurrentRow(neutral_row)
    application.processEvents()
    assert dialog.candidate_note_label.text() == "No candidate-specific AutoFit note."
    dialog.close()


def test_selected_candidates_run_in_background_cache_and_navigate_together(
    application: QApplication,
    auto_fit_gui_problem: AutoFitGuiProblem,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, _project, _sample, workflow, session, outcome, successful = (
        auto_fit_gui_problem
    )
    selected = successful[:3]
    calls: list[tuple[StandardModelCandidate, ...]] = []

    def complete_selected(
        _project: WorkflowProject,
        _draft: ManualFitDraft,
        _outcome: SingleQAutoFitOutcome,
        candidates: tuple[StandardModelCandidate, ...],
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> SelectedAutoFitMultiQResult:
        del cancel_requested
        calls.append(tuple(candidates))
        evidence = {
            item.candidate: item for item in _outcome.recommendation.candidate_results
        }
        branches = tuple(
            _candidate_branch(evidence[candidate], 10) for candidate in candidates
        )
        return SelectedAutoFitMultiQResult(
            MultiQExecutionStatus.COMPLETED,
            tuple(candidates),
            branches,
        )

    monkeypatch.setattr(
        auto_fit_module,
        "continue_selected_auto_fit_candidates",
        complete_selected,
    )
    dialog = AutoFitCandidateDialog(
        outcome,
        window,
        project=workflow,
        draft=session.draft,
    )
    _select_only(dialog, selected)
    initial_row_heights = tuple(row.minimumHeight() for row in dialog._candidate_rows)
    expected_order = tuple(
        item.candidate
        for item in outcome.recommendation.candidate_results
        if item in selected
    )
    assert dialog.run_selected_across_q()
    assert dialog.is_running
    assert dialog.progress_label.text() == "Calculated groups: 3 / 30"
    assert not dialog.progress_bar.isHidden()
    _wait_until(application, lambda: not dialog.is_running)
    assert calls == [expected_order]
    assert dialog.progress_label.text() == "Calculated groups: 30 / 30"
    assert len(dialog.executed_branches) == 3
    for candidate in selected:
        row = tuple(outcome.recommendation.candidate_results).index(candidate)
        assert (
            dialog._candidate_rows[row].status_dot.property("executionState") == "green"
        )
        assert (
            dialog._candidate_rows[row].status_dot.toolTip() == "All Q groups processed"
        )
    assert tuple(row.minimumHeight() for row in dialog._candidate_rows) == (
        initial_row_heights
    )

    anchor_colors = {
        candidate: next(
            line.get_color()
            for line in dialog.spectrum_axes_by_candidate[candidate].lines
            if line.get_label() == "Total fit"
        )
        for candidate in expected_order
    }
    assert set(anchor_colors.values()) == {SCIENTIFIC_TOTAL_FIT_COLOR}
    assert dialog.set_current_group(1)
    group_colors = {
        candidate: next(
            line.get_color()
            for line in dialog.spectrum_axes_by_candidate[candidate].lines
            if line.get_label() == "Total fit"
        )
        for candidate in expected_order
    }
    assert group_colors == anchor_colors
    assert all(
        next(
            line.get_color()
            for line in dialog.residual_axes_by_candidate[candidate].lines
            if line.get_label() == "Std. residual"
        )
        == SCIENTIFIC_RESIDUAL_COLOR
        for candidate in expected_order
    )

    first_row = tuple(outcome.recommendation.candidate_results).index(selected[0])
    assert dialog.set_candidate_checked(first_row, False)
    assert dialog.progress_label.text() == "Calculated groups: 20 / 20"
    assert dialog.set_candidate_checked(first_row, True)
    assert dialog.progress_label.text() == "Calculated groups: 30 / 30"
    assert not dialog.run_selected_button.isEnabled()
    assert not dialog.run_selected_across_q()
    assert calls == [expected_order]

    not_run = successful[3]
    not_run_row = tuple(outcome.recommendation.candidate_results).index(not_run)
    assert dialog.set_candidate_checked(not_run_row, True)
    assert (
        dialog._candidate_rows[not_run_row].status_dot.property("executionState")
        == "orange"
    )
    assert dialog._candidate_rows[not_run_row].status_dot.toolTip() == "Anchor ready"
    assert dialog.progress_label.text() == "Calculated groups: 31 / 40"
    assert "1 not run" in dialog.selection_status_label.text()
    assert dialog.run_selected_button.isEnabled()
    assert not dialog.use_candidate_button.isEnabled()
    assert dialog.set_candidate_checked(not_run_row, False)
    assert dialog.progress_label.text() == "Calculated groups: 30 / 30"
    assert dialog.use_candidate_button.text() == "Save selected results"
    assert dialog.use_candidate_button.isEnabled()
    dialog.use_candidate_button.click()
    assert tuple(item.candidate for item in dialog.accepted_branches) == expected_order


def test_selected_candidate_cancel_uses_core_contract_and_keeps_prefix(
    application: QApplication,
    auto_fit_gui_problem: AutoFitGuiProblem,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, _project, _sample, workflow, session, outcome, successful = (
        auto_fit_gui_problem
    )
    selected = successful[:2]
    worker_started = Event()
    cancel_seen = Event()

    def wait_for_cancel(
        _project: WorkflowProject,
        _draft: ManualFitDraft,
        _outcome: SingleQAutoFitOutcome,
        candidates: tuple[StandardModelCandidate, ...],
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> SelectedAutoFitMultiQResult:
        assert cancel_requested is not None
        worker_started.set()
        deadline = time.monotonic() + 2.0
        while not cancel_requested() and time.monotonic() < deadline:
            time.sleep(0.005)
        if cancel_requested():
            cancel_seen.set()
        evidence = next(
            item
            for item in _outcome.recommendation.candidate_results
            if item.candidate == candidates[0]
        )
        partial = _candidate_branch(
            evidence,
            10,
            status=MultiQExecutionStatus.CANCELLED,
        )
        return SelectedAutoFitMultiQResult(
            MultiQExecutionStatus.CANCELLED,
            tuple(candidates),
            (partial,),
        )

    monkeypatch.setattr(
        auto_fit_module,
        "continue_selected_auto_fit_candidates",
        wait_for_cancel,
    )
    dialog = AutoFitCandidateDialog(
        outcome,
        window,
        project=workflow,
        draft=session.draft,
    )
    _select_only(dialog, selected)
    assert dialog.run_selected_across_q()
    _wait_until(application, worker_started.is_set)
    ui_tick: list[bool] = []
    QTimer.singleShot(0, lambda: ui_tick.append(True))
    _wait_until(application, lambda: bool(ui_tick))
    assert dialog.is_running
    assert dialog.request_cancel()
    _wait_until(application, lambda: not dialog.is_running)
    assert cancel_seen.is_set()
    assert len(dialog.executed_branches) == 1
    assert dialog.executed_branches[0].candidate == selected[0].candidate
    assert selected[1].candidate not in {
        item.candidate for item in dialog.executed_branches
    }
    assert dialog.progress_label.text() == "Calculated groups: 2 / 20"
    assert "cancelled" in dialog.selection_status_label.text().lower()
    partial_row = tuple(outcome.recommendation.candidate_results).index(selected[0])
    assert (
        dialog._candidate_rows[partial_row].status_dot.property("executionState")
        == "orange"
    )
    assert (
        dialog._candidate_rows[partial_row].status_dot.toolTip() == "Partial Q coverage"
    )
    assert not dialog.use_candidate_button.isEnabled()
    dialog.reject()


def test_unavailable_per_q_outcome_is_not_plotted_as_success(
    auto_fit_gui_problem: AutoFitGuiProblem,
) -> None:
    window, _project, _sample, workflow, session, outcome, successful = (
        auto_fit_gui_problem
    )
    selected = successful[:2]
    dialog = AutoFitCandidateDialog(
        outcome,
        window,
        project=workflow,
        draft=session.draft,
    )
    _select_only(dialog, selected)
    first = _candidate_branch(selected[0], 10)
    second = _candidate_branch(selected[1], 10, failed_group=1)
    dialog._executed_branches = {
        first.candidate: first,
        second.candidate: second,
    }
    dialog._update_candidate_rows()
    dialog._sync_controls()
    assert dialog.set_current_group(1)
    failed_row = tuple(outcome.recommendation.candidate_results).index(selected[1])
    assert (
        dialog._candidate_rows[failed_row].status_dot.property("executionState")
        == "green"
    )
    assert (
        dialog._candidate_rows[failed_row].status_dot.toolTip()
        == "All Q groups processed"
    )
    assert (
        "successful"
        not in dialog._candidate_rows[failed_row].status_dot.toolTip().lower()
    )
    successful_axes = dialog.spectrum_axes_by_candidate[selected[0].candidate]
    failed_axes = dialog.spectrum_axes_by_candidate[selected[1].candidate]
    assert failed_axes.get_title(loc="left") == selected[1].candidate.name
    assert any("Failed" in text.get_text() for text in failed_axes.texts)
    assert "Total fit" in {line.get_label() for line in successful_axes.lines}
    assert "Total fit" not in {line.get_label() for line in failed_axes.lines}
    dialog.reject()


def test_single_apply_and_multi_save_use_existing_result_workspace(
    auto_fit_gui_problem: AutoFitGuiProblem,
) -> None:
    window, _project, _sample, _workflow, session, _outcome, successful = (
        auto_fit_gui_problem
    )
    branches = tuple(_candidate_branch(item, 10) for item in successful[:2])

    session.result_workspace = FittingWorkspaceState()
    window._retain_auto_fit_branches(session, branches[:1])
    assert session.result_workspace.current_result is branches[0].branch_result
    assert session.result_workspace.saved_results == ()

    session.result_workspace = FittingWorkspaceState()
    window._retain_auto_fit_branches(session, branches)
    assert session.result_workspace.current_result is branches[0].branch_result
    assert tuple(
        item.result for item in session.result_workspace.saved_results
    ) == tuple(item.branch_result for item in branches)
    for group_index in range(10):
        execution = session.execution_state(group_index)
        assert execution.lifecycle is ManualFitLifecycle.CURRENT
        assert (
            execution.fit_result
            is branches[0].branch_result.outcome(group_index).fit_result
        )
    window._activate_manual_session(session)
    window.dataset_view.set_current_group(1)
    assert window._manual_fit_result is branches[0].branch_result.outcome(1).fit_result
    assert (
        window.dataset_view._manual_fit_result
        is branches[0].branch_result.outcome(1).fit_result
    )
    saved = session.result_workspace.saved_results
    window._invalidate_manual_session(session, (1,))
    assert session.result_workspace.current_result is None
    assert session.result_workspace.saved_results == saved


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
    assert recommended_item.text() == ""
    assert "Recommended" in str(
        recommended_item.data(Qt.ItemDataRole.AccessibleDescriptionRole)
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
    active_row = dialog.candidate_list.currentRow()
    assert dialog.set_candidate_checked(chosen_row, True)
    assert dialog.candidate_list.currentRow() == active_row
    spectrum_axes = dialog.spectrum_axes_by_candidate[chosen.candidate]
    residual_axes = dialog.residual_axes_by_candidate[chosen.candidate]
    total_line = next(
        line for line in spectrum_axes.lines if line.get_label() == "Total fit"
    )
    np.testing.assert_array_equal(total_line.get_ydata(), chosen.fit.evaluation.total)
    residual_line = next(
        line for line in residual_axes.lines if line.get_label() == "Std. residual"
    )
    np.testing.assert_array_equal(
        residual_line.get_ydata(),
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
    assert not {"Linear", "SymLog", "Log"} & visible_scale_buttons
    preview_menu = dialog._build_preview_context_menu()
    scale_menu = preview_menu.actions()[0].menu()
    assert isinstance(scale_menu, QMenu)
    assert [action.text() for action in scale_menu.actions()] == [
        "Linear",
        "SymLog",
        "Log",
    ]
    scale_menu.actions()[1].trigger()
    application.processEvents()
    spectrum_axes = dialog.spectrum_axes_by_candidate[chosen.candidate]
    residual_axes = dialog.residual_axes_by_candidate[chosen.candidate]
    assert spectrum_axes.get_yscale() == "symlog"
    assert residual_axes.get_yscale() == "linear"
    assert all(
        axes.get_yscale() == "symlog"
        for axes in dialog.spectrum_axes_by_candidate.values()
    )
    assert dialog.y_scale == "symlog"
    scale_menu.actions()[2].trigger()
    application.processEvents()
    spectrum_axes = dialog.spectrum_axes_by_candidate[chosen.candidate]
    residual_axes = dialog.residual_axes_by_candidate[chosen.candidate]
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
    assert all(
        axes.get_xlim() == pytest.approx(zoomed_x_limits)
        and axes.get_ylim() == pytest.approx(zoomed_y_limits)
        for axes in dialog.spectrum_axes_by_candidate.values()
    )
    assert all(
        axes.get_xlim() == pytest.approx(zoomed_x_limits)
        for axes in dialog.residual_axes_by_candidate.values()
    )
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
    assert all(
        axes.get_xlim() == pytest.approx(dialog.spectrum_axes.get_xlim())
        for axes in dialog.spectrum_axes_by_candidate.values()
    )

    assert dialog.set_y_scale("linear")
    spectrum_axes = dialog.spectrum_axes_by_candidate[chosen.candidate]
    residual_axes = dialog.residual_axes_by_candidate[chosen.candidate]
    assert spectrum_axes.get_yscale() == "linear"
    assert residual_axes.get_yscale() == "linear"
    assert chosen.fit is fit_identity
    np.testing.assert_array_equal(outcome.measured_intensity, measured_snapshot)
    np.testing.assert_array_equal(chosen.fit.evaluation.total, evaluation_snapshot)

    opened_at: list[QPoint] = []

    class RecordingMenu:
        def exec(self, position: QPoint) -> None:
            opened_at.append(position)

    with monkeypatch.context() as context:
        context.setattr(
            dialog,
            "_build_preview_context_menu",
            lambda: RecordingMenu(),
        )
        dialog._show_preview_context_menu(
            QPoint(
                dialog.preview_canvas.width() // 2,
                dialog.preview_canvas.height() - 1,
            )
        )
    assert len(opened_at) == 1
    assert (
        dialog.preview_title.contextMenuPolicy()
        is not Qt.ContextMenuPolicy.CustomContextMenu
    )

    signed_measured = np.linspace(-1.0, 1.0, measured_snapshot.size)
    signed_curve = np.linspace(-0.75, 0.75, chosen.fit.evaluation.energy.size)
    signed_evaluation = replace(
        chosen.fit.evaluation,
        total=signed_curve,
        background=signed_curve,
        component_curves=tuple(
            replace(curve, values=signed_curve)
            for curve in chosen.fit.evaluation.component_curves
        ),
    )
    signed_fit = replace(chosen.fit, evaluation=signed_evaluation)
    signed_candidate = replace(chosen, fit=signed_fit)
    signed_outcome = replace(outcome, measured_intensity=signed_measured)
    signed_dialog = AutoFitCandidateDialog(signed_outcome, window)
    assert signed_dialog.set_y_scale("symlog")
    signed_dialog._draw_candidate(signed_candidate)
    signed_spectrum = signed_dialog.spectrum_axes_by_candidate[chosen.candidate]
    signed_residual = signed_dialog.residual_axes_by_candidate[chosen.candidate]
    measured_container = next(
        container
        for container in signed_spectrum.containers
        if container.get_label() == "Measured"
    )
    signed_measured_line = cast(ErrorbarContainer, measured_container).lines[0]
    assert np.any(np.asarray(signed_measured_line.get_ydata()) < 0.0)
    signed_model_lines = tuple(
        line
        for line in signed_spectrum.lines
        if line.get_label() in {"Total fit", "Elastic", "L1", "L2", "Background"}
    )
    assert signed_model_lines
    assert all(
        np.any(np.asarray(line.get_ydata()) < 0.0) for line in signed_model_lines
    )
    assert signed_spectrum.get_yscale() == "symlog"
    assert signed_residual.get_yscale() == "linear"
    np.testing.assert_array_equal(signed_outcome.measured_intensity, signed_measured)
    np.testing.assert_array_equal(signed_fit.evaluation.total, signed_curve)
    signed_dialog.close()

    nonpositive_measured = measured_snapshot.copy()
    nonpositive_measured[:2] = (-1.0, 0.0)
    nonpositive_outcome = replace(
        outcome,
        measured_intensity=nonpositive_measured,
    )
    nonpositive_dialog = AutoFitCandidateDialog(nonpositive_outcome, window)
    nonpositive_dialog.candidate_list.setCurrentRow(chosen_row)
    assert nonpositive_dialog.set_y_scale("log")
    displayed_candidate = next(iter(nonpositive_dialog.spectrum_axes_by_candidate))
    spectrum_axes = nonpositive_dialog.spectrum_axes_by_candidate[displayed_candidate]
    residual_axes = nonpositive_dialog.residual_axes_by_candidate[displayed_candidate]
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

        def __init__(
            self,
            _outcome: object,
            _parent: object,
            **_kwargs: object,
        ) -> None:
            self.accepted_candidate = self.selected
            self.accepted_branches: tuple[AutoFitCandidateBranchResult, ...] = ()

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
