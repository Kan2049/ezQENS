"""AutoFit finalization against real workflow-issued evidence and snapshots."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QTreeWidgetItem

import ezqens.gui.main_window as window_module
from ezqens.batch import MultiQBranchResult, MultiQFitStatus
from ezqens.domain import (
    DiagnosticSeverity,
    QBins,
    ReducedDataset,
    Spectrum,
    SpectrumRole,
)
from ezqens.fitting import (
    BACKGROUND_COMPONENT,
    BackgroundModel,
    CandidateFitResult,
    FitResult,
    ParameterFamily,
    ParameterReference,
)
from ezqens.gui import MainWindow, create_application
from ezqens.gui.auto_fit import AutoFitCandidateDialog, ResultNamesDialog
from ezqens.gui.main_window import ManualFitSession
from ezqens.gui.manual_fit import ManualFitLifecycle
from ezqens.gui.workspace import SAVED_RESULT_ENTRY_ROLE, DatasetState, ProjectState
from ezqens.workflow import (
    AutoFitCandidateBranchResult,
    FittingWorkspaceState,
    ManualFitDraft,
    ManualFitExecutionOutcome,
    MultiQConstraintTransferPolicy,
    ResultOrigin,
    SavedFitResult,
    SingleQAutoFitOutcome,
    WorkflowDiagnostic,
    WorkflowDiagnosticCode,
    WorkflowError,
    WorkflowProject,
    adopt_single_q_auto_fit_candidate,
    commit_fitting_selection,
    continue_selected_auto_fit_candidates,
    run_single_q_auto_fit,
    save_current_result_as,
)


@dataclass
class LifecycleProblem:
    app: QApplication
    window: MainWindow
    project: ProjectState
    sample: DatasetState
    workflow: WorkflowProject
    session: ManualFitSession
    draft: ManualFitDraft
    evaluation: SingleQAutoFitOutcome
    candidates: tuple[CandidateFitResult, ...]
    branches: tuple[AutoFitCandidateBranchResult, ...]


@pytest.fixture(scope="module")
def lifecycle_source() -> Iterator[LifecycleProblem]:
    app = create_application(["autofit-result-lifecycle-tests"])
    window = MainWindow()
    project = window.workspace.new_project()
    energy = np.linspace(-0.5, 0.5, 101)
    resolution_values = np.exp(-0.5 * np.square(energy / 0.04))
    normalized = resolution_values / np.trapezoid(resolution_values, energy)
    q_bins = QBins.from_q_values((0.4, 0.6))
    sample = window.workspace.add_dataset(
        project,
        ReducedDataset(
            role=SpectrumRole.SAMPLE,
            spectra=tuple(
                Spectrum(
                    role=SpectrumRole.SAMPLE,
                    group_index=index,
                    group_label=f"Group {index + 1}",
                    energy=energy,
                    intensity=(2.0 + 0.1 * index) * normalized + 0.1,
                    uncertainty=np.full(energy.size, 0.02),
                    energy_unit="meV",
                    intensity_unit="arb. unit",
                    uncertainty_unit="arb. unit",
                )
                for index in range(2)
            ),
            q_bins=q_bins,
        ),
    )
    resolution = window.workspace.add_dataset(
        project,
        ReducedDataset(
            role=SpectrumRole.RESOLUTION,
            spectra=tuple(
                Spectrum(
                    role=SpectrumRole.RESOLUTION,
                    group_index=index,
                    group_label=f"Group {index + 1}",
                    energy=energy,
                    intensity=resolution_values,
                    uncertainty=np.full(energy.size, 0.02),
                    energy_unit="meV",
                    intensity_unit="arb. unit",
                    uncertainty_unit="arb. unit",
                )
                for index in range(2)
            ),
            q_bins=q_bins,
        ),
    )
    assert window.open_dataset(project, sample)
    assert window.apply_resolution_for_sample(
        project, sample, resolution, replace_confirmed=True
    )
    window.show_manual_fit(project, sample)
    session = window._active_manual_session
    assert session is not None
    workflow = window._workflow_project_for(project)
    draft = session.draft
    evaluation = run_single_q_auto_fit(workflow, draft, 0)
    candidates = tuple(
        next(
            evidence
            for evidence in evaluation.recommendation.candidate_results
            if evidence.success
            and evidence.candidate.lorentzian_count == count
            and evidence.candidate.background is BackgroundModel.CONSTANT
        )
        for count in (1, 2)
    )
    executed = continue_selected_auto_fit_candidates(
        workflow, draft, evaluation, tuple(item.candidate for item in candidates)
    )
    assert len(executed.branches) == 2
    assert all(
        outcome.status is MultiQFitStatus.SUCCESS
        for branch in executed.branches
        for outcome in branch.branch_result.outcomes
    )
    yield LifecycleProblem(
        app,
        window,
        project,
        sample,
        workflow,
        session,
        draft,
        evaluation,
        candidates,
        executed.branches,
    )
    window.close()
    app.processEvents()


@pytest.fixture
def problem(lifecycle_source: LifecycleProblem) -> Iterator[LifecycleProblem]:
    state = lifecycle_source
    state.window._workflow_projects[state.project] = state.workflow
    state.session.draft = state.draft
    state.session.draft_available = True
    state.session.result_workspace = FittingWorkspaceState()
    state.session.execution_by_group.clear()
    state.session.viewed_saved_result = None
    state.window.open_dataset(state.project, state.sample)
    state.window.dataset_view.set_current_group(0)
    state.window._activate_manual_session(state.session, expand=True)
    state.window.workspace.refresh_dataset_analysis(state.project)
    yield state
    for dialog in state.window.findChildren(AutoFitCandidateDialog):
        assert not dialog.is_running
        dialog.close()
        dialog.deleteLater()
    state.app.processEvents()


def _dialog(
    problem: LifecycleProblem,
    selected: tuple[CandidateFitResult, ...],
    *,
    branches: bool = False,
) -> AutoFitCandidateDialog:
    dialog = problem.window._create_auto_fit_dialog(
        problem.project,
        problem.sample,
        problem.workflow,
        problem.session.draft,
        problem.evaluation,
    )
    if branches:
        dialog._executed_branches = {
            branch.candidate: branch
            for branch in problem.branches
            if any(branch.anchor_evidence is evidence for evidence in selected)
        }
    for index, evidence in enumerate(
        problem.evaluation.recommendation.candidate_results
    ):
        dialog.set_candidate_checked(index, any(evidence is item for item in selected))
    dialog.show()
    problem.app.processEvents()
    return dialog


def _name_results(
    problem: LifecycleProblem, dialog: AutoFitCandidateDialog, names: tuple[str, ...]
) -> None:
    def fill_names() -> None:
        naming = QApplication.activeModalWidget()
        assert isinstance(naming, ResultNamesDialog)
        assert tuple(edit.text() for edit in naming.name_edits) == tuple(
            evidence.candidate.name for evidence in dialog.checked_candidates
        )
        assert len(naming.name_edits) == len(names)
        naming.name_edits[0].clear()
        assert not naming.save_button.isEnabled()
        for edit, name in zip(naming.name_edits, names, strict=True):
            edit.setText(name)
        QTest.mouseClick(naming.save_button, Qt.MouseButton.LeftButton)

    QTimer.singleShot(0, fill_names)
    QTest.mouseClick(dialog.save_as_button, Qt.MouseButton.LeftButton)
    problem.app.processEvents()


def _saved_item(problem: LifecycleProblem, entry: SavedFitResult) -> QTreeWidgetItem:
    for item in problem.window.workspace.tree.findItems(
        entry.name, Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchRecursive, 0
    ):
        if item.data(0, SAVED_RESULT_ENTRY_ROLE) is entry:
            return item
    raise AssertionError("saved Result has no Workspace node")


def _apply_single(problem: LifecycleProblem) -> FitResult:
    dialog = _dialog(problem, (problem.candidates[1],))
    QTest.mouseClick(dialog.use_candidate_button, Qt.MouseButton.LeftButton)
    result = problem.session.result_workspace.current_result
    assert isinstance(result, FitResult)
    assert dialog.result() == QDialog.DialogCode.Accepted
    return result


def _wait_worker(problem: LifecycleProblem, dialog: AutoFitCandidateDialog) -> None:
    deadline = time.monotonic() + 20.0
    while dialog.is_running and time.monotonic() < deadline:
        problem.app.processEvents()
        time.sleep(0.005)
    problem.app.processEvents()
    assert not dialog.is_running


def _execution_state(problem: LifecycleProblem) -> dict[int, tuple[object, ...]]:
    return {
        group: (execution.lifecycle, execution.fit_result, execution.diagnostics)
        for group, execution in problem.session.execution_by_group.items()
    }


@pytest.mark.parametrize("across_q", (False, True))
def test_real_dialog_apply_projects_workflow_ties_bounds_and_b0(
    problem: LifecycleProblem,
    across_q: bool,
) -> None:
    candidate = problem.candidates[1]
    dialog = _dialog(problem, (candidate,))
    assert dialog.use_candidate_button.isEnabled()
    assert dialog.save_button.isHidden() and dialog.save_as_button.isHidden()
    if across_q:
        QTest.mouseClick(dialog.run_selected_button, Qt.MouseButton.LeftButton)
        _wait_worker(problem, dialog)
        assert len(dialog.executed_branches) == 1
        assert dialog.executed_branches[0].evaluation is problem.evaluation
        assert dialog.executed_branches[0].anchor_evidence is candidate
        assert not dialog.has_current_result_at_open
        assert dialog.use_candidate_button.isEnabled()
    QTest.mouseClick(dialog.use_candidate_button, Qt.MouseButton.LeftButton)
    snapshot = problem.session.result_workspace.current
    assert snapshot is not None
    assert snapshot.origin is (
        ResultOrigin.AUTO_FIT_BRANCH if across_q else ResultOrigin.AUTO_FIT_SINGLE_Q
    )
    assert snapshot.candidate is candidate.candidate
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert problem.session.result_workspace.saved_results == ()
    anchor = problem.session.draft.setup(0).model
    assert anchor is snapshot.working_states[0]
    assert anchor is not None and anchor.center_groups
    for reference in anchor.parameter_references():
        control = problem.window.manual_fit_editor.parameter_controls[reference]
        if reference.family is ParameterFamily.CENTER:
            assert control.chain is not None and control.chain.isChecked()
            assert control.bounds_enabled.isChecked()
            assert anchor.parameter_intent(reference).user_lower_limit is not None
            assert anchor.parameter_intent(reference).user_upper_limit is not None
        elif reference.family in {ParameterFamily.AREA, ParameterFamily.FWHM}:
            assert control.bounds_enabled.isChecked()
            assert anchor.parameter_intent(reference).user_lower_limit is not None
    assert problem.window.manual_fit_editor.model_label.text().endswith("B0")
    assert anchor.background is BackgroundModel.LINEAR
    assert (
        anchor.b1 is not None and not anchor.b1.free and anchor.b1.current_value == 0.0
    )
    slope = ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.SLOPE)
    slope_row = problem.window.manual_fit_editor.result_parameter_rows[slope]
    assert slope_row.name.text().endswith("Slope")
    assert slope_row.value.text() == "0"
    assert slope_row.uncertainty.text() == "—"
    if across_q:
        problem.window.dataset_view.set_current_group(1)
        model = problem.session.draft.setup(1).model
        assert model is snapshot.working_states[1]
        assert model is not None and model.center_groups
        controls = problem.window.manual_fit_editor.parameter_controls
        assert all(
            control.chain is not None and control.chain.isChecked()
            for reference, control in controls.items()
            if reference.family is ParameterFamily.CENTER
        )
        assert (
            problem.session.execution_state(1).lifecycle is ManualFitLifecycle.CURRENT
        )
        assert isinstance(snapshot.result, MultiQBranchResult)
        assert (
            problem.window.dataset_view._manual_fit_result
            is snapshot.result.outcome(1).fit_result
        )
    else:
        assert problem.session.draft.setup(1).model is None


def test_save_overwrites_exact_associated_entry_and_closes(
    problem: LifecycleProblem,
) -> None:
    _apply_single(problem)
    session = problem.session
    session.result_workspace = save_current_result_as(
        session.result_workspace, "Associated"
    )
    previous = session.result_workspace.saved_results[0]
    dialog = _dialog(problem, (problem.candidates[0],))
    assert dialog.use_candidate_button.isHidden()
    assert dialog.save_button.isEnabled() and dialog.save_as_button.isEnabled()
    QTest.mouseClick(dialog.save_button, Qt.MouseButton.LeftButton)
    workspace = session.result_workspace
    assert workspace.current_saved_result_index == 0
    assert len(workspace.saved_results) == 1
    assert workspace.saved_results[0] is not previous
    assert workspace.saved_results[0].name == previous.name
    assert workspace.saved_results[0].snapshot is workspace.current
    assert previous.snapshot.candidate is problem.candidates[1].candidate
    assert workspace.current is not None
    assert workspace.current.candidate is problem.candidates[0].candidate
    assert dialog.result() == QDialog.DialogCode.Accepted
    _saved_item(problem, workspace.saved_results[0])


def test_unassociated_save_disabled_and_button_rules_captured_at_open(
    problem: LifecycleProblem,
) -> None:
    dialog = _dialog(problem, (problem.candidates[0],))
    _apply_single(problem)
    dialog._sync_controls()
    assert not dialog.has_current_result_at_open
    assert not dialog.use_candidate_button.isHidden()
    assert dialog.save_button.isHidden()
    existing = _dialog(problem, (problem.candidates[1],))
    assert existing.has_current_result_at_open
    assert existing.use_candidate_button.isHidden()
    assert not existing.save_button.isEnabled()
    assert existing.save_as_button.isEnabled()
    QTest.mouseClick(existing.run_selected_button, Qt.MouseButton.LeftButton)
    _wait_worker(problem, existing)
    assert not existing.save_button.isEnabled()
    assert existing.has_current_result_at_open


def test_single_save_as_retains_current_and_its_association(
    problem: LifecycleProblem,
) -> None:
    _apply_single(problem)
    session = problem.session
    session.result_workspace = save_current_result_as(
        session.result_workspace, "Original"
    )
    dialog = _dialog(problem, (problem.candidates[0],))
    before = session.result_workspace
    draft = session.draft
    executions = _execution_state(problem)
    _name_results(problem, dialog, ("Alternative",))
    assert session.result_workspace.current is before.current
    assert (
        session.result_workspace.associated_saved_result
        is before.associated_saved_result
    )
    assert session.draft is draft
    assert _execution_state(problem) == executions
    assert len(session.result_workspace.saved_results) == 2
    entry = session.result_workspace.saved_results[-1]
    assert entry.name == "Alternative"
    assert entry.snapshot.origin is ResultOrigin.AUTO_FIT_SINGLE_Q
    assert entry.snapshot.candidate is problem.candidates[0].candidate
    _saved_item(problem, entry)
    assert dialog.isVisible()


@pytest.mark.parametrize("associated", (False, True))
def test_mixed_save_as_appends_without_adopting_or_closing(
    problem: LifecycleProblem,
    associated: bool,
) -> None:
    if associated:
        _apply_single(problem)
        problem.session.result_workspace = save_current_result_as(
            problem.session.result_workspace, "Original"
        )
    dialog = _dialog(problem, problem.candidates)
    branch = problem.branches[1]
    dialog._executed_branches[branch.candidate] = branch
    dialog._sync_controls()
    before = problem.session.result_workspace
    draft = problem.session.draft
    executions = _execution_state(problem)
    statuses = problem.window.dataset_view.fit_status_descriptions
    assert dialog.use_candidate_button.isHidden() and dialog.save_button.isHidden()
    _name_results(problem, dialog, ("Same name", "Same name"))
    workspace = problem.session.result_workspace
    assert workspace.current is before.current
    assert workspace.current_saved_result_index == before.current_saved_result_index
    assert problem.session.draft is draft
    assert _execution_state(problem) == executions
    assert problem.window.dataset_view.fit_status_descriptions == statuses
    assert workspace.saved_results[: len(before.saved_results)] == before.saved_results
    appended = workspace.saved_results[len(before.saved_results) :]
    assert len(appended) == 2 and appended[0] is not appended[1]
    assert appended[0].snapshot.origin is ResultOrigin.AUTO_FIT_SINGLE_Q
    assert appended[1].snapshot.origin is ResultOrigin.AUTO_FIT_BRANCH
    for entry in appended:
        _saved_item(problem, entry)
    assert dialog.isVisible() and dialog.result() == QDialog.DialogCode.Rejected


def test_save_as_adoption_failure_appends_nothing(
    problem: LifecycleProblem,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _apply_single(problem)
    problem.session.result_workspace = save_current_result_as(
        problem.session.result_workspace, "Existing association"
    )
    dialog = _dialog(problem, problem.candidates)
    branch = problem.branches[1]
    dialog._executed_branches[branch.candidate] = branch
    dialog._sync_controls()
    before = problem.session.result_workspace
    draft = problem.session.draft
    calls: list[CandidateFitResult] = []

    def first_adoption(
        project: WorkflowProject,
        draft: ManualFitDraft,
        outcome: SingleQAutoFitOutcome,
        candidate: CandidateFitResult,
        *,
        group_index: int | None = None,
    ) -> ManualFitExecutionOutcome:
        calls.append(candidate)
        return adopt_single_q_auto_fit_candidate(
            project, draft, outcome, candidate, group_index=group_index
        )

    def fail_branch(*_args: object, **_kwargs: object) -> object:
        raise WorkflowError(
            (
                WorkflowDiagnostic(
                    WorkflowDiagnosticCode.AUTO_FIT_CONTEXT_CHANGED,
                    DiagnosticSeverity.ERROR,
                    "typed second-selection adoption failure",
                ),
            )
        )

    messages: list[str] = []
    monkeypatch.setattr(
        window_module, "adopt_single_q_auto_fit_candidate", first_adoption
    )
    monkeypatch.setattr(window_module, "adopt_selected_auto_fit_branch", fail_branch)
    monkeypatch.setattr(
        window_module,
        "show_message_dialog",
        lambda _parent, _title, message: messages.append(message),
    )
    _name_results(problem, dialog, ("One", "Two"))
    assert calls
    assert messages and "second-selection" in messages[0]
    assert problem.session.result_workspace is before
    assert problem.session.draft is draft
    assert dialog.isVisible()


@pytest.mark.parametrize("branch", (False, True))
@pytest.mark.parametrize("action", ("apply", "save", "save_as"))
def test_changed_context_rejects_adoption_without_state_mutation(
    problem: LifecycleProblem,
    monkeypatch: pytest.MonkeyPatch,
    branch: bool,
    action: str,
) -> None:
    if action != "apply":
        _apply_single(problem)
        problem.session.result_workspace = save_current_result_as(
            problem.session.result_workspace, "Original"
        )
    dialog = _dialog(problem, (problem.candidates[1],), branches=branch)
    before = problem.session.result_workspace
    draft = problem.session.draft
    context = problem.evaluation.scientific_context
    changed = context.selection.with_group_range(0, lower_energy=-0.4, upper_energy=0.4)
    problem.window._workflow_projects[problem.project] = commit_fitting_selection(
        problem.workflow, context.sample, changed
    )
    messages: list[str] = []
    monkeypatch.setattr(
        window_module,
        "show_message_dialog",
        lambda _parent, _title, message: messages.append(message),
    )
    if action == "save_as":
        _name_results(problem, dialog, ("Rejected",))
    else:
        button = (
            dialog.use_candidate_button if action == "apply" else dialog.save_button
        )
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert messages and "context" in messages[0]
    assert problem.session.result_workspace is before
    assert problem.session.draft is draft
    assert dialog.isVisible()


@pytest.mark.parametrize("delete", ("viewed", "other", "associated", "stale"))
def test_workspace_delete_uses_exact_saved_entry(
    problem: LifecycleProblem,
    monkeypatch: pytest.MonkeyPatch,
    delete: str,
) -> None:
    _apply_single(problem)
    session = problem.session
    session.result_workspace = save_current_result_as(session.result_workspace, "A")
    session.result_workspace = save_current_result_as(session.result_workspace, "B")
    session.result_workspace = save_current_result_as(session.result_workspace, "C")
    a, b, c = session.result_workspace.saved_results
    problem.window.workspace.refresh_dataset_analysis(problem.project)
    problem.window.workspace._on_item_double_clicked(_saved_item(problem, b), 0)
    assert session.viewed_saved_result is b
    selected = b if delete == "viewed" else c if delete == "associated" else a
    menu = problem.window.workspace._create_context_menu(_saved_item(problem, selected))
    assert menu is not None and menu.actions()[0].text() == "Delete Result"
    current_item = problem.window.workspace.tree.currentItem()
    assert current_item is not None
    assert current_item.data(0, SAVED_RESULT_ENTRY_ROLE) is selected
    current = session.result_workspace.current
    draft = session.draft
    executions = _execution_state(problem)
    confirmations: list[bool] = []

    def confirm(*_args: object, **kwargs: object) -> bool:
        confirmations.append(kwargs["destructive"] is True)
        return True

    monkeypatch.setattr(window_module, "confirm_dialog", confirm)
    menu.actions()[0].trigger()
    assert confirmations == [True]
    assert not any(
        entry is selected for entry in session.result_workspace.saved_results
    )
    assert session.result_workspace.current is current
    assert session.draft is draft and _execution_state(problem) == executions
    if delete == "viewed":
        assert session.viewed_saved_result is None
        assert problem.window.dataset_view._manual_fit_result is current.result
    else:
        assert session.viewed_saved_result is b
        assert problem.window.dataset_view._manual_fit_result is b.result
    if delete == "associated":
        assert session.result_workspace.associated_saved_result is None
        assert not problem.window.manual_fit_editor.save_result_button.isEnabled()
    else:
        assert session.result_workspace.associated_saved_result is c
    for entry in session.result_workspace.saved_results:
        _saved_item(problem, entry)
    if delete == "stale":
        before = session.result_workspace
        menu.actions()[0].trigger()
        assert session.result_workspace is before
        assert confirmations == [True]


@pytest.mark.parametrize("origin", ("manual", "autofit"))
@pytest.mark.parametrize("edit", ("remove", "clear"))
def test_structural_edit_dismisses_current_branch_overlay_only(
    problem: LifecycleProblem,
    origin: str,
    edit: str,
) -> None:
    if origin == "manual":
        _apply_single(problem)
        anchor = problem.session.result_workspace.current_result
        assert isinstance(anchor, FitResult)
        model = problem.session.draft.setup(0).model
        assert model is not None
        problem.window._start_fit_all_q(
            problem.session, anchor, MultiQConstraintTransferPolicy(model)
        )
        deadline = time.monotonic() + 20.0
        while (
            problem.window._fit_all_q_thread is not None and time.monotonic() < deadline
        ):
            problem.app.processEvents()
            time.sleep(0.005)
        assert problem.window._fit_all_q_thread is None
    else:
        dialog = _dialog(problem, (problem.candidates[1],), branches=True)
        QTest.mouseClick(dialog.use_candidate_button, Qt.MouseButton.LeftButton)
    session = problem.session
    snapshot = session.result_workspace.current
    assert snapshot is not None and isinstance(snapshot.result, MultiQBranchResult)
    session.result_workspace = save_current_result_as(
        session.result_workspace, "Branch"
    )
    saved = session.result_workspace.saved_results[0]
    problem.window.workspace.refresh_dataset_analysis(problem.project)
    problem.window.dataset_view.set_current_group(1)
    statuses = problem.window.dataset_view.fit_status_descriptions
    assert problem.window.dataset_view._manual_fit_result is not None
    if edit == "clear":
        assert problem.window.clear_manual_model(confirmed=True)
    else:
        problem.window._remove_manual_component(BACKGROUND_COMPONENT)
    assert session.execution_state(1).lifecycle is ManualFitLifecycle.NEEDS_FIT
    assert session.execution_state(1).fit_result is None
    assert problem.window.dataset_view._manual_fit_result is None
    assert problem.window.dataset_view.residual_axes is None
    assert problem.window.manual_fit_editor.result_section.isHidden()
    assert session.result_workspace.current is snapshot
    assert session.result_workspace.saved_results[0] is saved
    assert problem.window.dataset_view.fit_status_descriptions == statuses
    problem.window.workspace._on_item_double_clicked(_saved_item(problem, saved), 0)
    problem.window.dataset_view.set_current_group(1)
    assert (
        problem.window.dataset_view._manual_fit_result
        is snapshot.result.outcome(1).fit_result
    )
    assert not problem.window.manual_fit_editor.result_section.isHidden()
