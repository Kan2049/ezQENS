"""AutoFit finalization against real workflow-issued evidence and snapshots."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Slot
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QTreeWidgetItem

import ezqens.gui.auto_fit as auto_fit_module
import ezqens.gui.main_window as window_module
from ezqens.batch import MultiQExecutionStatus, MultiQFitStatus
from ezqens.domain import (
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
    ResultOrigin,
    SavedFitResult,
    SingleQAutoFitOutcome,
    WorkflowProject,
    adopt_selected_auto_fit_branch,
    adopt_single_q_auto_fit_candidate,
    commit_fitting_selection,
    continue_selected_auto_fit_candidates,
    run_single_q_auto_fit,
    save_current_result_as,
    set_current_result,
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
    window._manual_sessions.clear()
    window.close()
    app.processEvents()


@pytest.fixture
def problem(lifecycle_source: LifecycleProblem) -> Iterator[LifecycleProblem]:
    state = lifecycle_source
    state.window._workflow_projects[state.project] = state.workflow
    state.session.result_workspace = FittingWorkspaceState()
    state.session.draft = state.draft
    state.session.draft_available = True
    state.session.execution_by_group.clear()
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
        assert len(naming.name_edits) == len(names)
        naming.name_edits[0].clear()
        assert not naming.save_button.isEnabled()
        for edit, name in zip(naming.name_edits, names, strict=True):
            edit.setText(name)
        QTest.mouseClick(naming.save_button, Qt.MouseButton.LeftButton)

    QTimer.singleShot(0, fill_names)
    QTest.mouseClick(dialog.primary_button, Qt.MouseButton.LeftButton)
    problem.app.processEvents()


def _saved_item(problem: LifecycleProblem, entry: SavedFitResult) -> QTreeWidgetItem:
    for item in problem.window.workspace.tree.findItems(
        "", Qt.MatchFlag.MatchContains | Qt.MatchFlag.MatchRecursive, 0
    ):
        if item.data(0, SAVED_RESULT_ENTRY_ROLE) is entry:
            return item
    raise AssertionError("saved Result has no Workspace node")


def _seed_single(problem: LifecycleProblem, name: str | None = None) -> FitResult:
    """Install real issued evidence, independent of the dialog button contract."""
    adopted = adopt_single_q_auto_fit_candidate(
        problem.workflow,
        problem.session.draft,
        problem.evaluation,
        problem.candidates[1],
    )
    assert adopted.success and adopted.snapshot is not None
    assert adopted.adopted_draft is not None
    problem.session.result_workspace = set_current_result(
        problem.session.result_workspace, adopted.snapshot, draft=adopted.adopted_draft
    )
    if name is not None:
        problem.session.result_workspace = save_current_result_as(
            problem.session.result_workspace, name
        )
    problem.window._activate_manual_session(problem.session)
    assert adopted.fit_result is not None
    return adopted.fit_result


@pytest.mark.parametrize("active", ("raw", "untitled", "saved"))
def test_apply_replaces_anchor_only_in_active_saved_entry(
    problem: LifecycleProblem, active: str, qt_slot_errors: list[BaseException]
) -> None:
    if active != "raw":
        _seed_single(problem, "Active" if active == "saved" else None)
    before = problem.session.result_workspace
    dialog = _dialog(problem, (problem.candidates[0],))
    assert dialog.primary_button.isEnabled()
    assert dialog.primary_button.text() == "Apply"
    QTest.mouseClick(dialog.primary_button, Qt.MouseButton.LeftButton)
    workspace = problem.session.result_workspace
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert workspace.current is not None
    assert (
        workspace.current.result.point(0).candidate is problem.candidates[0].candidate
    )
    assert workspace.current.result.point(1).fit_result is None
    if active == "saved":
        assert workspace.saved_results[0].snapshot is before.saved_results[0].snapshot
        assert workspace.saved_results[0].modified
    else:
        assert workspace.untitled is workspace.active_item
    assert not qt_slot_errors
    model = problem.session.draft.setup(0).model
    assert model is workspace.current.working_states[0]
    assert model is not None and model.center_groups
    assert model.background is BackgroundModel.LINEAR
    assert model.b1 is not None and model.b1.current_value == 0 and not model.b1.free
    slope = ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.SLOPE)
    row = problem.window.manual_fit_editor.result_parameter_rows[slope]
    assert row.name.text().endswith("Slope")
    assert row.value.text() == "0" and row.uncertainty.text() == "—"


@pytest.mark.parametrize("saved", (False, True))
def test_apply_preserves_other_q_of_open_item(
    problem: LifecycleProblem, saved: bool, qt_slot_errors: list[BaseException]
) -> None:
    adopted = adopt_selected_auto_fit_branch(
        problem.workflow,
        problem.session.draft,
        problem.evaluation,
        problem.branches[1],
        center_group_id="adopted-center",
    )
    assert adopted.snapshot is not None and adopted.adopted_draft is not None
    workspace = set_current_result(
        problem.session.result_workspace, adopted.snapshot, draft=adopted.adopted_draft
    )
    if saved:
        workspace = save_current_result_as(workspace, "Open branch")
    problem.session.result_workspace = workspace
    problem.window._activate_manual_session(problem.session)
    before = workspace.current_result
    assert before is not None
    dialog = _dialog(problem, (problem.candidates[0],))
    dialog.primary_button.click()
    current = problem.session.result_workspace.current_result
    assert current is not None
    assert current.point(1) is before.point(1)
    assert problem.session.draft.setup(1).model is before.point(1).working_state
    assert current.point(0).candidate is problem.candidates[0].candidate
    assert dialog.result() == QDialog.DialogCode.Accepted and not qt_slot_errors


@pytest.mark.parametrize("state", ("raw", "untitled", "saved", "modified"))
@pytest.mark.parametrize("multiple", (False, True))
@pytest.mark.parametrize("across_q", (False, True))
def test_primary_action_follows_checked_scope_not_open_item(
    problem: LifecycleProblem, state: str, multiple: bool, across_q: bool
) -> None:
    if state != "raw":
        _seed_single(problem, None if state == "untitled" else "Saved")
    if state == "modified":
        problem.window._run_manual_fit()
    dialog = _dialog(
        problem,
        problem.candidates if multiple else (problem.candidates[0],),
        branches=across_q,
    )
    assert dialog.primary_button.isEnabled()
    assert dialog.primary_button.text() == (
        "Apply" if not multiple and not across_q else "Save"
    )
    for index in range(len(problem.evaluation.recommendation.candidate_results)):
        dialog.set_candidate_checked(index, False)
    assert not dialog.primary_button.isEnabled()


@pytest.mark.parametrize("active_work", ("raw", "saved", "untitled", "draft-only"))
def test_save_branches_is_atomic_and_preserves_existing_work(
    problem: LifecycleProblem, active_work: str, qt_slot_errors: list[BaseException]
) -> None:
    if active_work in {"saved", "untitled"}:
        _seed_single(problem, "Existing" if active_work == "saved" else None)
    elif active_work == "draft-only":
        adopted = adopt_single_q_auto_fit_candidate(
            problem.workflow,
            problem.session.draft,
            problem.evaluation,
            problem.candidates[0],
        )
        assert adopted.adopted_draft is not None
        problem.session.draft = adopted.adopted_draft
    dialog = _dialog(problem, problem.candidates, branches=True)
    before = problem.session.result_workspace
    draft = problem.session.draft
    _name_results(problem, dialog, ("Single", "Branch"))
    workspace = problem.session.result_workspace
    assert len(workspace.saved_results) == len(before.saved_results) + 2
    single, whole = workspace.saved_results[-2:]
    assert single.snapshot.origin is ResultOrigin.AUTO_FIT_BRANCH
    assert whole.snapshot.origin is ResultOrigin.AUTO_FIT_BRANCH
    assert whole.result.fitted_count == whole.result.total_count == 2
    if active_work != "raw":
        assert workspace.active_item is before.active_item
        assert workspace.current is before.current and problem.session.draft is draft
    else:
        assert workspace.active_item is single
        assert (
            problem.session.draft.setup(0).model is single.working_copy.draft_states[0]
        )
    assert not qt_slot_errors
    assert not dialog.isVisible() and dialog.result() == QDialog.DialogCode.Accepted
    _saved_item(problem, single)
    _saved_item(problem, whole)


def test_save_default_names_use_first_free_suffix(
    problem: LifecycleProblem, qt_slot_errors: list[BaseException]
) -> None:
    base = problem.candidates[0].candidate.name
    _seed_single(problem, base)
    problem.session.result_workspace = save_current_result_as(
        problem.session.result_workspace, base + " (2)"
    )
    for expected in (base + " (1)", base + " (3)"):
        dialog = _dialog(problem, (problem.candidates[0],), branches=True)

        def accept_default(expected: str = expected) -> None:
            naming = QApplication.activeModalWidget()
            assert isinstance(naming, ResultNamesDialog)
            assert naming.names == (expected,)
            QTest.mouseClick(naming.save_button, Qt.MouseButton.LeftButton)

        QTimer.singleShot(0, accept_default)
        QTest.mouseClick(dialog.primary_button, Qt.MouseButton.LeftButton)
        assert not qt_slot_errors
        assert not dialog.isVisible() and dialog.result() == QDialog.DialogCode.Accepted
        assert problem.session.result_workspace.saved_results[-1].name == expected
        dialog.close()


@pytest.mark.parametrize("choice", ("Discard them and Save", "Cancel"))
@pytest.mark.parametrize("has_branch", (False, True))
def test_pending_save_prompt_discards_only_unrun_or_cancels(
    problem: LifecycleProblem,
    choice: str,
    has_branch: bool,
    qt_slot_errors: list[BaseException],
) -> None:
    _seed_single(problem, "Open")
    dialog = _dialog(problem, problem.candidates)
    if has_branch:
        branch = problem.branches[1]
        dialog._executed_branches[branch.candidate] = branch
        dialog._sync_controls()
    before, draft = problem.session.result_workspace, problem.session.draft
    saved = choice == "Discard them and Save" and has_branch

    def choose() -> None:
        prompt = QApplication.activeModalWidget()
        assert isinstance(prompt, QMessageBox)
        assert prompt.text() == (
            f"{1 if has_branch else 2} selected candidates have not been fit to all Q"
        )
        assert {button.text() for button in prompt.buttons()} == {
            "Fit to all Q",
            "Discard them and Save",
            "Cancel",
        }
        if saved:
            QTimer.singleShot(0, name_result)
        QTest.mouseClick(
            next(button for button in prompt.buttons() if button.text() == choice),
            Qt.MouseButton.LeftButton,
        )

    def name_result() -> None:
        naming = QApplication.activeModalWidget()
        assert isinstance(naming, ResultNamesDialog)
        assert naming.windowTitle() == "Save Results"
        assert naming.save_button.text() == "Save"
        assert naming.names == (problem.candidates[1].candidate.name,)
        naming.save_button.click()

    QTimer.singleShot(0, choose)
    QTest.mouseClick(dialog.primary_button, Qt.MouseButton.LeftButton)
    after = problem.session.result_workspace
    if saved:
        assert len(after.saved_results) == len(before.saved_results) + 1
        assert after.saved_results[-1].snapshot.origin is ResultOrigin.AUTO_FIT_BRANCH
        assert dialog.result() == QDialog.DialogCode.Accepted
    else:
        assert after is before and dialog.isVisible()
        if choice == "Discard them and Save":
            assert dialog.selection_status_label.text() == (
                "No selected candidates remain to save."
            )
    assert after.active_item is before.active_item and problem.session.draft is draft
    assert not qt_slot_errors


@pytest.mark.parametrize("execution", ("success", "cancel", "failure"))
def test_pending_save_runs_missing_and_only_saves_after_completed_run(
    problem: LifecycleProblem,
    execution: str,
    monkeypatch: pytest.MonkeyPatch,
    qt_slot_errors: list[BaseException],
) -> None:
    _seed_single(problem, "Open")
    dialog = _dialog(problem, problem.candidates)
    existing = problem.branches[0]
    dialog._executed_branches[existing.candidate] = existing
    dialog._sync_controls()
    before, draft = problem.session.result_workspace, problem.session.draft
    original = continue_selected_auto_fit_candidates
    called: list[tuple[object, ...]] = []

    def execute(*args: object, **kwargs: object) -> object:
        called.append(args)
        if execution == "failure":
            raise RuntimeError("deliberate run failure")
        if execution == "cancel":
            kwargs["cancel_requested"] = lambda: True
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        auto_fit_module, "continue_selected_auto_fit_candidates", execute
    )

    def fill_names_when_ready() -> None:
        naming = QApplication.activeModalWidget()
        if not isinstance(naming, ResultNamesDialog):
            QTimer.singleShot(20, fill_names_when_ready)
            return
        assert problem.session.result_workspace is before
        assert len(naming.names) == 2
        naming.save_button.click()

    def choose() -> None:
        prompt = QApplication.activeModalWidget()
        assert isinstance(prompt, QMessageBox)
        assert prompt.text() == "1 selected candidates have not been fit to all Q"
        if execution == "success":
            QTimer.singleShot(20, fill_names_when_ready)
        QTest.mouseClick(
            next(b for b in prompt.buttons() if b.text() == "Fit to all Q"),
            Qt.MouseButton.LeftButton,
        )

    QTimer.singleShot(0, choose)
    dialog.primary_button.click()
    deadline = time.monotonic() + 20
    while dialog.is_running and time.monotonic() < deadline:
        problem.app.processEvents()
        time.sleep(0.005)
    assert not dialog.is_running and len(called) == 1
    assert called[0][3] == (problem.candidates[1].candidate,)
    if execution == "success":
        assert dialog.result() == QDialog.DialogCode.Accepted
        assert len(problem.session.result_workspace.saved_results) == 3
    else:
        assert problem.session.result_workspace is before and dialog.isVisible()
        assert ("failed" if execution == "failure" else "cancelled") in (
            dialog.selection_status_label.text().lower()
        )
    assert problem.session.result_workspace.active_item is before.active_item
    assert problem.session.draft is draft and not qt_slot_errors


def test_cancel_after_real_worker_completion_prevents_pending_automatic_save(
    problem: LifecycleProblem,
    monkeypatch: pytest.MonkeyPatch,
    qt_slot_errors: list[BaseException],
) -> None:
    _seed_single(problem, "Open")
    before, draft = problem.session.result_workspace, problem.session.draft
    clicked: list[bool] = []
    naming_calls: list[ResultNamesDialog] = []

    class CancelOnCompletion(QObject):
        @Slot(object)
        def completed(self, _result: object) -> None:
            # Queued after the dialog's completion slot, before thread teardown.
            assert QThread.currentThread() is problem.app.thread()
            assert dialog._run_result is not None
            assert dialog._run_result.status is MultiQExecutionStatus.COMPLETED
            assert dialog.is_running and dialog._worker_thread is not None
            assert dialog._save_after_run and dialog.cancel_button.isEnabled()
            assert dialog.cancel_button.text() == "Cancel run"
            QTest.mouseClick(dialog.cancel_button, Qt.MouseButton.LeftButton)
            assert dialog._cancel_event is not None and dialog._cancel_event.is_set()
            assert dialog.cancel_button.text() == "Cancelling…"
            clicked.append(True)

    def unexpected_naming(self: ResultNamesDialog) -> int:
        # Fail safely rather than hanging in a modal dialog on the buggy path.
        naming_calls.append(self)
        return int(QDialog.DialogCode.Rejected)

    monkeypatch.setattr(ResultNamesDialog, "exec", unexpected_naming)
    dialog = _dialog(problem, problem.candidates)
    observer = CancelOnCompletion(dialog)
    original_start = QThread.start

    def start(thread: QThread) -> None:
        assert thread is dialog._worker_thread and dialog._worker is not None
        dialog._worker.completed.connect(
            observer.completed, Qt.ConnectionType.QueuedConnection
        )
        original_start(thread)  # real QThread and real worker execution

    monkeypatch.setattr(QThread, "start", start)
    finalizations: list[str] = []
    dialog.finalization_requested.connect(
        lambda action, _branches: finalizations.append(action)
    )

    def choose_run() -> None:
        prompt = QApplication.activeModalWidget()
        assert isinstance(prompt, QMessageBox)
        QTest.mouseClick(
            next(b for b in prompt.buttons() if b.text() == "Fit to all Q"),
            Qt.MouseButton.LeftButton,
        )

    QTimer.singleShot(0, choose_run)
    QTest.mouseClick(dialog.primary_button, Qt.MouseButton.LeftButton)
    deadline = time.monotonic() + 20
    while dialog.is_running and time.monotonic() < deadline:
        problem.app.processEvents()
        time.sleep(0.005)
    assert not dialog.is_running and dialog._worker_thread is None
    assert clicked == [True] and not qt_slot_errors
    assert not naming_calls and not finalizations
    assert not dialog._save_after_run
    assert dialog.isVisible()
    assert "cancelled" in dialog.selection_status_label.text().lower()
    assert dialog._run_result is not None
    assert dialog._run_result.status is MultiQExecutionStatus.COMPLETED
    assert problem.session.result_workspace is before and problem.session.draft is draft


def test_save_second_adoption_failure_changes_nothing(
    problem: LifecycleProblem,
    monkeypatch: pytest.MonkeyPatch,
    qt_slot_errors: list[BaseException],
) -> None:
    _seed_single(problem, "Existing")
    dialog = _dialog(problem, problem.candidates, branches=True)
    before = problem.session.result_workspace
    draft = problem.session.draft

    def fail_branch(*_args: object, **_kwargs: object) -> object:
        raise ValueError("second adoption failed")

    monkeypatch.setattr(window_module, "adopt_selected_auto_fit_branch", fail_branch)
    # Drive the actual error dialog nested after the real naming dialog.
    messages: list[str] = []

    def dismiss_error() -> None:
        from PySide6.QtWidgets import QLabel, QPushButton

        error = QApplication.activeModalWidget()
        assert isinstance(error, QDialog)
        try:
            assert error.windowTitle() == "AutoFit"
            messages.extend(
                label.text()
                for label in error.findChildren(QLabel)
                if label.objectName() == "ezqensDialogMessage"
            )
            QTest.mouseClick(
                next(
                    button
                    for button in error.findChildren(QPushButton)
                    if button.text() == "OK"
                ),
                Qt.MouseButton.LeftButton,
            )
        finally:
            error.reject()

    def accept_names() -> None:
        naming = QApplication.activeModalWidget()
        assert isinstance(naming, ResultNamesDialog)
        QTimer.singleShot(0, dismiss_error)
        for edit, name in zip(naming.name_edits, ("One", "Two"), strict=True):
            edit.setText(name)
        QTest.mouseClick(naming.save_button, Qt.MouseButton.LeftButton)

    QTimer.singleShot(0, accept_names)
    QTest.mouseClick(dialog.primary_button, Qt.MouseButton.LeftButton)
    assert not qt_slot_errors
    assert messages == ["second adoption failed"]
    assert problem.session.result_workspace is before
    assert problem.session.draft is draft and dialog.isVisible()


@pytest.mark.parametrize(("branch", "action"), ((False, "apply"), (True, "save")))
def test_changed_context_rejects_finalization(
    problem: LifecycleProblem,
    monkeypatch: pytest.MonkeyPatch,
    branch: bool,
    action: str,
) -> None:
    _seed_single(problem, "Existing")
    dialog = _dialog(problem, (problem.candidates[1],), branches=branch)
    before = problem.session.result_workspace
    context = problem.evaluation.scientific_context
    changed = context.selection.with_group_range(0, lower_energy=-0.4, upper_energy=0.4)
    problem.window._workflow_projects[problem.project] = commit_fitting_selection(
        problem.workflow, context.sample, changed
    )
    messages: list[str] = []
    monkeypatch.setattr(
        window_module,
        "show_message_dialog",
        lambda _p, _t, message: messages.append(message),
    )
    if action == "save":
        _name_results(problem, dialog, ("Rejected",))
    else:
        QTest.mouseClick(dialog.primary_button, Qt.MouseButton.LeftButton)
    assert messages and "context" in messages[0]
    assert problem.session.result_workspace is before
    assert dialog.isVisible()


def test_structural_edit_hides_overlay_but_retains_result(
    problem: LifecycleProblem,
) -> None:
    _seed_single(problem, "Saved")
    snapshot = problem.session.result_workspace.current
    assert snapshot is not None
    problem.window._remove_manual_component(BACKGROUND_COMPONENT)
    assert problem.session.lifecycle is ManualFitLifecycle.NEEDS_FIT
    assert problem.window.dataset_view._manual_fit_result is None
    assert problem.window.manual_fit_editor.result_section.isHidden()
    assert problem.session.result_workspace.current is snapshot
    entry = problem.session.result_workspace.saved_results[0]
    assert entry.modified_copy is not None and entry.has_draft_edits
    assert "*" in _saved_item(problem, entry).text(0)
    problem.window._open_saved_result(problem.project, problem.sample, entry)
    assert problem.window.dataset_view._manual_fit_result is None
    assert problem.window.manual_fit_editor.result_section.isHidden()
    assert problem.session.result_workspace.current is snapshot
