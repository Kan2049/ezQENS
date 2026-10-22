"""Real-dialog working-copy/Workspace lifecycle over issued Result evidence."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import replace
from threading import Event

import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QInputDialog, QLabel, QPushButton
from test_gui_autofit_result_lifecycle import (
    LifecycleProblem,
    _saved_item,
    _seed_single,
)
from test_gui_autofit_result_lifecycle import (
    lifecycle_source as lifecycle_source,
)

from ezqens.domain import DiagnosticSeverity, QBins, SpectrumRole
from ezqens.gui.auto_fit import ResultNamesDialog
from ezqens.gui.manual_fit import ManualFitLifecycle
from ezqens.gui.workspace import (
    ACTIVE_DATASET_ROLE,
    DatasetAnalysisState,
    _analysis_state_icon,
)
from ezqens.workflow import (
    FittingWorkspaceState,
    ManualParameterEdit,
    WorkflowDiagnostic,
    WorkflowDiagnosticCode,
    active_manual_draft,
    adopt_selected_auto_fit_branch,
    append_saved_result,
    continue_selected_auto_fit_candidates,
    save_current_result_as,
    set_best_saved_result,
    set_current_result,
)


@pytest.fixture
def state(lifecycle_source: LifecycleProblem) -> Iterator[LifecycleProblem]:
    state = lifecycle_source
    state.window._workflow_projects[state.project] = state.workflow
    state.session.result_workspace = FittingWorkspaceState()
    state.session.resume(state.draft)
    state.session.execution_by_group.clear()
    state.window.open_dataset(state.project, state.sample)
    state.window.dataset_view.set_current_group(0)
    state.window._activate_manual_session(state.session)
    state.window.show()
    yield state
    state.session.result_workspace = FittingWorkspaceState()
    for dialog in state.window.findChildren(QDialog):
        dialog.reject()
    state.window.close()
    state.app.processEvents()


def _respond(
    *steps: tuple[str, str | tuple[str, ...]], wait: bool = False
) -> list[str]:
    """Drive actual modal dialogs, including nested naming/confirmation paths."""
    messages: list[str] = []
    remaining = list(steps)
    deadline = time.monotonic() + 5

    def answer() -> None:
        dialog = QApplication.activeModalWidget()
        if wait and dialog is None and time.monotonic() < deadline:
            QTimer.singleShot(5, answer)
            return
        assert isinstance(dialog, QDialog)
        title, value = remaining.pop(0)
        assert dialog.windowTitle() == title
        messages.extend(label.text() for label in dialog.findChildren(QLabel))
        if remaining:
            QTimer.singleShot(0, answer)
        if isinstance(dialog, QInputDialog):
            assert isinstance(value, str)
            dialog.setTextValue(value)
            dialog.accept()
        elif isinstance(dialog, ResultNamesDialog) and isinstance(value, tuple):
            for edit, name in zip(dialog.name_edits, value, strict=True):
                edit.setText(name)
            QTest.mouseClick(dialog.save_button, Qt.MouseButton.LeftButton)
        else:
            assert isinstance(value, str)
            button = next(
                button
                for button in dialog.findChildren(QPushButton)
                if button.text() == value
            )
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)

    QTimer.singleShot(0, answer)
    return messages


def _edit(state: LifecycleProblem) -> None:
    model = state.session.draft.setup(0).model
    assert model is not None
    reference = model.parameter_references()[0]
    intent = model.parameter_intent(reference)
    state.window._update_manual_parameter(
        reference,
        ManualParameterEdit(
            intent.current_value + 0.01,
            intent.user_lower_limit,
            intent.user_upper_limit,
            intent.free,
            intent.user_bounds_enabled,
        ),
    )


def test_default_best_first_and_fresh_views(state: LifecycleProblem) -> None:
    _seed_single(state, "First")
    state.session.result_workspace = save_current_result_as(
        state.session.result_workspace, "Best"
    )
    workspace = set_best_saved_result(
        state.session.result_workspace, state.session.result_workspace.saved_results[1]
    )
    state.session.result_workspace = replace(workspace, active_item=None)
    state.window._open_workspace_dataset(state.project, state.sample)
    assert (
        state.session.result_workspace.associated_saved_result
        is workspace.best_saved_result
    )
    state.window._fresh_workspace_dataset(state.project, state.sample)
    assert state.session.result_workspace.current is None
    assert all(setup.model is None for setup in state.session.draft.setups)
    assert state.window.dataset_view._manual_fit_result is None
    assert not state.window.manual_fit_editor.parameter_controls
    fresh = state.session.result_workspace.active_item
    state.window._open_workspace_dataset(state.project, state.sample)
    assert state.session.result_workspace.active_item is fresh
    state.session.result_workspace = replace(
        set_best_saved_result(workspace, None),
        active_item=None,
    )
    state.window._open_workspace_dataset(state.project, state.sample)
    assert state.session.result_workspace.current_saved_result_index == 0


def test_real_workspace_click_loads_and_dataset_double_click_is_fresh(
    state: LifecycleProblem,
) -> None:
    _seed_single(state, "A")
    state.session.result_workspace = save_current_result_as(
        state.session.result_workspace, "B"
    )
    a = state.session.result_workspace.saved_results[0]
    state.window.workspace.refresh_dataset_analysis(state.project)
    tree = state.window.workspace.tree
    row = _saved_item(state, a)
    tree.scrollToItem(row)
    state.app.processEvents()
    QTest.mouseClick(
        tree.viewport(),
        Qt.MouseButton.LeftButton,
        pos=tree.visualItemRect(row).center(),
    )
    assert state.session.result_workspace.active_item is a
    row = _saved_item(state, a)
    dataset_row = row.parent()
    assert dataset_row is not None
    position = tree.visualItemRect(dataset_row).center()
    QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=position)
    assert state.session.result_workspace.active_item is a
    QTest.mouseDClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=position)
    assert state.session.result_workspace.current is None
    assert all(setup.model is None for setup in state.session.draft.setups)


@pytest.mark.parametrize("choice", ("Save", "Discard", "Cancel"))
def test_untitled_load_consent(state: LifecycleProblem, choice: str) -> None:
    _seed_single(state, "Saved")
    saved = state.session.result_workspace.saved_results[0]
    assert saved.draft_states is not None
    state.window._fresh_workspace_dataset(state.project, state.sample)
    _seed_single(state)
    before = state.session.result_workspace
    steps: list[tuple[str, str | tuple[str, ...]]] = [("Replace Untitled", choice)]
    if choice == "Save":
        steps.append(("Save Result As", "Retained"))
    _respond(*steps)
    state.window._open_saved_result(state.project, state.sample, saved)
    if choice == "Cancel":
        assert state.session.result_workspace is before
    else:
        assert state.session.result_workspace.active_item is saved
        assert state.session.result_workspace.untitled is None
        assert len(state.session.result_workspace.saved_results) == (
            2 if choice == "Save" else 1
        )


@pytest.mark.parametrize("choice", ("Discard", "Cancel"))
def test_draft_only_fresh_consent_and_cached_dataset_switch(
    state: LifecycleProblem, choice: str
) -> None:
    _seed_single(state, "Saved")
    saved = state.session.result_workspace.saved_results[0]
    assert saved.draft_states is not None
    state.window._fresh_workspace_dataset(state.project, state.sample)
    state.session.draft = state.session.draft.with_model(0, saved.draft_states[0])
    draft = state.session.draft
    resolution = next(
        item
        for item in state.project.datasets
        if item.dataset.role is SpectrumRole.RESOLUTION
    )
    assert state.window.open_dataset(state.project, resolution)
    state.window._open_workspace_dataset(state.project, state.sample)
    assert state.session.draft is draft
    assert active_manual_draft(state.session.result_workspace, draft) is draft
    messages = _respond(("Replace Untitled", choice))
    state.window._fresh_workspace_dataset(state.project, state.sample)
    assert any("cannot be saved" in message for message in messages)
    assert (state.session.draft.setup(0).model is None) == (choice == "Discard")


def test_edit_switch_revert_rename_best_delete(state: LifecycleProblem) -> None:
    _seed_single(state, "A")
    state.session.result_workspace = save_current_result_as(
        state.session.result_workspace, "B"
    )
    b = state.session.result_workspace.saved_results[1]
    _edit(state)
    modified = state.session.result_workspace.saved_results[1]
    draft = state.session.draft
    snapshot = state.session.result_workspace.current
    assert modified.modified_copy is not None
    assert state.session.lifecycle is ManualFitLifecycle.NEEDS_FIT
    assert "*" in _saved_item(state, modified).text(0)
    a = state.session.result_workspace.saved_results[0]
    state.window._open_saved_result(state.project, state.sample, a)
    state.window._open_saved_result(state.project, state.sample, modified)
    assert state.session.draft.setup(0).model is draft.setup(0).model
    assert state.session.result_workspace.current is snapshot
    menu = state.window.workspace._create_context_menu(_saved_item(state, a))
    assert menu is not None
    assert (
        state.session.result_workspace.active_item is modified
    )  # right-click never loads
    _respond(("Rename Result", "Renamed"))
    menu.actions()[0].trigger()
    a = state.session.result_workspace.saved_results[0]
    assert a.name == "Renamed"
    state.window._best_saved_result(state.project, state.sample, a)
    best = state.session.result_workspace.best_saved_result
    assert best is not None and best.name == "Renamed"
    state.window._revert_saved_result(state.project, state.sample, modified)
    restored = state.session.result_workspace.saved_results[1]
    assert restored.modified_copy is None
    assert b.draft_states is not None
    assert state.session.draft.setup(0).model is b.draft_states[0]
    assert state.session.execution_state().lifecycle is ManualFitLifecycle.CURRENT
    _respond(("Delete Result", "Delete Result"))
    state.window._delete_saved_result(state.project, state.sample, restored)
    assert state.session.result_workspace.active_item is best
    state.window.workspace.refresh_dataset_analysis(state.project)
    assert _saved_item(state, best).data(0, ACTIVE_DATASET_ROLE)


@pytest.mark.parametrize("cancel_names", (False, True))
def test_close_save_all_naming_is_transactional(
    state: LifecycleProblem, cancel_names: bool
) -> None:
    _seed_single(state, "Saved")
    _edit(state)
    modified = state.session.result_workspace.saved_results[0]
    state.window._fresh_workspace_dataset(state.project, state.sample)
    _seed_single(state)
    before = state.session.result_workspace
    names: str | tuple[str, ...] = "Cancel" if cancel_names else ("New",)
    messages = _respond(("Close Results", "Save All"), ("Save Results As", names))
    result = state.window._resolve_result_close((state.session,), quitting=True)
    assert result is not cancel_names
    assert any("All Results" in message for message in messages)
    assert any("Saved · unfitted edits" in message for message in messages)
    if cancel_names:
        assert state.session.result_workspace is before
    else:
        workspace = state.session.result_workspace
        assert [entry.name for entry in workspace.saved_results] == ["Saved", "New"]
        assert all(entry.modified_copy is None for entry in workspace.saved_results)
        assert workspace.saved_results[0].draft_states is not None
        assert (
            workspace.saved_results[0].draft_states[0]
            is modified.working_copy.draft_states[0]
        )


def test_clean_saved_result_still_requires_quit_warning(
    state: LifecycleProblem,
) -> None:
    _seed_single(state, "Saved")
    _respond(("Close Results", "Cancel"))
    assert not state.window.close()
    assert state.window.isVisible()
    _respond(("Close Results", "Discard"))
    assert state.window.close()


def test_manual_run_replaces_only_active_q_and_failure_retains_result(
    state: LifecycleProblem, monkeypatch: pytest.MonkeyPatch
) -> None:
    branch = state.branches[1]
    from ezqens.workflow import (
        ManualFitExecutionOutcome,
        adopt_selected_auto_fit_branch,
    )

    adopted = adopt_selected_auto_fit_branch(
        state.workflow,
        state.session.draft,
        state.evaluation,
        branch,
        center_group_id="manual-branch",
    )
    state.session.result_workspace = set_current_result(
        state.session.result_workspace, adopted.snapshot, draft=adopted.adopted_draft
    )
    state.session.result_workspace = save_current_result_as(
        state.session.result_workspace, "Across Q"
    )
    state.window._activate_manual_session(state.session)
    old = state.session.result_workspace.current
    assert old is not None
    state.window._run_manual_fit()
    current = state.session.result_workspace.current
    assert current is not None
    assert current.result.point(0).fit_result is not old.result.point(0).fit_result
    assert current.result.point(1) is old.result.point(1)
    import ezqens.gui.main_window as module

    monkeypatch.setattr(
        module,
        "run_and_adopt_manual_fit",
        lambda *_args: ManualFitExecutionOutcome(
            None,
            None,
            (
                WorkflowDiagnostic(
                    WorkflowDiagnosticCode.MANUAL_FIT_EXECUTION_FAILED,
                    DiagnosticSeverity.ERROR,
                    "failed run",
                ),
            ),
        ),
    )
    state.window._run_manual_fit()
    assert state.session.result_workspace.current is current


def test_q_count_mismatch_entry_stays_visible_but_load_is_blocked(
    state: LifecycleProblem,
) -> None:
    _seed_single(state, "Two Qs")
    saved = state.session.result_workspace.saved_results[0]
    dataset = replace(
        state.sample.dataset,
        spectra=(state.sample.dataset.spectra[0],),
        q_bins=QBins.from_q_values((0.4,)),
    )
    resolution = next(
        item
        for item in state.project.datasets
        if item.dataset.role is SpectrumRole.RESOLUTION
    )
    assert state.window.open_dataset(state.project, resolution)
    updated = state.window.workspace._replace_dataset(
        state.project, state.sample, dataset
    )
    state.window.show_manual_fit(state.project, updated)
    item = _saved_item(state, saved)
    assert "different q-group count" in item.toolTip(0).lower()
    before = state.session.result_workspace
    _respond(("Open Saved Result", "OK"))
    state.window._open_saved_result(state.project, updated, saved)
    assert state.session.result_workspace is before
    # Restore the owner's real dataset for the module-scoped fixture.
    assert state.window.open_dataset(state.project, resolution)
    state.sample = state.window.workspace._replace_dataset(
        state.project, updated, state.sample.dataset
    )


def test_dataset_readiness_is_independent_of_fit_coverage(
    state: LifecycleProblem,
) -> None:
    _seed_single(state, "Single")
    sidebar = state.window.workspace
    assert (
        sidebar.dataset_analysis_state_for(state.project, state.sample)
        is DatasetAnalysisState.READY
    )
    entry = state.session.result_workspace.saved_results[0]
    item = _saved_item(state, entry)
    assert item.text(1) == "1/2 Q"
    assert "success: 1" in item.toolTip(1).lower()
    assert "not_run: 1" in item.toolTip(1).lower()
    assert not item.icon(1).isNull()
    assert (
        item.icon(1).cacheKey()
        == _analysis_state_icon(sidebar._result_partial_color).cacheKey()
    )
    assert item.data(0, ACTIVE_DATASET_ROLE)


def test_excluded_qs_are_removed_only_from_row_denominator(
    state: LifecycleProblem,
) -> None:
    evidence = state.candidates[0]
    executed = continue_selected_auto_fit_candidates(
        state.workflow,
        state.session.draft,
        state.evaluation,
        (evidence.candidate,),
        fit_excluded_groups=(1,),
    )
    adopted = adopt_selected_auto_fit_branch(
        state.workflow,
        state.session.draft,
        state.evaluation,
        executed.branches[0],
        center_group_id="excluded-branch",
    )
    state.session.result_workspace = append_saved_result(
        state.session.result_workspace, "Excluded", adopted.snapshot
    )
    state.window.workspace.refresh_dataset_analysis(state.project)
    saved = state.session.result_workspace.saved_results[0]
    item = _saved_item(state, saved)
    assert saved.result.total_count == 2
    assert item.text(1) == "1/1 Q"
    assert "excluded: 1" in item.toolTip(1).lower()
    assert "success: 1" in item.toolTip(1).lower()
    success_color = state.window.workspace._analysis_state_colors[
        DatasetAnalysisState.FULLY_FIT
    ]
    assert item.icon(1).cacheKey() == _analysis_state_icon(success_color).cacheKey()


def test_role_cleanup_preserves_saved_history_and_blocks_projection(
    state: LifecycleProblem,
) -> None:
    _seed_single(state, "History")
    saved = state.session.result_workspace.saved_results[0]
    updated = state.window.change_dataset_role(
        state.project, state.sample, SpectrumRole.RESOLUTION
    )
    assert updated is not None
    assert not state.session.draft_available
    assert state.session.result_workspace.saved_results == (saved,)
    assert state.window.dataset_view._manual_fit_result is None
    _saved_item(state, saved)
    _respond(("Open Saved Result", "OK"))
    state.window._open_saved_result(state.project, updated, saved)
    assert state.session.result_workspace.current is None
    restored = state.window.change_dataset_role(
        state.project, updated, SpectrumRole.SAMPLE
    )
    assert restored is not None
    state.sample = restored


def test_equal_scientific_context_replacement_keeps_core_current_status(
    state: LifecycleProblem,
) -> None:
    result = _seed_single(state, "Current")
    resolution = next(
        item
        for item in state.project.datasets
        if item.dataset.role is SpectrumRole.RESOLUTION
    )
    updated = state.window.workspace._replace_dataset(
        state.project, resolution, replace(resolution.dataset)
    )
    assert updated is not resolution
    assert state.session.lifecycle is ManualFitLifecycle.CURRENT
    assert state.window.dataset_view._manual_fit_result is result


def test_project_close_collects_sessions_and_saves_all_with_real_dialogs(
    state: LifecycleProblem,
) -> None:
    from test_gui_slice3_manual_fit import (
        _set_runnable_background,
        _window_with_manual_resolution,
    )

    window, project, first = _window_with_manual_resolution(state.app)
    _set_runnable_background(window)
    window._run_manual_fit()
    first_session = window._active_manual_session
    assert first_session is not None
    second = window.workspace.add_dataset(project, first.dataset)
    resolution = next(
        item
        for item in project.datasets
        if item.dataset.role is SpectrumRole.RESOLUTION
    )
    assert window.open_dataset(project, second)
    assert window.apply_resolution_for_sample(
        project, second, resolution, replace_confirmed=True
    )
    window.show_manual_fit(project, second)
    _set_runnable_background(window)
    window._run_manual_fit()
    second_session = window._active_manual_session
    assert second_session is not None
    second_session.result_workspace = save_current_result_as(
        second_session.result_workspace, "Named"
    )
    draft = second_session.draft
    model = draft.setup(0).model
    assert model is not None
    reference = model.parameter_references()[0]
    intent = model.parameter_intent(reference)
    window._update_manual_parameter(
        reference,
        ManualParameterEdit(
            intent.current_value + 0.1,
            intent.user_lower_limit,
            intent.user_upper_limit,
            intent.free,
            intent.user_bounds_enabled,
        ),
    )
    original = second_session.result_workspace
    messages = _respond(("Close Results", "Save All"), ("Save Results As", "Cancel"))
    assert not window.remove_project(project, confirmed=True)
    assert second_session.result_workspace is original
    assert any("Untitled" in message and "Named" in message for message in messages)
    _respond(("Close Results", "Save All"), ("Save Results As", ("First saved",)))
    assert window.remove_project(project, confirmed=True)
    assert first_session.result_workspace.saved_results[0].name == "First saved"
    assert second_session.result_workspace.saved_results[0].modified_copy is None
    assert not window._manual_sessions
    window.close()


def test_quit_cancel_after_worker_shutdown_restores_interaction(
    state: LifecycleProblem, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ezqens.gui.manual_fit_all_q as worker_module
    from ezqens.workflow import MultiQConstraintTransferPolicy

    fit = _seed_single(state, "Saved")
    model = state.session.draft.setup(0).model
    assert model is not None
    started = Event()
    cancelled = Event()
    finish_release = Event()

    def run(*_args: object, **kwargs: object) -> object:
        cancel_requested = kwargs["cancel_requested"]
        assert callable(cancel_requested)
        started.set()
        deadline = time.monotonic() + 10
        while not cancel_requested() and time.monotonic() < deadline:
            time.sleep(0.002)
        assert cancel_requested()
        cancelled.set()
        assert finish_release.wait(timeout=3)
        raise RuntimeError("late cancelled worker")

    monkeypatch.setattr(worker_module, "fit_all_q_from_current_with_adoption", run)
    state.window._start_fit_all_q(
        state.session, fit, MultiQConstraintTransferPolicy(model)
    )
    assert started.wait(timeout=2)
    _respond(("Close Results", "Cancel"), wait=True)
    state.window.close()
    assert cancelled.wait(timeout=2)
    finish_release.set()
    deadline = time.monotonic() + 3
    while state.window._fit_all_q_thread is not None and time.monotonic() < deadline:
        state.app.processEvents()
        time.sleep(0.002)
    state.app.processEvents()
    assert state.window._fit_all_q_thread is None
    assert state.window.isVisible() and state.window.isEnabled()
    assert state.window.dataset_view.group_spinbox.isEnabled()
    assert state.window.manual_fit_editor.run_button.isEnabled()
    assert state.window.dataset_view._manual_fit_result is fit
