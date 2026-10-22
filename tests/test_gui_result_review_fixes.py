"""Pre-publication retirement consent and stale saved-entry menu regressions."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QInputDialog, QMenu
from test_gui_slice3_manual_fit import (
    _set_runnable_background,
    _window_with_manual_resolution,
)
from test_gui_unified_result import _respond

from ezqens.domain import QBins, SpectrumRole
from ezqens.gui import MainWindow, create_application
from ezqens.gui.main_window import ManualFitSession
from ezqens.gui.workspace import (
    SAVED_RESULT_ENTRY_ROLE,
    DatasetState,
    ProjectState,
    _dataset_with_role,
)
from ezqens.workflow import SavedFitResult, fresh_analysis, save_current_result_as

type Problem = tuple[MainWindow, ProjectState, DatasetState, ManualFitSession]


@pytest.fixture
def problem() -> Iterator[Problem]:
    app = create_application(["result-review-fixes"])
    window, project, sample = _window_with_manual_resolution(app)
    for dataset in tuple(project.datasets):
        current = window.workspace._replace_dataset(
            project,
            dataset,
            replace(
                dataset.dataset, q_bins=QBins.from_edges((0.3, 0.5, 0.7))
            ).confirm_unrebinned_source(),
        )
        if dataset is sample:
            sample = current
    _set_runnable_background(window)
    session = window._active_manual_session
    assert session is not None
    yield window, project, sample, session
    window._manual_sessions.clear()
    window.close()
    app.processEvents()


def _replacement_action(problem: Problem, route: str) -> Callable[[], object]:
    window, project, sample, _session = problem
    reduced = replace(
        sample.dataset,
        spectra=(sample.dataset.spectra[0],),
        q_bins=QBins.from_edges((0.3, 0.7)),
        fractional_coverage=None,
    )
    if route == "replace":
        return lambda: window.workspace._replace_dataset(project, sample, reduced)
    if route == "role-workspace":
        return lambda: window.workspace.set_dataset_role(
            project, sample, SpectrumRole.RESOLUTION
        )
    if route == "role-action":
        return lambda: window.change_dataset_role(
            project, sample, SpectrumRole.RESOLUTION
        )
    if route == "role-replace":
        return lambda: window.workspace._replace_dataset(
            project, sample, _dataset_with_role(sample.dataset, SpectrumRole.RESOLUTION)
        )
    dialog = window.show_q_rebin(project, sample)
    assert dialog is not None
    dialog.set_target_edges((0.3, 0.7))
    preview = dialog.preview_result
    assert preview is not None
    if route == "rebin-workspace":
        # Exercise the Workspace publication seam with real rebin-preview output.
        resolution = next(
            item
            for item in project.datasets
            if item.dataset.role is SpectrumRole.RESOLUTION
        )
        assert preview.resolution is not None
        rebinned_resolution = preview.resolution.dataset

        def apply_workspace() -> object:
            result = window.workspace.apply_q_rebin_transaction(
                project,
                sample,
                preview.sample.dataset,
                resolution=resolution,
                rebinned_resolution=rebinned_resolution,
            )
            dialog.reject()
            return result

        return apply_workspace
    assert route == "rebin-action"

    def apply_dialog() -> object:
        result = dialog.apply_rebin()
        if not result:
            dialog.reject()
        return result

    return apply_dialog


ROUTES = (
    "replace",
    "role-workspace",
    "role-action",
    "role-replace",
    "rebin-workspace",
    "rebin-action",
)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("choice", ("Cancel", "Discard", "Save"))
def test_retiring_change_is_consented_before_publication(
    problem: Problem, route: str, choice: str
) -> None:
    window, project, sample, session = problem
    window._run_manual_fit()
    assert session.result_workspace.current is not None
    action = _replacement_action(problem, route)
    before = session.result_workspace
    draft = session.draft
    workflow = window._workflow_project_for(project)
    datasets = tuple(project.datasets)
    published: list[object] = []
    window.workspace.dataset_updated.connect(lambda *_args: published.append(object()))
    window.workspace.dataset_role_changed.connect(
        lambda *_args: published.append(object())
    )

    # At the actual modal consent boundary neither the owner nor the workflow
    # has been replaced, even for the paired Q-rebin publication path.
    def check_unpublished() -> None:
        assert tuple(project.datasets) == datasets
        assert window._open_dataset is sample
        assert window._workflow_projects[project] is workflow
        assert session.result_workspace is before
        assert session.draft is draft
        assert not published

    QTimer.singleShot(0, check_unpublished)
    steps: list[tuple[str, str | tuple[str, ...]]] = [("Replace Untitled", choice)]
    if choice == "Save":
        steps.append(("Save Result As", "Protected"))
    _respond(*steps, wait=True)
    action()

    if choice == "Cancel":
        assert all(
            current is previous
            for current, previous in zip(project.datasets, datasets, strict=True)
        )
        assert window._open_dataset is sample
        assert window.dataset_view.dataset is sample.dataset
        assert window._workflow_projects[project] is workflow
        assert session.result_workspace is before and session.draft is draft
        assert session.draft_available
        assert not published
    else:
        current = next(
            item
            for item in project.datasets
            if item.workflow_dataset_id == sample.workflow_dataset_id
        )
        assert current is not sample and published
        assert window._open_dataset is current
        assert window.dataset_view.dataset is current.dataset
        assert not session.draft_available
        assert session.result_workspace.current is None
        assert window._active_manual_session is None
        assert len(window.dataset_view.fit_status_descriptions) == len(
            current.dataset.spectra
        )
        assert len(session.result_workspace.saved_results) == (
            1 if choice == "Save" else 0
        )
        if choice == "Save":
            saved = session.result_workspace.saved_results[0]
            assert saved.name == "Protected" and saved.result.total_count == 2
            assert saved.snapshot is before.current


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("choice", ("Cancel", "Discard"))
def test_unfitted_edits_use_the_same_retirement_gate(
    problem: Problem, route: str, choice: str
) -> None:
    window, project, sample, session = problem
    before, draft = session.result_workspace, session.draft
    assert before.untitled is not None and before.untitled.has_draft_edits
    action = _replacement_action(problem, route)
    messages = _respond(("Replace Untitled", choice), wait=True)
    action()
    assert any("cannot be saved" in message for message in messages)
    if choice == "Cancel":
        assert session.result_workspace is before and session.draft is draft
        assert window._open_dataset is sample
        assert any(item is sample for item in project.datasets)
    else:
        assert not session.draft_available
        assert session.result_workspace.current is None


@pytest.mark.parametrize("route", ROUTES)
def test_empty_draft_needs_no_retirement_prompt(
    problem: Problem, route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ezqens.gui.main_window as module

    window, _project, _sample, session = problem
    session.result_workspace, _draft = fresh_analysis(
        session.result_workspace, session.draft
    )

    def unexpected(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("nothing to protect: consent must not be requested")

    monkeypatch.setattr(module, "choose_dialog", unexpected)
    _replacement_action(problem, route)()
    assert not session.draft_available


def test_q_assignment_keeps_count_and_invalid_count_cannot_publish(
    problem: Problem,
) -> None:
    window, project, sample, session = problem
    window._run_manual_fit()
    draft = session.draft
    workspace = session.result_workspace
    with pytest.raises(ValueError):
        window.workspace.assign_q_bins(project, sample, QBins.from_edges((0.3, 0.7)))
    assert window._open_dataset is sample
    assert session.draft is draft and session.result_workspace is workspace
    assert window._apply_q_bins(QBins.from_edges((0.2, 0.5, 0.8)), ())
    assert session.draft is draft and session.result_workspace is workspace
    assert session.draft_available


def _saved_menu(problem: Problem) -> tuple[QMenu, SavedFitResult]:
    window, project, _sample, session = problem
    window._run_manual_fit()
    assert session.result_workspace.current is not None
    session.result_workspace = save_current_result_as(session.result_workspace, "Saved")
    # A real rerun replaces the saved entry with its modified working copy.
    window._run_manual_fit()
    entry = session.result_workspace.saved_results[0]
    assert entry.modified_copy is not None
    window.workspace.refresh_dataset_analysis(project)
    row = next(
        item
        for item in window.workspace.tree.findItems(
            "", Qt.MatchFlag.MatchContains | Qt.MatchFlag.MatchRecursive, 0
        )
        if item.data(0, SAVED_RESULT_ENTRY_ROLE) is entry
    )
    menu = window.workspace._create_context_menu(row)
    assert menu is not None
    return menu, entry


@pytest.mark.parametrize("action_name", ("Rename…", "Mark as Best", "Revert to Saved"))
@pytest.mark.parametrize("stale", ("replaced", "missing-session"))
def test_saved_menu_ignores_stale_entries(
    problem: Problem,
    action_name: str,
    stale: str,
    monkeypatch: pytest.MonkeyPatch,
    qt_slot_errors: list[BaseException],
) -> None:
    window, _project, _sample, session = problem
    menu, entry = _saved_menu(problem)
    if stale == "replaced":
        window._run_manual_fit()
        assert session.result_workspace.saved_results[0] is not entry
    else:
        window._manual_sessions.pop(session.owner)
    before, draft = session.result_workspace, session.draft

    def unexpected(*_args: object, **_kwargs: object) -> tuple[str, bool]:
        raise AssertionError("a stale Rename action must not open its name dialog")

    monkeypatch.setattr(QInputDialog, "getText", unexpected)
    next(action for action in menu.actions() if action.text() == action_name).trigger()
    assert not qt_slot_errors
    assert session.result_workspace is before and session.draft is draft
    menu.close()


@pytest.mark.parametrize("stale", ("replaced", "missing-session"))
def test_rename_rechecks_identity_after_real_modal_dialog(
    problem: Problem, stale: str, qt_slot_errors: list[BaseException]
) -> None:
    window, _project, _sample, session = problem
    menu, entry = _saved_menu(problem)
    after: list[object] = []

    def replace_while_naming() -> None:
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, QInputDialog)
        if stale == "replaced":
            window._run_manual_fit()
            assert session.result_workspace.saved_results[0] is not entry
        else:
            window._manual_sessions.pop(session.owner)
        after.append(session.result_workspace)
        dialog.setTextValue("Must not rename")
        QTest.keyClick(dialog, Qt.Key.Key_Return)

    QTimer.singleShot(0, replace_while_naming)
    next(action for action in menu.actions() if action.text() == "Rename…").trigger()
    assert not qt_slot_errors
    assert after == [session.result_workspace]
    assert session.result_workspace.saved_results[0].name == "Saved"
    menu.close()
