"""Dataset-scoped Saved Result selection and real bulk-delete interaction."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMenu, QTreeWidgetItem
from test_gui_result_review_fixes import Problem
from test_gui_result_review_fixes import problem as problem
from test_gui_slice3_manual_fit import _set_runnable_background
from test_gui_unified_result import _respond

from ezqens.gui.main_window import ManualFitSession
from ezqens.gui.workspace import SAVED_RESULT_ENTRY_ROLE
from ezqens.workflow import (
    ManualParameterEdit,
    SavedFitResult,
    rename_saved_result,
    save_current_result_as,
)


@pytest.fixture
def state(problem: Problem, qt_slot_errors: list[BaseException]) -> Iterator[Problem]:
    window, project, _sample, session = problem
    window.show()
    window._run_manual_fit()
    assert session.result_workspace.current is not None
    for name in ("A", "B", "C", "D"):
        session.result_workspace = save_current_result_as(
            session.result_workspace, name
        )
    window.workspace.refresh_dataset_analysis(project)
    window._refresh_manual_fit()
    QApplication.processEvents()
    yield problem
    assert not qt_slot_errors


def _row(state: Problem, entry: SavedFitResult) -> QTreeWidgetItem:
    tree = state[0].workspace.tree
    return next(
        item
        for item in tree.findItems(
            "", Qt.MatchFlag.MatchContains | Qt.MatchFlag.MatchRecursive, 0
        )
        if item.data(0, SAVED_RESULT_ENTRY_ROLE) is entry
    )


def _click(
    state: Problem,
    entry: SavedFitResult,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    tree = state[0].workspace.tree
    row = _row(state, entry)
    tree.scrollToItem(row)
    QApplication.processEvents()
    QTest.mouseClick(
        tree.viewport(),
        Qt.MouseButton.LeftButton,
        modifiers,
        tree.visualItemRect(row).center(),
    )
    QApplication.processEvents()


def _selected(state: Problem) -> set[int]:
    return {
        id(item.data(0, SAVED_RESULT_ENTRY_ROLE))
        for item in state[0].workspace.tree.selectedItems()
    }


def _context(
    state: Problem, entry: SavedFitResult, interact: Callable[[QMenu], None]
) -> None:
    """Dispatch the viewport's real context-menu event and operate its popup."""

    tree = state[0].workspace.tree
    row = _row(state, entry)
    tree.scrollToItem(row)
    QApplication.processEvents()
    position = tree.visualItemRect(row).center()

    def opened() -> None:
        menu = QApplication.activePopupWidget()
        assert isinstance(menu, QMenu)
        try:
            interact(menu)
        finally:
            menu.close()

    QTimer.singleShot(0, opened)
    QApplication.sendEvent(
        tree.viewport(),
        QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse,
            position,
            tree.viewport().mapToGlobal(position),
        ),
    )


def _edit(state: Problem) -> None:
    window, _project, _sample, session = state
    model = session.draft.setup(0).model
    assert model is not None
    reference = model.parameter_references()[0]
    intent = model.parameter_intent(reference)
    window._update_manual_parameter(
        reference,
        ManualParameterEdit(
            intent.current_value + 0.01,
            intent.user_lower_limit,
            intent.user_upper_limit,
            intent.free,
            intent.user_bounds_enabled,
        ),
    )


def test_save_and_load_preserves_shift_anchor_across_rebuild(state: Problem) -> None:
    window, project, sample, session = state
    a, b, c, _d = session.result_workspace.saved_results
    window._fresh_workspace_dataset(project, sample)
    _set_runnable_background(window)
    window._run_manual_fit()
    assert session.result_workspace.untitled is not None
    messages = _respond(
        ("Replace Untitled", "Save"), ("Save Result As", "Retained"), wait=True
    )
    _click(state, a)
    assert messages and session.result_workspace.active_item is a
    assert len(session.result_workspace.saved_results) == 5
    _click(state, c, Qt.KeyboardModifier.ShiftModifier)
    assert _selected(state) == {id(a), id(b), id(c)}
    assert session.result_workspace.active_item is a


def test_deleted_shift_anchor_starts_new_selection_without_loading(
    state: Problem,
) -> None:
    window, _project, _sample, session = state
    a, _b, c, d = session.result_workspace.saved_results
    _click(state, a)
    _respond(("Delete Result", "Delete Result"))
    QTest.keyClick(window.workspace.tree, Qt.Key.Key_Delete)
    assert all(entry is not a for entry in session.result_workspace.saved_results)
    before = session.result_workspace
    _click(state, c, Qt.KeyboardModifier.ShiftModifier)
    assert _selected(state) == {id(c)}
    assert session.result_workspace is before
    _click(state, d, Qt.KeyboardModifier.ShiftModifier)
    assert _selected(state) == {id(c), id(d)}
    assert session.result_workspace is before


@pytest.mark.parametrize(
    "modifier", (Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.MetaModifier)
)
def test_toggle_and_shift_select_without_loading_or_consent(
    state: Problem,
    modifier: Qt.KeyboardModifier,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ezqens.gui.main_window as module

    window, _project, _sample, session = state
    a, b, c, d = session.result_workspace.saved_results
    _click(state, a)
    assert session.result_workspace.active_item is a
    before, draft = session.result_workspace, session.draft

    def unexpected(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("selection-only gestures must not prompt")

    monkeypatch.setattr(module, "choose_dialog", unexpected)
    _click(state, b, modifier)
    assert _selected(state) == {id(a), id(b)}
    _click(state, b, modifier)
    assert _selected(state) == {id(a)}
    assert session.result_workspace is before and session.draft is draft
    _click(state, a)  # establish the range anchor using an ordinary click
    before, draft = session.result_workspace, session.draft
    _click(state, c, Qt.KeyboardModifier.ShiftModifier)
    assert _selected(state) == {id(a), id(b), id(c)}
    assert session.result_workspace is before and session.draft is draft
    assert window.dataset_view._manual_fit_result is a.result.point(0).fit_result
    _click(state, d)
    assert _selected(state) == {id(d)}
    assert session.result_workspace.active_item is d


@pytest.mark.parametrize(
    "modifier",
    (
        Qt.KeyboardModifier.ControlModifier,
        Qt.KeyboardModifier.MetaModifier,
        Qt.KeyboardModifier.ShiftModifier,
    ),
)
def test_cross_dataset_selection_resets_without_loading(
    state: Problem,
    modifier: Qt.KeyboardModifier,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ezqens.gui.main_window as module

    window, project, sample, session = state
    second = window.workspace.add_dataset(project, sample.dataset)
    resolution = next(
        item for item in project.datasets if item is not sample and item is not second
    )
    assert window.open_dataset(project, second)
    assert window.apply_resolution_for_sample(
        project, second, resolution, replace_confirmed=True
    )
    window.show_manual_fit(project, second)
    _set_runnable_background(window)
    window._run_manual_fit()
    other = window._active_manual_session
    assert other is not None and other is not session
    other.result_workspace = save_current_result_as(other.result_workspace, "Other")
    entry = other.result_workspace.saved_results[0]
    window.workspace.refresh_dataset_analysis(project)
    window._open_workspace_dataset(project, sample)
    a, b, _c, _d = session.result_workspace.saved_results
    _click(state, a)
    _click(state, b, Qt.KeyboardModifier.ControlModifier)
    before, other_before = session.result_workspace, other.result_workspace

    def unexpected(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("cross-Dataset selection must not prompt")

    monkeypatch.setattr(module, "choose_dialog", unexpected)
    _click(state, entry, modifier)
    assert _selected(state) == {id(entry)}
    assert window._open_dataset is sample and window._active_manual_session is session
    assert session.result_workspace is before and other.result_workspace is other_before


def test_context_menu_keeps_selection_or_reduces_without_loading(
    state: Problem,
) -> None:
    window, _project, _sample, session = state
    a, b, c, _d = session.result_workspace.saved_results
    _click(state, a)
    _click(state, b, Qt.KeyboardModifier.ControlModifier)
    before = session.result_workspace
    tree = window.workspace.tree
    QTest.mouseClick(
        tree.viewport(),
        Qt.MouseButton.RightButton,
        pos=tree.visualItemRect(_row(state, b)).center(),
    )
    assert _selected(state) == {id(a), id(b)}
    assert session.result_workspace is before

    def multiple(menu: QMenu) -> None:
        assert [action.text() for action in menu.actions()] == ["Delete 2 Results"]
        assert _selected(state) == {id(a), id(b)}

    _context(state, b, multiple)

    def single(menu: QMenu) -> None:
        assert [action.text() for action in menu.actions()] == [
            "Rename…",
            "Mark as Best",
            "Delete Result",
        ]
        assert _selected(state) == {id(c)}

    _context(state, c, single)
    assert session.result_workspace is before
    assert window.dataset_view._manual_fit_result is a.result.point(0).fit_result


@pytest.mark.parametrize("draft_only", (False, True))
def test_modified_clicks_do_not_resolve_protected_untitled_work(
    state: Problem,
    draft_only: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ezqens.gui.main_window as module

    window, project, sample, session = state
    a, b, c, _d = session.result_workspace.saved_results
    window._fresh_workspace_dataset(project, sample)
    _set_runnable_background(window)
    if not draft_only:
        window._run_manual_fit()
    before, draft = session.result_workspace, session.draft

    def unexpected(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("selection must not ask to replace protected work")

    monkeypatch.setattr(module, "choose_dialog", unexpected)
    _click(state, a, Qt.KeyboardModifier.ControlModifier)
    _click(state, c, Qt.KeyboardModifier.ShiftModifier)
    assert _selected(state) == {id(a), id(b), id(c)}
    assert session.result_workspace is before and session.draft is draft


@pytest.mark.parametrize("choice", ("Cancel", "Delete 2 Results"))
def test_bulk_delete_confirms_counts_and_restores_default_once(
    state: Problem,
    choice: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, project, sample, session = state
    a, b, c, d = session.result_workspace.saved_results
    window._best_saved_result(project, sample, d)
    best = session.result_workspace.best_saved_result
    assert best is not None
    _click(state, a)
    _edit(state)
    a = session.result_workspace.saved_results[0]
    _click(state, a)
    _click(state, b, Qt.KeyboardModifier.ControlModifier)
    before, draft = session.result_workspace, session.draft
    defaults: list[ManualFitSession] = []
    original = window._default_result_view

    def default(current: ManualFitSession) -> None:
        defaults.append(current)
        original(current)

    monkeypatch.setattr(window, "_default_result_view", default)
    messages: list[str] = []

    def delete(menu: QMenu) -> None:
        nonlocal messages
        messages = _respond(("Delete Results", choice))
        action = menu.actions()[0]
        QTest.mouseClick(
            menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center()
        )

    _context(state, b, delete)
    assert any(
        "Delete 2 Results?" in text and "unsaved modifications: 1." in text
        for text in messages
    )
    if choice == "Cancel":
        assert session.result_workspace is before and session.draft is draft
        assert not defaults
    else:
        assert defaults == [session]
        assert session.result_workspace.saved_results == (c, best)
        assert session.result_workspace.active_item is best
        assert window.dataset_view._manual_fit_result is best.result.point(0).fit_result


def test_bulk_delete_skips_entry_replaced_while_menu_is_open(state: Problem) -> None:
    window, project, _sample, session = state
    a, b, c, d = session.result_workspace.saved_results
    _click(state, a)
    _click(state, b, Qt.KeyboardModifier.ControlModifier)
    _click(state, c, Qt.KeyboardModifier.ControlModifier)
    messages: list[str] = []

    def delete(menu: QMenu) -> None:
        nonlocal messages
        assert menu.actions()[0].text() == "Delete 3 Results"
        session.result_workspace = rename_saved_result(
            session.result_workspace, c, "New identity"
        )
        window.workspace.refresh_dataset_analysis(project)
        messages = _respond(("Delete Results", "Delete 2 Results"))
        QTest.mouseClick(
            menu,
            Qt.MouseButton.LeftButton,
            pos=menu.actionGeometry(menu.actions()[0]).center(),
        )

    _context(state, b, delete)
    assert any(
        "Delete 2 Results?" in text and "unsaved modifications: 0." in text
        for text in messages
    )
    remaining = session.result_workspace.saved_results
    assert [entry.name for entry in remaining] == ["New identity", "D"]
    assert remaining[0].snapshot is c.snapshot and remaining[1] is d
    assert session.result_workspace.active_item is remaining[0]


def test_bulk_delete_rechecks_identity_after_confirmation_and_counts_inactive_edits(
    state: Problem,
) -> None:
    window, project, _sample, session = state
    a, b, c, d = session.result_workspace.saved_results
    _click(state, a)
    _edit(state)
    a = session.result_workspace.saved_results[0]
    _click(state, b)
    _edit(state)
    b = session.result_workspace.saved_results[1]
    _click(state, a)
    _click(state, b, Qt.KeyboardModifier.ControlModifier)
    messages: list[str] = []

    def delete(menu: QMenu) -> None:
        # Simulate a queued replacement during the actual confirmation dialog.
        def replace_during_confirmation() -> None:
            session.result_workspace = rename_saved_result(
                session.result_workspace, a, "Retained"
            )
            window.workspace.refresh_dataset_analysis(project)

        nonlocal messages
        QTimer.singleShot(0, replace_during_confirmation)
        messages = _respond(("Delete Results", "Delete 2 Results"))
        QTest.mouseClick(
            menu,
            Qt.MouseButton.LeftButton,
            pos=menu.actionGeometry(menu.actions()[0]).center(),
        )

    _context(state, b, delete)
    assert any(
        "Delete 2 Results?" in text and "unsaved modifications: 2." in text
        for text in messages
    )
    retained = session.result_workspace.saved_results[0]
    assert retained.name == "Retained" and retained.snapshot is a.snapshot
    assert session.result_workspace.active_item is retained
    assert session.result_workspace.saved_results == (retained, c, d)
