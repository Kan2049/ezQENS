"""Focus-scoped Workspace deletion and displayed/active selection regressions."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QItemSelectionModel, QRect, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QInputDialog,
    QLineEdit,
    QStyle,
    QStyleOptionViewItem,
)
from test_gui_result_multiselection import _click, _edit, _row, _selected
from test_gui_result_multiselection import state as state
from test_gui_result_review_fixes import Problem
from test_gui_result_review_fixes import problem as problem
from test_gui_unified_result import _respond

from ezqens.gui.workspace import (
    ACTIVE_DATASET_ROLE,
    ITEM_KIND_ROLE,
    SAVED_RESULT_ENTRY_ROLE,
)
from ezqens.workflow import rename_saved_result


@pytest.mark.parametrize("key", (Qt.Key.Key_Delete, Qt.Key.Key_Backspace))
@pytest.mark.parametrize("multiple", (False, True))
@pytest.mark.parametrize("cancel", (False, True))
def test_result_keys_use_existing_confirmation_and_removal(
    state: Problem,
    key: Qt.Key,
    multiple: bool,
    cancel: bool,
) -> None:
    window, _project, _sample, session = state
    a, b, c, d = session.result_workspace.saved_results
    _click(state, a)
    _edit(state)
    a = session.result_workspace.saved_results[0]
    _click(state, a)
    if multiple:
        _click(state, b, Qt.KeyboardModifier.ControlModifier)
    before = session.result_workspace
    tree = window.workspace.tree
    tree.setFocus()
    QApplication.processEvents()
    assert tree.hasFocus()
    title = "Delete Results" if multiple else "Delete Result"
    accept = "Delete 2 Results" if multiple else "Delete Result"
    messages = _respond((title, "Cancel" if cancel else accept))
    QTest.keyClick(tree, key)
    if multiple:
        assert any(
            "Delete 2 Results?" in text and "unsaved modifications: 1." in text
            for text in messages
        )
    else:
        assert any('Delete Result "A"?' in text for text in messages)
    if cancel:
        assert session.result_workspace is before
    else:
        remaining = (c, d) if multiple else (b, c, d)
        assert session.result_workspace.saved_results == remaining
        assert session.result_workspace.active_item is remaining[0]


@pytest.mark.parametrize("key", (Qt.Key.Key_Delete, Qt.Key.Key_Backspace))
def test_key_deletion_skips_stale_selected_entry(state: Problem, key: Qt.Key) -> None:
    window, _project, _sample, session = state
    a, b, c, d = session.result_workspace.saved_results
    _click(state, a)
    _click(state, b, Qt.KeyboardModifier.ControlModifier)
    # Leave the tree with the old entry, as can happen before a queued refresh.
    session.result_workspace = rename_saved_result(
        session.result_workspace, b, "Retained"
    )
    messages = _respond(("Delete Results", "Delete 1 Results"))
    QTest.keyClick(window.workspace.tree, key)
    assert any("Delete 1 Results?" in text for text in messages)
    retained = session.result_workspace.saved_results[0]
    assert retained.name == "Retained" and retained.snapshot is b.snapshot
    assert session.result_workspace.saved_results == (retained, c, d)


@pytest.mark.parametrize("key", (Qt.Key.Key_Delete, Qt.Key.Key_Backspace))
@pytest.mark.parametrize(
    ("kind", "mixed"),
    (
        ("project", True),
        ("dataset", True),
        ("q_method", True),
        ("empty", False),
        ("data", False),
        ("methods", False),
    ),
)
def test_mixed_result_and_object_or_other_selection_has_no_key_removal(
    state: Problem,
    key: Qt.Key,
    kind: str,
    mixed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ezqens.gui.main_window as module

    window, project, sample, session = state
    assert sample.dataset.q_bins is not None
    window.workspace.save_q_method(project, sample.dataset.q_bins)
    tree = window.workspace.tree
    tree.clearSelection()
    if kind != "empty":
        row = next(
            item
            for item in tree.findItems(
                "", Qt.MatchFlag.MatchContains | Qt.MatchFlag.MatchRecursive, 0
            )
            if item.data(0, ITEM_KIND_ROLE) == kind
        )
        tree.setCurrentItem(row, 0, QItemSelectionModel.SelectionFlag.NoUpdate)
        row.setSelected(True)
        if mixed:
            _row(state, session.result_workspace.saved_results[0]).setSelected(True)
    before = session.result_workspace
    datasets, methods = tuple(project.datasets), tuple(project.q_methods)

    def unexpected(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("invalid selection must not request removal consent")

    monkeypatch.setattr(module, "choose_dialog", unexpected)
    monkeypatch.setattr(module, "confirm_dialog", unexpected)
    tree.setFocus()
    QApplication.processEvents()
    assert tree.hasFocus()
    QTest.keyClick(tree, key)
    assert session.result_workspace is before
    assert tuple(project.datasets) == datasets and tuple(project.q_methods) == methods
    assert window.workspace.projects == (project,)


@pytest.mark.parametrize("key", (Qt.Key.Key_Delete, Qt.Key.Key_Backspace))
@pytest.mark.parametrize("kind", ("project", "dataset", "q_method"))
@pytest.mark.parametrize("cancel", (False, True))
def test_object_keys_use_existing_removal_confirmation(
    state: Problem,
    key: Qt.Key,
    kind: str,
    cancel: bool,
) -> None:
    window, project, sample, session = state
    assert sample.dataset.q_bins is not None
    method = window.workspace.save_q_method(project, sample.dataset.q_bins)
    tree = window.workspace.tree
    row = next(
        item
        for item in tree.findItems(
            "", Qt.MatchFlag.MatchContains | Qt.MatchFlag.MatchRecursive, 0
        )
        if item.data(0, ITEM_KIND_ROLE) == kind
    )
    if kind == "dataset":
        dataset_row = _row(state, session.result_workspace.saved_results[0]).parent()
        assert dataset_row is not None
        row = dataset_row
    tree.setCurrentItem(row)
    tree.setFocus()
    QApplication.processEvents()
    assert tree.hasFocus() and tree.selectedItems() == [row]
    before = session.result_workspace
    title = "Delete Project" if kind == "project" else "Remove from Project"
    accept = "Delete Project" if kind == "project" else "Remove"
    messages = _respond((title, "Cancel" if cancel else accept))
    QTest.keyClick(tree, key)
    assert messages
    assert (project in window.workspace.projects) == (cancel or kind != "project")
    if kind == "dataset":
        assert (sample in project.datasets) == cancel
    if kind == "q_method":
        assert (method in project.q_methods) == cancel
    if cancel:
        assert session.result_workspace is before
        assert window._manual_sessions[session.owner] is session


@pytest.mark.parametrize("key", (Qt.Key.Key_Delete, Qt.Key.Key_Backspace))
@pytest.mark.parametrize("kind", ("project", "dataset"))
@pytest.mark.parametrize("cancel", (False, True))
def test_object_keys_preserve_unsaved_result_close_prompt(
    state: Problem,
    key: Qt.Key,
    kind: str,
    cancel: bool,
) -> None:
    window, project, sample, session = state
    _click(state, session.result_workspace.saved_results[0])
    _edit(state)
    before, draft = session.result_workspace, session.draft
    tree = window.workspace.tree
    row = next(
        item
        for item in tree.findItems(
            "", Qt.MatchFlag.MatchContains | Qt.MatchFlag.MatchRecursive, 0
        )
        if item.data(0, ITEM_KIND_ROLE) == kind
    )
    if kind == "dataset":
        dataset_row = _row(state, session.result_workspace.saved_results[0]).parent()
        assert dataset_row is not None
        row = dataset_row
    tree.setCurrentItem(row)
    tree.setFocus()
    QApplication.processEvents()
    title = "Delete Project" if kind == "project" else "Remove from Project"
    accept = "Delete Project" if kind == "project" else "Remove"
    messages = _respond(
        (title, accept), ("Close Results", "Cancel" if cancel else "Discard")
    )
    QTest.keyClick(tree, key)
    assert any("unfitted edits" in text for text in messages)
    if cancel:
        assert project in window.workspace.projects and sample in project.datasets
        assert session.result_workspace is before and session.draft is draft
        assert window._manual_sessions[session.owner] is session
    else:
        assert session.owner not in window._manual_sessions
        assert (project in window.workspace.projects) == (kind != "project")
        if kind == "dataset":
            assert sample not in project.datasets


@pytest.mark.parametrize("key", (Qt.Key.Key_Delete, Qt.Key.Key_Backspace))
@pytest.mark.parametrize("editing_tree", (False, True))
def test_object_keys_do_not_intercept_input_editing(
    state: Problem,
    key: Qt.Key,
    editing_tree: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ezqens.gui.main_window as module

    window, project, _sample, session = state
    tree = window.workspace.tree
    row = tree.topLevelItem(0)
    assert row is not None
    tree.setCurrentItem(row)
    if editing_tree:
        # Exercise Qt's native editing guard even though current object renaming
        # normally opens a separate dialog rather than an inline editor.
        row.setFlags(row.flags() | Qt.ItemFlag.ItemIsEditable)
        tree.editItem(row)
        editor = tree.findChild(QLineEdit)
        assert editor is not None
    else:
        editor = next(
            iter(window.manual_fit_editor.parameter_controls.values())
        ).current
    editor.setFocus()
    QApplication.processEvents()
    assert editor.hasFocus() and not tree.hasFocus()
    editor.deselect()
    original = editor.text()
    editor.setCursorPosition(0 if key is Qt.Key.Key_Delete else len(original))
    before = session.result_workspace

    def unexpected(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("input editing must not request object removal")

    monkeypatch.setattr(module, "choose_dialog", unexpected)
    monkeypatch.setattr(module, "confirm_dialog", unexpected)
    QTest.keyClick(editor, key)
    assert len(editor.text()) == len(original) - 1
    QTest.keyClick(tree, key)
    assert window.workspace.projects == (project,)
    assert session.result_workspace is before
    if editing_tree:
        QTest.keyClick(editor, Qt.Key.Key_Escape)


@pytest.mark.parametrize("key", (Qt.Key.Key_Delete, Qt.Key.Key_Backspace))
@pytest.mark.parametrize("target", ("parameter", "dialog"))
def test_typing_outside_tree_never_deletes_selected_results(
    state: Problem,
    key: Qt.Key,
    target: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ezqens.gui.main_window as module

    window, _project, _sample, session = state
    a, b, _c, _d = session.result_workspace.saved_results
    _click(state, a)
    _click(state, b, Qt.KeyboardModifier.ControlModifier)
    snapshots = tuple(
        entry.snapshot for entry in session.result_workspace.saved_results
    )
    dialog = None
    if target == "dialog":
        dialog = QInputDialog(window)
        dialog.setTextValue("Result name")
        dialog.open()
        editor = dialog.findChild(QLineEdit)
        assert editor is not None
    else:
        editor = next(
            iter(window.manual_fit_editor.parameter_controls.values())
        ).current
    editor.setFocus()
    QApplication.processEvents()
    assert editor.hasFocus() and not window.workspace.tree.hasFocus()
    original_text = editor.text()
    editor.setCursorPosition(0 if key is Qt.Key.Key_Delete else len(original_text))

    def unexpected(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("typing in an input must not request Result deletion")

    monkeypatch.setattr(module, "choose_dialog", unexpected)
    try:
        QTest.keyClick(editor, key)
        assert len(editor.text()) == len(original_text) - 1
        # Even a misdirected key event must respect the actual focus owner.
        QTest.keyClick(window.workspace.tree, key)
        assert (
            tuple(entry.snapshot for entry in session.result_workspace.saved_results)
            == snapshots
        )
    finally:
        if dialog is not None:
            dialog.reject()


@pytest.mark.parametrize("reverse", (False, True))
def test_shift_range_follows_star_first_display_not_storage_order(
    state: Problem,
    reverse: bool,
) -> None:
    window, project, sample, session = state
    a, b, c, d = session.result_workspace.saved_results
    window._best_saved_result(project, sample, d)
    best = session.result_workspace.best_saved_result
    assert best is not None
    assert session.result_workspace.saved_results == (a, b, c, best)
    parent = _row(state, best).parent()
    assert parent is not None
    displayed = []
    for index in range(4):
        child = parent.child(index)
        assert child is not None
        displayed.append(child.data(0, SAVED_RESULT_ENTRY_ROLE))
    assert displayed == [best, a, b, c]
    assert _row(state, best).text(0).startswith("★ ")
    first, last = (b, best) if reverse else (best, b)
    _click(state, first)
    before, draft = session.result_workspace, session.draft
    fit = window.dataset_view._manual_fit_result
    _click(state, last, Qt.KeyboardModifier.ShiftModifier)
    assert _selected(state) == {id(best), id(a), id(b)}
    assert session.result_workspace is before and session.draft is draft
    assert session.result_workspace.active_item is first
    assert window.dataset_view._manual_fit_result is fit


@pytest.mark.parametrize("include_active", (False, True))
def test_active_entry_and_view_stay_distinct_during_multi_selection(
    state: Problem,
    include_active: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, _project, sample, session = state
    a, b, c, _d = session.result_workspace.saved_results
    _click(state, a)
    before, draft = session.result_workspace, session.draft
    fit = window.dataset_view._manual_fit_result
    group = window.dataset_view.current_group_index
    if not include_active:
        _click(state, a, Qt.KeyboardModifier.ControlModifier)
    _click(state, b, Qt.KeyboardModifier.ControlModifier)
    _click(state, c, Qt.KeyboardModifier.ControlModifier)
    assert session.result_workspace is before and session.draft is draft
    assert session.result_workspace.active_item is a
    assert window._open_dataset is sample
    assert window.dataset_view._manual_fit_result is fit
    assert window.dataset_view.current_group_index == group
    assert _row(state, a).isSelected() == include_active
    assert _row(state, b).isSelected() and _row(state, c).isSelected()
    assert all(_row(state, a).data(column, ACTIVE_DATASET_ROLE) for column in (0, 1))
    assert not _row(state, b).data(0, ACTIVE_DATASET_ROLE)

    delegate = window.workspace.item_delegate
    assert delegate._active_background != delegate._selection_background
    colors: list[str] = []
    original = delegate._paint_row_background

    def record(painter: QPainter, option: QStyleOptionViewItem, color: str) -> None:
        colors.append(color)
        original(painter, option, color)

    monkeypatch.setattr(delegate, "_paint_row_background", record)
    for entry, expected in (
        (a, delegate._active_background),
        (b, delegate._selection_background),
    ):
        row = _row(state, entry)
        option = QStyleOptionViewItem()
        option.initFrom(window.workspace.tree)
        option.rect = QRect(0, 0, 260, window.workspace.tree.fontMetrics().height() + 8)
        if row.isSelected():
            option.state |= QStyle.StateFlag.State_Selected
        pixmap = QPixmap(option.rect.size())
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        colors.clear()
        try:
            delegate.paint(painter, option, window.workspace.tree.indexFromItem(row))
        finally:
            painter.end()
        assert colors == [expected]
