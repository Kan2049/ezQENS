"""Verify Q Rebin is not reachable from the v1.0 analysis UI."""

from __future__ import annotations

from PySide6.QtCore import QObject, QPoint, QTimer
from PySide6.QtGui import QShortcut
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QMenu,
    QPushButton,
    QTreeWidgetItemIterator,
    QWidget,
)
from test_gui_q_rebin import _q_dataset
from test_gui_q_rebin import application as application

from ezqens.domain import SpectrumRole
from ezqens.gui import MainWindow


def _rebin_entry_points(window: MainWindow) -> set[QObject]:
    entries: set[QObject] = {
        button
        for button in window.findChildren(QAbstractButton)
        if button.isVisible()
        and button.isEnabled()
        and any(
            "rebin" in text.casefold() for text in (button.text(), button.toolTip())
        )
    }
    entries.update(
        shortcut for shortcut in window.findChildren(QShortcut) if shortcut.isEnabled()
    )
    entries.update(
        action
        for widget in (window, *window.findChildren(QWidget))
        for action in widget.actions()
        if action.isVisible()
        and action.isEnabled()
        and "rebin" in action.text().casefold()
    )
    return entries


def test_q_rebin_has_no_user_entry_with_resolution_applied(
    application: QApplication,
    qt_slot_errors: list[BaseException],
) -> None:
    window = MainWindow()
    try:
        project = window.workspace.new_project()
        sample = window.workspace.add_dataset(project, _q_dataset())
        resolution = window.workspace.add_dataset(
            project, _q_dataset(SpectrumRole.RESOLUTION)
        )
        assert window.open_dataset(project, sample)
        assert window.apply_resolution_for_sample(project, sample, resolution)
        assert window.q_rebin_action.isEnabled()  # Eligible, but not offered in v1.0.
        window.show()
        application.processEvents()

        window.workspace._create_context_menu(None)
        rows = QTreeWidgetItemIterator(window.workspace.tree)
        while (row := rows.value()) is not None:
            window.workspace._create_context_menu(row)
            rows += 1

        view = window.dataset_view
        view._build_spectrum_context_menu()
        view._build_overview_context_menu()

        def close_navigation_menu() -> None:
            for menu in view.findChildren(QMenu):
                if menu.isVisible():
                    menu.close()

        QTimer.singleShot(0, close_navigation_menu)
        view._show_navigation_q_context_menu(view, QPoint())

        user_actions = [
            action
            for menu in window.findChildren(QMenu)
            for action in menu.actions()
            if action.isVisible() and action.isEnabled()
        ]
        assert any(action.text() == "Edit Q…" for action in user_actions)
        assert not any("q rebin" in action.text().lower() for action in user_actions)
        assert not window.q_rebin_action.associatedObjects()
        assert not window.q_rebin_action.shortcuts()
        assert not _rebin_entry_points(window)

        window.set_inspector_visible(True)
        application.processEvents()
        assert window.inspector.isVisible()
        assert not _rebin_entry_points(window)

        # Self-check the case that a menu-only scan missed in review.
        button = QPushButton("Q Rebin…", window)
        button.clicked.connect(lambda: window.show_q_rebin())
        button.show()
        assert _rebin_entry_points(window) == {button}
    finally:
        window.close()
    assert not qt_slot_errors
