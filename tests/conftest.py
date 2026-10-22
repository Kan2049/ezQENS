"""Keep legacy GUI teardown non-interactive; lifecycle tests exercise dialogs."""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest


@pytest.fixture
def qt_slot_errors(monkeypatch: pytest.MonkeyPatch) -> list[BaseException]:
    """Capture exceptions Qt delivers to Python's hook instead of the caller."""

    errors: list[BaseException] = []
    monkeypatch.setattr(
        sys, "excepthook", lambda _kind, error, _traceback: errors.append(error)
    )
    return errors


@pytest.fixture(autouse=True)
def legacy_window_teardown(
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    if request.module.__name__ in {"test_gui_unified_result"}:
        yield
        return
    if not request.module.__name__.startswith("test_gui"):
        yield
        return
    from ezqens.gui.main_window import MainWindow, ManualFitSession

    original = MainWindow._resolve_result_close

    def resolve(
        self: MainWindow,
        sessions: tuple[ManualFitSession, ...],
        *,
        quitting: bool = False,
    ) -> bool:
        if quitting:
            return True
        return original(self, sessions, quitting=quitting)

    with pytest.MonkeyPatch.context() as teardown_patch:
        teardown_patch.setattr(MainWindow, "_resolve_result_close", resolve)
        yield
        from PySide6.QtWidgets import QApplication

        application = QApplication.instance()
        if isinstance(application, QApplication):
            for widget in application.topLevelWidgets():
                if isinstance(widget, MainWindow) and widget.isVisible():
                    widget.close()
            application.processEvents()
