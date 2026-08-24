"""Minimal application launcher for the ezQENS desktop GUI."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from ezqens.gui.main_window import MainWindow
from ezqens.gui.theme import application_appearance_controller


def create_application(argv: Sequence[str] | None = None) -> QApplication:
    """Return the process QApplication, creating and styling it when needed."""
    existing = QCoreApplication.instance()
    if existing is not None:
        if not isinstance(existing, QApplication):
            message = (
                "ezQENS GUI requires QApplication, but a non-GUI "
                "QCoreApplication already owns this process."
            )
            raise RuntimeError(message)
        application_appearance_controller(existing)
        return existing

    QCoreApplication.setOrganizationName("ezQENS")
    QCoreApplication.setApplicationName("ezQENS")
    application = QApplication(list(argv) if argv is not None else sys.argv)
    application.setApplicationDisplayName("ezQENS")
    application_appearance_controller(application)
    return application


def main() -> int:
    """Launch the native ezQENS desktop application."""
    application = create_application()
    window = MainWindow()
    window.show()
    return application.exec()
