"""Small application-owned dialogs for consistent ezQENS workflow feedback."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class DialogChoice:
    """One explicit choice in a compact application dialog."""

    text: str
    value: str
    destructive: bool = False
    default: bool = False


def choose_dialog(
    parent: QWidget | None,
    title: str,
    message: str,
    choices: tuple[DialogChoice, ...],
    *,
    cancel_value: str,
) -> str:
    """Present themed choices with conventional default and Escape behavior."""

    dialog = QDialog(parent)
    dialog.setObjectName("ezqensDialog")
    dialog.setWindowTitle(title)
    dialog.setModal(True)
    dialog.setMinimumWidth(340)
    selected = cancel_value

    heading = QLabel(title)
    heading.setObjectName("ezqensDialogTitle")
    body = QLabel(message)
    body.setObjectName("ezqensDialogMessage")
    body.setWordWrap(True)

    buttons = QHBoxLayout()
    buttons.setContentsMargins(0, 0, 0, 0)
    buttons.setSpacing(6)
    buttons.addStretch(1)
    for choice in choices:
        button = QPushButton(choice.text)
        button.setProperty("destructive", choice.destructive)
        button.setDefault(choice.default)
        button.setAutoDefault(choice.default)

        def accept(value: str = choice.value) -> None:
            nonlocal selected
            selected = value
            dialog.accept()

        button.clicked.connect(accept)
        buttons.addWidget(button)

    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(8)
    layout.addWidget(heading)
    layout.addWidget(body)
    layout.addLayout(buttons)
    dialog.setWindowModality(Qt.WindowModality.WindowModal)
    dialog.exec()
    return selected


def confirm_dialog(
    parent: QWidget | None,
    title: str,
    message: str,
    *,
    accept_text: str = "Continue",
    destructive: bool = False,
) -> bool:
    """Return whether the user accepts one ordinary or destructive action."""

    return (
        choose_dialog(
            parent,
            title,
            message,
            (
                DialogChoice("Cancel", "cancel"),
                DialogChoice(
                    accept_text,
                    "accept",
                    destructive=destructive,
                    default=True,
                ),
            ),
            cancel_value="cancel",
        )
        == "accept"
    )


def show_message_dialog(parent: QWidget | None, title: str, message: str) -> None:
    """Show one themed acknowledgement without native message-box styling."""

    choose_dialog(
        parent,
        title,
        message,
        (DialogChoice("OK", "ok", default=True),),
        cancel_value="ok",
    )
