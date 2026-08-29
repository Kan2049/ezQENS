"""Small, theme-aware seam for the shell's restrained line icons."""

from __future__ import annotations

from enum import Enum
from importlib import resources

from PySide6.QtCore import QByteArray, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QAbstractButton


class IconName(Enum):
    """The small set of icons used by the application shell."""

    PROJECT = "plus"
    IMPORT = "folder-input"
    UNDO = "undo-2"
    REDO = "redo-2"
    INSPECTOR_SHOW = "panel-right-open"
    INSPECTOR_HIDE = "panel-right-close"
    RESOLUTION = "activity"
    LOCK = "lock-keyhole"
    CHAIN = "link-2"
    CHEVRON_RIGHT = "chevron-right"
    CHEVRON_DOWN = "chevron-down"


_ICON_SIZE = 18
_icon_cache: dict[tuple[IconName, str], QIcon] = {}


def load_icon(name: IconName, color: str) -> QIcon:
    """Return a cached, token-colored Lucide line icon."""
    cache_key = (name, color)
    cached = _icon_cache.get(cache_key)
    if cached is not None:
        return cached

    icon_file = resources.files("ezqens.gui").joinpath(
        "assets",
        "icons",
        f"{name.value}.svg",
    )
    svg = icon_file.read_text(encoding="utf-8").replace("currentColor", color)
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pixmap = QPixmap(_ICON_SIZE, _ICON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()

    icon = QIcon(pixmap)
    _icon_cache[cache_key] = icon
    return icon


def apply_disclosure_icon(
    button: QAbstractButton,
    *,
    expanded: bool,
    color: str,
) -> None:
    """Apply one restrained chevron while retaining a comfortable hit target."""

    button.setProperty("compactDisclosure", True)
    button.setProperty("disclosureExpanded", expanded)
    button.setIcon(
        load_icon(
            IconName.CHEVRON_DOWN if expanded else IconName.CHEVRON_RIGHT,
            color,
        ),
    )
    button.setIconSize(QSize(9, 9))
    button.setMinimumHeight(22)
