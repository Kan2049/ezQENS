"""Small, theme-aware seam for the shell's restrained line icons."""

from __future__ import annotations

from enum import Enum
from importlib import resources

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer


class IconName(Enum):
    """The small set of icons used by the application shell."""

    PROJECT = "plus"
    IMPORT = "folder-input"
    UNDO = "undo-2"
    REDO = "redo-2"
    INSPECTOR_SHOW = "panel-right-open"
    INSPECTOR_HIDE = "panel-right-close"
    RESOLUTION = "activity"


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
