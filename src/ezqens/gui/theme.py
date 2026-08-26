"""Central application-chrome tokens and lightweight appearance control."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication


class ColorScheme(Enum):
    """Application chrome color schemes."""

    LIGHT = "light"
    DARK = "dark"


class Appearance(Enum):
    """User-selectable application appearance modes."""

    SYSTEM = "system"
    LIGHT = "light"
    DARK = "dark"


@dataclass(frozen=True)
class TypographyTokens:
    """Small shared hierarchy for application chrome typography."""

    task_title_size: int = 13
    control_size: int = 13
    secondary_size: int = 11
    tooltip_size: int = 11
    task_title_weight: int = 600
    control_weight: int = 400


TYPOGRAPHY = TypographyTokens()


@dataclass(frozen=True)
class LayoutTokens:
    """Small set of static production dimensions for application chrome."""

    control_height: int = 28
    control_horizontal_padding: int = 8
    control_icon_left_inset: int = 11
    control_icon_size: int = 14
    row_spacing: int = 5
    section_spacing: int = 8
    toolbar_spacing: int = 4


DEFAULT_LAYOUT_TOKENS = LayoutTokens()


@dataclass(frozen=True)
class DesignTokens:
    """Small semantic token set for integrated application chrome."""

    surface: str
    surface_sidebar: str
    surface_central: str
    surface_inspector: str
    surface_hover: str
    surface_selected: str
    surface_input: str
    text_primary: str
    text_secondary: str
    text_muted: str
    border_subtle: str
    border_focus: str
    canvas_boundary: str
    accent: str
    success: str
    warning: str
    danger: str
    disabled: str
    masked: str
    control_radius: int
    space_small: int
    space_medium: int
    space_large: int


LIGHT_TOKENS = DesignTokens(
    surface="#f4f4f2",
    surface_sidebar="#eeeeeb",
    surface_central="#f7f7f5",
    surface_inspector="#f2f2f0",
    surface_hover="#ececea",
    surface_selected="#e6edf5",
    surface_input="#ffffff",
    text_primary="#292928",
    text_secondary="#676764",
    text_muted="#898985",
    border_subtle="#deded9",
    border_focus="#6c89a9",
    canvas_boundary="#ededea",
    accent="#496d91",
    success="#47785a",
    warning="#a56c1d",
    danger="#a54a4a",
    disabled="#aaa9a5",
    masked="#8d819d",
    control_radius=5,
    space_small=4,
    space_medium=8,
    space_large=12,
)

DARK_TOKENS = DesignTokens(
    surface="#2b2b29",
    surface_sidebar="#292927",
    surface_central="#2d2d2b",
    surface_inspector="#2a2a28",
    surface_hover="#373735",
    surface_selected="#35404c",
    surface_input="#20201e",
    text_primary="#ecece8",
    text_secondary="#b4b4ae",
    text_muted="#8d8d87",
    border_subtle="#464643",
    border_focus="#7b9abd",
    canvas_boundary="#41413e",
    accent="#86a9cd",
    success="#74a887",
    warning="#d0a24f",
    danger="#d27b7b",
    disabled="#6b6b66",
    masked="#aa9db8",
    control_radius=5,
    space_small=4,
    space_medium=8,
    space_large=12,
)


def _color_scheme_from_qt(
    scheme: Qt.ColorScheme,
    application: QApplication,
) -> ColorScheme:
    if scheme is Qt.ColorScheme.Dark:
        return ColorScheme.DARK
    if scheme is Qt.ColorScheme.Light:
        return ColorScheme.LIGHT
    window_color = application.palette().color(QPalette.ColorRole.Window)
    return ColorScheme.DARK if window_color.lightness() < 128 else ColorScheme.LIGHT


def system_color_scheme(application: QApplication) -> ColorScheme:
    """Return Qt's current system scheme, with a palette fallback."""
    return _color_scheme_from_qt(application.styleHints().colorScheme(), application)


def tokens_for(scheme: ColorScheme) -> DesignTokens:
    """Return semantic chrome tokens for ``scheme``."""
    return DARK_TOKENS if scheme is ColorScheme.DARK else LIGHT_TOKENS


def application_stylesheet(tokens: DesignTokens) -> str:
    """Build the restrained stylesheet for application chrome only."""

    layout = DEFAULT_LAYOUT_TOKENS
    return f"""
QMainWindow, #applicationShell, QSplitter {{
    background: {tokens.surface};
    color: {tokens.text_primary};
}}
#centralScientificWorkspace, #centralHeader {{
    background: {tokens.surface_central};
}}
#workspaceSidebar {{
    background: {tokens.surface_sidebar};
}}
#inspectorPanel {{
    background: {tokens.surface_inspector};
}}
QLabel {{
    color: {tokens.text_primary};
}}
QLabel[secondary="true"] {{
    color: {tokens.text_secondary};
}}
QLabel[muted="true"] {{
    color: {tokens.text_muted};
}}
#workspaceTitle, #inspectorTitle, #maskTaskTitle, #qAssignmentTitle,
#ezqensDialogTitle {{
    font-size: {TYPOGRAPHY.task_title_size}px;
    font-weight: {TYPOGRAPHY.task_title_weight};
}}
QPushButton, QToolButton {{
    background: transparent;
    border: 1px solid {tokens.border_subtle};
    border-radius: {tokens.control_radius}px;
    color: {tokens.text_primary};
    font-size: {TYPOGRAPHY.control_size}px;
    font-weight: {TYPOGRAPHY.control_weight};
    min-height: {layout.control_height - 4}px;
    padding: 4px {layout.control_horizontal_padding}px;
    qproperty-iconSize: {layout.control_icon_size}px;
}}
QPushButton:hover, QToolButton:hover {{
    background: {tokens.surface_hover};
}}
QPushButton:pressed, QToolButton:pressed {{
    background: {tokens.surface_selected};
}}
QPushButton:focus, QToolButton:focus {{
    border-color: {tokens.border_focus};
}}
QPushButton:disabled, QToolButton:disabled {{
    background: transparent;
    border-color: {tokens.canvas_boundary};
    color: {tokens.disabled};
}}
QComboBox, QLineEdit, QPlainTextEdit, QSpinBox {{
    background: {tokens.surface_input};
    border: 1px solid {tokens.border_subtle};
    border-radius: {tokens.control_radius}px;
    color: {tokens.text_primary};
    font-size: {TYPOGRAPHY.control_size}px;
    font-weight: {TYPOGRAPHY.control_weight};
    padding: 3px {layout.control_horizontal_padding - 2}px;
}}
QComboBox:hover, QLineEdit:hover, QPlainTextEdit:hover, QSpinBox:hover {{
    border-color: {tokens.border_focus};
}}
QComboBox:focus, QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus {{
    border: 1px solid {tokens.border_focus};
}}
QComboBox:disabled, QLineEdit:disabled, QPlainTextEdit:disabled, QSpinBox:disabled {{
    background: {tokens.surface};
    color: {tokens.disabled};
}}
QComboBox::drop-down {{
    border: 0;
    width: 18px;
}}
QComboBox::down-arrow {{
    width: 7px;
    height: 7px;
}}
#newProjectButton, #importDataButton {{
    background: {tokens.surface};
    min-height: {layout.control_height}px;
    text-align: center;
}}
QToolButton[controlKind="split"]::menu-button {{
    width: 22px;
    border-left: 1px solid transparent;
}}
QToolButton[controlKind="split"]::menu-button:hover {{
    background: {tokens.surface_hover};
}}
QToolButton[controlKind="split"]::menu-button:pressed {{
    background: {tokens.surface_selected};
}}
QToolButton[controlKind="split"]::menu-arrow {{
    width: 7px;
}}
#groupNavigationLabel {{
    font-size: {TYPOGRAPHY.control_size}px;
    font-weight: {TYPOGRAPHY.control_weight};
}}
#datasetMetadataStatus, #qEditorPreview, #qEditorStatus,
#inspectorSourceTitle, #inspectorSourceMetadata {{
    font-size: {TYPOGRAPHY.secondary_size}px;
}}
#groupSpinBox {{
    background: {tokens.surface_input};
    color: {tokens.text_primary};
    padding: 2px 4px;
}}
#inspectorButton {{
    border-color: transparent;
    border-radius: {tokens.control_radius}px;
    padding: 3px;
}}
#inspectorButton:hover {{
    border-color: {tokens.border_subtle};
}}
#inspectorButton:checked {{
    background: {tokens.surface_selected};
    border-color: transparent;
}}
#maskTaskBar QToolButton:checked {{
    background: {tokens.surface_selected};
    border-color: {tokens.border_focus};
}}
#maskTaskBar QToolButton {{
    min-height: {layout.control_height - 4}px;
}}
#qAssignmentEditor {{
    border: 0;
}}
#inspectorSourceTitle {{
    font-weight: 600;
    margin-top: 4px;
}}
#inspectorSourceMetadata {{
    line-height: 1.2;
}}
QTreeWidget {{
    background: transparent;
    border: 0;
    color: {tokens.text_primary};
    show-decoration-selected: 0;
}}
QTreeWidget::viewport {{
    background: transparent;
}}
QTreeWidget::item {{
    border-radius: 4px;
    padding: 4px;
}}
QTreeWidget::item:hover {{
    background: {tokens.surface_hover};
}}
QTreeWidget::item:focus {{
    border: 1px solid {tokens.border_focus};
}}
QTreeWidget::branch:selected {{
    background: transparent;
}}
QSplitter::handle {{
    background: transparent;
    border: 0;
}}
QSplitter::handle:horizontal {{
    width: 8px;
    border-left: 1px solid {tokens.border_subtle};
}}
QSplitter::handle:horizontal:hover {{
    border-left-color: {tokens.border_focus};
}}
QSplitter::handle:horizontal:pressed {{
    border-left-color: {tokens.accent};
}}
QMenuBar, QMenu {{
    background: {tokens.surface};
    color: {tokens.text_primary};
}}
QMenu::item:selected {{
    background: {tokens.surface_selected};
}}
QMenu::separator {{
    height: 1px;
    background: {tokens.border_subtle};
    margin: 4px 8px;
}}
#ezqensDialog {{
    background: {tokens.surface};
    border: 1px solid {tokens.border_subtle};
}}
#ezqensDialogMessage {{
    color: {tokens.text_secondary};
}}
#ezqensDialog QPushButton[destructive="true"] {{
    color: {tokens.danger};
}}
#ezqensDialog QPushButton[destructive="true"]:hover {{
    background: {tokens.surface_hover};
    border-color: {tokens.danger};
}}
QToolTip {{
    background: {tokens.surface};
    border: 1px solid {tokens.border_subtle};
    color: {tokens.text_primary};
    font-size: {TYPOGRAPHY.tooltip_size}px;
    font-weight: {TYPOGRAPHY.control_weight};
}}
#scientificCanvas {{
    background: #ffffff;
    border: 0;
}}
#scientificCanvasSurface {{
    border: 1px solid {tokens.canvas_boundary};
}}
"""


def apply_application_theme(
    application: QApplication,
    scheme: ColorScheme | None = None,
) -> ColorScheme:
    """Apply centralized chrome styling while retaining native Qt behavior."""
    selected_scheme = scheme or system_color_scheme(application)
    application.setStyleSheet(application_stylesheet(tokens_for(selected_scheme)))
    return selected_scheme


class AppearanceController(QObject):
    """Small process-local controller for the View > Appearance choice."""

    theme_changed = Signal(object)

    def __init__(self, application: QApplication) -> None:
        super().__init__(application)
        self.application = application
        self._appearance = Appearance.SYSTEM
        self._current_scheme = ColorScheme.LIGHT
        application.styleHints().colorSchemeChanged.connect(
            self.on_system_color_scheme_changed,
        )
        self._apply()

    @property
    def appearance(self) -> Appearance:
        """Return the selected appearance mode."""
        return self._appearance

    @property
    def current_scheme(self) -> ColorScheme:
        """Return the currently applied chrome color scheme."""
        return self._current_scheme

    def set_appearance(self, appearance: Appearance) -> None:
        """Select System, Light, or Dark application chrome."""
        self._appearance = appearance
        self._apply()

    def on_system_color_scheme_changed(self, scheme: Qt.ColorScheme) -> None:
        """Reapply System appearance when Qt reports an OS theme change."""
        if self._appearance is Appearance.SYSTEM:
            self._apply(_color_scheme_from_qt(scheme, self.application))

    def _apply(self, system_scheme: ColorScheme | None = None) -> None:
        if self._appearance is Appearance.LIGHT:
            scheme = ColorScheme.LIGHT
        elif self._appearance is Appearance.DARK:
            scheme = ColorScheme.DARK
        else:
            scheme = system_scheme or system_color_scheme(self.application)
        self._current_scheme = apply_application_theme(self.application, scheme)
        self.theme_changed.emit(self._current_scheme)


_appearance_controller: AppearanceController | None = None


def application_appearance_controller(
    application: QApplication,
) -> AppearanceController:
    """Return the one lightweight appearance controller for ``application``."""
    global _appearance_controller
    if (
        _appearance_controller is None
        or _appearance_controller.application is not application
    ):
        _appearance_controller = AppearanceController(application)
    return _appearance_controller
