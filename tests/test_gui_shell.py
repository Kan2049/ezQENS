"""Focused headless tests for the initial desktop application shell."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from matplotlib.colors import to_hex
from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import QApplication, QLabel

from ezqens.gui import MainWindow, create_application
from ezqens.gui.main_window import (
    INSPECTOR_PREFERRED_WIDTH,
    inspector_outward_expansion_width,
)
from ezqens.gui.scientific_canvas import SCIENTIFIC_BACKGROUND
from ezqens.gui.theme import (
    DARK_TOKENS,
    DEFAULT_LAYOUT_TOKENS,
    LIGHT_TOKENS,
    TYPOGRAPHY,
    Appearance,
    application_appearance_controller,
)


@pytest.fixture(scope="module")
def application() -> Iterator[QApplication]:
    app = create_application(["ezqens-tests"])
    yield app
    app.closeAllWindows()


@pytest.fixture(autouse=True)
def reset_appearance(application: QApplication) -> Iterator[None]:
    controller = application_appearance_controller(application)
    controller.set_appearance(Appearance.SYSTEM)
    yield
    controller.set_appearance(Appearance.SYSTEM)


def test_gui_package_and_application_imports(application: QApplication) -> None:
    assert QApplication.instance() is application
    assert application.applicationName() == "ezQENS"
    assert create_application(["already-running"]) is application


def test_main_window_constructs_with_expected_shell_regions(
    application: QApplication,
) -> None:
    window = MainWindow()
    assert window.workspace.objectName() == "workspaceSidebar"
    assert window.central_workspace.objectName() == "centralScientificWorkspace"
    assert window.central_header.objectName() == "centralHeader"
    assert window.scientific_canvas.objectName() == "scientificCanvas"
    assert window.scientific_canvas_container.isHidden()
    assert window.inspector.objectName() == "inspectorPanel"
    assert window.splitter.count() == 3
    assert window.splitter.widget(0) is window.workspace
    assert window.splitter.widget(1) is window.central_workspace
    assert window.splitter.widget(2) is window.inspector
    assert window.central_header.parentWidget() is window.central_workspace
    assert any(label.text() == "ezQENS" for label in window.findChildren(QLabel))
    window.close()


def test_inspector_starts_hidden_and_can_be_shown(
    application: QApplication,
) -> None:
    window = MainWindow()
    window.show()
    application.processEvents()
    assert window.inspector.isHidden()

    window.set_inspector_visible(True)
    application.processEvents()
    assert window.inspector.isVisible()
    assert window.toggle_inspector_action.isChecked()
    assert window.inspector_button.toolTip() == "Hide Inspector"

    window.set_inspector_visible(False)
    application.processEvents()
    assert window.inspector.isHidden()
    assert window.inspector_button.toolTip() == "Show Inspector"
    window.close()


def test_repeated_inspector_show_hide_restores_automatic_window_expansion(
    application: QApplication,
) -> None:
    window = MainWindow()
    window.show()
    application.processEvents()
    original_size = window.size()

    for _ in range(2):
        window.set_inspector_visible(True)
        application.processEvents()
        window.set_inspector_visible(False)
        application.processEvents()

    assert window.size() == original_size
    window.close()


@pytest.mark.parametrize("appearance", list(Appearance))
def test_scientific_canvas_remains_white_in_all_appearance_modes(
    application: QApplication,
    appearance: Appearance,
) -> None:
    application_appearance_controller(application).set_appearance(appearance)
    window = MainWindow()
    assert to_hex(window.scientific_canvas.figure.get_facecolor()) == (
        SCIENTIFIC_BACKGROUND
    )
    window.close()


def test_scientific_figure_surface_is_available_only_when_requested(
    application: QApplication,
) -> None:
    window = MainWindow()
    assert window.scientific_canvas_container.isHidden()

    window.set_scientific_figure_visible(True)

    assert not window.scientific_canvas_container.isHidden()
    assert to_hex(window.scientific_canvas.figure.get_facecolor()) == (
        SCIENTIFIC_BACKGROUND
    )
    window.close()


def test_scientific_canvas_wrapper_is_not_forced_white(
    application: QApplication,
) -> None:
    window = MainWindow()
    window.set_scientific_figure_visible(True)
    assert "#scientificCanvasSurface {\n    background: #ffffff;" not in (
        application.styleSheet()
    )
    assert to_hex(window.scientific_canvas.figure.get_facecolor()) == (
        SCIENTIFIC_BACKGROUND
    )
    window.close()


@pytest.mark.parametrize("appearance", [Appearance.LIGHT, Appearance.DARK])
def test_shared_control_chrome_and_action_icons_follow_appearance(
    application: QApplication,
    appearance: Appearance,
) -> None:
    application_appearance_controller(application).set_appearance(appearance)
    window = MainWindow()

    stylesheet = application.styleSheet()
    assert "QComboBox:hover" in stylesheet
    assert "QLineEdit:focus" in stylesheet
    assert 'QToolButton[controlKind="split"]::menu-button:hover' in stylesheet
    assert not window.workspace.new_project_button.icon().isNull()
    assert not window.workspace.import_data_button.icon().isNull()
    assert (
        window.workspace.import_data_button.toolButtonStyle()
        is Qt.ToolButtonStyle.ToolButtonTextBesideIcon
    )
    assert not window.mask_undo_button.icon().isNull()
    assert not window.mask_redo_button.icon().isNull()
    assert (
        window.mask_undo_button.toolButtonStyle()
        is Qt.ToolButtonStyle.ToolButtonTextBesideIcon
    )
    window.close()


def test_theme_uses_one_shared_typography_hierarchy_for_task_and_controls(
    application: QApplication,
) -> None:
    stylesheet = application.styleSheet()

    assert "#maskTaskTitle, #qAssignmentTitle" in stylesheet
    assert f"font-size: {TYPOGRAPHY.task_title_size}px;" in stylesheet
    assert f"font-size: {TYPOGRAPHY.control_size}px;" in stylesheet
    assert f"font-weight: {TYPOGRAPHY.control_weight};" in stylesheet
    assert f"font-size: {TYPOGRAPHY.secondary_size}px;" in stylesheet
    assert "QToolTip" in stylesheet
    assert "qproperty-iconSize: 14px;" in stylesheet
    assert "#newProjectButton, #importDataButton" in stylesheet
    assert "text-align: center;" in stylesheet


def test_main_window_uses_static_layout_defaults_without_runtime_tuner_hooks(
    application: QApplication,
) -> None:
    window = MainWindow()

    assert not hasattr(window, "_layout_controller")
    assert "Developer" not in {
        action.text().replace("&", "") for action in window.menuBar().actions()
    }
    assert window.workspace.new_project_button.iconSize().width() == (
        DEFAULT_LAYOUT_TOKENS.control_icon_size
    )
    assert window.workspace._layout.spacing() == DEFAULT_LAYOUT_TOKENS.section_spacing
    assert window.dataset_view._navigation_layout.spacing() == (
        DEFAULT_LAYOUT_TOKENS.row_spacing
    )
    assert window._mask_task_layout.spacing() == (DEFAULT_LAYOUT_TOKENS.toolbar_spacing)
    window.close()
    assert not window._appearance_listener_connected


@pytest.mark.parametrize(
    ("appearance", "central_surface"),
    [
        (Appearance.LIGHT, LIGHT_TOKENS.surface_central),
        (Appearance.DARK, DARK_TOKENS.surface_central),
    ],
)
def test_central_workspace_follows_application_appearance(
    application: QApplication,
    appearance: Appearance,
    central_surface: str,
) -> None:
    application_appearance_controller(application).set_appearance(appearance)
    window = MainWindow()
    assert central_surface in application.styleSheet()
    assert window.scientific_canvas_container.isHidden()
    assert to_hex(window.scientific_canvas.figure.get_facecolor()) == (
        SCIENTIFIC_BACKGROUND
    )
    window.close()


def test_light_workspace_and_central_surfaces_are_distinct(
    application: QApplication,
) -> None:
    application_appearance_controller(application).set_appearance(Appearance.LIGHT)
    assert LIGHT_TOKENS.surface_sidebar != LIGHT_TOKENS.surface_central
    assert LIGHT_TOKENS.surface_sidebar in application.styleSheet()
    assert LIGHT_TOKENS.surface_central in application.styleSheet()


def test_central_header_and_inspector_toggle_are_top_aligned(
    application: QApplication,
) -> None:
    window = MainWindow()
    window.show()
    application.processEvents()

    assert window.central_header.geometry().top() == 0
    assert window.inspector_button.parentWidget() is window.central_header
    assert window.inspector_button.geometry().top() >= 0
    assert window.inspector_button.geometry().bottom() < window.central_header.height()
    window.close()


def test_central_header_has_no_explicit_separator(
    application: QApplication,
) -> None:
    assert "border-bottom" not in application.styleSheet()


def test_native_window_chrome_uses_platform_safe_configuration(
    application: QApplication,
) -> None:
    window = MainWindow()
    if sys.platform == "darwin":
        assert window.unifiedTitleAndToolBarOnMac()
        assert window.windowTitle() == ""
    else:
        assert window.windowTitle() == "ezQENS"
    window.close()


def test_inspector_icon_is_recolored_when_appearance_changes(
    application: QApplication,
) -> None:
    controller = application_appearance_controller(application)
    controller.set_appearance(Appearance.LIGHT)
    window = MainWindow()
    light_icon_key = window.inspector_button.icon().cacheKey()
    assert not window.inspector_button.icon().isNull()

    controller.set_appearance(Appearance.DARK)

    assert not window.inspector_button.icon().isNull()
    assert window.inspector_button.icon().cacheKey() != light_icon_key
    window.close()


def test_appearance_actions_are_exclusive_and_functional(
    application: QApplication,
) -> None:
    window = MainWindow()
    controller = application_appearance_controller(application)
    assert window.system_appearance_action.isChecked()

    window.dark_appearance_action.trigger()

    assert controller.appearance.value == "dark"
    assert window.dark_appearance_action.isChecked()
    assert not window.system_appearance_action.isChecked()
    assert not window.light_appearance_action.isChecked()

    window.light_appearance_action.trigger()

    assert controller.appearance.value == "light"
    assert window.light_appearance_action.isChecked()
    assert not window.dark_appearance_action.isChecked()
    window.close()


def test_system_appearance_change_reapplies_theme(
    application: QApplication,
) -> None:
    controller = application_appearance_controller(application)
    controller.set_appearance(Appearance.SYSTEM)

    application.styleHints().colorSchemeChanged.emit(Qt.ColorScheme.Dark)

    assert controller.current_scheme.value == "dark"
    assert DARK_TOKENS.surface in application.styleSheet()

    controller.set_appearance(Appearance.LIGHT)
    application.styleHints().colorSchemeChanged.emit(Qt.ColorScheme.Dark)
    assert controller.current_scheme.value == "light"


@pytest.mark.parametrize(
    ("window_geometry", "available_geometry", "maximized", "expected"),
    [
        (QRect(100, 100, 800, 600), QRect(0, 0, 1920, 1080), False, 280),
        (QRect(900, 100, 800, 600), QRect(0, 0, 1920, 1080), False, 0),
        (QRect(100, 100, 800, 600), QRect(0, 0, 1920, 1080), True, 0),
        (QRect(100, 100, 1800, 600), QRect(0, 0, 1920, 1080), False, 0),
        (QRect(-10, 100, 800, 600), QRect(0, 0, 1920, 1080), False, 0),
    ],
)
def test_inspector_outward_expansion_is_bounded_by_available_geometry(
    window_geometry: QRect,
    available_geometry: QRect,
    maximized: bool,
    expected: int,
) -> None:
    assert (
        inspector_outward_expansion_width(
            window_geometry,
            available_geometry,
            INSPECTOR_PREFERRED_WIDTH,
            is_maximized=maximized,
        )
        == expected
    )


def test_non_gui_qcoreapplication_produces_clear_lifecycle_error() -> None:
    program = """
from PySide6.QtCore import QCoreApplication
from ezqens.gui.application import create_application

QCoreApplication([])
try:
    create_application([])
except RuntimeError as error:
    print(error)
else:
    raise SystemExit(1)
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "requires QApplication" in completed.stdout


def test_new_project_transitions_workspace_from_empty_state(
    application: QApplication,
) -> None:
    window = MainWindow()
    assert window.workspace.project_count == 0
    assert window.workspace.shows_empty_state

    project = window.workspace.new_project()

    assert project.name == "Untitled Project"
    assert window.project_context_label.isHidden()
    assert window.workspace.project_count == 1
    assert not window.workspace.shows_empty_state
    tree = window.workspace.tree
    assert tree.topLevelItemCount() == 1
    first_project = tree.topLevelItem(0)
    assert first_project is not None
    assert first_project.text(0) == project.name
    assert tree.isVisibleTo(window.workspace)
    window.close()


def test_workspace_project_selection_does_not_create_open_view_context(
    application: QApplication,
) -> None:
    window = MainWindow()
    window.workspace.new_project()
    window.workspace.new_project()
    assert window.project_context_label.isHidden()

    first_item = window.workspace.tree.topLevelItem(0)
    assert first_item is not None
    window.workspace.tree.setCurrentItem(first_item)
    application.processEvents()

    assert window.project_context_label.isHidden()
    window.close()
