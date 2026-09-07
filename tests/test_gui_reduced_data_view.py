"""Focused tests for Slice 1 reduced-data import and read-only viewing."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.colors import LogNorm, to_hex
from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QFileDialog,
    QHeaderView,
    QMenu,
    QToolButton,
)

from ezqens.domain import (
    ImportValidationError,
    QBins,
    ReducedDataset,
    Spectrum,
    SpectrumRole,
)
from ezqens.gui import MainWindow, create_application
from ezqens.gui.dataset_view import (
    OVERVIEW_ACTIVE_ALPHA,
    OVERVIEW_ACTIVE_COLOR,
    OVERVIEW_COLORMAP,
    _intensity_normalization,
)
from ezqens.gui.main_window import ImportBatchResult, _supported_reduced_data_files
from ezqens.gui.scientific_canvas import SCIENTIFIC_BACKGROUND
from ezqens.gui.theme import (
    DARK_TOKENS,
    LIGHT_TOKENS,
    Appearance,
    DesignTokens,
    application_appearance_controller,
)
from ezqens.gui.workspace import (
    ACTIVE_DATASET_ROLE,
    DatasetAnalysisState,
    DatasetState,
    ProjectState,
    SplitDataButton,
    WorkspaceItemDelegate,
    dataset_analysis_state,
)

FIXTURES = Path(__file__).parent / "fixtures" / "reduced_data"


@pytest.fixture(scope="module")
def application() -> Iterator[QApplication]:
    app = create_application(["ezqens-slice-1-tests"])
    yield app
    app.closeAllWindows()


def _import_two_group_dataset(
    window: MainWindow,
) -> tuple[ProjectState, DatasetState]:
    project = window.workspace.new_project()
    dataset = window.import_data(FIXTURES / "wide_multiple_pairs.txt", project=project)
    return project, dataset


def _highlight_x_bounds(window: MainWindow) -> tuple[float, float]:
    highlight = window.dataset_view.overview_active_highlight
    assert highlight is not None
    return (highlight.get_x(), highlight.get_x() + highlight.get_width())


def test_import_uses_core_and_adds_project_data_hierarchy_without_opening(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, dataset = _import_two_group_dataset(window)

    assert dataset.dataset.source_reference == "wide_multiple_pairs.txt"
    assert dataset.dataset.role is SpectrumRole.SAMPLE
    assert project.datasets == [dataset]
    project_item = window.workspace.tree.topLevelItem(0)
    assert project_item is not None
    data_item = project_item.child(0)
    assert data_item is not None
    assert data_item.text(0) == "Data"
    assert project_item.isFirstColumnSpanned()
    assert data_item.isFirstColumnSpanned()
    assert project_item.isExpanded()
    assert data_item.isExpanded()
    header = window.workspace.tree.header()
    assert not header.stretchLastSection()
    assert header.sectionResizeMode(0) is QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(1) is QHeaderView.ResizeMode.ResizeToContents
    assert (
        window.workspace.tree.horizontalScrollBarPolicy()
        is Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    assert data_item.childCount() == 1
    dataset_item = data_item.child(0)
    assert dataset_item is not None
    assert dataset_item.text(0) == "wide_multiple_pairs.txt"
    assert dataset_item.text(1) == ""
    assert not dataset_item.icon(1).isNull()
    assert "Q required" in dataset_item.toolTip(1)
    assert "Units required" in dataset_item.toolTip(1)
    assert window.dataset_view.dataset is None
    assert window.scientific_canvas_container.isHidden()
    window.close()


def test_single_click_selects_but_double_click_opens_at_group_one(
    application: QApplication,
) -> None:
    window = MainWindow()
    _project, dataset = _import_two_group_dataset(window)
    project_item = window.workspace.tree.topLevelItem(0)
    assert project_item is not None
    data_item = project_item.child(0)
    assert data_item is not None
    dataset_item = data_item.child(0)
    assert dataset_item is not None

    window.workspace.tree.setCurrentItem(dataset_item)
    application.processEvents()
    assert window.dataset_view.dataset is None
    assert window.scientific_canvas_container.isHidden()

    window.workspace.tree.itemDoubleClicked.emit(dataset_item, 0)
    application.processEvents()
    assert window.dataset_view.dataset is dataset.dataset
    assert window.dataset_view.current_group_index == 0
    assert window.dataset_view.group_navigation_label.text() == "Group"
    assert window.dataset_view.group_spinbox.value() == 1
    assert not window.scientific_canvas_container.isHidden()
    assert "wide_multiple_pairs.txt" in window.project_context_label.text()
    window.close()


def test_group_navigation_and_overview_selection_remain_synchronized(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, dataset = _import_two_group_dataset(window)
    window.open_dataset(project, dataset)
    view = window.dataset_view

    view.next_button.click()
    assert view.current_group_index == 1
    assert view.group_spinbox.value() == 2
    assert "Group 2" in window.inspector_context_label.text()
    assert _highlight_x_bounds(window) == view.overview_x_cell_bounds[1]
    assert view.next_button.isEnabled() is False

    view.previous_button.click()
    assert view.current_group_index == 0
    view.set_current_group(1)
    assert view.current_group_index == 1

    view.select_group_from_overview(1.0)
    assert view.current_group_index == 0
    assert view.group_spinbox.value() == 1
    assert "Group 1" in window.inspector_context_label.text()
    assert _highlight_x_bounds(window) == view.overview_x_cell_bounds[0]
    window.close()


def test_group_navigation_has_one_compact_editable_control(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, dataset = _import_two_group_dataset(window)
    window.open_dataset(project, dataset)
    view = window.dataset_view

    assert hasattr(view, "group_spinbox")
    assert not hasattr(view, "current_group_label")
    assert view.group_navigation_label.text() == "Group"
    assert view.group_spinbox.value() == 1
    assert view.group_spinbox.maximum() == 2
    assert view.spectrum_axes is not None
    assert view.spectrum_axes.get_title(loc="left") == "Group 1"
    window.close()


def test_manual_group_entry_clamps_to_real_group_range(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, dataset = _import_two_group_dataset(window)
    window.open_dataset(project, dataset)
    view = window.dataset_view

    view.group_spinbox.lineEdit().setText("2")
    view.group_spinbox.interpretText()

    assert view.current_group_index == 1
    assert view.group_spinbox.value() == 2
    assert (
        view.group_spinbox.buttonSymbols() is QAbstractSpinBox.ButtonSymbols.NoButtons
    )
    view.group_spinbox.lineEdit().setText("99")
    view.group_spinbox.interpretText()
    assert view.current_group_index == 1
    assert view.group_spinbox.value() == 2
    window.close()


@pytest.mark.parametrize(
    ("appearance", "input_background", "text_color"),
    [
        (Appearance.LIGHT, LIGHT_TOKENS.surface_input, LIGHT_TOKENS.text_primary),
        (Appearance.DARK, DARK_TOKENS.surface_input, DARK_TOKENS.text_primary),
    ],
)
def test_group_input_uses_the_centralized_appearance_tokens(
    application: QApplication,
    appearance: Appearance,
    input_background: str,
    text_color: str,
) -> None:
    controller = application_appearance_controller(application)
    controller.set_appearance(appearance)
    window = MainWindow()

    stylesheet = application.styleSheet()
    group_spinbox_style = stylesheet[stylesheet.index("#groupSpinBox") :]
    assert f"background: {input_background};" in group_spinbox_style
    assert f"color: {text_color};" in group_spinbox_style
    assert (
        window.dataset_view.group_spinbox.buttonSymbols()
        is QAbstractSpinBox.ButtonSymbols.NoButtons
    )
    window.close()
    controller.set_appearance(Appearance.SYSTEM)


def test_overview_visibility_preserves_current_group_and_expands_spectrum(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, dataset = _import_two_group_dataset(window)
    window.open_dataset(project, dataset)
    view = window.dataset_view
    view.set_current_group(1)
    window.show()
    application.processEvents()
    navigation_top = view.group_spinbox.geometry().top()
    expanded_overview_height = view.overview_canvas.height()
    expanded_spectrum_height = view.canvas.height()

    view.set_overview_visible(False)

    assert view.overview_axes is None
    assert view.overview_meshes == ()
    assert view.overview_active_highlight is None
    assert view.current_group_index == 1
    assert view.navigator_axes is not None
    assert view.group_spinbox.value() == 2
    assert view.group_spinbox.geometry().top() == navigation_top
    assert view.overview_toggle_button.text() == "Show Overview"
    application.processEvents()
    assert view.overview_canvas.height() < expanded_overview_height
    assert view.canvas.height() > expanded_spectrum_height

    view.set_overview_visible(True)

    assert view.overview_axes is not None
    assert view.current_group_index == 1
    assert _highlight_x_bounds(window) == view.overview_x_cell_bounds[1]
    assert view.overview_toggle_button.text() == "Hide Overview"
    window.close()


def test_collapsed_navigator_click_and_drag_use_same_discrete_group_state(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, dataset = _import_two_group_dataset(window)
    window.open_dataset(project, dataset)
    view = window.dataset_view
    view.set_overview_visible(False)
    assert view.navigator_axes is not None

    view.select_group_from_navigator(2.0)
    assert view.current_group_index == 1
    view.select_group_from_navigator(1.0)
    assert view.current_group_index == 0
    press = cast(
        MouseEvent,
        SimpleNamespace(
            button=MouseButton.LEFT,
            inaxes=view.navigator_axes,
            xdata=1.0,
        ),
    )
    drag = cast(
        MouseEvent,
        SimpleNamespace(inaxes=view.navigator_axes, xdata=2.0),
    )
    view._on_button_press(press)
    view._on_mouse_motion(drag)

    assert view.current_group_index == 1
    assert view.group_spinbox.value() == 2
    assert [label.get_text() for label in view.navigator_axes.get_xticklabels()] == [
        "1",
        "2",
    ]
    assert view.spectrum_axes is not None
    np.testing.assert_array_equal(
        view.spectrum_axes.lines[0].get_ydata(),
        dataset.dataset.spectra[1].intensity,
    )
    window.close()


def test_collapsed_navigator_uses_real_q_centers_as_labels(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, imported = _import_two_group_dataset(window)
    assigned = imported.dataset.assign_q_bins(QBins.from_q_values([0.42, 1.18]))
    q_dataset = window.workspace.add_dataset(project, assigned)
    window.open_dataset(project, q_dataset)
    view = window.dataset_view
    view.set_overview_visible(False)
    assert view.navigator_axes is not None

    labels = {text.get_text() for text in view.navigator_axes.texts}
    assert {"0.42", "1.18", "Å⁻¹"}.issubset(labels)
    view.select_group_from_navigator(2.0)
    assert view.current_q_label.text() == "Q = 1.18 Å⁻¹"
    window.close()


def test_dragging_overview_snaps_to_discrete_groups_and_updates_spectrum(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, dataset = _import_two_group_dataset(window)
    window.open_dataset(project, dataset)
    view = window.dataset_view
    assert view.overview_axes is not None

    press = cast(
        MouseEvent,
        SimpleNamespace(
            button=MouseButton.LEFT,
            inaxes=view.overview_axes,
            xdata=1.0,
        ),
    )
    drag = cast(
        MouseEvent,
        SimpleNamespace(inaxes=view.overview_axes, xdata=2.0),
    )
    release = cast(MouseEvent, SimpleNamespace())
    view._on_button_press(press)
    view._on_mouse_motion(drag)
    view._on_button_release(release)

    assert view.current_group_index == 1
    assert view._navigation_drag_active is False
    assert _highlight_x_bounds(window) == view.overview_x_cell_bounds[1]
    assert view.spectrum_axes is not None
    np.testing.assert_array_equal(
        view.spectrum_axes.lines[0].get_ydata(),
        dataset.dataset.spectra[1].intensity,
    )
    window.close()


def test_spectrum_view_scale_and_lock_controls_preserve_source_arrays(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, dataset = _import_two_group_dataset(window)
    window.open_dataset(project, dataset)
    view = window.dataset_view
    snapshots = [spectrum.intensity.copy() for spectrum in dataset.dataset.spectra]
    menu = view._build_spectrum_context_menu()
    assert [action.text() for action in menu.actions()] == [
        "Assign Q…",
        "",
        "Reset View",
        "Y Scale",
        "",
        "Lock Y Range",
        "",
        "Edit Mask…",
    ]

    assert view.set_y_scale("log") is True
    assert view.spectrum_axes is not None
    assert view.spectrum_axes.get_yscale() == "log"
    assert view.set_y_scale("linear") is True
    assert view.spectrum_axes.get_yscale() == "linear"
    for spectrum, snapshot in zip(dataset.dataset.spectra, snapshots, strict=True):
        np.testing.assert_array_equal(spectrum.intensity, snapshot)

    default_locked_limits = view.spectrum_axes.get_ylim()
    assert view.y_range_locked
    view.set_current_group(1)
    assert view.spectrum_axes.get_ylim() == pytest.approx(default_locked_limits)
    view.set_y_range_locked(False)
    view.set_current_group(0)
    view.set_current_group(1)
    assert view.spectrum_axes.get_ylim() != pytest.approx(default_locked_limits)
    window.close()


def test_plot_context_menus_use_their_qt_canvas_as_the_full_hit_region(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = ReducedDataset(
        role=SpectrumRole.SAMPLE,
        spectra=(
            Spectrum(
                role=SpectrumRole.SAMPLE,
                group_index=0,
                group_label="Group 1",
                energy=np.linspace(-1.0, 1.0, 5),
                intensity=np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
                uncertainty=np.full(5, 0.1),
                energy_unit="meV",
                intensity_unit="arb. unit",
                uncertainty_unit="arb. unit",
            ),
        ),
        q_bins=QBins.from_q_values((0.5,)),
    )
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, dataset)
    assert window.open_dataset(project, state)
    window.show()
    application.processEvents()
    view = window.dataset_view
    assert view.spectrum_axes is not None
    assert view.overview_axes is not None

    opened: list[tuple[str, QPoint]] = []

    class RecordingMenu:
        def __init__(self, role: str) -> None:
            self.role = role

        def exec(self, position: QPoint) -> None:
            opened.append((self.role, position))

    monkeypatch.setattr(
        view,
        "_build_spectrum_context_menu",
        lambda: RecordingMenu("spectrum"),
    )
    monkeypatch.setattr(
        view,
        "_build_overview_context_menu",
        lambda: RecordingMenu("overview"),
    )

    negative_display = view.spectrum_axes.transData.transform((0.0, -1.0))
    negative_position = QPoint(
        round(float(negative_display[0])),
        round(view.canvas.height() - float(negative_display[1])),
    )
    assert view.canvas.rect().contains(negative_position)
    view._show_spectrum_context_menu(negative_position)
    view._show_spectrum_context_menu(
        QPoint(view.canvas.width() // 2, view.canvas.height() - 1),
    )
    view._show_overview_context_menu(
        QPoint(view.overview_canvas.width() // 2, view.overview_canvas.height() - 1),
    )
    view.set_overview_visible(False)
    assert view.navigator_axes is not None
    view._show_overview_context_menu(
        QPoint(view.overview_canvas.width() // 2, view.overview_canvas.height() - 1),
    )

    assert [role for role, _position in opened] == [
        "spectrum",
        "spectrum",
        "overview",
        "overview",
    ]
    assert (
        view.controls_container.contextMenuPolicy()
        is not Qt.ContextMenuPolicy.CustomContextMenu
    )
    window.close()


def test_no_q_dataset_uses_group_and_unknown_units_stay_truthful(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, dataset = _import_two_group_dataset(window)
    window.open_dataset(project, dataset)
    view = window.dataset_view
    assert view.overview_axes is not None
    assert view.spectrum_axes is not None

    assert view.overview_axes.get_xlabel() == ""
    assert view.overview_axes.get_ylabel() == "Energy"
    assert view.overview_axes.get_title() == ""
    assert [label.get_text() for label in view.overview_axes.get_xticklabels()] == [
        "1",
        "2",
    ]
    assert view.overview_q_axis is None
    assert view.spectrum_axes.get_xlabel() == "Energy"
    assert "meV" not in view.spectrum_axes.get_xlabel()
    assert "Q required" in view.metadata_status_label.text()
    assert "Units required" in view.metadata_status_label.text()
    window.close()


def test_q_assigned_dataset_uses_real_representatives_and_keeps_group_identity(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, imported = _import_two_group_dataset(window)
    assigned = imported.dataset.assign_q_bins(QBins.from_q_values([0.42, 1.18]))
    q_dataset = window.workspace.add_dataset(project, assigned)
    window.open_dataset(project, q_dataset)
    view = window.dataset_view
    assert view.overview_axes is not None

    assert view.overview_axes.get_xlabel() == ""
    assert view.overview_axes.get_ylabel() == "Energy"
    assert view.overview_q_axis is not None
    assert view.group_spinbox.value() == 1
    assert view.current_q_label.text() == "Q = 0.42 Å⁻¹"
    view.select_group_from_overview(1.18)
    assert view.current_group_index == 1
    assert view.group_spinbox.value() == 2
    assert view.current_q_label.text() == "Q = 1.18 Å⁻¹"
    assert _highlight_x_bounds(window) == view.overview_x_cell_bounds[1]
    window.close()


def test_explicit_q_bin_bounds_select_the_rendered_cell_before_nearest_q(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, imported = _import_two_group_dataset(window)
    q_bins = QBins(
        q_values=np.array([0.36, 0.92]),
        edges=np.array([0.2, 0.5, 1.2]),
    )
    assigned = imported.dataset.assign_q_bins(q_bins)
    q_dataset = window.workspace.add_dataset(project, assigned)
    window.open_dataset(project, q_dataset)
    view = window.dataset_view
    assert view.overview_x_cell_bounds == ((0.2, 0.5), (0.5, 1.2))

    view.select_group_from_overview(0.55)

    assert view.current_group_index == 1
    assert _highlight_x_bounds(window) == (0.5, 1.2)
    view.select_group_from_overview(0.1)
    assert view.current_group_index == 0
    window.close()


def test_overview_drag_uses_rendered_q_bin_bounds(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, imported = _import_two_group_dataset(window)
    assigned = imported.dataset.assign_q_bins(
        QBins(
            q_values=np.array([0.36, 0.92]),
            edges=np.array([0.2, 0.5, 1.2]),
        ),
    )
    q_dataset = window.workspace.add_dataset(project, assigned)
    window.open_dataset(project, q_dataset)
    view = window.dataset_view
    assert view.overview_axes is not None
    press = cast(
        MouseEvent,
        SimpleNamespace(
            button=MouseButton.LEFT,
            inaxes=view.overview_axes,
            xdata=0.36,
        ),
    )
    drag = cast(
        MouseEvent,
        SimpleNamespace(inaxes=view.overview_axes, xdata=0.55),
    )
    view._on_button_press(press)
    view._on_mouse_motion(drag)

    assert view.current_group_index == 1
    assert view.current_q_label.text() == "Q = 0.92 Å⁻¹"
    window.close()


@pytest.mark.parametrize(
    ("appearance", "tokens"),
    [
        (Appearance.LIGHT, LIGHT_TOKENS),
        (Appearance.DARK, DARK_TOKENS),
    ],
)
def test_overview_and_collapsed_navigator_follow_theme_but_spectrum_stays_white(
    application: QApplication,
    appearance: Appearance,
    tokens: DesignTokens,
) -> None:
    controller = application_appearance_controller(application)
    controller.set_appearance(appearance)
    window = MainWindow()
    project, dataset = _import_two_group_dataset(window)
    window.open_dataset(project, dataset)
    view = window.dataset_view
    assert view.overview_axes is not None
    assert view.spectrum_axes is not None

    assert to_hex(view.overview_axes.figure.get_facecolor()) == tokens.surface_central
    assert to_hex(view.overview_axes.get_facecolor()) == tokens.surface_central
    assert all(
        to_hex(label.get_color()) == tokens.text_primary
        for label in view.overview_axes.get_xticklabels()
    )
    assert to_hex(view.overview_axes.yaxis.label.get_color()) == tokens.text_primary
    assert all(
        to_hex(spine.get_edgecolor()) == tokens.canvas_boundary
        for spine in view.overview_axes.spines.values()
    )
    assert to_hex(view.spectrum_axes.figure.get_facecolor()) == SCIENTIFIC_BACKGROUND
    assert to_hex(view.spectrum_axes.get_facecolor()) == SCIENTIFIC_BACKGROUND
    assert to_hex(view.spectrum_axes.xaxis.label.get_color()) == "#000000"
    assert to_hex(view.spectrum_axes.yaxis.label.get_color()) == "#000000"
    view.set_overview_visible(False)
    assert view.navigator_axes is not None
    assert to_hex(view.navigator_axes.figure.get_facecolor()) == tokens.surface_central
    assert to_hex(view.navigator_axes.get_facecolor()) == tokens.surface_central
    assert all(
        to_hex(label.get_color()) == tokens.text_primary
        for label in view.navigator_axes.get_xticklabels()
    )
    window.close()
    controller.set_appearance(Appearance.SYSTEM)


def test_switching_groups_plots_real_values_and_preserves_arrays(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, dataset = _import_two_group_dataset(window)
    snapshots = [
        (
            spectrum.energy.copy(),
            spectrum.intensity.copy(),
            spectrum.uncertainty.copy(),
        )
        for spectrum in dataset.dataset.spectra
    ]
    window.open_dataset(project, dataset)
    view = window.dataset_view
    assert view.spectrum_axes is not None
    line = view.spectrum_axes.lines[0]
    np.testing.assert_array_equal(line.get_xdata(), snapshots[0][0])
    np.testing.assert_array_equal(line.get_ydata(), snapshots[0][1])

    view.set_current_group(1)
    assert view.spectrum_axes is not None
    line = view.spectrum_axes.lines[0]
    np.testing.assert_array_equal(line.get_xdata(), snapshots[1][0])
    np.testing.assert_array_equal(line.get_ydata(), snapshots[1][1])
    for spectrum, snapshot in zip(dataset.dataset.spectra, snapshots, strict=True):
        np.testing.assert_array_equal(spectrum.energy, snapshot[0])
        np.testing.assert_array_equal(spectrum.intensity, snapshot[1])
        np.testing.assert_array_equal(spectrum.uncertainty, snapshot[2])
    window.close()


def test_overview_preserves_unequal_grids_without_interpolation(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    dataset = window.import_data(FIXTURES / "dave_multiple_groups.dat", project=project)
    window.open_dataset(project, dataset)
    view = window.dataset_view
    assert view.overview_axes is not None
    assert len(view.overview_meshes) == 2
    np.testing.assert_array_equal(
        np.asarray(view.overview_meshes[0].get_array()).ravel(),
        dataset.dataset.spectra[0].intensity,
    )
    np.testing.assert_array_equal(
        np.asarray(view.overview_meshes[1].get_array()).ravel(),
        dataset.dataset.spectra[1].intensity,
    )
    assert view.overview_x_cell_bounds == ((0.5, 1.5), (1.5, 2.5))
    assert view.overview_x_cell_bounds[0][1] == view.overview_x_cell_bounds[1][0]
    window.close()


def test_overview_active_band_brightens_full_current_cell_without_changing_data(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, dataset = _import_two_group_dataset(window)
    window.open_dataset(project, dataset)
    view = window.dataset_view
    assert view.overview_axes is not None
    assert _highlight_x_bounds(window) == view.overview_x_cell_bounds[0]
    highlight = view.overview_active_highlight
    assert highlight is not None
    facecolor = np.asarray(highlight.get_facecolor(), dtype=float)
    edgecolor = np.asarray(highlight.get_edgecolor(), dtype=float)
    assert to_hex(highlight.get_facecolor()) == OVERVIEW_ACTIVE_COLOR
    assert facecolor[-1] == pytest.approx(OVERVIEW_ACTIVE_ALPHA)
    assert to_hex(highlight.get_edgecolor()) == OVERVIEW_ACTIVE_COLOR
    assert edgecolor[-1] == pytest.approx(1.0)
    assert highlight.get_linewidth() == pytest.approx(1.25)
    rendered_values = tuple(
        np.array(mesh.get_array(), copy=True) for mesh in view.overview_meshes
    )
    assert not view.overview_axes.lines
    assert all(mesh.cmap.name == OVERVIEW_COLORMAP for mesh in view.overview_meshes)

    view.set_current_group(1)
    assert _highlight_x_bounds(window) == view.overview_x_cell_bounds[1]
    for rendered, spectrum in zip(
        rendered_values,
        dataset.dataset.spectra,
        strict=True,
    ):
        np.testing.assert_array_equal(rendered.ravel(), spectrum.intensity)
    window.close()


def test_overview_does_not_treat_negative_intensity_as_a_mask(
    application: QApplication,
) -> None:
    dataset = ReducedDataset(
        role=SpectrumRole.SAMPLE,
        spectra=(
            Spectrum(
                role=SpectrumRole.SAMPLE,
                group_index=0,
                group_label="1",
                energy=np.array([-1.0, 0.0]),
                intensity=np.array([-2.0, 1.0]),
                uncertainty=np.array([0.1, 0.1]),
                energy_unit="unknown",
                intensity_unit="unknown",
                uncertainty_unit="unknown",
            ),
        ),
    )
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, dataset)
    window.open_dataset(project, state)
    mesh_values = window.dataset_view.overview_meshes[0].get_array()

    assert mesh_values is not None
    np.testing.assert_array_equal(np.asarray(mesh_values).ravel(), [-2.0, 1.0])
    assert not np.any(np.ma.getmaskarray(mesh_values))
    window.close()


def test_heatmap_scale_is_independent_reversible_and_display_only(
    application: QApplication,
) -> None:
    dataset = ReducedDataset(
        role=SpectrumRole.SAMPLE,
        spectra=(
            Spectrum(
                role=SpectrumRole.SAMPLE,
                group_index=0,
                group_label="1",
                energy=np.arange(5, dtype=float),
                intensity=np.array([-2.0, 0.0, 1.0, 10.0, 100.0]),
                uncertainty=np.full(5, 0.1),
                energy_unit="unknown",
                intensity_unit="unknown",
                uncertainty_unit="unknown",
            ),
        ),
    )
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, dataset)
    window.open_dataset(project, state)
    view = window.dataset_view
    intensity_snapshot = dataset.spectra[0].intensity.copy()
    selection_snapshot = view.selection
    linear_norm = view.overview_meshes[0].norm
    assert not isinstance(linear_norm, LogNorm)
    linear_midpoint = float(np.asarray(linear_norm(np.array([10.0])))[0])

    assert view.set_y_scale("log")
    assert view.heatmap_scale == "linear"
    assert view.set_heatmap_scale("log")
    assert view.y_scale == "log"
    log_norm = view.overview_meshes[0].norm
    assert isinstance(log_norm, LogNorm)
    normalized = np.ma.asarray(log_norm(np.array([-2.0, 0.0, 1.0, 10.0, 100.0])))
    assert not np.any(np.ma.getmaskarray(normalized))
    assert normalized[:2].tolist() == [-1.0, -1.0]
    assert normalized[2:].tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert float(np.asarray(log_norm(np.array([10.0])))[0]) != pytest.approx(
        linear_midpoint
    )
    np.testing.assert_array_equal(
        np.asarray(view.overview_meshes[0].get_array()).ravel(),
        intensity_snapshot,
    )
    np.testing.assert_array_equal(dataset.spectra[0].intensity, intensity_snapshot)
    assert view.selection is selection_snapshot

    visible_scale_buttons = {
        button.text() for button in view.findChildren(QToolButton) if button.isVisible()
    }
    assert not {"Linear", "Log"} & visible_scale_buttons
    overview_menu = view._build_overview_context_menu()
    heatmap_menu = overview_menu.actions()[0].menu()
    assert isinstance(heatmap_menu, QMenu)
    assert heatmap_menu.title() == "Heatmap Scale"
    assert [action.text() for action in heatmap_menu.actions()] == ["Linear", "Log"]
    assert heatmap_menu.actions()[1].isChecked()

    assert view.set_heatmap_scale("linear")
    assert view.y_scale == "log"
    restored_norm = view.overview_meshes[0].norm
    assert not isinstance(restored_norm, LogNorm)
    assert float(np.asarray(restored_norm(np.array([10.0])))[0]) == pytest.approx(
        linear_midpoint
    )
    np.testing.assert_array_equal(dataset.spectra[0].intensity, intensity_snapshot)
    assert view.selection is selection_snapshot

    no_positive = _intensity_normalization(np.array([-2.0, 0.0]), "log")
    assert no_positive is not None
    np.testing.assert_array_equal(no_positive(np.array([-2.0, 0.0])), [0.0, 0.0])
    window.close()


def test_sample_view_state_is_per_sample_before_and_after_manual_session(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, imported = _import_two_group_dataset(window)
    template = imported.dataset.spectra[0]

    def sample(name: str) -> ReducedDataset:
        return ReducedDataset(
            role=SpectrumRole.SAMPLE,
            spectra=tuple(
                replace(
                    template,
                    group_index=index,
                    group_label=f"{name} {index + 1}",
                    intensity=template.intensity + index,
                )
                for index in range(4)
            ),
        )

    first = window.workspace.add_dataset(project, sample("A"))
    second = window.workspace.add_dataset(project, sample("B"))
    assert window.open_dataset(project, first)
    window.dataset_view.set_current_group(2)
    assert window.dataset_view.set_y_scale("log")
    assert window.dataset_view.set_heatmap_scale("log")
    assert not window._manual_sessions

    assert window.open_dataset(project, second)
    window.dataset_view.set_current_group(1)
    assert window.dataset_view.current_group_index == 1
    assert window.dataset_view.y_scale == "linear"
    assert window.dataset_view.heatmap_scale == "linear"
    assert not window._manual_sessions

    assert window.open_dataset(project, first)
    assert window.dataset_view.current_group_index == 2
    assert window.dataset_view.y_scale == "log"
    assert window.dataset_view.heatmap_scale == "log"
    window.show_manual_fit(project, first)
    assert len(window._manual_sessions) == 1

    assert window.open_dataset(project, second)
    assert window.dataset_view.current_group_index == 1
    assert window.dataset_view.y_scale == "linear"
    assert window.dataset_view.heatmap_scale == "linear"
    assert window.open_dataset(project, first)
    assert window.dataset_view.current_group_index == 2
    assert window.dataset_view.y_scale == "log"
    assert window.dataset_view.heatmap_scale == "log"
    window.close()


def test_switching_dataset_resets_to_group_one_and_updates_values(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, first = _import_two_group_dataset(window)
    second = window.import_data(FIXTURES / "single_valid.csv", project=project)
    window.open_dataset(project, first)
    window.dataset_view.set_current_group(1)
    window.open_dataset(project, second)

    assert window.dataset_view.dataset is second.dataset
    assert window.dataset_view.current_group_index == 0
    assert window.dataset_view.group_spinbox.value() == 1
    assert window.dataset_view.spectrum_axes is not None
    np.testing.assert_array_equal(
        window.dataset_view.spectrum_axes.lines[0].get_ydata(),
        second.dataset.spectra[0].intensity,
    )
    window.close()


def test_workspace_selection_stays_distinct_from_open_view_context(
    application: QApplication,
) -> None:
    window = MainWindow()
    first_project, dataset = _import_two_group_dataset(window)
    window.open_dataset(first_project, dataset)
    open_header = window.project_context_label.text()
    open_inspector = window.inspector_context_label.text()
    second_project = window.workspace.new_project()
    second_dataset = window.import_data(
        FIXTURES / "single_valid.csv",
        project=second_project,
    )
    second_item = window.workspace.tree.topLevelItem(1)
    assert second_item is not None
    window.workspace.tree.setCurrentItem(second_item)
    application.processEvents()

    assert window.project_context_label.text() == open_header
    assert window.inspector_context_label.text() == open_inspector
    assert window.dataset_view.dataset is dataset.dataset
    assert window.dataset_view.current_group_index == 0

    second_data_item = second_item.child(0)
    assert second_data_item is not None
    second_dataset_item = second_data_item.child(0)
    assert second_dataset_item is not None
    window.workspace.tree.setCurrentItem(second_dataset_item)
    application.processEvents()
    assert window.project_context_label.text() == open_header
    assert window.inspector_context_label.text() == open_inspector
    assert window.dataset_view.dataset is dataset.dataset

    window.open_dataset(second_project, second_dataset)
    assert window.project_context_label.text() == (
        f"{second_project.name} · {second_dataset.name}"
    )
    assert window.dataset_view.dataset is second_dataset.dataset
    assert second_project.name in window.inspector_context_label.text()
    assert second_dataset.name in window.inspector_context_label.text()
    window.close()


def test_workspace_active_dataset_state_persists_across_selection_and_moves_on_open(
    application: QApplication,
) -> None:
    controller = application_appearance_controller(application)
    controller.set_appearance(Appearance.LIGHT)
    window = MainWindow()
    first_project, first_dataset = _import_two_group_dataset(window)
    window.open_dataset(first_project, first_dataset)
    second_project = window.workspace.new_project()
    second_dataset = window.import_data(
        FIXTURES / "single_valid.csv",
        project=second_project,
    )
    first_project_item = window.workspace.tree.topLevelItem(0)
    second_project_item = window.workspace.tree.topLevelItem(1)
    assert first_project_item is not None
    assert second_project_item is not None
    first_data_item = first_project_item.child(0)
    second_data_item = second_project_item.child(0)
    assert first_data_item is not None
    assert second_data_item is not None
    first_item = first_data_item.child(0)
    second_item = second_data_item.child(0)
    assert first_item is not None
    assert second_item is not None

    assert first_item.data(0, ACTIVE_DATASET_ROLE) is True
    window.workspace.tree.setCurrentItem(second_item)
    application.processEvents()
    assert first_item.data(0, ACTIVE_DATASET_ROLE) is True
    assert second_item.data(0, ACTIVE_DATASET_ROLE) is False
    assert (
        window.workspace.item_delegate._active_background
        == LIGHT_TOKENS.surface_selected
    )
    assert (
        window.workspace.item_delegate._selection_background
        == LIGHT_TOKENS.surface_hover
    )

    window.open_dataset(second_project, second_dataset)
    first_item = first_data_item.child(0)
    second_item = second_data_item.child(0)
    assert first_item is not None
    assert second_item is not None
    assert first_item.data(0, ACTIVE_DATASET_ROLE) is False
    assert second_item.data(0, ACTIVE_DATASET_ROLE) is True
    window.close()
    controller.set_appearance(Appearance.SYSTEM)


def test_core_import_diagnostics_propagate_from_programmatic_gui_boundary(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()

    with pytest.raises(ImportValidationError) as caught:
        window.import_data(
            FIXTURES / "wide_missing_yerr.txt",
            project=project,
        )

    assert {diagnostic.code for diagnostic in caught.value.diagnostics} == {
        "wide_uncertainty_column_missing"
    }
    assert project.datasets == []
    window.close()


def test_open_scientific_figure_stays_white_inside_themed_dark_workspace(
    application: QApplication,
) -> None:
    controller = application_appearance_controller(application)
    controller.set_appearance(Appearance.DARK)
    window = MainWindow()
    project, dataset = _import_two_group_dataset(window)
    window.open_dataset(project, dataset)
    assert window.dataset_view.overview_axes is not None
    assert window.dataset_view.spectrum_axes is not None

    assert to_hex(window.scientific_canvas.figure.get_facecolor()) == (
        SCIENTIFIC_BACKGROUND
    )
    assert to_hex(window.dataset_view.overview_axes.get_facecolor()) != (
        SCIENTIFIC_BACKGROUND
    )
    assert to_hex(window.dataset_view.spectrum_axes.get_facecolor()) == (
        SCIENTIFIC_BACKGROUND
    )
    assert "#scientificCanvasSurface {\n    background: #ffffff;" not in (
        application.styleSheet()
    )
    window.close()
    controller.set_appearance(Appearance.SYSTEM)


def test_open_dataset_rejects_foreign_membership_without_mutating_view_context(
    application: QApplication,
) -> None:
    window = MainWindow()
    first_project, first_dataset = _import_two_group_dataset(window)
    window.open_dataset(first_project, first_dataset)
    original_header = window.project_context_label.text()
    original_inspector = window.inspector_context_label.text()
    second_project = window.workspace.new_project()
    second_dataset = window.import_data(
        FIXTURES / "single_valid.csv",
        project=second_project,
    )

    with pytest.raises(ValueError, match="dataset does not belong"):
        window.open_dataset(second_project, first_dataset)

    assert window.project_context_label.text() == original_header
    assert window.inspector_context_label.text() == original_inspector
    assert window.dataset_view.dataset is first_dataset.dataset
    assert window.dataset_view.current_group_index == 0
    assert second_dataset.dataset.role is SpectrumRole.SAMPLE
    window.close()


def test_import_action_adds_ordinary_data_without_a_role_dialog(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileNames",
        lambda *_args: ([str(FIXTURES / "single_valid.csv")], ""),
    )

    window._prompt_import_files()

    assert len(project.datasets) == 1
    assert project.datasets[0].dataset.role is SpectrumRole.SAMPLE
    window.close()


def test_files_prompt_processes_multiple_selected_paths(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    results: list[ImportBatchResult] = []
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileNames",
        lambda *_args: (
            [
                str(FIXTURES / "single_valid.csv"),
                str(FIXTURES / "wide_multiple_pairs.txt"),
            ],
            "",
        ),
    )
    monkeypatch.setattr(window, "_show_import_result", results.append)

    window._prompt_import_files()

    assert len(project.datasets) == 2
    assert len(results) == 1
    assert len(results[0].imported) == 2
    assert results[0].failures == ()
    assert window.dataset_view.dataset is None
    assert all(
        dataset.dataset.role is SpectrumRole.SAMPLE for dataset in project.datasets
    )
    window.close()


def test_workspace_switches_from_empty_project_action_to_import_menu(
    application: QApplication,
) -> None:
    window = MainWindow()
    workspace = window.workspace
    window.show()
    application.processEvents()

    assert not workspace.new_project_button.isHidden()
    assert workspace.new_project_button.text() == "Project"
    assert workspace.import_data_button.isHidden()
    assert workspace.new_project_button.width() >= workspace.width() - 26
    empty_font = workspace.new_project_button.font()

    workspace.new_project()
    application.processEvents()

    assert workspace.new_project_button.isHidden()
    assert not workspace.import_data_button.isHidden()
    assert workspace.import_data_button.text() == "Import"
    assert workspace.import_data_button.width() >= workspace.width() - 26
    populated_font = workspace.import_data_button.font()
    assert populated_font.pointSizeF() == empty_font.pointSizeF()
    assert populated_font.weight() == empty_font.weight()
    menu = workspace.import_data_button.menu()
    assert menu is not None
    assert [action.text() for action in menu.actions()] == ["Files…", "Folder…"]
    window.close()


def test_primary_import_button_opens_files_directly(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileNames",
        lambda *_args: ([str(FIXTURES / "single_valid.csv")], ""),
    )

    window.workspace.import_data_button.click()

    assert len(project.datasets) == 1
    assert project.datasets[0].dataset.source_reference == "single_valid.csv"
    spectrum = project.datasets[0].dataset.spectra[0]
    assert spectrum.intensity_unit == "arb. unit"
    assert spectrum.uncertainty_unit == "arb. unit"
    window.close()


def test_import_split_button_dropdown_uses_its_files_and_folder_actions(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow()
    window.workspace.new_project()
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *_args: ([], ""))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args: "")
    requested: list[str] = []
    window.workspace.import_files_requested.connect(
        lambda: requested.append("files"),
    )
    window.workspace.import_folder_requested.connect(
        lambda: requested.append("folder"),
    )
    menu = window.workspace.import_data_button.menu()
    assert menu is not None

    menu.actions()[0].trigger()
    menu.actions()[1].trigger()

    assert requested == ["files", "folder"]
    window.close()


def test_import_split_button_calculates_a_right_aligned_popup_position(
    application: QApplication,
) -> None:
    window = MainWindow()
    window.workspace.new_project()
    button = window.workspace.import_data_button
    assert isinstance(button, SplitDataButton)
    assert button.property("controlKind") == "split"
    window.show()
    application.processEvents()

    menu_size = QSize(96, 48)
    position = button.menu_popup_position(menu_size)
    local_position = button.mapFromGlobal(position)

    assert local_position.x() + menu_size.width() == button.width()
    assert local_position.y() == button.height()
    window.close()


def test_new_project_remains_available_from_file_and_blank_workspace_menu(
    application: QApplication,
) -> None:
    window = MainWindow()
    file_menu = window.menuBar().actions()[0].menu()
    assert isinstance(file_menu, QMenu)
    assert window.new_project_action in file_menu.actions()

    blank_menu = window.workspace._create_context_menu(None)
    assert blank_menu is not None
    assert [action.text() for action in blank_menu.actions()] == ["New Project"]
    blank_menu.actions()[0].trigger()
    assert window.workspace.project_count == 1
    window.close()


def test_dataset_rename_changes_only_workspace_display_name(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, dataset = _import_two_group_dataset(window)
    window.open_dataset(project, dataset)

    renamed = window.workspace.rename_dataset(project, dataset, "Reference scan")

    assert renamed.name == "Reference scan"
    assert renamed.dataset is dataset.dataset
    assert renamed.dataset.source_reference == "wide_multiple_pairs.txt"
    assert renamed.source_path == FIXTURES / "wide_multiple_pairs.txt"
    assert "Reference scan" in window.project_context_label.text()
    assert "Source:" in window.inspector_context_label.text()
    assert "wide_multiple_pairs.txt" in window.inspector_context_label.text()
    project_item = window.workspace.tree.topLevelItem(0)
    assert project_item is not None
    data_item = project_item.child(0)
    assert data_item is not None
    dataset_item = data_item.child(0)
    assert dataset_item is not None
    assert dataset_item.text(0) == "Reference scan"
    assert "wide_multiple_pairs.txt" in dataset_item.toolTip(0)
    menu = window.workspace._create_context_menu(dataset_item)
    assert menu is not None
    assert menu.actions()[0].text() == "Rename"
    window.close()


def test_project_rename_updates_open_context_without_changing_view_state(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, dataset = _import_two_group_dataset(window)
    window.open_dataset(project, dataset)
    window.dataset_view.set_current_group(1)

    window.workspace.rename_project(project, "Calibration run")

    assert project.name == "Calibration run"
    assert window.dataset_view.dataset is dataset.dataset
    assert window.dataset_view.current_group_index == 1
    assert window.project_context_label.text() == (
        "Calibration run · wide_multiple_pairs.txt"
    )
    assert "Calibration run" in window.inspector_context_label.text()
    project_item = window.workspace.tree.topLevelItem(0)
    assert project_item is not None
    assert project_item.text(0) == "Calibration run"
    menu = window.workspace._create_context_menu(project_item)
    assert menu is not None
    assert [action.text() for action in menu.actions()] == [
        "Rename",
        "Delete Project",
    ]
    window.close()


def test_dataset_state_indicator_is_compact_and_truthful(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, imported = _import_two_group_dataset(window)
    project_item = window.workspace.tree.topLevelItem(0)
    assert project_item is not None
    data_item = project_item.child(0)
    assert data_item is not None
    imported_item = data_item.child(0)
    assert imported_item is not None

    assert dataset_analysis_state(imported.dataset) is DatasetAnalysisState.REQUIRED
    assert imported_item.text(1) == ""
    assert not imported_item.icon(1).isNull()
    assert "Information required" in imported_item.toolTip(1)
    assert "Q required" not in imported_item.text(1)
    assert "Units required" not in imported_item.text(1)

    complete = replace(
        imported.dataset.assign_q_bins(QBins.from_q_values([0.42, 1.18])),
        spectra=tuple(
            replace(
                spectrum,
                energy_unit="meV",
                intensity_unit="counts",
                uncertainty_unit="counts",
            )
            for spectrum in imported.dataset.spectra
        ),
    )
    complete_state = window.workspace.add_dataset(project, complete)
    assert dataset_analysis_state(complete) is DatasetAnalysisState.READY
    assert (
        dataset_analysis_state(complete, fitted_group_count=1)
        is DatasetAnalysisState.PARTIALLY_FIT
    )
    assert (
        dataset_analysis_state(complete, fitted_group_count=2)
        is DatasetAnalysisState.FULLY_FIT
    )
    assert complete_state.dataset is complete
    complete_item = data_item.child(1)
    assert complete_item is not None
    assert "Ready for Fit" in complete_item.toolTip(1)
    window.close()


def test_resolution_identity_is_separate_from_dataset_state_indicator(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, dataset = _import_two_group_dataset(window)
    resolution = window.workspace.set_dataset_role(
        project,
        dataset,
        SpectrumRole.RESOLUTION,
    )
    assert resolution is not None
    project_item = window.workspace.tree.topLevelItem(0)
    assert project_item is not None
    data_item = project_item.child(0)
    assert data_item is not None
    resolution_item = data_item.child(0)
    assert resolution_item is not None

    assert resolution_item.text(1) == "Resolution"
    assert not resolution_item.icon(0).isNull()
    assert not resolution_item.icon(1).isNull()
    assert dataset_analysis_state(resolution.dataset) is DatasetAnalysisState.REQUIRED
    assert "Information required" in resolution_item.toolTip(1)
    window.close()


def test_import_target_comes_from_workspace_selection_not_open_view_context(
    application: QApplication,
) -> None:
    window = MainWindow()
    first_project, first_dataset = _import_two_group_dataset(window)
    second_project = window.workspace.new_project()
    window.open_dataset(first_project, first_dataset)
    second_item = window.workspace.tree.topLevelItem(1)
    assert second_item is not None
    second_data_item = second_item.child(0)
    assert second_data_item is not None
    window.workspace.tree.setCurrentItem(second_data_item)

    result = window.import_paths((FIXTURES / "single_valid.csv",))

    assert result.failures == ()
    assert len(result.imported) == 1
    assert second_project.datasets == [result.imported[0]]
    assert first_project.datasets == [first_dataset]
    assert window.dataset_view.dataset is first_dataset.dataset
    window.close()


def test_batch_import_keeps_valid_files_when_one_file_fails(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()

    result = window.import_paths(
        (
            FIXTURES / "single_valid.csv",
            FIXTURES / "wide_missing_yerr.txt",
        ),
        project=project,
    )

    assert len(result.imported) == 1
    assert len(result.failures) == 1
    assert "yerr3" in result.failures[0].detail
    assert result.summary == "1 imported · 1 skipped"
    assert project.datasets == [result.imported[0]]
    assert window.dataset_view.dataset is None
    assert result.imported[0].dataset.role is SpectrumRole.SAMPLE
    window.close()


def test_folder_import_scans_supported_files_without_descending(
    application: QApplication,
    tmp_path: Path,
) -> None:
    top_level = tmp_path / "top-level.csv"
    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    nested = nested_directory / "nested.txt"
    ignored = tmp_path / "notes.md"
    shutil.copy(FIXTURES / "single_valid.csv", top_level)
    shutil.copy(FIXTURES / "wide_multiple_pairs.txt", nested)
    ignored.write_text("not reduced data", encoding="utf-8")

    paths = _supported_reduced_data_files(tmp_path)

    assert paths == (top_level,)
    window = MainWindow()
    project = window.workspace.new_project()
    result = window.import_paths(paths, project=project)
    assert len(result.imported) == 1
    assert result.failures == ()
    assert window.dataset_view.dataset is None
    window.close()


def test_marking_dataset_resolution_updates_order_and_can_be_reverted(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, first = _import_two_group_dataset(window)
    second = window.import_data(FIXTURES / "single_valid.csv", project=project)
    assert project.datasets == [first, second]
    assert (
        window.workspace.tree.contextMenuPolicy()
        is Qt.ContextMenuPolicy.CustomContextMenu
    )

    resolution = window.workspace.set_dataset_role(
        project,
        second,
        SpectrumRole.RESOLUTION,
    )
    assert resolution is not None

    assert resolution.dataset.role is SpectrumRole.RESOLUTION
    assert all(
        spectrum.role is SpectrumRole.RESOLUTION
        for spectrum in resolution.dataset.spectra
    )
    assert project.datasets == [resolution, first]
    project_item = window.workspace.tree.topLevelItem(0)
    assert project_item is not None
    data_item = project_item.child(0)
    assert data_item is not None
    resolution_item = data_item.child(0)
    assert resolution_item is not None
    assert resolution_item.text(0) == resolution.name
    assert "Resolution" in resolution_item.text(1)
    assert not resolution_item.icon(0).isNull()
    assert resolution_item.font(0).weight() == QFont.Weight.DemiBold

    restored = window.workspace.set_dataset_role(
        project,
        resolution,
        SpectrumRole.SAMPLE,
    )
    assert restored is not None

    assert restored.dataset.role is SpectrumRole.SAMPLE
    assert project.datasets == [first, restored]
    first_item = data_item.child(0)
    assert first_item is not None
    assert first_item.text(0) == first.name
    window.close()


def test_workspace_selection_style_has_no_corner_focus_decoration(
    application: QApplication,
) -> None:
    window = MainWindow()
    assert isinstance(window.workspace.item_delegate, WorkspaceItemDelegate)
    assert "QTreeWidget::item:selected" not in application.styleSheet()
    assert "outline:" not in application.styleSheet()
    assert "QTreeWidget::branch:selected" in application.styleSheet()
    assert "show-decoration-selected: 0;" in application.styleSheet()
    window.close()


def test_resolution_dataset_remains_viewable_and_updates_open_inspector_context(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, dataset = _import_two_group_dataset(window)
    window.open_dataset(project, dataset)
    resolution = window.workspace.set_dataset_role(
        project,
        dataset,
        SpectrumRole.RESOLUTION,
    )
    assert resolution is not None

    assert window.dataset_view.dataset is resolution.dataset
    assert window.dataset_view.current_group_index == 0
    assert "Resolution data" in window.inspector_context_label.text()
    assert dataset.name in window.project_context_label.text()
    assert window.dataset_view.spectrum_axes is not None
    np.testing.assert_array_equal(
        window.dataset_view.spectrum_axes.lines[0].get_ydata(),
        resolution.dataset.spectra[0].intensity,
    )
    window.close()
