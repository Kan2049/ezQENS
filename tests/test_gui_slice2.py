"""Focused GUI/workflow tests for Slice 2 Q, units, and mask behavior."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QPushButton,
    QToolButton,
)

from ezqens.domain import QBins, ReducedDataset, Spectrum, SpectrumRole
from ezqens.gui import MainWindow, create_application
from ezqens.gui import masking as masking_module
from ezqens.gui.masking import AutoMaskState, MaskTaskDraft, create_auto_mask_state
from ezqens.gui.q_editor import Q_CENTER, Q_EDGES, QAssignmentEditor
from ezqens.gui.units_editor import SourceUnitsDialog
from ezqens.gui.workspace import DatasetAnalysisState, dataset_analysis_state
from ezqens.io import parse_dave_q_bins
from ezqens.preprocessing import BoundarySide, FittingSelection

Q_FIXTURES = Path(__file__).parent / "fixtures" / "q_bins"
REDUCED_FIXTURES = Path(__file__).parent / "fixtures" / "reduced_data"


@pytest.fixture(scope="module")
def application() -> Iterator[QApplication]:
    app = create_application(["ezqens-slice-2-tests"])
    yield app
    app.closeAllWindows()


def _dataset(
    *,
    group_count: int = 2,
    q_bins: QBins | None = None,
    units: tuple[str, str] = ("unknown", "unknown"),
) -> ReducedDataset:
    spectra = tuple(
        Spectrum(
            role=SpectrumRole.SAMPLE,
            group_index=index,
            group_label=f"Group {index + 1}",
            energy=np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
            intensity=np.array([1.0 + index, 2.0, 3.0, 2.0, 1.0]),
            uncertainty=np.full(5, 0.1),
            energy_unit=units[0],
            intensity_unit=units[1],
            uncertainty_unit=units[1],
        )
        for index in range(group_count)
    )
    return ReducedDataset(role=SpectrumRole.SAMPLE, spectra=spectra, q_bins=q_bins)


def _padding_dataset() -> ReducedDataset:
    return ReducedDataset(
        role=SpectrumRole.SAMPLE,
        spectra=(
            Spectrum(
                role=SpectrumRole.SAMPLE,
                group_index=0,
                group_label="Group 1",
                energy=np.arange(7, dtype=np.float64),
                intensity=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0]),
                uncertainty=np.ones(7),
                energy_unit="meV",
                intensity_unit="counts",
                uncertainty_unit="counts",
            ),
        ),
    )


def _signed_dataset() -> ReducedDataset:
    source = _dataset(group_count=1, units=("meV", "counts"))
    spectrum = source.spectra[0]
    return ReducedDataset(
        role=SpectrumRole.SAMPLE,
        spectra=(
            Spectrum(
                role=SpectrumRole.SAMPLE,
                group_index=spectrum.group_index,
                group_label=spectrum.group_label,
                energy=spectrum.energy,
                intensity=np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
                uncertainty=spectrum.uncertainty,
                energy_unit=spectrum.energy_unit,
                intensity_unit=spectrum.intensity_unit,
                uncertainty_unit=spectrum.uncertainty_unit,
            ),
        ),
    )


def test_source_units_assignment_updates_metadata_without_array_conversion(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _dataset())
    before = tuple(spectrum.intensity.copy() for spectrum in state.dataset.spectra)

    assigned = window.workspace.assign_source_units(
        project,
        state,
        energy_unit="meV",
        intensity_unit="counts",
    )
    assert assigned is not None
    assigned = window.workspace.assign_q_bins(
        project,
        assigned,
        QBins.from_q_values([0.42, 0.58]),
    )
    assert assigned is not None

    assert all(spectrum.energy_unit == "meV" for spectrum in assigned.dataset.spectra)
    assert all(
        spectrum.intensity_unit == "counts" for spectrum in assigned.dataset.spectra
    )
    assert all(
        spectrum.uncertainty_unit == "counts" for spectrum in assigned.dataset.spectra
    )
    for original, spectrum in zip(before, assigned.dataset.spectra, strict=True):
        np.testing.assert_array_equal(original, spectrum.intensity)
    assert dataset_analysis_state(assigned.dataset) is DatasetAnalysisState.READY
    window.close()


def test_direct_and_batch_gui_imports_share_arb_unit_defaults(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    direct = window.import_data(REDUCED_FIXTURES / "single_valid.csv", project=project)
    batch = window.import_paths(
        (REDUCED_FIXTURES / "single_valid.csv",),
        project=project,
    )
    assert len(batch.imported) == 1

    for state in (direct, batch.imported[0]):
        spectrum = state.dataset.spectra[0]
        assert spectrum.intensity_unit == "arb. unit"
        assert spectrum.uncertainty_unit == "arb. unit"
    window.close()


@pytest.mark.parametrize("energy_unit", ["µeV", "eV"])
def test_gui_preserves_authoritative_dave_units_over_import_defaults(
    application: QApplication,
    tmp_path: Path,
    energy_unit: str,
) -> None:
    source = tmp_path / f"authoritative-{energy_unit}.dat"
    source.write_text(
        f"#X Units: Energy / {energy_unit}\n"
        "#Y Units: counts\n"
        "#Begin\n"
        "#Group Number: 1\n"
        "#X Value Intensity dIntensity\n"
        "-1.0 3.0 0.2\n"
        "0.0 4.0 0.2\n",
        encoding="utf-8",
    )
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.import_data(source, project=project)
    spectrum = state.dataset.spectra[0]

    assert spectrum.energy_unit == energy_unit
    assert spectrum.intensity_unit == "counts"
    assert spectrum.uncertainty_unit == "counts"
    np.testing.assert_array_equal(spectrum.energy, [-1.0, 0.0])
    np.testing.assert_array_equal(spectrum.intensity, [3.0, 4.0])
    window.close()


def test_source_units_dialog_uses_supported_energy_choices_without_conversion(
    application: QApplication,
) -> None:
    source = _dataset(group_count=1, units=("unknown", "arb. unit"))
    dialog = SourceUnitsDialog(source)
    assert [
        dialog.energy_unit_combo.itemText(index)
        for index in range(dialog.energy_unit_combo.count())
    ] == ["Not set", "µeV", "meV", "eV"]
    emitted: list[tuple[str, str]] = []
    dialog.units_applied.connect(
        lambda energy, intensity: emitted.append((energy, intensity))
    )
    dialog.energy_unit_combo.setCurrentText("µeV")

    assert dialog.apply()
    assert emitted == [("µeV", "arb. unit")]
    np.testing.assert_array_equal(source.spectra[0].energy, np.array([-2, -1, 0, 1, 2]))
    dialog.close()


def test_imported_dave_metadata_populates_the_existing_gui_workflow(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.import_data(
        REDUCED_FIXTURES / "dave_rich_metadata.dat",
        project=project,
    )
    dataset = state.dataset
    original_energy = tuple(spectrum.energy.copy() for spectrum in dataset.spectra)
    original_intensity = tuple(
        spectrum.intensity.copy() for spectrum in dataset.spectra
    )

    assert dataset.q_bins is not None
    assert dataset.q_bins.edges is None
    assert dataset.q_bins.q_values.tolist() == pytest.approx([0.575, 0.825])
    assert project.q_methods == []
    assert all(spectrum.energy_unit == "meV" for spectrum in dataset.spectra)
    assert all(
        spectrum.intensity_unit == "arbitrary units"
        and spectrum.uncertainty_unit == "arbitrary units"
        for spectrum in dataset.spectra
    )

    window.open_dataset(project, state)
    view = window.dataset_view
    assert view.current_q_label.text().startswith("Q = 0.575")
    assert view.overview_q_axis is not None
    assert view.spectrum_axes is not None
    assert view.spectrum_axes.get_title(loc="left").startswith("Q = 0.575")
    assert "Instrument: SYNTHETIC-SPECTROMETER" in (
        window.inspector_source_metadata_label.text()
    )
    assert "Temperature: 301.193 K" in window.inspector_source_metadata_label.text()
    assert (
        "first repeated-key value" not in window.inspector_source_metadata_label.text()
    )

    window.show_q_editor()
    draft = view.q_editor.build_q_bins()
    assert draft.edges is None
    assert draft.q_values.tolist() == pytest.approx([0.575, 0.825])
    view.q_editor.hide()
    assert state.dataset.q_bins is not None
    assert state.dataset.q_bins.edges is None
    for before, spectrum in zip(original_energy, state.dataset.spectra, strict=True):
        np.testing.assert_array_equal(spectrum.energy, before)
    for before, spectrum in zip(
        original_intensity,
        state.dataset.spectra,
        strict=True,
    ):
        np.testing.assert_array_equal(spectrum.intensity, before)
    window.close()


def test_formal_dave_and_legacy_samples_share_post_import_q_and_mask_lifecycle(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    legacy = window.import_data(
        REDUCED_FIXTURES / "wide_multiple_pairs.txt",
        project=project,
    )
    result = window.import_paths(
        (REDUCED_FIXTURES / "dave_rich_metadata.dat",),
        project=project,
    )
    assert result.failures == ()
    assert len(result.imported) == 1
    formal = result.imported[0]

    for state in (legacy, formal):
        assert state.dataset.role is SpectrumRole.SAMPLE
        assert state.auto_mask is not None
        assert state.auto_mask.selection is not None
        assert state.mask_editable
    assert formal.dataset.q_bins is not None
    assert formal.dataset.q_bins.edges is None

    project_item = window.workspace.tree.topLevelItem(0)
    assert project_item is not None
    data_item = project_item.child(0)
    assert data_item is not None
    items = tuple(
        item
        for index in range(data_item.childCount())
        if (item := data_item.child(index)) is not None
    )
    formal_item = next(item for item in items if item.text(0) == formal.name)
    legacy_item = next(item for item in items if item.text(0) == legacy.name)
    formal_menu = window.workspace._create_context_menu(formal_item)
    legacy_menu = window.workspace._create_context_menu(legacy_item)
    assert formal_menu is not None
    assert legacy_menu is not None
    assert "Edit Q…" in [action.text() for action in formal_menu.actions()]
    assert "Edit Mask…" in [action.text() for action in formal_menu.actions()]
    assert "Assign Q…" in [action.text() for action in legacy_menu.actions()]

    next(
        action for action in formal_menu.actions() if action.text() == "Edit Q…"
    ).trigger()
    assert window._open_dataset is formal
    draft = window.dataset_view.q_editor.build_q_bins()
    assert draft.edges is None
    assert draft.q_values.tolist() == pytest.approx([0.575, 0.825])
    window.dataset_view.q_editor.hide()

    next(
        action for action in formal_menu.actions() if action.text() == "Edit Mask…"
    ).trigger()
    assert window._mask_draft is not None
    assert formal.auto_mask is not None
    assert window._mask_draft.selection == formal.auto_mask.selection
    assert window.close_mask_task("discard")
    window.close()


def test_double_clicking_visible_q_text_opens_the_shared_q_editor_without_apply(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q_bins = QBins.from_q_values([0.42, 0.58])
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _dataset(q_bins=q_bins))
    window.open_dataset(project, state)
    view = window.dataset_view
    monkeypatch.setattr(
        "ezqens.gui.dataset_view._event_hits_text",
        lambda _event, _artist: True,
    )
    overview_event = cast(
        MouseEvent,
        SimpleNamespace(
            dblclick=True,
            button=MouseButton.LEFT,
            inaxes=view.overview_axes,
        ),
    )
    assert view._overview_q_label_artists
    view._on_button_press(overview_event)
    assert not view.q_editor.isHidden()
    view.q_editor.hide()

    view.set_overview_visible(False)
    navigator_event = cast(
        MouseEvent,
        SimpleNamespace(
            dblclick=True,
            button=MouseButton.LEFT,
            inaxes=view.navigator_axes,
        ),
    )
    assert view._navigator_q_label_artists
    view._on_button_press(navigator_event)
    assert not view.q_editor.isHidden()
    view.q_editor.hide()

    spectrum_event = cast(
        MouseEvent,
        SimpleNamespace(dblclick=True, button=MouseButton.LEFT),
    )
    assert view.spectrum_q_title is not None
    view._on_spectrum_q_press(spectrum_event)
    assert not view.q_editor.isHidden()
    draft = view.q_editor.build_q_bins()
    assert draft.edges is None
    assert draft.q_values.tolist() == pytest.approx([0.42, 0.58])
    assert state.dataset.q_bins is q_bins
    window.close()


def test_energy_axis_double_clicks_request_the_shared_units_editor(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _dataset(units=("meV", "counts")))
    window.open_dataset(project, state)
    view = window.dataset_view
    assert view.spectrum_axes is not None
    assert view.overview_axes is not None
    requested: list[bool] = []
    view.units_requested.connect(lambda: requested.append(True))
    monkeypatch.setattr(
        view.spectrum_axes.xaxis.label, "contains", lambda _event: (True, {})
    )
    monkeypatch.setattr(
        view.overview_axes.yaxis.label, "contains", lambda _event: (True, {})
    )
    event = cast(
        MouseEvent,
        SimpleNamespace(dblclick=True, button=MouseButton.LEFT),
    )

    view._on_spectrum_units_press(event)
    view._on_overview_units_press(event)

    assert requested == [True, True]
    window.close()


def test_q_editor_preserves_non_midpoint_representatives_when_switching_views(
    application: QApplication,
) -> None:
    q_bins = QBins(
        q_values=np.array([0.41, 0.63]),
        edges=np.array([0.3, 0.5, 0.8]),
    )
    editor = QAssignmentEditor()
    editor.open_for_dataset(_dataset(q_bins=q_bins))

    editor.representation_combo.setCurrentText(Q_EDGES)
    editor.representation_combo.setCurrentText(Q_CENTER)
    rebuilt = editor.build_q_bins()

    assert rebuilt.q_values.tolist() == pytest.approx([0.41, 0.63])
    assert rebuilt.edges is not None
    assert rebuilt.edges.tolist() == pytest.approx([0.3, 0.5, 0.8])
    editor.close()


def test_q_editor_validates_pasted_count_before_emitting_any_assignment(
    application: QApplication,
) -> None:
    editor = QAssignmentEditor()
    editor.open_for_dataset(_dataset())
    emitted: list[QBins] = []
    editor.q_bins_applied.connect(lambda q_bins, _diagnostics: emitted.append(q_bins))
    editor.advanced_values_button.setChecked(True)
    editor.values_edit.setPlainText("0.42")

    assert not editor.apply()
    assert emitted == []
    assert "count" in editor.status_label.text().casefold()
    editor.close()


def test_q_center_slots_are_drafts_until_all_groups_are_entered(
    application: QApplication,
) -> None:
    editor = QAssignmentEditor()
    editor.open_for_dataset(_dataset(group_count=2))
    editor.set_q_center_slot(1, "0.58")

    assert editor.q_center_slots() == (None, 0.58)
    assert not editor.apply()
    editor.set_q_center_slot(0, "0.42")

    assert editor.apply()
    built = editor.build_q_bins()
    assert built.q_values.tolist() == pytest.approx([0.42, 0.58])
    editor.close()


def test_q_editor_default_uniform_centers_resolve_through_qbins(
    application: QApplication,
) -> None:
    editor = QAssignmentEditor()
    editor.open_for_dataset(_dataset())
    editor.start_edit.setText("0.4")
    editor.step_edit.setText("0.2")
    editor.end_edit.setText("0.6")

    q_bins = editor.build_q_bins()

    assert not editor.advanced_values_button.isChecked()
    assert q_bins.q_values.tolist() == pytest.approx([0.4, 0.6])
    assert q_bins.edges is not None
    assert q_bins.edges.tolist() == pytest.approx([0.3, 0.5, 0.7])
    assert "Centers" in editor.preview_label.text()
    editor.close()


def test_q_editor_default_uniform_edges_require_group_count_consistency(
    application: QApplication,
) -> None:
    editor = QAssignmentEditor()
    editor.open_for_dataset(_dataset())
    editor.representation_combo.setCurrentText(Q_EDGES)
    editor.start_edit.setText("0.25")
    editor.step_edit.setText("0.25")
    editor.end_edit.setText("0.75")

    q_bins = editor.build_q_bins()

    assert q_bins.q_values.tolist() == pytest.approx([0.375, 0.625])
    assert q_bins.edges is not None
    assert q_bins.edges.tolist() == pytest.approx([0.25, 0.5, 0.75])
    editor.step_edit.setText("0.2")
    assert not editor.apply()
    assert "inconsistent" in editor.status_label.text().casefold()
    editor.close()


def test_q_editor_switches_uniform_center_and_edge_representations(
    application: QApplication,
) -> None:
    editor = QAssignmentEditor()
    editor.open_for_dataset(_dataset())
    editor.start_edit.setText("0.4")
    editor.step_edit.setText("0.2")
    editor.end_edit.setText("0.6")

    editor.representation_combo.setCurrentText(Q_EDGES)

    assert editor.start_edit.text() == "0.3"
    assert editor.step_edit.text() == "0.2"
    assert editor.end_edit.text() == "0.7"
    q_bins = editor.build_q_bins()
    assert q_bins.q_values.tolist() == pytest.approx([0.4, 0.6])
    editor.close()


def test_dave_q_import_uses_the_validated_core_reader(
    application: QApplication,
) -> None:
    expected = parse_dave_q_bins(Q_FIXTURES / "dave_valid.txt")
    editor = QAssignmentEditor()
    editor.open_for_dataset(_dataset(group_count=expected.q_bins.group_count))

    result = editor.import_dave_file(str(Q_FIXTURES / "dave_valid.txt"))
    built = editor.build_q_bins()

    assert result.q_bins.q_values.tolist() == pytest.approx(
        expected.q_bins.q_values.tolist(),
    )
    assert built.edges is not None
    assert expected.q_bins.edges is not None
    assert built.edges.tolist() == pytest.approx(expected.q_bins.edges.tolist())
    editor.close()


def test_dave_warning_diagnostics_remain_attached_after_q_apply(
    application: QApplication,
) -> None:
    window = MainWindow()
    result = parse_dave_q_bins(Q_FIXTURES / "dave_count_mismatch.txt")
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(
        project,
        _dataset(group_count=result.q_bins.group_count, units=("meV", "counts")),
    )
    window.open_dataset(project, state)
    window.show_q_editor()
    window.dataset_view.q_editor.import_dave_file(
        str(Q_FIXTURES / "dave_count_mismatch.txt"),
    )

    assert len(project.q_methods) == 1
    assert window.dataset_view.q_editor.apply()
    assert window._open_dataset is not None
    assert "dave_q_bins_group_count_mismatch" in {
        diagnostic.code for diagnostic in window._open_dataset.dataset.diagnostics
    }
    window.close()


def test_q_application_updates_viewer_but_does_not_implicitly_save_method(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _dataset(units=("meV", "counts")))
    window.open_dataset(project, state)
    window.show_q_editor()
    window.dataset_view.q_editor.advanced_values_button.setChecked(True)
    window.dataset_view.q_editor.values_edit.setPlainText("0.42\n0.58")

    assert window.dataset_view.q_editor.apply()
    assert window._open_dataset is not None
    assert window._open_dataset.dataset.q_bins is not None
    assert window.dataset_view.current_q_label.text() == "Q = 0.42 Å⁻¹"
    assert project.q_methods == []
    window.close()


def test_q_method_menu_loads_a_compatible_draft_without_assignment(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _dataset(units=("meV", "counts")))
    method = window.workspace.save_q_method(
        project,
        QBins.from_q_values([0.42, 0.58]),
        name="Instrument Q",
    )
    window.open_dataset(project, state)
    window.show_q_editor()

    assert isinstance(window.dataset_view.q_editor.import_button, QPushButton)
    assert window.dataset_view.q_editor.import_button.text() == "Import…"
    assert window.dataset_view.q_editor.import_button.menu() is None
    assert isinstance(window.dataset_view.q_editor.use_method_button, QPushButton)
    assert window.dataset_view.q_editor.use_method_button.text() == "Use Method…"
    assert window.dataset_view.q_editor.use_method_button.menu() is None
    method_actions = window.dataset_view.q_editor._method_menu.actions()
    assert [action.text() for action in method_actions] == [method.name]
    method_actions[0].trigger()

    built = window.dataset_view.q_editor.build_q_bins()
    assert built.q_values.tolist() == pytest.approx([0.42, 0.58])
    assert state.dataset.q_bins is None
    window.close()


def test_removing_q_method_refreshes_an_open_editor_without_changing_its_draft(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _dataset(units=("meV", "counts")))
    retained = window.workspace.save_q_method(
        project,
        QBins.from_q_values([0.42, 0.58]),
        name="Retained",
    )
    removed = window.workspace.save_q_method(
        project,
        QBins.from_q_values([0.44, 0.62]),
        name="Removed",
    )
    window.open_dataset(project, state)
    window.show_q_editor()
    editor = window.dataset_view.q_editor
    editor.load_q_bins_draft(retained.q_bins)

    assert window.remove_q_method(project, removed, confirmed=True)
    assert [action.text() for action in editor._method_menu.actions()] == ["Retained"]
    assert editor.build_q_bins().q_values.tolist() == pytest.approx([0.42, 0.58])
    window.close()


def test_standalone_dave_q_bins_route_to_reusable_method_without_assignment(
    application: QApplication,
    tmp_path: Path,
) -> None:
    dave_path = tmp_path / "dave-q-bins.txt"
    dave_path.write_text("1.00000\n4.60000\n14\n0.250000\n", encoding="utf-8")
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _dataset(units=("meV", "counts")))
    window.open_dataset(project, state)

    result = window.import_paths((dave_path,), project=project)

    assert result.imported == ()
    assert len(result.q_methods) == 1
    assert result.q_methods[0].q_bins.group_count == 14
    assert project.datasets == [state]
    assert state.dataset.q_bins is None
    assert window._open_dataset is state
    window.close()


def test_ambiguous_import_content_is_not_silently_routed(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dave_result = parse_dave_q_bins(Q_FIXTURES / "dave_valid.txt")
    monkeypatch.setattr(
        "ezqens.gui.main_window.parse_dave_q_bins",
        lambda _path: dave_result,
    )
    window = MainWindow()
    project = window.workspace.new_project()

    result = window.import_paths(
        (REDUCED_FIXTURES / "single_valid.csv",),
        project=project,
    )

    assert result.imported == ()
    assert result.q_methods == ()
    assert len(result.failures) == 1
    assert "both reduced data and DAVE" in result.failures[0].detail
    window.close()


def test_imported_auto_mask_is_core_derived_and_normal_view_has_no_bridged_points(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    source = _padding_dataset()
    original = source.spectra[0].intensity.copy()
    state = window.workspace.add_dataset(project, source)
    assert state.auto_mask is not None
    assert state.auto_mask.selection is not None
    assert np.array_equal(state.auto_mask.padding.spectra[0].auto_mask[:5], [True] * 5)

    window.open_dataset(project, state)
    axes = window.dataset_view.spectrum_axes
    assert axes is not None
    assert all(not np.any(line.get_ydata() == 0.0) for line in axes.lines)
    np.testing.assert_array_equal(source.spectra[0].intensity, original)
    overview_axes = window.dataset_view.overview_axes
    assert overview_axes is not None
    assert len(overview_axes.collections) >= 2  # data + mask overlay
    window.close()


def test_singleton_edge_auto_mask_is_rendered_and_remains_restorable(
    application: QApplication,
) -> None:
    singleton_edge = ReducedDataset(
        role=SpectrumRole.SAMPLE,
        spectra=(
            Spectrum(
                role=SpectrumRole.SAMPLE,
                group_index=0,
                group_label="Group 1",
                energy=np.array([-1.0, 0.0, 1.0, 2.0]),
                intensity=np.array([-3.0, 2.0, 3.0, 4.0]),
                uncertainty=np.ones(4),
                energy_unit="meV",
                intensity_unit="counts",
                uncertainty_unit="counts",
            ),
        ),
    )
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, singleton_edge)
    assert state.auto_mask is not None
    np.testing.assert_array_equal(
        state.auto_mask.padding.spectra[0].auto_mask,
        [True, False, False, False],
    )

    window.open_dataset(project, state)
    axes = window.dataset_view.spectrum_axes
    assert axes is not None
    assert all(not np.any(np.isclose(line.get_ydata(), -3.0)) for line in axes.lines)
    window.enter_mask_task()
    assert window._mask_draft is not None
    assert window._mask_draft.restore_points(0, [True, False, False, False])
    assert window._mask_draft.selection.retained_mask(0)[0]
    window.close()


def test_mask_draft_uses_reversible_core_selection_and_keeps_history_local() -> None:
    state = create_auto_mask_state(_padding_dataset())
    draft = MaskTaskDraft(state)
    before = draft.selection.excluded_mask(0).copy()

    assert draft.set_auto_boundary(0, side=BoundarySide.LEFT, energy=-1.0)
    assert not np.any(draft.selection.excluded_mask(0)[:5])
    assert draft.can_undo
    draft.undo()
    np.testing.assert_array_equal(draft.selection.excluded_mask(0), before)
    draft.redo()
    assert not np.any(draft.selection.excluded_mask(0)[:5])


def test_reset_group_restores_only_its_existing_auto_mask_baseline() -> None:
    state = create_auto_mask_state(_dataset(group_count=2, units=("meV", "counts")))
    draft = MaskTaskDraft(state)
    assert draft.exclude_points(0, [False, False, True, False, False])
    assert draft.exclude_points(1, [False, True, False, False, False])
    proposal = state.padding

    assert draft.reset_group(0)
    assert not np.any(draft.selection.manual_exclusion_mask(0))
    assert draft.selection.manual_exclusion_mask(1)[1]
    assert draft._padding is proposal
    assert draft.can_undo
    draft.undo()
    assert draft.selection.manual_exclusion_mask(0)[2]


def test_disable_auto_mask_reincludes_valid_auto_points_as_one_reversible_edit() -> (
    None
):
    source = _padding_dataset()
    spectrum = source.spectra[0]
    second = Spectrum(
        role=SpectrumRole.SAMPLE,
        group_index=1,
        group_label="Group 2",
        energy=spectrum.energy,
        intensity=spectrum.intensity + 1.0,
        uncertainty=spectrum.uncertainty,
        energy_unit="meV",
        intensity_unit="counts",
        uncertainty_unit="counts",
    )
    proposed = create_auto_mask_state(
        ReducedDataset(role=SpectrumRole.SAMPLE, spectra=(spectrum, second)),
    )
    assert proposed.selection is not None
    dataset = ReducedDataset(
        role=SpectrumRole.SAMPLE,
        spectra=(
            Spectrum(
                role=SpectrumRole.SAMPLE,
                group_index=0,
                group_label="Group 1",
                energy=spectrum.energy,
                intensity=spectrum.intensity,
                uncertainty=np.array([np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                energy_unit="meV",
                intensity_unit="counts",
                uncertainty_unit="counts",
            ),
            second,
        ),
    )
    state = AutoMaskState(
        padding=proposed.padding,
        selection=FittingSelection(
            dataset=dataset,
            padding=proposed.padding,
            ranges=proposed.selection.ranges,
        ),
    )
    assert state.selection is not None
    draft = MaskTaskDraft(state)
    assert draft.exclude_points(1, [False, False, False, False, False, True, False])
    auto_before = tuple(item.auto_mask.copy() for item in state.padding.spectra)
    manual_before = tuple(
        draft.selection.manual_exclusion_mask(group_index).copy()
        for group_index in range(2)
    )

    assert draft.disable_auto_mask()
    for group_index, auto_mask in enumerate(auto_before):
        invalid = draft.selection.invalid_mask(group_index)
        eligible = auto_mask & ~invalid & ~manual_before[group_index]
        assert np.all(
            draft.selection.manual_auto_reinclusion_mask(group_index)[eligible]
        )
    assert draft.selection.manual_exclusion_mask(1)[5]
    assert draft.selection.excluded_mask(0)[0]
    for group_index, auto_mask in enumerate(auto_before):
        np.testing.assert_array_equal(
            state.padding.spectra[group_index].auto_mask, auto_mask
        )
        assert not np.any(state.selection.manual_auto_reinclusion_mask(group_index))

    draft.undo()
    for group_index in range(2):
        assert not np.any(draft.selection.manual_auto_reinclusion_mask(group_index))
    draft.redo()
    assert draft.reset_all()
    assert not draft.has_manual_edits


def test_disable_auto_mask_is_safe_when_the_proposal_has_no_auto_points() -> None:
    draft = MaskTaskDraft(
        create_auto_mask_state(_dataset(group_count=1, units=("meV", "counts"))),
    )

    assert not draft.can_disable_auto_mask
    assert not draft.disable_auto_mask()


def test_disable_auto_mask_stays_local_until_saved(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _padding_dataset())
    assert state.auto_mask is not None
    assert state.auto_mask.selection is not None
    original_auto_mask = state.auto_mask.padding.spectra[0].auto_mask.copy()
    window.open_dataset(project, state)
    window.enter_mask_task()

    window._disable_auto_mask()
    assert window._mask_draft is not None
    assert np.any(window._mask_draft.selection.manual_auto_reinclusion_mask(0))
    assert not np.any(state.auto_mask.selection.manual_auto_reinclusion_mask(0))

    assert window.save_mask_task()
    assert window._open_dataset is not None
    assert window._open_dataset.auto_mask is not None
    saved = window._open_dataset.auto_mask
    assert saved.selection is not None
    assert np.all(saved.selection.manual_auto_reinclusion_mask(0)[original_auto_mask])
    np.testing.assert_array_equal(
        saved.padding.spectra[0].auto_mask, original_auto_mask
    )
    window.close()


@pytest.mark.parametrize(
    ("choice", "commits_draft"),
    [("save", True), ("discard", False)],
)
def test_mask_close_button_resolves_dirty_disable_auto_mask_draft(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    choice: str,
    commits_draft: bool,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _padding_dataset())
    window.open_dataset(project, state)
    window.enter_mask_task()
    window._disable_auto_mask()
    assert window._mask_draft is not None
    assert window._mask_draft.is_dirty

    monkeypatch.setattr(
        "ezqens.gui.main_window.choose_dialog", lambda *_args, **_kw: choice
    )
    window.mask_close_button.click()

    assert window._mask_draft is None
    assert window.mask_task_bar.isHidden()
    assert window._open_dataset is not None
    assert window._open_dataset.auto_mask is not None
    assert window._open_dataset.auto_mask.selection is not None
    assert (
        bool(
            window._open_dataset.auto_mask.selection.manual_auto_reinclusion_mask(
                0
            ).any()
        )
        is commits_draft
    )
    window.close()


def test_mask_close_button_cancel_preserves_dirty_disable_auto_mask_draft(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _padding_dataset())
    window.open_dataset(project, state)
    window.enter_mask_task()
    window._disable_auto_mask()
    assert window._mask_draft is not None
    draft = window._mask_draft
    reinclusion_before = draft.selection.manual_auto_reinclusion_mask(0).copy()
    exclusion_before = draft.selection.manual_exclusion_mask(0).copy()
    assert draft.is_dirty

    monkeypatch.setattr(
        "ezqens.gui.main_window.choose_dialog",
        lambda *_args, **_kw: "cancel",
    )
    window.mask_close_button.click()

    assert window._mask_draft is draft
    assert window._mask_draft_owner is state
    assert window._open_dataset is state
    assert draft.is_dirty
    assert window.dataset_view.mask_inspection_mode
    assert not window.mask_task_bar.isHidden()
    np.testing.assert_array_equal(
        draft.selection.manual_auto_reinclusion_mask(0),
        reinclusion_before,
    )
    np.testing.assert_array_equal(
        draft.selection.manual_exclusion_mask(0),
        exclusion_before,
    )
    assert state.auto_mask is not None
    assert state.auto_mask.selection is not None
    assert not state.auto_mask.selection.manual_auto_reinclusion_mask(0).any()
    window.close()


def test_reset_all_requires_confirmation_and_preserves_existing_proposal(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(
        project,
        _dataset(group_count=2, units=("meV", "counts")),
    )
    window.open_dataset(project, state)
    window.enter_mask_task()
    assert window._mask_draft is not None
    assert window._mask_draft.exclude_points(0, [False, False, True, False, False])
    proposal = window._mask_draft._padding
    dialog_titles: list[str] = []

    def reject_dialog(
        _parent: object, title: str, _message: str, **_kwargs: object
    ) -> bool:
        dialog_titles.append(title)
        return False

    monkeypatch.setattr(
        "ezqens.gui.main_window.confirm_dialog",
        reject_dialog,
    )

    assert not window._reset_mask_all()
    assert dialog_titles == ["Reset All Groups"]
    assert window._mask_draft.selection.manual_exclusion_mask(0)[2]
    assert window._reset_mask_all(confirmed=True)
    assert not window._mask_draft.has_manual_edits
    assert window._mask_draft._padding is proposal
    window.close()


def test_reset_all_button_absorbs_checked_and_uses_real_confirmation(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(
        project,
        _dataset(group_count=2, units=("meV", "counts")),
    )
    window.open_dataset(project, state)
    window.enter_mask_task()
    assert window._mask_draft is not None
    assert window._mask_draft.exclude_points(0, [False, False, True, False, False])
    seen_dialogs: list[QDialog] = []

    def confirm_reset() -> None:
        dialog = application.activeModalWidget()
        assert isinstance(dialog, QDialog)
        assert dialog.windowTitle() == "Reset All Groups"
        seen_dialogs.append(dialog)
        button = next(
            item
            for item in dialog.findChildren(QPushButton)
            if item.text() == "Reset All"
        )
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)

    window.mask_reset_all_button.setCheckable(True)
    window.show()
    application.processEvents()
    QTimer.singleShot(0, confirm_reset)
    QTest.mouseClick(window.mask_reset_all_button, Qt.MouseButton.LeftButton)
    application.processEvents()

    assert window.mask_reset_all_button.isChecked()
    assert seen_dialogs
    assert window._mask_draft is not None
    assert not window._mask_draft.has_manual_edits
    window.close()


def test_rerun_auto_mask_invokes_the_core_proposal_path_and_replaces_baseline(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _padding_dataset())
    window.open_dataset(project, state)
    window.enter_mask_task()
    assert window._mask_draft is not None
    assert window._mask_draft.exclude_points(
        0,
        [False, False, False, False, False, True, False],
    )
    proposals: list[AutoMaskState] = []
    original = masking_module.create_auto_mask_state

    def rerun(dataset: ReducedDataset) -> AutoMaskState:
        proposal = original(dataset)
        proposals.append(proposal)
        return proposal

    monkeypatch.setattr(masking_module, "create_auto_mask_state", rerun)

    assert window.rerun_auto_mask(confirmed=True)
    assert len(proposals) == 1
    assert window._mask_draft._padding is proposals[0].padding
    assert not window._mask_draft.has_manual_edits
    window.close()


def test_rerun_button_absorbs_checked_and_uses_real_confirmation(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _padding_dataset())
    window.open_dataset(project, state)
    window.enter_mask_task()
    assert window._mask_draft is not None
    assert window._mask_draft.exclude_points(
        0,
        [False, False, False, False, False, True, False],
    )
    original = masking_module.create_auto_mask_state
    proposals: list[AutoMaskState] = []

    def rerun(dataset: ReducedDataset) -> AutoMaskState:
        proposal = original(dataset)
        proposals.append(proposal)
        return proposal

    monkeypatch.setattr(masking_module, "create_auto_mask_state", rerun)
    seen_dialogs: list[QDialog] = []

    def confirm_rerun() -> None:
        dialog = application.activeModalWidget()
        assert isinstance(dialog, QDialog)
        assert dialog.windowTitle() == "Re-run AutoMask"
        seen_dialogs.append(dialog)
        button = next(
            item
            for item in dialog.findChildren(QPushButton)
            if item.text() == "Re-run AutoMask"
        )
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)

    window.mask_rerun_button.setCheckable(True)
    window.show()
    application.processEvents()
    QTimer.singleShot(0, confirm_rerun)
    QTest.mouseClick(window.mask_rerun_button, Qt.MouseButton.LeftButton)
    application.processEvents()

    assert window.mask_rerun_button.isChecked()
    assert seen_dialogs
    assert len(proposals) == 1
    assert window._mask_draft is not None
    assert window._mask_draft._padding is proposals[0].padding
    assert not window._mask_draft.has_manual_edits
    window.close()


def test_rerun_auto_mask_commits_its_new_baseline_before_mask_task_closes(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _padding_dataset())
    window.open_dataset(project, state)
    window.enter_mask_task()
    assert window._mask_draft is not None
    previous_padding = window._mask_draft._padding
    original = masking_module.create_auto_mask_state
    proposals: list[AutoMaskState] = []

    def rerun(dataset: ReducedDataset) -> AutoMaskState:
        proposal = original(dataset)
        proposals.append(proposal)
        return proposal

    monkeypatch.setattr(masking_module, "create_auto_mask_state", rerun)

    assert window.rerun_auto_mask(confirmed=True)
    assert window._open_dataset is not None
    assert window._open_dataset.auto_mask is not None
    assert window._open_dataset.auto_mask.padding is proposals[0].padding
    assert window._open_dataset.auto_mask.padding is not previous_padding
    assert window.close_mask_task("discard")
    window.enter_mask_task()
    assert window._mask_draft is not None
    assert window._mask_draft._padding is proposals[0].padding
    window.close()


def test_workspace_removal_clears_open_context_without_touching_source_file(
    application: QApplication,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "reduced.csv"
    source_path.write_text("external source", encoding="utf-8")
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(
        project,
        _dataset(units=("meV", "counts")),
        source_path=source_path,
    )
    window.open_dataset(project, state)

    assert window.remove_dataset(project, state, confirmed=True)
    assert source_path.exists()
    assert project.datasets == []
    assert window._open_project is None
    assert window._open_dataset is None
    assert window.dataset_view.dataset is None
    window.close()


def test_workspace_method_and_project_removal_are_in_memory_only(
    application: QApplication,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "method.txt"
    source_path.write_text("external method", encoding="utf-8")
    window = MainWindow()
    project = window.workspace.new_project()
    method = window.workspace.save_q_method(
        project,
        QBins.from_q_values([0.42]),
        name="Imported method",
    )

    assert window.remove_q_method(project, method, confirmed=True)
    assert source_path.exists()
    assert project.q_methods == []
    assert window.remove_project(project, confirmed=True)
    assert source_path.exists()
    assert window.workspace.projects == ()
    window.close()


def test_removing_selected_dataset_rebuilds_tree_without_changing_open_dataset(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    open_state = window.workspace.add_dataset(
        project, _dataset(units=("meV", "counts"))
    )
    selected_state = window.workspace.add_dataset(
        project,
        _dataset(units=("meV", "counts")),
    )
    window.open_dataset(project, open_state)
    project_item = window.workspace.tree.topLevelItem(0)
    assert project_item is not None
    data_item = project_item.child(0)
    assert data_item is not None
    selected_item = data_item.child(1)
    assert selected_item is not None
    window.workspace.tree.setCurrentItem(selected_item)

    assert window.remove_dataset(project, selected_state, confirmed=True)
    assert project.datasets == [open_state]
    assert window._open_dataset is open_state
    assert window.workspace.tree.currentItem() is data_item
    window.close()


def test_removing_last_selected_dataset_leaves_an_empty_data_node(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _dataset(units=("meV", "counts")))
    project_item = window.workspace.tree.topLevelItem(0)
    assert project_item is not None
    data_item = project_item.child(0)
    assert data_item is not None
    selected_item = data_item.child(0)
    assert selected_item is not None
    window.workspace.tree.setCurrentItem(selected_item)

    assert window.remove_dataset(project, state, confirmed=True)
    assert data_item.childCount() == 0
    assert window.workspace.tree.currentItem() is data_item
    window.close()


def test_workspace_removal_menu_and_keyboard_share_removable_object_scope(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    dataset = window.workspace.add_dataset(project, _dataset(units=("meV", "counts")))
    method = window.workspace.save_q_method(project, QBins.from_q_values([0.42]))
    project_item = window.workspace.tree.topLevelItem(0)
    assert project_item is not None
    data_item = project_item.child(0)
    methods_item = project_item.child(1)
    assert data_item is not None
    assert methods_item is not None
    dataset_item = data_item.child(0)
    method_item = methods_item.child(0)
    assert dataset_item is not None
    assert method_item is not None
    assert window.workspace._create_context_menu(data_item) is None
    assert window.workspace._create_context_menu(methods_item) is None
    dataset_menu = window.workspace._create_context_menu(dataset_item)
    method_menu = window.workspace._create_context_menu(method_item)
    assert dataset_menu is not None
    assert method_menu is not None
    assert "Remove from Project" in [action.text() for action in dataset_menu.actions()]
    assert [action.text() for action in method_menu.actions()] == [
        "Remove from Project"
    ]
    requested: list[tuple[object, object]] = []
    window.workspace.dataset_removal_requested.connect(
        lambda requested_project, requested_dataset: requested.append(
            (requested_project, requested_dataset),
        ),
    )
    monkeypatch.setattr(
        "ezqens.gui.main_window.confirm_dialog",
        lambda *_args, **_kwargs: False,
    )
    window.workspace.tree.setCurrentItem(dataset_item)
    event = QKeyEvent(
        QKeyEvent.Type.KeyPress,
        Qt.Key.Key_Delete,
        Qt.KeyboardModifier.NoModifier,
    )
    window.workspace.tree.keyPressEvent(event)

    assert requested == [(project, dataset)]
    assert method in project.q_methods
    window.close()


def test_rectangle_and_lasso_edits_change_only_the_task_local_mask() -> None:
    source = _dataset(group_count=1, units=("meV", "counts"))
    original = source.spectra[0].intensity.copy()
    draft = MaskTaskDraft(create_auto_mask_state(source))

    assert draft.exclude_rectangle(
        0,
        lower_energy=-0.1,
        upper_energy=0.1,
        lower_intensity=2.9,
        upper_intensity=3.1,
    )
    assert draft.selection.manual_exclusion_mask(0)[2]
    assert draft.exclude_lasso(
        0,
        [(-2.5, 0.5), (-1.5, 0.5), (-1.5, 1.5), (-2.5, 1.5)],
    )
    assert draft.selection.manual_exclusion_mask(0)[0]
    np.testing.assert_array_equal(source.spectra[0].intensity, original)


def test_mask_task_previews_then_discard_restores_normal_hidden_mask_view(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _padding_dataset())
    window.open_dataset(project, state)
    window.enter_mask_task()

    assert window.dataset_view.mask_inspection_mode
    assert window._mask_draft is not None
    assert window._mask_draft.exclude_points(
        0, [False, False, False, False, False, True, False]
    )
    window._refresh_mask_preview()
    assert window.close_mask_task("discard")
    assert not window.dataset_view.mask_inspection_mode
    assert window._open_dataset is state
    window.close()


def test_mask_task_save_commits_the_preview_to_project_memory(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _padding_dataset())
    window.open_dataset(project, state)
    window.enter_mask_task()
    assert window._mask_draft is not None
    assert window._mask_draft.exclude_points(
        0,
        [False, False, False, False, False, True, False],
    )
    window._refresh_mask_preview()

    assert window.save_mask_task()
    assert window._open_dataset is not None
    assert window._open_dataset.auto_mask is not None
    assert window._open_dataset.auto_mask.selection is not None
    assert window._open_dataset.auto_mask.selection.manual_exclusion_mask(0)[5]
    assert window.close_mask_task("discard")
    window.close()


@pytest.mark.parametrize(
    ("decision", "commits_draft"),
    [("save", True), ("discard", False)],
)
def test_mask_task_resolves_before_workspace_edit_mask_opens_another_dataset(
    application: QApplication,
    decision: str,
    commits_draft: bool,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    first = window.workspace.add_dataset(project, _padding_dataset())
    second = window.workspace.add_dataset(project, _dataset(units=("meV", "counts")))
    window.open_dataset(project, first)
    window.enter_mask_task()
    assert window._mask_draft is not None
    first_draft = window._mask_draft
    assert first_draft.exclude_points(
        0,
        [False, False, False, False, False, True, False],
    )

    window.show_mask_editor(project, second, mask_task_decision=decision)

    assert window._open_dataset is second
    assert window._mask_draft is not None
    assert window._mask_draft is not first_draft
    assert window._mask_draft_owner is second
    assert not window._mask_draft.can_undo
    current_first = next(
        dataset for dataset in project.datasets if dataset is not second
    )
    assert current_first.auto_mask is not None
    assert current_first.auto_mask.selection is not None
    assert (
        bool(current_first.auto_mask.selection.manual_exclusion_mask(0)[5])
        is commits_draft
    )
    assert second.auto_mask is not None
    assert second.auto_mask.selection is not None
    assert not second.auto_mask.selection.manual_exclusion_mask(0).any()
    window.close()


def test_mask_task_cancel_keeps_its_owner_open_during_workspace_edit_mask(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    first = window.workspace.add_dataset(project, _padding_dataset())
    second = window.workspace.add_dataset(project, _dataset(units=("meV", "counts")))
    window.open_dataset(project, first)
    window.enter_mask_task()
    assert window._mask_draft is not None
    first_draft = window._mask_draft
    assert first_draft.exclude_points(
        0,
        [False, False, False, False, False, True, False],
    )

    window.show_mask_editor(project, second, mask_task_decision="cancel")

    assert window._open_dataset is first
    assert window._mask_draft is first_draft
    assert window._mask_draft_owner is first
    assert first_draft.is_dirty
    assert window.dataset_view.mask_inspection_mode
    assert not window.mask_task_bar.isHidden()
    window.close()


@pytest.mark.parametrize(
    ("decision", "commits_draft"),
    [("save", True), ("discard", False)],
)
def test_mask_task_resolves_before_current_sample_becomes_resolution(
    application: QApplication,
    decision: str,
    commits_draft: bool,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _padding_dataset())
    window.open_dataset(project, state)
    window.enter_mask_task()
    assert window._mask_draft is not None
    assert window._mask_draft.exclude_points(
        0,
        [False, False, False, False, False, True, False],
    )
    window.dataset_view._mask_points = [(0.0, 0.0)]

    current = window.change_dataset_role(
        project,
        state,
        SpectrumRole.RESOLUTION,
        mask_task_decision=decision,
    )

    assert current is not None
    assert current.dataset.role is SpectrumRole.RESOLUTION
    assert current.auto_mask is not None
    assert current.auto_mask.selection is not None
    assert (
        bool(current.auto_mask.selection.manual_exclusion_mask(0)[5]) is commits_draft
    )
    assert window._mask_draft is None
    assert window._mask_draft_owner is None
    assert not window.dataset_view.mask_inspection_mode
    assert window.dataset_view._mask_tool is None
    assert window.dataset_view._mask_points == []
    assert window.mask_task_bar.isHidden()
    assert not window.mask_boundary_button.isChecked()
    assert not window.mask_exclude_button.isChecked()
    window.close()


def test_mask_task_cancel_keeps_current_sample_editable_and_active(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _padding_dataset())
    window.open_dataset(project, state)
    window.enter_mask_task()
    assert window._mask_draft is not None
    draft = window._mask_draft
    assert draft.exclude_points(
        0,
        [False, False, False, False, False, True, False],
    )

    current = window.change_dataset_role(
        project,
        state,
        SpectrumRole.RESOLUTION,
        mask_task_decision="cancel",
    )

    assert current is None
    assert state.dataset.role is SpectrumRole.SAMPLE
    assert window._open_dataset is state
    assert window._mask_draft is draft
    assert window._mask_draft_owner is state
    assert draft.is_dirty
    assert window.dataset_view.mask_inspection_mode
    assert not window.mask_task_bar.isHidden()
    window.close()


def test_clean_mask_task_closes_before_current_sample_becomes_resolution(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _padding_dataset())
    window.open_dataset(project, state)
    window.enter_mask_task()

    current = window.change_dataset_role(project, state, SpectrumRole.RESOLUTION)

    assert current is not None
    assert current.dataset.role is SpectrumRole.RESOLUTION
    assert window._mask_draft is None
    assert not window.dataset_view.mask_inspection_mode
    assert window.mask_task_bar.isHidden()
    window.close()


def test_renaming_an_active_mask_dataset_preserves_owner_and_selection_identity(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _padding_dataset())
    window.open_dataset(project, state)
    window.enter_mask_task()
    assert window._mask_draft is not None
    assert window._mask_draft.exclude_points(
        0,
        [False, False, False, False, False, True, False],
    )

    renamed = window.workspace.rename_dataset(project, state, "Renamed sample")

    assert window._open_dataset is renamed
    assert window._mask_draft_owner is renamed
    assert window._mask_draft.selection.dataset is renamed.dataset
    assert window._mask_draft.restore_points(
        0,
        [True, False, False, False, False, False, False],
    )
    assert window.save_mask_task()
    assert window.close_mask_task("discard")
    assert project.datasets[0].name == "Renamed sample"
    assert project.datasets[0].auto_mask is not None
    assert project.datasets[0].auto_mask.selection is not None
    assert project.datasets[0].auto_mask.selection.manual_exclusion_mask(0)[5]
    assert project.datasets[0].auto_mask.selection.manual_auto_reinclusion_mask(0)[0]
    window.close()


@pytest.mark.parametrize(
    ("decision", "commits_draft"),
    [("save", True), ("discard", False)],
)
def test_q_apply_resolves_an_active_mask_task_before_replacing_dataset(
    application: QApplication,
    decision: str,
    commits_draft: bool,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _padding_dataset())
    window.open_dataset(project, state)
    window.enter_mask_task()
    assert window._mask_draft is not None
    assert window._mask_draft.exclude_points(
        0,
        [False, False, False, False, False, True, False],
    )

    assert window._apply_q_bins(
        QBins.from_q_values([0.5]),
        (),
        mask_task_decision=decision,
    )

    assert window._open_dataset is not None
    assert window._open_dataset.dataset.q_bins is not None
    assert window._mask_draft is None
    assert window._mask_draft_owner is None
    assert window._open_dataset.auto_mask is not None
    assert window._open_dataset.auto_mask.selection is not None
    assert (
        window._open_dataset.auto_mask.selection.dataset is window._open_dataset.dataset
    )
    assert (
        bool(window._open_dataset.auto_mask.selection.manual_exclusion_mask(0)[5])
        is commits_draft
    )
    window.close()


def test_q_apply_cancel_keeps_the_active_mask_task_and_dataset_unchanged(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _padding_dataset())
    window.open_dataset(project, state)
    window.enter_mask_task()
    assert window._mask_draft is not None
    draft = window._mask_draft

    assert not window._apply_q_bins(
        QBins.from_q_values([0.5]),
        (),
        mask_task_decision="cancel",
    )

    assert window._open_dataset is state
    assert state.dataset.q_bins is None
    assert window._mask_draft is draft
    assert window._mask_draft_owner is state
    assert draft.selection.dataset is state.dataset
    window.close()


@pytest.mark.parametrize(
    ("decision", "commits_draft"),
    [("save", True), ("discard", False)],
)
def test_units_apply_resolves_an_active_mask_task_before_replacing_dataset(
    application: QApplication,
    decision: str,
    commits_draft: bool,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _padding_dataset())
    window.open_dataset(project, state)
    window.enter_mask_task()
    assert window._mask_draft is not None
    assert window._mask_draft.exclude_points(
        0,
        [False, False, False, False, False, True, False],
    )

    assert window._apply_source_units(
        project,
        state,
        "µeV",
        "counts",
        mask_task_decision=decision,
    )

    assert window._open_dataset is not None
    assert window._open_dataset.dataset.spectra[0].energy_unit == "µeV"
    assert window._mask_draft is None
    assert window._mask_draft_owner is None
    assert window._open_dataset.auto_mask is not None
    assert window._open_dataset.auto_mask.selection is not None
    assert (
        window._open_dataset.auto_mask.selection.dataset is window._open_dataset.dataset
    )
    assert (
        bool(window._open_dataset.auto_mask.selection.manual_exclusion_mask(0)[5])
        is commits_draft
    )
    window.close()


def test_units_apply_cancel_keeps_the_active_mask_task_and_dataset_unchanged(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _padding_dataset())
    window.open_dataset(project, state)
    window.enter_mask_task()
    assert window._mask_draft is not None
    draft = window._mask_draft

    assert not window._apply_source_units(
        project,
        state,
        "µeV",
        "counts",
        mask_task_decision="cancel",
    )

    assert window._open_dataset is state
    assert state.dataset.spectra[0].energy_unit == "meV"
    assert window._mask_draft is draft
    assert window._mask_draft_owner is state
    assert draft.selection.dataset is state.dataset
    window.close()


def test_mask_save_rejects_a_stale_selection_dataset(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _padding_dataset())
    foreign = window.workspace.add_dataset(project, _padding_dataset())
    window.open_dataset(project, state)
    window.enter_mask_task()
    assert foreign.auto_mask is not None
    window._mask_draft = MaskTaskDraft(foreign.auto_mask)
    window._mask_draft_owner = state

    assert not window.save_mask_task()
    assert project.datasets[0] is state
    assert window._open_dataset is state
    window.close()


def test_non_open_role_change_keeps_the_open_mask_task_on_its_requested_dataset(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    first = window.workspace.add_dataset(project, _padding_dataset())
    second = window.workspace.add_dataset(project, _padding_dataset())
    window.open_dataset(project, first)
    window.enter_mask_task()
    assert window._mask_draft is not None
    draft = window._mask_draft
    assert draft.exclude_points(
        0,
        [False, False, False, False, False, True, False],
    )

    resolution = window.change_dataset_role(project, second, SpectrumRole.RESOLUTION)

    assert resolution is not None
    assert resolution.dataset.role is SpectrumRole.RESOLUTION
    assert first.dataset.role is SpectrumRole.SAMPLE
    assert window._open_dataset is first
    assert window._mask_draft is draft
    assert window._mask_draft_owner is first
    assert draft.selection.dataset is first.dataset
    window.close()


def test_mask_toolbar_groups_direct_tools_above_task_actions(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _dataset(units=("meV", "counts")))
    window.open_dataset(project, state)
    window.enter_mask_task()

    layout = window.mask_task_bar.layout()
    assert layout is not None
    assert layout.indexOf(window.mask_tool_row) == 0
    assert layout.indexOf(window.mask_action_row) == 1
    assert [
        button.text() for button in window.mask_tool_row.findChildren(QToolButton)
    ] == [
        "Boundary",
        "Rectangle",
        "Lasso",
        "Exclude",
        "Restore",
    ]
    assert [
        button.text() for button in window.mask_action_row.findChildren(QToolButton)
    ] == [
        "Undo",
        "Redo",
        "Disable AutoMask",
        "Reset Group",
        "Reset All",
        "Re-run AutoMask…",
        "Save",
        "Close",
    ]
    assert window.mask_boundary_button.isChecked()
    assert window.mask_exclude_button.isChecked()
    assert not hasattr(window, "mask_operation_label")
    tool_layout = window.mask_tool_row.layout()
    action_layout = window.mask_action_row.layout()
    assert tool_layout is not None
    assert action_layout is not None
    tool_spacer = tool_layout.itemAt(1)
    action_spacer = action_layout.itemAt(0)
    assert tool_spacer is not None
    assert action_spacer is not None
    assert tool_spacer.spacerItem() is not None
    assert action_spacer.spacerItem() is not None

    window.show()
    application.processEvents()
    assert window.mask_restore_button.geometry().right() == (
        window.mask_close_button.geometry().right()
    )
    title = window.mask_task_bar.findChild(QLabel, "maskTaskTitle")
    assert title is not None
    assert title.geometry().left() < window.mask_boundary_button.geometry().left()

    window._set_mask_tool("lasso", True)
    window._set_mask_operation("restore", True)
    assert window.mask_lasso_button.isChecked()
    assert window.mask_restore_button.isChecked()
    window.close()


def test_mask_edit_direct_boundary_preview_commits_once_and_restores_outward(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _dataset(units=("meV", "counts")))
    window.open_dataset(project, state)
    window.enter_mask_task()
    view = window.dataset_view
    axes = view.spectrum_axes
    assert axes is not None
    assert set(view._boundary_handle_artists) == {"left", "right"}

    press = cast(
        MouseEvent,
        SimpleNamespace(
            button=MouseButton.LEFT,
            inaxes=axes,
            xdata=-2.0,
            x=None,
        ),
    )
    drag = cast(MouseEvent, SimpleNamespace(inaxes=axes, xdata=-1.0, x=None))
    view._on_spectrum_button_press(press)
    view._on_spectrum_mouse_motion(drag)
    assert view.selection is not None
    assert view.selection.excluded_mask(0)[0]
    view._on_spectrum_button_release(
        cast(
            MouseEvent,
            SimpleNamespace(inaxes=view.spectrum_axes, xdata=-1.0, x=None),
        ),
    )
    assert window._mask_draft is not None
    assert window._mask_draft.can_undo
    assert window._mask_draft.selection.excluded_mask(0)[0]

    axes = view.spectrum_axes
    assert axes is not None
    outward = cast(
        MouseEvent,
        SimpleNamespace(inaxes=axes, xdata=-2.0, x=None),
    )
    view._on_spectrum_button_press(
        cast(
            MouseEvent,
            SimpleNamespace(
                button=MouseButton.LEFT,
                inaxes=axes,
                xdata=-1.0,
                x=None,
            ),
        ),
    )
    view._on_spectrum_mouse_motion(outward)
    view._on_spectrum_button_release(
        cast(
            MouseEvent,
            SimpleNamespace(inaxes=view.spectrum_axes, xdata=-2.0, x=None),
        ),
    )
    assert not window._mask_draft.selection.manual_exclusion_mask(0)[0]
    window.close()


def test_mask_rectangle_lasso_show_provisional_geometry_and_escape_cancels(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _dataset(units=("meV", "counts")))
    window.open_dataset(project, state)
    window.enter_mask_task()
    view = window.dataset_view
    axes = view.spectrum_axes
    assert axes is not None
    window._set_mask_tool("rectangle", True)
    press = cast(
        MouseEvent,
        SimpleNamespace(
            button=MouseButton.LEFT,
            inaxes=axes,
            xdata=-1.0,
            ydata=1.0,
        ),
    )
    drag = cast(
        MouseEvent,
        SimpleNamespace(inaxes=axes, xdata=1.0, ydata=3.0),
    )
    view._on_spectrum_button_press(press)
    view._on_spectrum_mouse_motion(drag)
    assert view._mask_preview_rectangle is not None
    assert view._mask_preview_rectangle.get_width() > 0.0
    view._on_spectrum_key_press(cast(KeyEvent, SimpleNamespace(key="escape")))
    assert view._mask_preview_rectangle is None
    assert window._mask_draft is not None
    assert not window._mask_draft.is_dirty

    window._set_mask_tool("lasso", True)
    view._on_spectrum_button_press(press)
    view._on_spectrum_mouse_motion(drag)
    assert view._mask_preview_line is not None
    window.close()


def test_mask_rectangle_and_lasso_commit_the_last_visible_geometry_outside_axes(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _dataset(units=("meV", "counts")))
    window.open_dataset(project, state)
    window.enter_mask_task()
    view = window.dataset_view
    axes = view.spectrum_axes
    assert axes is not None

    window._set_mask_tool("rectangle", True)
    rectangle_requests: list[tuple[float, float, float, float]] = []
    view.mask_rectangle_requested.connect(
        lambda _group, start_x, end_x, start_y, end_y, _operation: (
            rectangle_requests.append((start_x, end_x, start_y, end_y))
        ),
    )
    view._on_spectrum_button_press(
        cast(
            MouseEvent,
            SimpleNamespace(
                button=MouseButton.LEFT,
                inaxes=axes,
                xdata=-1.0,
                ydata=1.0,
            ),
        ),
    )
    view._on_spectrum_mouse_motion(
        cast(
            MouseEvent,
            SimpleNamespace(inaxes=axes, xdata=1.0, ydata=3.0),
        ),
    )
    rectangle = view._mask_preview_rectangle
    assert rectangle is not None
    visible_bounds = rectangle.get_bbox().bounds
    view._on_spectrum_mouse_motion(
        cast(MouseEvent, SimpleNamespace(inaxes=None, xdata=None, ydata=None)),
    )
    assert rectangle.get_bbox().bounds == visible_bounds
    view._on_spectrum_button_release(
        cast(MouseEvent, SimpleNamespace(inaxes=None, xdata=None, ydata=None)),
    )
    assert rectangle_requests == [(-1.0, 1.0, 1.0, 3.0)]
    assert view._mask_preview_rectangle is None
    assert window._mask_draft is not None
    assert window._mask_draft.can_undo
    assert np.any(window._mask_draft.selection.manual_exclusion_mask(0))

    window._set_mask_tool("lasso", True)
    lasso_requests: list[tuple[tuple[float, float], ...]] = []
    view.mask_lasso_requested.connect(
        lambda _group, vertices, _operation: lasso_requests.append(vertices),
    )
    view._on_spectrum_button_press(
        cast(
            MouseEvent,
            SimpleNamespace(
                button=MouseButton.LEFT,
                inaxes=view.spectrum_axes,
                xdata=-0.5,
                ydata=2.5,
            ),
        ),
    )
    for x_data, y_data in ((0.5, 2.5), (0.0, 3.5)):
        view._on_spectrum_mouse_motion(
            cast(
                MouseEvent,
                SimpleNamespace(inaxes=view.spectrum_axes, xdata=x_data, ydata=y_data),
            ),
        )
    line = view._mask_preview_line
    assert line is not None
    visible_vertices = tuple(view._mask_points)
    view._on_spectrum_button_release(
        cast(MouseEvent, SimpleNamespace(inaxes=None, xdata=None, ydata=None)),
    )
    assert lasso_requests == [visible_vertices]
    assert view._mask_preview_line is None
    assert window._mask_draft.can_undo
    window.close()


def test_mask_gesture_outside_release_cancellation_and_start_policy(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _dataset(units=("meV", "counts")))
    window.open_dataset(project, state)
    window.enter_mask_task()
    view = window.dataset_view
    axes = view.spectrum_axes
    assert axes is not None
    window._set_mask_tool("rectangle", True)

    view._on_spectrum_button_press(
        cast(
            MouseEvent,
            SimpleNamespace(
                button=MouseButton.LEFT,
                inaxes=None,
                xdata=None,
                ydata=None,
            ),
        ),
    )
    assert view._mask_points == []
    assert view._mask_preview_rectangle is None

    view._on_spectrum_button_press(
        cast(
            MouseEvent,
            SimpleNamespace(
                button=MouseButton.LEFT,
                inaxes=axes,
                xdata=-1.0,
                ydata=1.0,
            ),
        ),
    )
    view._on_spectrum_mouse_motion(
        cast(MouseEvent, SimpleNamespace(inaxes=axes, xdata=1.0, ydata=3.0)),
    )
    view._on_spectrum_mouse_motion(
        cast(MouseEvent, SimpleNamespace(inaxes=None, xdata=None, ydata=None)),
    )
    view._on_spectrum_key_press(cast(KeyEvent, SimpleNamespace(key="escape")))
    assert view._mask_points == []
    assert view._mask_preview_rectangle is None
    assert window._mask_draft is not None
    assert not window._mask_draft.is_dirty

    view._on_spectrum_button_press(
        cast(
            MouseEvent,
            SimpleNamespace(
                button=MouseButton.LEFT,
                inaxes=axes,
                xdata=-1.0,
                ydata=1.0,
            ),
        ),
    )
    view._on_spectrum_button_release(
        cast(MouseEvent, SimpleNamespace(inaxes=axes, xdata=1.0, ydata=3.0)),
    )
    assert window._mask_draft.is_dirty
    window.close()


def test_mask_restore_and_option_alt_inversion_keep_persistent_operation(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _dataset(units=("meV", "counts")))
    window.open_dataset(project, state)
    window.enter_mask_task()
    window._exclude_rectangle(0, -0.1, 0.1, 2.9, 3.1, "exclude")
    assert window._mask_draft is not None
    assert window._mask_draft.selection.manual_exclusion_mask(0)[2]
    window._exclude_rectangle(0, -0.1, 0.1, 2.9, 3.1, "restore")
    assert not window._mask_draft.selection.manual_exclusion_mask(0)[2]
    window.dataset_view.set_mask_operation("exclude")
    monkeypatch.setattr(window.dataset_view, "_modifier_active", lambda: True)
    assert window.dataset_view._effective_mask_operation() == "restore"
    assert window.dataset_view.mask_operation == "exclude"
    window.close()


def test_wheel_zoom_respects_plot_and_axis_regions_and_not_lock_y(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _dataset(units=("meV", "counts")))
    window.open_dataset(project, state)
    view = window.dataset_view
    axes = view.spectrum_axes
    assert axes is not None
    view.canvas.draw()  # type: ignore[no-untyped-call]
    bbox = axes.bbox
    initial_x = axes.get_xlim()
    initial_y = axes.get_ylim()
    body = cast(
        MouseEvent,
        SimpleNamespace(
            x=(bbox.x0 + bbox.x1) / 2,
            y=(bbox.y0 + bbox.y1) / 2,
            button="up",
            inaxes=axes,
        ),
    )
    view._on_scroll(body)
    assert axes.get_xlim() != pytest.approx(initial_x)
    assert axes.get_ylim() != pytest.approx(initial_y)

    x_before_axis = axes.get_xlim()
    y_before_axis = axes.get_ylim()
    x_axis = cast(
        MouseEvent,
        SimpleNamespace(
            x=(bbox.x0 + bbox.x1) / 2,
            y=bbox.y0 - 4,
            button="up",
            inaxes=None,
        ),
    )
    view._on_scroll(x_axis)
    assert axes.get_xlim() != pytest.approx(x_before_axis)
    assert axes.get_ylim() == pytest.approx(y_before_axis)

    view.set_y_range_locked(True)
    y_before_axis = axes.get_ylim()
    y_axis = cast(
        MouseEvent,
        SimpleNamespace(
            x=bbox.x0 - 4,
            y=(bbox.y0 + bbox.y1) / 2,
            button="up",
            inaxes=None,
        ),
    )
    view._on_scroll(y_axis)
    assert axes.get_ylim() != pytest.approx(y_before_axis)
    window.close()


def test_symlog_and_log_are_explicit_display_only_for_signed_data(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _signed_dataset())
    before = state.dataset.spectra[0].intensity.copy()
    window.open_dataset(project, state)
    view = window.dataset_view
    selection = view.selection
    assert selection is not None
    manual_mask = np.array([True, True, True, False, False])
    masked_selection = FittingSelection(
        dataset=selection.dataset,
        padding=selection.padding,
        ranges=selection.ranges,
        manual_exclusion_masks=(manual_mask,),
        manual_auto_reinclusion_masks=selection.manual_auto_reinclusion_masks,
    )
    view.set_selection(masked_selection)
    mask_before = masked_selection.manual_exclusion_mask(0).copy()

    assert view.set_y_scale("symlog")
    assert view.spectrum_axes is not None
    assert view.spectrum_axes.get_yscale() == "symlog"
    assert view.set_y_scale("log")
    assert view.y_scale == "log"
    assert view.spectrum_axes.get_yscale() == "log"
    displayed = np.asarray(view.spectrum_axes.lines[0].get_ydata(), dtype=float)
    assert np.all(displayed > 0.0)
    np.testing.assert_array_equal(state.dataset.spectra[0].intensity, before)
    np.testing.assert_array_equal(
        masked_selection.manual_exclusion_mask(0),
        mask_before,
    )
    assert view.selection is masked_selection
    view.set_selection(selection)
    assert view.y_scale == "log"
    assert view.spectrum_axes is not None
    displayed_runs = tuple(
        np.asarray(line.get_ydata(), dtype=float) for line in view.spectrum_axes.lines
    )
    assert displayed_runs
    assert all(np.all(run > 0.0) for run in displayed_runs)
    window.close()


def test_all_nonpositive_data_keeps_log_with_an_empty_display_state(
    application: QApplication,
) -> None:
    source = _signed_dataset()
    spectrum = source.spectra[0]
    dataset = ReducedDataset(
        role=SpectrumRole.SAMPLE,
        spectra=(
            Spectrum(
                role=SpectrumRole.SAMPLE,
                group_index=0,
                group_label="Group 1",
                energy=spectrum.energy,
                intensity=np.array([-2.0, -1.0, 0.0, -3.0, -0.5]),
                uncertainty=spectrum.uncertainty,
                energy_unit=spectrum.energy_unit,
                intensity_unit=spectrum.intensity_unit,
                uncertainty_unit=spectrum.uncertainty_unit,
            ),
        ),
    )
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, dataset)
    window.open_dataset(project, state)
    view = window.dataset_view
    arrays = (
        spectrum.energy.copy(),
        state.dataset.spectra[0].intensity.copy(),
        state.dataset.spectra[0].uncertainty.copy(),
    )
    selection = view.selection
    workflow = window._workflow_project_for(project)
    lock_state = view.y_range_locked

    assert view.set_y_scale("log")

    assert view.y_scale == "log"
    assert view.y_range_locked is lock_state
    assert view.spectrum_axes is not None
    assert view.spectrum_axes.get_yscale() == "log"
    assert view.spectrum_log_empty_message is not None
    assert "No positive values" in view.spectrum_log_empty_message.get_text()
    assert np.asarray(view.spectrum_axes.lines[0].get_ydata()).size == 0
    assert view.selection is selection
    assert window._workflow_project_for(project) is workflow
    np.testing.assert_array_equal(state.dataset.spectra[0].energy, arrays[0])
    np.testing.assert_array_equal(state.dataset.spectra[0].intensity, arrays[1])
    np.testing.assert_array_equal(state.dataset.spectra[0].uncertainty, arrays[2])
    window.close()


def test_metadata_complete_resolution_uses_neutral_not_sample_fit_coverage() -> None:
    complete = _dataset(
        q_bins=QBins.from_q_values([0.42, 0.58]),
        units=("meV", "counts"),
    )
    resolution = ReducedDataset(
        role=SpectrumRole.RESOLUTION,
        spectra=tuple(
            Spectrum(
                role=SpectrumRole.RESOLUTION,
                group_index=spectrum.group_index,
                group_label=spectrum.group_label,
                energy=spectrum.energy,
                intensity=spectrum.intensity,
                uncertainty=spectrum.uncertainty,
                energy_unit=spectrum.energy_unit,
                intensity_unit=spectrum.intensity_unit,
                uncertainty_unit=spectrum.uncertainty_unit,
            )
            for spectrum in complete.spectra
        ),
        q_bins=complete.q_bins,
    )

    assert dataset_analysis_state(resolution) is DatasetAnalysisState.COMPLETE
