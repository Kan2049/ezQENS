"""Focused GUI/workflow coverage for the minimal M6 Q-Rebin task."""

from __future__ import annotations

import os
from collections.abc import Iterator

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid

from ezqens.domain import (
    FractionalCoverageOrigin,
    QBins,
    ReducedDataset,
    Spectrum,
    SpectrumRole,
)
from ezqens.gui import MainWindow, create_application
from ezqens.gui.q_rebin import (
    EXPLICIT_EDGES,
    REGULAR_GRID,
    QRebinDialog,
    adaptive_navigation_indices,
    navigation_markers_are_separable,
)
from ezqens.gui.workspace import DatasetState, ProjectState
from ezqens.workflow import applied_resolution


@pytest.fixture(scope="module")
def application() -> Iterator[QApplication]:
    app = create_application(["ezqens-q-rebin-tests"])
    yield app
    app.closeAllWindows()


def _q_dataset(
    role: SpectrumRole = SpectrumRole.SAMPLE,
    *,
    group_count: int = 4,
    q_edges: tuple[float, ...] | None = None,
    confirmed: bool = True,
    incompatible_group: int | None = None,
) -> ReducedDataset:
    spectra = []
    for index in range(group_count):
        energy = np.array([-1.0, 0.0, 1.0])
        if index == incompatible_group:
            energy = np.array([-1.0, 0.25, 1.0])
        spectra.append(
            Spectrum(
                role=role,
                group_index=index,
                group_label=f"Group {index + 1}",
                energy=energy,
                intensity=np.array([1.0, 3.0 + index, 1.0]),
                uncertainty=np.full(3, 0.2),
                energy_unit="meV",
                intensity_unit="counts",
                uncertainty_unit="counts",
            )
        )
    dataset = ReducedDataset(
        role=role,
        spectra=tuple(spectra),
        q_bins=QBins.from_edges(
            np.arange(group_count + 1, dtype=np.float64) if q_edges is None else q_edges
        ),
    )
    return dataset.confirm_unrebinned_source() if confirmed else dataset


def _open_sample(
    window: MainWindow,
    dataset: ReducedDataset,
) -> tuple[ProjectState, DatasetState]:
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, dataset)
    assert window.open_dataset(project, state)
    return project, state


def test_q_rebin_is_a_conditional_low_frequency_analysis_command(
    application: QApplication,
) -> None:
    window = MainWindow()
    assert not window.q_rebin_action.isEnabled()
    _project, _state = _open_sample(window, _q_dataset())

    assert window.q_rebin_action.isEnabled()
    assert window.q_rebin_action.text() == "Q Rebin…"
    window.q_rebin_action.trigger()
    dialog = window.q_rebin_dialog

    assert dialog is not None
    assert dialog.isModal()
    assert dialog.source_label.text().startswith("Current:")
    dialog.reject()
    window.close()


def test_missing_coverage_requires_explicit_confirmation_before_preview(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, state = _open_sample(window, _q_dataset(confirmed=False))
    dialog = window.show_q_rebin(project, state)
    assert dialog is not None

    dialog.set_target_edges((0.0, 2.0, 4.0))
    assert dialog.preview_result is None
    assert not dialog.apply_button.isEnabled()
    assert "confirmation required" in dialog.legality_label.text().lower()
    assert state.dataset.fractional_coverage is None

    assert dialog.confirm_sample_unrebinned(confirmed=True)
    assert dialog.preview_result is not None
    assert dialog.apply_button.isEnabled()
    assert state.dataset.fractional_coverage is None
    assert dialog.preview_result.sample_source.fractional_coverage is not None
    assert (
        dialog.preview_result.sample_source.fractional_coverage.origin
        is FractionalCoverageOrigin.CONFIRMED_UNREBINNED_SOURCE
    )
    assert dialog.apply_rebin()
    assert project.datasets[0].dataset.fractional_coverage is not None
    window.close()


def test_target_count_is_independent_and_success_keeps_one_workspace_row(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, state = _open_sample(window, _q_dataset())
    dialog = window.show_q_rebin(project, state)
    assert dialog is not None

    dialog.set_target_edges((0.0, 2.0, 4.0))
    assert dialog.target_q_bins().group_count == 2
    assert dialog.preview_result is not None
    assert dialog.preview_result.sample.dataset.q_bins is not None
    assert dialog.preview_result.sample.dataset.q_bins.group_count == 2
    assert dialog.heatmap_source_group_count == 4
    assert len(dialog.preview_canvas.figure.axes[0].lines) == 3
    assert dialog.visible_label_indices == (0, 1)
    assert dialog.apply_rebin()

    assert len(project.datasets) == 1
    assert not project.q_methods
    assert len(project.datasets[0].dataset.spectra) == 2
    assert window.dataset_view.dataset is project.datasets[0].dataset
    assert window.dataset_view.overview_source_dataset is not None
    assert len(window.dataset_view.overview_source_dataset.spectra) == 4
    assert len(window.dataset_view.overview_meshes) == 4
    assert len(window.dataset_view.overview_x_cell_bounds) == 2
    assert window.dataset_view.group_spinbox.maximum() == 2
    overview_source = window.dataset_view.overview_source_dataset
    overview_meshes = window.dataset_view.overview_meshes
    highlight = window.dataset_view.overview_active_highlight
    assert highlight is not None
    window.dataset_view.set_current_group(1)
    assert window.dataset_view.overview_source_dataset is overview_source
    assert window.dataset_view.overview_meshes == overview_meshes
    assert (highlight.get_x(), highlight.get_x() + highlight.get_width()) == (
        window.dataset_view.overview_x_cell_bounds[1]
    )
    window.close()


def test_regular_grid_step_generates_exact_edges_for_existing_workflow(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, state = _open_sample(window, _q_dataset())
    dialog = window.show_q_rebin(project, state)
    assert dialog is not None

    assert dialog.target_mode_combo.currentText() == REGULAR_GRID
    dialog.set_regular_grid(0.0, 4.0, step=2.0)

    assert dialog.regular_step_edit.isEnabled()
    assert dialog.regular_group_count_edit.isEnabled()
    assert dialog.regular_group_count_edit.text() == "2"
    np.testing.assert_array_equal(dialog.target_q_bins().edges, [0.0, 2.0, 4.0])
    assert dialog.preview_result is not None
    assert dialog.preview_result.specification.target_edges == (0.0, 2.0, 4.0)
    assert dialog.preview_result.sample.dataset.q_bins is not None
    np.testing.assert_array_equal(
        dialog.preview_result.sample.dataset.q_bins.edges,
        [0.0, 2.0, 4.0],
    )
    dialog.reject()
    window.close()


def test_regular_grid_group_count_generates_n_plus_one_edges_and_step(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, state = _open_sample(window, _q_dataset())
    dialog = window.show_q_rebin(project, state)
    assert dialog is not None

    dialog.set_regular_grid(0.0, 4.0, group_count=4)

    assert dialog.regular_step_edit.isEnabled()
    assert dialog.regular_group_count_edit.isEnabled()
    assert float(dialog.regular_step_edit.text()) == pytest.approx(1.0)
    edges = dialog.target_q_bins().edges
    assert edges is not None
    assert len(edges) == 5
    np.testing.assert_array_equal(edges, [0.0, 1.0, 2.0, 3.0, 4.0])

    dialog.regular_step_edit.setText("2")
    assert dialog.regular_group_count_edit.text() == "2"
    np.testing.assert_array_equal(dialog.target_q_bins().edges, [0.0, 2.0, 4.0])
    dialog.reject()
    window.close()


@pytest.mark.parametrize("group_count", [3, 7])
def test_initial_fractional_regular_step_retains_exact_grid(
    application: QApplication,
    group_count: int,
) -> None:
    edges = tuple(float(value) for value in np.linspace(0.0, 1.0, group_count + 1))
    window = MainWindow()
    project, state = _open_sample(
        window,
        _q_dataset(group_count=group_count, q_edges=edges),
    )
    dialog = window.show_q_rebin(project, state)
    assert dialog is not None

    assert dialog.target_mode_combo.currentText() == REGULAR_GRID
    assert dialog.regular_group_count_edit.text() == str(group_count)
    assert dialog.preview_result is not None
    assert dialog.apply_button.isEnabled()
    target_edges = dialog.target_q_bins().edges
    assert target_edges is not None
    np.testing.assert_array_equal(target_edges, edges)
    dialog.reject()
    window.close()


@pytest.mark.parametrize("group_count", [3, 7])
def test_group_count_to_step_uses_exact_derived_value(
    application: QApplication,
    group_count: int,
) -> None:
    window = MainWindow()
    project, state = _open_sample(
        window,
        _q_dataset(
            group_count=group_count,
            q_edges=tuple(
                float(value) for value in np.linspace(0.0, 1.0, group_count + 1)
            ),
        ),
    )
    dialog = window.show_q_rebin(project, state)
    assert dialog is not None

    dialog.set_regular_grid(0.0, 1.0, group_count=group_count)
    displayed_step = dialog.regular_step_edit.text()
    exact_step = dialog._regular_exact_values["step"]
    dialog.target_mode_combo.setCurrentText(EXPLICIT_EDGES)
    dialog.target_mode_combo.setCurrentText(REGULAR_GRID)

    assert dialog.regular_step_edit.text() == displayed_step
    assert dialog._regular_exact_values["step"] == exact_step
    assert dialog.regular_group_count_edit.text() == str(group_count)
    assert dialog.preview_result is not None
    target_edges = dialog.target_q_bins().edges
    assert target_edges is not None
    np.testing.assert_array_equal(
        target_edges,
        np.linspace(0.0, 1.0, group_count + 1),
    )
    dialog.reject()
    window.close()


def test_regular_grid_step_uses_only_complete_fixed_width_bins(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, state = _open_sample(window, _q_dataset())
    dialog = window.show_q_rebin(project, state)
    assert dialog is not None

    dialog.set_regular_grid(0.0, 0.3, step=0.1)
    accepted_edges = dialog.target_q_bins().edges
    assert accepted_edges is not None
    assert len(accepted_edges) == 4
    assert accepted_edges[-1] == pytest.approx(0.3)

    dialog.set_regular_grid(0.0, 1.0, step=0.3)

    target = dialog.target_q_bins()
    assert target.edges is not None
    np.testing.assert_allclose(target.edges, [0.0, 0.3, 0.6, 0.9])
    assert dialog.regular_group_count_edit.text() == "3"
    assert "Requested Upper: 1" in dialog.regular_coverage_label.text()
    assert "Actual covered upper edge: 0.9" in dialog.regular_coverage_label.text()
    dialog.reject()
    window.close()


def test_regular_grid_fields_switch_drivers_and_preserve_requested_limits(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, state = _open_sample(window, _q_dataset())
    dialog = window.show_q_rebin(project, state)
    assert dialog is not None

    dialog.set_regular_grid(0.0, 1.0, step=0.3)
    dialog.regular_lower_edit.setText("0.1")
    assert dialog.regular_step_edit.text() == "0.3"
    assert dialog.regular_group_count_edit.text() == "3"
    dialog.regular_upper_edit.setText("0.95")
    assert dialog.regular_step_edit.text() == "0.3"
    assert dialog.regular_group_count_edit.text() == "2"
    step_target = dialog.target_q_bins()
    assert step_target.edges is not None
    np.testing.assert_allclose(step_target.edges, [0.1, 0.4, 0.7])

    dialog.regular_group_count_edit.setText("4")
    target = dialog.target_q_bins()
    assert target.edges is not None
    np.testing.assert_allclose(target.edges, np.linspace(0.1, 0.95, 5))
    assert dialog._regular_exact_values["step"] == (0.95 - 0.1) / 4
    dialog.reject()
    window.close()


def test_q_rebin_regular_explicit_round_trip_keeps_exact_requested_grid_draft(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, state = _open_sample(window, _q_dataset())
    dialog = window.show_q_rebin(project, state)
    assert dialog is not None
    dialog.set_regular_grid(0.0, 1.0, step=0.3)

    dialog.target_mode_combo.setCurrentText(EXPLICIT_EDGES)
    explicit_target = dialog.target_q_bins()
    assert explicit_target.edges is not None
    np.testing.assert_allclose(
        explicit_target.edges,
        [0.0, 0.3, 0.6, 0.9],
    )
    dialog.target_mode_combo.setCurrentText(REGULAR_GRID)

    assert dialog.regular_upper_edit.text() == "1"
    assert dialog._regular_exact_values["step"] == 0.3
    regular_target = dialog.target_q_bins()
    assert regular_target.edges is not None
    np.testing.assert_allclose(
        regular_target.edges,
        [0.0, 0.3, 0.6, 0.9],
    )

    dialog.regular_group_count_edit.setText("4")
    dialog.regular_group_count_edit.setText("3")
    exact_step = 1.0 / 3.0
    assert dialog._regular_exact_values["step"] == exact_step
    dialog.target_mode_combo.setCurrentText(EXPLICIT_EDGES)
    dialog.target_mode_combo.setCurrentText(REGULAR_GRID)
    assert dialog._regular_exact_values["step"] == exact_step

    dialog.regular_step_edit.setText("0.3")
    assert dialog._regular_driver == "step"
    assert "step" not in dialog._regular_exact_values
    edited_target = dialog.target_q_bins()
    assert edited_target.edges is not None
    assert edited_target.edges[-1] == pytest.approx(0.9)
    dialog.reject()
    window.close()


def test_q_rebin_surfaces_fixed_width_constructor_rejection(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, state = _open_sample(window, _q_dataset())
    dialog = window.show_q_rebin(project, state)
    assert dialog is not None

    dialog.set_regular_grid(0.0, 1.0, step=2.0)

    with pytest.raises(ValueError, match="one complete bin"):
        dialog.target_q_bins()
    assert dialog.preview_result is None
    assert not dialog.apply_button.isEnabled()
    assert "one complete bin" in dialog.legality_label.text()
    dialog.reject()
    window.close()


def test_explicit_edge_mode_remains_available_for_nonuniform_grids(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, state = _open_sample(window, _q_dataset())
    dialog = window.show_q_rebin(project, state)
    assert dialog is not None

    dialog.set_target_edges((0.0, 1.0, 3.0, 4.0))

    assert dialog.target_mode_combo.currentText() == EXPLICIT_EDGES
    assert not dialog.regular_fields.isVisible()
    assert dialog.target_edges_edit.isVisible()
    np.testing.assert_array_equal(
        dialog.target_q_bins().edges,
        [0.0, 1.0, 3.0, 4.0],
    )
    assert dialog.preview_result is not None
    assert dialog.preview_result.specification.target_edges == (0.0, 1.0, 3.0, 4.0)
    dialog.target_mode_combo.setCurrentText(REGULAR_GRID)
    assert dialog.regular_step_edit.text() == ""
    dialog.target_mode_combo.setCurrentText(EXPLICIT_EDGES)
    np.testing.assert_array_equal(
        dialog.target_q_bins().edges,
        [0.0, 1.0, 3.0, 4.0],
    )
    dialog.reject()
    window.close()


def test_blocked_refinement_redirects_to_preserved_source(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, state = _open_sample(window, _q_dataset())
    first = window.show_q_rebin(project, state)
    assert first is not None
    first.set_target_edges((0.0, 2.0, 4.0))
    assert first.apply_rebin()

    current = project.datasets[0]
    second = window.show_q_rebin(project, current)
    assert second is not None
    second.set_target_edges((0.0, 1.0, 2.0, 3.0, 4.0))
    assert second.preview_result is None
    assert second.use_preserved_source_button.isVisible()
    assert "refine" in second.legality_label.text().lower()

    second.use_preserved_source()
    assert second.using_preserved_source
    assert second.preview_result is not None
    assert second.apply_button.isEnabled()
    second.reject()
    window.close()


def test_q_coverage_exclusions_are_distinct_from_fitting_masks(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, state = _open_sample(window, _q_dataset())
    assert state.auto_mask is not None
    fitting_selection = state.auto_mask.selection
    assert fitting_selection is not None
    before = tuple(
        fitting_selection.excluded_mask(index).copy()
        for index in range(len(state.dataset.spectra))
    )
    dialog = window.show_q_rebin(project, state)
    assert dialog is not None

    assert dialog.add_exclusion(0.25, 0.75)
    assert len(dialog.exclusions) == 1
    assert dialog.preview_result is not None
    assert window._mask_draft is None
    for index, expected in enumerate(before):
        np.testing.assert_array_equal(
            fitting_selection.excluded_mask(index),
            expected,
        )
    dialog.reject()
    window.close()


def test_associated_resolution_rebins_as_one_successful_transaction(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, sample = _open_sample(window, _q_dataset())
    resolution = window.workspace.add_dataset(
        project,
        _q_dataset(SpectrumRole.RESOLUTION),
    )
    assert window.apply_resolution_for_sample(
        project,
        sample,
        resolution,
        replace_confirmed=True,
    )
    dialog = window.show_q_rebin(project, sample)
    assert dialog is not None
    dialog.set_target_edges((0.0, 2.0, 4.0))

    assert dialog.preview_result is not None
    assert dialog.preview_result.resolution is not None
    assert "transactionally" in dialog.association_label.text()
    assert dialog.apply_rebin()

    assert len(project.datasets) == 2
    assert all(len(state.dataset.spectra) == 2 for state in project.datasets)
    workflow = window._workflow_project_for(project)
    sample_state = next(
        state for state in project.datasets if state.dataset.role is SpectrumRole.SAMPLE
    )
    current_sample = window._workflow_dataset(workflow, sample_state)
    rebinned_resolution = applied_resolution(workflow, current_sample)
    assert rebinned_resolution is not None
    assert len(rebinned_resolution.dataset.spectra) == 2
    window.close()


def test_missing_resolution_coverage_requires_its_own_confirmation(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, sample = _open_sample(window, _q_dataset(confirmed=False))
    resolution = window.workspace.add_dataset(
        project,
        _q_dataset(SpectrumRole.RESOLUTION, confirmed=False),
    )
    assert window.apply_resolution_for_sample(
        project,
        sample,
        resolution,
        replace_confirmed=True,
    )
    dialog = window.show_q_rebin(project, sample)
    assert dialog is not None
    dialog.set_target_edges((0.0, 2.0, 4.0))

    assert dialog.preview_result is None
    assert dialog.confirm_sample_button.isVisible()
    assert dialog.confirm_resolution_button.isVisible()
    assert dialog.confirm_sample_unrebinned(confirmed=True)
    assert dialog.preview_result is None
    assert dialog.confirm_resolution_unrebinned(confirmed=True)
    assert dialog.preview_result is not None
    assert sample.dataset.fractional_coverage is None
    assert resolution.dataset.fractional_coverage is None
    dialog.reject()
    window.close()


def test_shared_resolution_branch_stays_workflow_only(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, first_sample = _open_sample(window, _q_dataset())
    second_sample = window.workspace.add_dataset(project, _q_dataset())
    resolution = window.workspace.add_dataset(
        project,
        _q_dataset(SpectrumRole.RESOLUTION),
    )
    assert window.apply_resolution_for_sample(
        project,
        first_sample,
        resolution,
        replace_confirmed=True,
    )
    assert window.apply_resolution_for_sample(
        project,
        second_sample,
        resolution,
        replace_confirmed=True,
    )
    dialog = window.show_q_rebin(project, first_sample)
    assert dialog is not None
    dialog.set_target_edges((0.0, 2.0, 4.0))
    assert dialog.preview_result is not None
    assert dialog.preview_result.resolution is not None
    assert dialog.preview_result.resolution.dataset_id != resolution.workflow_dataset_id
    assert dialog.apply_rebin()

    assert len(project.datasets) == 3
    visible_resolution = next(
        state
        for state in project.datasets
        if state.dataset.role is SpectrumRole.RESOLUTION
    )
    assert visible_resolution.dataset is resolution.dataset
    workflow = window._workflow_project_for(project)
    current_first = window._workflow_dataset(
        workflow,
        next(
            state
            for state in project.datasets
            if state.workflow_dataset_id == first_sample.workflow_dataset_id
        ),
    )
    current_second = window._workflow_dataset(workflow, second_sample)
    first_resolution = applied_resolution(workflow, current_first)
    second_resolution = applied_resolution(workflow, current_second)
    assert first_resolution is not None
    assert second_resolution is not None
    assert first_resolution.dataset_id != second_resolution.dataset_id
    assert len(first_resolution.dataset.spectra) == 2
    assert len(second_resolution.dataset.spectra) == 4
    window.close()


def test_associated_resolution_failure_is_atomic(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, sample = _open_sample(window, _q_dataset())
    resolution = window.workspace.add_dataset(
        project,
        _q_dataset(
            SpectrumRole.RESOLUTION,
            incompatible_group=1,
        ),
    )
    assert window.apply_resolution_for_sample(
        project,
        sample,
        resolution,
        replace_confirmed=True,
    )
    sample_before = sample.dataset
    resolution_before = resolution.dataset
    dialog = window.show_q_rebin(project, sample)
    assert dialog is not None
    dialog.set_target_edges((0.0, 2.0, 4.0))

    assert dialog.preview_result is None
    assert not dialog.apply_button.isEnabled()
    assert "unchanged" in dialog.association_label.text()
    current_sample_state = next(
        state for state in project.datasets if state.dataset.role is SpectrumRole.SAMPLE
    )
    current_resolution_state = next(
        state
        for state in project.datasets
        if state.dataset.role is SpectrumRole.RESOLUTION
    )
    assert current_sample_state.dataset is sample_before
    assert current_resolution_state.dataset is resolution_before
    dialog.reject()
    window.close()


def test_target_navigation_density_uses_rendered_width_not_group_threshold() -> None:
    wide = adaptive_navigation_indices(37, 1800.0, 34.0)
    narrow = adaptive_navigation_indices(37, 220.0, 34.0)

    assert wide[0] == narrow[0] == 0
    assert wide[-1] == narrow[-1] == 36
    assert len(wide) == 20
    assert 2 <= len(narrow) < len(wide)
    assert navigation_markers_are_separable(18, 1200.0, 100.0)
    assert not navigation_markers_are_separable(18, 120.0, 100.0)


@pytest.mark.parametrize("shared", [False, True])
def test_late_resolution_replay_survives_gui_synchronization(
    application: QApplication,
    shared: bool,
) -> None:
    window = MainWindow()
    project, sample = _open_sample(window, _q_dataset(group_count=8))
    resolution = window.workspace.add_dataset(
        project,
        _q_dataset(SpectrumRole.RESOLUTION, group_count=8),
    )
    original = resolution.dataset
    if shared:
        other = window.workspace.add_dataset(project, _q_dataset(group_count=8))
        assert window.apply_resolution_for_sample(project, other, resolution)
    for edges in ((0.0, 2.0, 4.0, 6.0, 8.0), (0.0, 4.0, 8.0)):
        dialog = window.show_q_rebin()
        assert dialog is not None
        dialog.set_target_edges(edges)
        assert dialog.apply_rebin()
    current = next(
        s
        for s in project.datasets
        if s.workflow_dataset_id == sample.workflow_dataset_id
    )
    assert window.apply_resolution_for_sample(project, current, resolution)
    for _ in range(3):
        workflow = window._workflow_project_for(project)
        sample_ref = window._workflow_dataset(workflow, current)
        replayed = applied_resolution(workflow, sample_ref)
        assert replayed is not None
        assert replayed.dataset.q_bins is not None
        assert current.dataset.q_bins is not None
        np.testing.assert_array_equal(
            replayed.dataset.q_bins.edges,
            current.dataset.q_bins.edges,
        )
        np.testing.assert_array_equal(
            replayed.dataset.q_bins.q_values,
            current.dataset.q_bins.q_values,
        )
        coverage = replayed.dataset.fractional_coverage
        sample_coverage = current.dataset.fractional_coverage
        assert coverage is not None and sample_coverage is not None
        assert coverage.origin is FractionalCoverageOrigin.PROPAGATED_Q_REBIN
        assert len(coverage.q_rebin_history) == 2
        assert coverage.q_rebin_history == sample_coverage.q_rebin_history
        for actual, expected in zip(
            coverage.values, sample_coverage.values, strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
        assert window._preserved_q_sources[(project, replayed.dataset_id)] is original
        visible = next(
            s
            for s in project.datasets
            if s.workflow_dataset_id == resolution.workflow_dataset_id
        )
        assert visible.dataset is (original if shared else replayed.dataset)
    assert len(original.spectra) == 8
    if shared:
        other_resolution = applied_resolution(
            workflow,
            window._workflow_dataset(workflow, other),
        )
        assert other_resolution is not None
        assert other_resolution.dataset is original
    else:
        assert window.open_dataset(project, visible)
        assert replayed is not None
        assert window.dataset_view.dataset is replayed.dataset
    window.close()


def test_failed_late_resolution_replay_preserves_previous_state(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = MainWindow()
    project, sample = _open_sample(window, _q_dataset())
    good = window.workspace.add_dataset(project, _q_dataset(SpectrumRole.RESOLUTION))
    assert window.apply_resolution_for_sample(project, sample, good)
    dialog = window.show_q_rebin()
    assert dialog is not None
    dialog.set_target_edges((0.0, 2.0, 4.0))
    assert dialog.apply_rebin()
    bad = window.workspace.add_dataset(
        project,
        _q_dataset(SpectrumRole.RESOLUTION, incompatible_group=1),
    )
    current = next(
        s
        for s in project.datasets
        if s.workflow_dataset_id == sample.workflow_dataset_id
    )
    before = window._workflow_project_for(project)
    rows = tuple(project.datasets)
    sources = dict(window._preserved_q_sources)
    errors = []
    monkeypatch.setattr(
        window, "_show_workflow_error", lambda title, error: errors.append(error)
    )
    assert not window.apply_resolution_for_sample(
        project,
        current,
        bad,
        replace_confirmed=True,
    )
    assert errors
    assert window._workflow_project_for(project) is before
    assert tuple(project.datasets) == rows
    assert window._preserved_q_sources == sources
    assert window.dataset_view.dataset is current.dataset
    window.close()


def test_post_rebin_provenance_survives_navigation_and_reopening(
    application: QApplication,
) -> None:
    window = MainWindow()
    project, _sample = _open_sample(window, _q_dataset())
    window.show()
    dialog = window.show_q_rebin()
    assert dialog is not None
    dialog.set_target_edges((0.0, 2.0, 4.0))
    assert dialog.apply_rebin()
    label = window.dataset_view.overview_provenance_label
    assert label.isVisible()
    assert (
        label.text() == "Preserved source · 4 groups   |   Current Q grouping · 2 bins"
    )
    window.dataset_view.next_group()
    window.dataset_view.set_overview_visible(False)
    assert label.isVisible()
    window._clear_open_context()
    assert window.open_dataset(project, project.datasets[0])
    assert label.isVisible()
    assert "Preserved source · 4 groups" in label.text()
    window.close()


def test_closed_q_rebin_dialogs_are_destroyed(application: QApplication) -> None:
    window = MainWindow()
    _open_sample(window, _q_dataset())
    for _ in range(4):
        dialog = window.show_q_rebin()
        assert dialog is not None
        assert window.show_q_rebin() is dialog
        assert len(window.findChildren(QRebinDialog)) == 1
        dialog.reject()
        assert window.q_rebin_dialog is None
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert not isValid(dialog)
        assert not window.findChildren(QRebinDialog)
    window.close()
