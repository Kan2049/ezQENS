"""Focused GUI boundaries for Slice 3A-1 Manual Fit working state."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor, QFocusEvent
from PySide6.QtWidgets import QApplication, QFrame, QGridLayout, QLabel, QToolButton

from ezqens.domain import QBins, ReducedDataset, Spectrum, SpectrumRole
from ezqens.fitting import (
    BackgroundModel,
    ManualParameterIntent,
    ParameterFamily,
    ParameterReference,
)
from ezqens.gui import MainWindow, create_application
from ezqens.gui.manual_fit import CompactNumberEdit
from ezqens.gui.theme import indicator_tokens_for, tokens_for
from ezqens.gui.workspace import DatasetAnalysisState, DatasetState, ProjectState
from ezqens.workflow import (
    ManualComponentKind,
    ManualLorentzianState,
    ManualModelState,
    ManualParameterEdit,
    create_parameter_tie,
)


@pytest.fixture(scope="module")
def application() -> Iterator[QApplication]:
    app = create_application(["ezqens-slice-3-tests"])
    yield app
    app.closeAllWindows()


def _sample(
    group_count: int = 2, *, q_values: tuple[float, ...] = (0.4, 0.6)
) -> ReducedDataset:
    spectra = tuple(
        Spectrum(
            role=SpectrumRole.SAMPLE,
            group_index=index,
            group_label=f"Group {index + 1}",
            energy=np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
            intensity=np.array([1.0, 2.0, 3.0, 2.0, 1.0]),
            uncertainty=np.full(5, 0.1),
            energy_unit="meV",
            intensity_unit="counts",
            uncertainty_unit="counts",
        )
        for index in range(group_count)
    )
    return ReducedDataset(
        role=SpectrumRole.SAMPLE,
        spectra=spectra,
        q_bins=QBins.from_q_values(q_values),
    )


def _resolution(
    sample: ReducedDataset, *, q_values: tuple[float, ...] | None = None
) -> ReducedDataset:
    if sample.q_bins is None:
        raise ValueError("test Sample requires Q bins")
    return ReducedDataset(
        role=SpectrumRole.RESOLUTION,
        spectra=tuple(
            Spectrum(
                role=SpectrumRole.RESOLUTION,
                group_index=item.group_index,
                group_label=item.group_label,
                energy=item.energy,
                intensity=item.intensity,
                uncertainty=item.uncertainty,
                energy_unit=item.energy_unit,
                intensity_unit=item.intensity_unit,
                uncertainty_unit=item.uncertainty_unit,
            )
            for item in sample.spectra
        ),
        q_bins=QBins.from_q_values(q_values or tuple(sample.q_bins.q_values)),
    )


def _window_with_sample(
    application: QApplication,
) -> tuple[MainWindow, ProjectState, DatasetState]:
    window = MainWindow()
    project = window.workspace.new_project()
    state = window.workspace.add_dataset(project, _sample())
    window.open_dataset(project, state)
    application.processEvents()
    return window, project, state


def _window_with_manual_resolution(
    application: QApplication,
) -> tuple[MainWindow, ProjectState, DatasetState]:
    window, project, sample = _window_with_sample(application)
    resolution = window.workspace.add_dataset(project, _resolution(sample.dataset))
    assert window.apply_resolution_for_sample(
        project,
        sample,
        resolution,
        replace_confirmed=True,
    )
    window.show_manual_fit(project, sample)
    return window, project, sample


def _spectrum_event(
    window: MainWindow,
    x_value: float,
    y_value: float,
    *,
    pressed: bool = False,
) -> MouseEvent:
    return cast(
        MouseEvent,
        SimpleNamespace(
            button=MouseButton.LEFT if pressed else None,
            inaxes=window.dataset_view.spectrum_axes,
            xdata=x_value,
            ydata=y_value,
        ),
    )


def _chain_control(window: MainWindow, reference: ParameterReference) -> QToolButton:
    controls = window.manual_fit_editor.parameter_controls[reference]
    assert controls.chain is not None
    return controls.chain


def test_resolution_stays_normal_and_is_pinned_before_samples(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    resolution = window.workspace.add_dataset(project, _resolution(sample.dataset))

    assert project.datasets == [resolution, sample]
    project_item = window.workspace.tree.topLevelItem(0)
    assert project_item is not None
    data = project_item.child(0)
    assert data is not None
    resolution_item = data.child(0)
    assert resolution_item is not None
    assert resolution_item.text(0) == resolution.name
    assert resolution_item.text(1) == "Resolution"
    assert window.open_dataset(project, resolution)
    assert window.dataset_view.dataset is resolution.dataset
    assert not window.manual_fit_action.isEnabled()
    window.close()


def test_manual_fit_opens_empty_without_resolution_and_cancels_geometry(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)

    window.show_manual_fit(project, sample)

    assert not window.manual_fit_editor.isHidden()
    assert window.manual_fit_editor.model_label.text().endswith("N/A")
    assert "Resolution required" in window.manual_fit_editor.status_label.text()
    assert window._manual_draft is not None
    window._begin_manual_component_interaction(ManualComponentKind.ELASTIC)
    assert window._pending_manual_interaction is not None
    assert window._manual_draft.setup(0).model is None
    window.dataset_view._on_spectrum_key_press(
        KeyEvent("key_press_event", window.dataset_view.canvas, key="escape"),
    )
    assert window._pending_manual_interaction is None
    assert window._manual_draft.setup(0).model is None
    assert window._manual_preview is None
    window.close()


def test_background_can_be_committed_without_applied_resolution(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    window.show_manual_fit(project, sample)
    window._begin_manual_component_interaction(ManualComponentKind.BACKGROUND)

    window._complete_manual_component_interaction(
        ManualComponentKind.BACKGROUND,
        {
            "first_energy": -1.0,
            "first_height": 0.1,
            "second_energy": 1.0,
            "second_height": 0.2,
        },
    )

    assert window._manual_draft is not None
    model = window._manual_draft.setup(0).model
    assert model is not None
    assert model.background is BackgroundModel.LINEAR
    assert window._manual_preview is None
    assert "Resolution required" in window.manual_fit_editor.status_label.text()
    window.close()


def test_all_component_creation_flows_delegate_to_workflow(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    resolution = window.workspace.add_dataset(project, _resolution(sample.dataset))
    assert window.apply_resolution_for_sample(
        project,
        sample,
        resolution,
        replace_confirmed=True,
    )
    window.show_manual_fit(project, sample)

    window._begin_manual_component_interaction(ManualComponentKind.ELASTIC)
    window._complete_manual_component_interaction(
        ManualComponentKind.ELASTIC,
        {"component_peak_center": 0.0, "component_peak_height": 1.0},
    )
    window._begin_manual_component_interaction(ManualComponentKind.LORENTZIAN)
    window._complete_manual_component_interaction(
        ManualComponentKind.LORENTZIAN,
        {
            "component_peak_center": 0.0,
            "component_peak_height": 0.25,
            "width_endpoint_energy": 0.5,
        },
    )
    window._begin_manual_component_interaction(ManualComponentKind.BACKGROUND)
    window._complete_manual_component_interaction(
        ManualComponentKind.BACKGROUND,
        {
            "first_energy": -1.0,
            "first_height": 0.1,
            "second_energy": 1.0,
            "second_height": 0.2,
        },
    )

    assert window._manual_draft is not None
    model = window._manual_draft.setup(0).model
    assert model is not None
    assert model.elastic_area is not None
    assert len(model.lorentzians) == 1
    assert model.background is BackgroundModel.LINEAR
    assert window._manual_preview is not None
    window.close()


def test_invalid_display_energy_preview_is_contained_at_workflow_boundary(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    sample_dataset = _sample()
    invalid_energy = np.array(sample_dataset.spectra[0].energy, copy=True)
    invalid_energy[0] = np.nan
    invalid_spectrum = replace(sample_dataset.spectra[0], energy=invalid_energy)
    sample_dataset = replace(
        sample_dataset,
        spectra=(invalid_spectrum, *sample_dataset.spectra[1:]),
    )
    sample = window.workspace.add_dataset(project, sample_dataset)
    resolution = window.workspace.add_dataset(project, _resolution(sample_dataset))
    window.open_dataset(project, sample)
    assert window.apply_resolution_for_sample(
        project,
        sample,
        resolution,
        replace_confirmed=True,
    )
    window.show_manual_fit(project, sample)

    window._begin_manual_component_interaction(ManualComponentKind.BACKGROUND)
    window._complete_manual_component_interaction(
        ManualComponentKind.BACKGROUND,
        {
            "first_energy": -1.0,
            "first_height": 0.1,
            "second_energy": 1.0,
            "second_height": 0.2,
        },
    )

    assert window._manual_draft is not None
    assert window._manual_draft.setup(0).model is not None
    assert window._manual_preview is None
    assert window.dataset_view._manual_preview is None
    assert window.manual_fit_editor.isEnabled()
    window.close()


def test_task_transition_cancels_provisional_component_geometry(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    window.show_manual_fit(project, sample)
    window._begin_manual_component_interaction(ManualComponentKind.LORENTZIAN)

    assert window._pending_manual_interaction is not None
    window.enter_mask_task()

    assert window._pending_manual_interaction is None
    assert window.dataset_view._manual_component_kind is None
    assert window._manual_draft is not None
    assert window._manual_draft.setup(0).model is None
    window.close()


def test_saved_mask_baseline_cancels_pending_scientific_preview(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_manual_resolution(application)
    window.enter_mask_task()
    window._begin_manual_component_interaction(ManualComponentKind.BACKGROUND)
    window._preview_manual_component_interaction(
        ManualComponentKind.BACKGROUND,
        {
            "first_energy": -1.0,
            "first_height": 0.1,
            "second_energy": 1.0,
            "second_height": 0.2,
        },
    )
    assert window.dataset_view._manual_pending_evaluation is not None

    assert window.save_mask_task()

    assert window._pending_manual_interaction is None
    assert window.dataset_view._manual_pending_evaluation is None
    workflow = window._workflow_project_for(project)
    committed = next(
        item
        for item in workflow.fitting_selections
        if item.sample_id == sample.workflow_dataset_id
    )
    current = next(
        item
        for item in project.datasets
        if item.workflow_dataset_id == sample.workflow_dataset_id
    )
    assert current.auto_mask is not None
    assert committed.selection is current.auto_mask.selection
    window.close()


def test_q_replacement_preserves_workflow_point_selection(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    window.show_manual_fit(project, sample)

    updated = window.workspace.assign_q_bins(
        project,
        sample,
        QBins.from_q_values((0.45, 0.65)),
    )

    assert updated is not None
    assert updated.auto_mask is not None
    assert updated.auto_mask.selection is not None
    assert updated.auto_mask.selection.dataset is updated.dataset
    workflow = window._workflow_project_for(project)
    committed = next(
        item
        for item in workflow.fitting_selections
        if item.sample_id == updated.workflow_dataset_id
    )
    assert committed.selection.dataset is updated.dataset

    window.close()


def test_cold_import_registers_workflow_dataset_and_selection_immediately(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    sample = window.workspace.add_dataset(project, _sample())
    application.processEvents()

    workflow = window._workflow_projects[project]
    reference = next(
        item
        for item in workflow.datasets
        if item.dataset_id == sample.workflow_dataset_id
    )
    committed = next(
        item
        for item in workflow.fitting_selections
        if item.sample_id == sample.workflow_dataset_id
    )
    assert reference.dataset is sample.dataset
    assert sample.auto_mask is not None
    assert committed.selection is sample.auto_mask.selection
    assert window._open_dataset is None
    window.close()


def test_cold_import_unit_replacement_invalidates_workflow_selection(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    sample = window.workspace.add_dataset(project, _sample())

    updated = window.workspace.assign_source_units(
        project,
        sample,
        energy_unit="microelectronvolt",
        intensity_unit="counts",
    )

    assert updated is not None
    workflow = window._workflow_projects[project]
    assert all(
        item.sample_id != updated.workflow_dataset_id
        for item in workflow.fitting_selections
    )
    assert window._open_dataset is None
    window.close()


def test_cold_import_scientific_replacement_invalidates_workflow_selection(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    sample = window.workspace.add_dataset(project, _sample())
    changed_intensity = np.array(sample.dataset.spectra[0].intensity, copy=True)
    changed_intensity[2] += 0.25
    changed_dataset = replace(
        sample.dataset,
        spectra=(
            replace(sample.dataset.spectra[0], intensity=changed_intensity),
            *sample.dataset.spectra[1:],
        ),
    )

    updated = window.workspace._replace_dataset(project, sample, changed_dataset)

    workflow = window._workflow_projects[project]
    assert all(
        item.sample_id != updated.workflow_dataset_id
        for item in workflow.fitting_selections
    )
    assert window._open_dataset is None
    window.close()


def test_cold_import_q_only_replacement_preserves_workflow_selection(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    sample = window.workspace.add_dataset(project, _sample())

    updated = window.workspace.assign_q_bins(
        project,
        sample,
        QBins.from_q_values((0.45, 0.65)),
    )

    assert updated is not None
    workflow = window._workflow_projects[project]
    committed = next(
        item
        for item in workflow.fitting_selections
        if item.sample_id == updated.workflow_dataset_id
    )
    assert committed.selection.dataset is updated.dataset
    assert window._open_dataset is None
    window.close()


def test_unit_replacement_invalidates_workflow_point_selection(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_manual_resolution(application)
    window._begin_manual_component_interaction(ManualComponentKind.BACKGROUND)
    window._preview_manual_component_interaction(
        ManualComponentKind.BACKGROUND,
        {
            "first_energy": -1.0,
            "first_height": 0.1,
            "second_energy": 1.0,
            "second_height": 0.2,
        },
    )
    assert window.dataset_view._manual_pending_evaluation is not None

    unit_replacement = window.workspace.assign_source_units(
        project,
        sample,
        energy_unit="microelectronvolt",
        intensity_unit="counts",
    )

    assert unit_replacement is not None
    assert unit_replacement.auto_mask is not None
    assert unit_replacement.auto_mask.selection is not None
    assert unit_replacement.auto_mask.selection.dataset is unit_replacement.dataset
    workflow = window._workflow_project_for(project)
    assert all(
        item.sample_id != unit_replacement.workflow_dataset_id
        for item in workflow.fitting_selections
    )
    assert window._pending_manual_interaction is None
    assert window.dataset_view._manual_pending_evaluation is None
    window.close()


def test_q_context_replacement_cancels_pending_scientific_preview(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_manual_resolution(application)
    window._begin_manual_component_interaction(ManualComponentKind.BACKGROUND)
    window._preview_manual_component_interaction(
        ManualComponentKind.BACKGROUND,
        {
            "first_energy": -1.0,
            "first_height": 0.1,
            "second_energy": 1.0,
            "second_height": 0.2,
        },
    )
    assert window._pending_manual_interaction is not None
    assert window.dataset_view._manual_pending_evaluation is not None

    updated = window.workspace.assign_q_bins(
        project,
        sample,
        QBins.from_q_values((0.45, 0.65)),
    )

    assert updated is not None
    assert window._pending_manual_interaction is None
    assert window.dataset_view._manual_pending_evaluation is None
    window._complete_manual_component_interaction(
        ManualComponentKind.BACKGROUND,
        {
            "first_energy": -1.0,
            "first_height": 0.1,
            "second_energy": 1.0,
            "second_height": 0.2,
        },
    )
    assert window._manual_draft is not None
    assert window._manual_draft.setup(0).model is None
    window.close()


def test_associated_resolution_removal_clears_manual_context_and_preview(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_manual_resolution(application)
    resolution = next(
        item
        for item in project.datasets
        if item.dataset.role is SpectrumRole.RESOLUTION
    )
    window._begin_manual_component_interaction(ManualComponentKind.ELASTIC)
    window._complete_manual_component_interaction(
        ManualComponentKind.ELASTIC,
        {"component_peak_center": 0.0, "component_peak_height": 1.0},
    )
    assert window._manual_preview is not None
    window._begin_manual_component_interaction(ManualComponentKind.LORENTZIAN)
    window._preview_manual_component_interaction(
        ManualComponentKind.LORENTZIAN,
        {
            "component_peak_center": 0.0,
            "component_peak_height": 0.3,
            "width_endpoint_energy": 0.5,
        },
    )
    assert window.dataset_view._manual_pending_evaluation is not None

    assert window.remove_dataset(project, resolution, confirmed=True)

    assert window._open_dataset is sample
    assert window._pending_manual_interaction is None
    assert window.dataset_view._manual_pending_evaluation is None
    assert window._manual_preview is None
    assert window.dataset_view._manual_preview is None
    assert "Resolution required" in window.manual_fit_editor.status_label.text()
    window.close()


def test_associated_resolution_replacement_refreshes_manual_preview(
    application: QApplication,
) -> None:
    window, project, _sample = _window_with_manual_resolution(application)
    resolution = next(
        item
        for item in project.datasets
        if item.dataset.role is SpectrumRole.RESOLUTION
    )
    window._begin_manual_component_interaction(ManualComponentKind.ELASTIC)
    window._complete_manual_component_interaction(
        ManualComponentKind.ELASTIC,
        {"component_peak_center": 0.0, "component_peak_height": 1.0},
    )
    assert window._manual_preview is not None
    previous_curve = np.array(
        window._manual_preview.display_evaluation.total,
        copy=True,
    )
    window._begin_manual_component_interaction(ManualComponentKind.LORENTZIAN)
    window._preview_manual_component_interaction(
        ManualComponentKind.LORENTZIAN,
        {
            "component_peak_center": 0.0,
            "component_peak_height": 0.3,
            "width_endpoint_energy": 0.5,
        },
    )
    changed_spectra = tuple(
        replace(
            spectrum,
            intensity=np.array([0.5, 1.0, 4.0, 1.5, 0.25]),
        )
        for spectrum in resolution.dataset.spectra
    )

    replacement = window.workspace._replace_dataset(
        project,
        resolution,
        replace(resolution.dataset, spectra=changed_spectra),
    )

    assert replacement.dataset.role is SpectrumRole.RESOLUTION
    assert window._pending_manual_interaction is None
    assert window.dataset_view._manual_pending_evaluation is None
    assert window._manual_preview is not None
    assert not np.array_equal(
        window._manual_preview.display_evaluation.total,
        previous_curve,
    )
    window.close()


def test_associated_resolution_rerole_clears_manual_context(
    application: QApplication,
) -> None:
    window, project, _sample = _window_with_manual_resolution(application)
    resolution = next(
        item
        for item in project.datasets
        if item.dataset.role is SpectrumRole.RESOLUTION
    )
    window._begin_manual_component_interaction(ManualComponentKind.BACKGROUND)
    window._preview_manual_component_interaction(
        ManualComponentKind.BACKGROUND,
        {
            "first_energy": -1.0,
            "first_height": 0.1,
            "second_energy": 1.0,
            "second_height": 0.2,
        },
    )

    rerolled = window.workspace.set_dataset_role(
        project,
        resolution,
        SpectrumRole.SAMPLE,
    )

    assert rerolled is not None
    assert window._pending_manual_interaction is None
    assert window.dataset_view._manual_pending_evaluation is None
    assert window._manual_preview is None
    assert "Resolution required" in window.manual_fit_editor.status_label.text()
    window.close()


def test_same_shape_measurement_replacement_cannot_rebind_gui_selection(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_manual_resolution(application)
    window._begin_manual_component_interaction(ManualComponentKind.BACKGROUND)
    window._preview_manual_component_interaction(
        ManualComponentKind.BACKGROUND,
        {
            "first_energy": -1.0,
            "first_height": 0.1,
            "second_energy": 1.0,
            "second_height": 0.2,
        },
    )
    assert window.dataset_view._manual_pending_evaluation is not None
    changed_intensity = np.array(sample.dataset.spectra[0].intensity, copy=True)
    changed_intensity[2] += 0.25
    changed_spectrum = replace(
        sample.dataset.spectra[0],
        intensity=changed_intensity,
    )
    changed_dataset = replace(
        sample.dataset,
        spectra=(changed_spectrum, *sample.dataset.spectra[1:]),
    )

    updated = window.workspace._replace_dataset(project, sample, changed_dataset)

    assert updated.auto_mask is not None
    assert updated.auto_mask.selection is None
    workflow = window._workflow_project_for(project)
    assert all(
        item.sample_id != updated.workflow_dataset_id
        for item in workflow.fitting_selections
    )
    assert window._pending_manual_interaction is None
    assert window.dataset_view._manual_pending_evaluation is None
    window.close()


def test_resolution_application_is_transactional_on_failed_replacement(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, project, sample = _window_with_sample(application)
    resolution = window.workspace.add_dataset(project, _resolution(sample.dataset))
    bad_resolution = window.workspace.add_dataset(
        project,
        _resolution(sample.dataset, q_values=(0.41, 0.6)),
    )

    assert window.apply_resolution_for_sample(
        project,
        sample,
        resolution,
        replace_confirmed=True,
    )
    window.show_manual_fit(project, sample)
    window._begin_manual_component_interaction(ManualComponentKind.BACKGROUND)
    window._complete_manual_component_interaction(
        ManualComponentKind.BACKGROUND,
        {
            "first_energy": -1.0,
            "first_height": 1.0,
            "second_energy": 1.0,
            "second_height": 3.0,
        },
    )
    assert window._manual_preview is not None
    assert window.dataset_view._manual_preview is window._manual_preview
    draft = window._manual_draft
    assert draft is not None
    model = draft.setup(0).model
    assert model is not None
    offset = next(
        item
        for item in model.parameter_references()
        if item.family is ParameterFamily.OFFSET
    )
    window._emphasize_manual_parameter(offset)
    assert window.dataset_view._manual_emphasized_components == frozenset(
        (offset.component,),
    )
    workflow = window._workflow_project_for(project)
    association = workflow.resolution_associations[0]
    assert association.resolution_id == resolution.workflow_dataset_id
    replacement_validation_calls: list[object] = []
    monkeypatch.setattr(
        "ezqens.gui.main_window.apply_resolution",
        lambda *_args, **_kwargs: replacement_validation_calls.append(object()),
    )
    assert not window.apply_resolution_for_sample(
        project,
        sample,
        bad_resolution,
        replace_confirmed=False,
    )
    assert replacement_validation_calls == []
    monkeypatch.undo()
    monkeypatch.setattr(
        "ezqens.gui.main_window.show_message_dialog", lambda *_args: None
    )
    assert not window.apply_resolution_for_sample(
        project,
        sample,
        bad_resolution,
        replace_confirmed=True,
    )
    workflow = window._workflow_project_for(project)
    assert (
        workflow.resolution_associations[0].resolution_id
        == resolution.workflow_dataset_id
    )
    window.close()


def test_background_interaction_and_group_drafts_remain_independent(
    application: QApplication,
) -> None:
    window, _project, _sample = _window_with_manual_resolution(application)
    window._begin_manual_component_interaction(ManualComponentKind.BACKGROUND)
    window._complete_manual_component_interaction(
        ManualComponentKind.BACKGROUND,
        {
            "first_energy": -1.0,
            "first_height": 1.0,
            "second_energy": 1.0,
            "second_height": 3.0,
        },
    )
    assert window._manual_draft is not None
    assert window._manual_draft.setup(0).model is not None
    window.dataset_view.set_current_group(1)
    assert window._manual_draft.setup(1).model is None
    window.dataset_view.set_current_group(0)
    assert window._manual_draft.setup(0).model is not None
    window.close()


def test_current_limits_and_ties_use_typed_shared_manual_state(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    window.show_manual_fit(project, sample)
    assert window._manual_draft is not None
    lorentzians = tuple(
        ManualLorentzianState(
            area=ManualParameterIntent(1.0),
            fwhm=ManualParameterIntent(0.5),
            center=ManualParameterIntent(0.0),
        )
        for _ in range(3)
    )
    window._manual_draft = window._manual_draft.with_model(
        0,
        ManualModelState(lorentzians=lorentzians),
    )
    window._refresh_manual_fit()
    model = window._manual_draft.setup(0).model
    assert model is not None
    areas = tuple(
        item
        for item in model.parameter_references()
        if item.family is ParameterFamily.AREA
    )
    window._handle_manual_chain(areas[0])
    window._handle_manual_chain(areas[1])
    window._handle_manual_chain(areas[2])
    model = window._manual_draft.setup(0).model
    assert model is not None
    assert model.tie_for(areas[0]) is not None
    assert len(model.tie_for(areas[0]).members) == 3  # type: ignore[union-attr]
    window._update_manual_parameter(
        areas[1],
        ManualParameterEdit(0.3, None, 0.2, False),
    )
    model = window._manual_draft.setup(0).model
    assert model is not None
    for reference in areas:
        intent = model.parameter_intent(reference)
        assert intent.current_value == 0.3
        assert intent.user_upper_limit == 0.2
        assert not intent.free
        controls = window.manual_fit_editor.parameter_controls[reference]
        assert controls.fixed.isChecked()
        assert not controls.lower.isEnabled()
        assert not controls.upper.isEnabled()

    window.manual_fit_editor.parameter_controls[areas[2]].fixed.click()

    model = window._manual_draft.setup(0).model
    assert model is not None
    for reference in areas:
        intent = model.parameter_intent(reference)
        assert intent.free
        assert intent.user_lower_limit is None
        assert intent.user_upper_limit == 0.2
        controls = window.manual_fit_editor.parameter_controls[reference]
        assert not controls.fixed.isChecked()
        assert controls.lower.isEnabled()
        assert controls.upper.isEnabled()
    window.close()


def test_lock_toggle_round_trips_current_value_and_limits_without_rounding(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    window.show_manual_fit(project, sample)
    assert window._manual_draft is not None
    current_value = 0.12345678901234566
    lower_limit = 0.00000012345678901234566
    upper_limit = 0.9876543210987654
    lorentzian = ManualLorentzianState(
        area=ManualParameterIntent(current_value, lower_limit, upper_limit),
        fwhm=ManualParameterIntent(0.5),
        center=ManualParameterIntent(0.0),
    )
    window._manual_draft = window._manual_draft.with_model(
        0,
        ManualModelState(lorentzians=(lorentzian,)),
    )
    window._refresh_manual_fit()
    model = window._manual_draft.setup(0).model
    assert model is not None
    area_reference = next(
        item
        for item in model.parameter_references()
        if item.family is ParameterFamily.AREA
    )
    controls = window.manual_fit_editor.parameter_controls[area_reference]

    controls.fixed.click()

    model = window._manual_draft.setup(0).model
    assert model is not None
    updated = model.parameter_intent(area_reference)
    assert updated.current_value == current_value
    assert updated.user_lower_limit == lower_limit
    assert updated.user_upper_limit == upper_limit
    assert not updated.free
    window.close()


def test_tied_emphasis_untie_and_component_removal_use_stable_references(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    window.show_manual_fit(project, sample)
    assert window._manual_draft is not None
    lorentzians = tuple(
        ManualLorentzianState(
            area=ManualParameterIntent(value),
            fwhm=ManualParameterIntent(0.5),
            center=ManualParameterIntent(0.0),
        )
        for value in (0.2, 0.3, 0.4)
    )
    window._manual_draft = window._manual_draft.with_model(
        0,
        ManualModelState(lorentzians=lorentzians),
    )
    window._refresh_manual_fit()
    model = window._manual_draft.setup(0).model
    assert model is not None
    areas = tuple(
        item
        for item in model.parameter_references()
        if item.family is ParameterFamily.AREA
    )
    window._handle_manual_chain(areas[0])
    window._handle_manual_chain(areas[1])
    window._handle_manual_chain(areas[2])
    model = window._manual_draft.setup(0).model
    assert model is not None
    shared_intent = model.parameter_intent(areas[0])

    window._emphasize_manual_parameter(areas[1])

    assert window.dataset_view._manual_emphasized_components == frozenset(
        item.component for item in areas
    )
    window._handle_manual_chain(areas[1])
    model = window._manual_draft.setup(0).model
    assert model is not None
    assert model.tie_for(areas[1]) is None
    assert model.parameter_intent(areas[1]) == shared_intent

    window._remove_manual_component(areas[0].component)
    model = window._manual_draft.setup(0).model
    assert model is not None
    assert all(
        reference.component != areas[0].component
        for reference in model.parameter_references()
    )
    assert len(model.parameter_ties) == 1
    assert model.parameter_ties[0].members == (areas[2],)
    assert model.parameter_intent(areas[2]) == shared_intent
    window.close()


def test_chain_left_click_is_immediate_nonmodal_and_member_local(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    window.show_manual_fit(project, sample)
    assert window._manual_draft is not None
    window._manual_draft = window._manual_draft.with_model(
        0,
        ManualModelState(
            lorentzians=tuple(
                ManualLorentzianState(
                    area=ManualParameterIntent(0.2 + index / 10),
                    center=ManualParameterIntent(index / 100),
                    fwhm=ManualParameterIntent(0.4 + index / 10),
                )
                for index in range(3)
            ),
        ),
    )
    window._refresh_manual_fit()
    model = window._manual_draft.setup(0).model
    assert model is not None
    centers = tuple(
        reference
        for reference in model.parameter_references()
        if reference.family is ParameterFamily.CENTER
    )
    unrelated = next(
        reference
        for reference in model.parameter_references()
        if reference.family is ParameterFamily.FWHM
    )

    _chain_control(window, centers[0]).click()

    model = window._manual_draft.setup(0).model
    assert model is not None
    assert len(model.parameter_ties) == 1
    assert model.parameter_ties[0].members == (centers[0],)
    assert _chain_control(window, centers[0]).isChecked()
    assert _chain_control(window, centers[0]).property("tieColor") == "accent"
    assert not hasattr(window, "_manual_tie_pick_source")
    assert not hasattr(window.manual_fit_editor, "_tie_pick_source")
    assert all(_chain_control(window, reference).isEnabled() for reference in centers)

    unrelated_edit = window.manual_fit_editor.parameter_controls[unrelated].current
    assert unrelated_edit.isEnabled()
    unrelated_edit.setText("0.73")
    unrelated_edit.textEdited.emit("0.73")
    unrelated_edit.editingFinished.emit()
    model = window._manual_draft.setup(0).model
    assert model is not None
    assert model.parameter_intent(unrelated).current_value == 0.73
    assert model.parameter_ties[0].members == (centers[0],)

    _chain_control(window, centers[1]).click()
    window._update_manual_parameter(
        centers[0],
        ManualParameterEdit(0.31, -0.2, 0.25, False),
    )
    _chain_control(window, centers[1]).click()

    model = window._manual_draft.setup(0).model
    assert model is not None
    assert model.parameter_ties[0].members == (centers[0],)
    assert model.tie_for(centers[1]) is None
    assert model.parameter_intent(centers[1]) == ManualParameterIntent(
        0.31,
        -0.2,
        0.25,
        False,
    )

    _chain_control(window, centers[1]).click()
    _chain_control(window, centers[2]).click()
    model = window._manual_draft.setup(0).model
    assert model is not None
    assert model.parameter_ties[0].members == centers

    _chain_control(window, centers[1]).click()
    model = window._manual_draft.setup(0).model
    assert model is not None
    assert model.parameter_ties[0].members == (centers[0], centers[2])
    assert model.tie_for(centers[1]) is None

    _chain_control(window, centers[2]).click()
    model = window._manual_draft.setup(0).model
    assert model is not None
    assert model.parameter_ties[0].members == (centers[0],)
    _chain_control(window, centers[0]).click()
    model = window._manual_draft.setup(0).model
    assert model is not None
    assert not model.parameter_ties
    window.close()


def test_multiple_same_family_chains_remain_independent(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    window.show_manual_fit(project, sample)
    assert window._manual_draft is not None
    window._manual_draft = window._manual_draft.with_model(
        0,
        ManualModelState(
            lorentzians=tuple(
                ManualLorentzianState(
                    area=ManualParameterIntent(0.2 + index / 10),
                    center=ManualParameterIntent(index / 100),
                    fwhm=ManualParameterIntent(0.4),
                )
                for index in range(5)
            ),
        ),
    )
    window._refresh_manual_fit()
    model = window._manual_draft.setup(0).model
    assert model is not None
    centers = tuple(
        reference
        for reference in model.parameter_references()
        if reference.family is ParameterFamily.CENTER
    )

    _chain_control(window, centers[0]).click()
    _chain_control(window, centers[1]).click()
    model = window._manual_draft.setup(0).model
    assert model is not None
    first_group_id = model.parameter_ties[0].group_id

    window.manual_fit_editor.new_tie_group_requested.emit(centers[2])
    model = window._manual_draft.setup(0).model
    assert model is not None
    assert len(model.parameter_ties) == 2
    second_group_id = model.parameter_ties[1].group_id
    assert model.parameter_ties[1].members == (centers[2],)

    _chain_control(window, centers[3]).click()
    window.manual_fit_editor.join_tie_requested.emit(centers[4], second_group_id)
    model = window._manual_draft.setup(0).model
    assert model is not None
    groups = {group.group_id: group.members for group in model.parameter_ties}
    assert groups[first_group_id] == (centers[0], centers[1], centers[3])
    assert groups[second_group_id] == (centers[2], centers[4])

    _chain_control(window, centers[3]).click()
    model = window._manual_draft.setup(0).model
    assert model is not None
    groups = {group.group_id: group.members for group in model.parameter_ties}
    assert groups[first_group_id] == (centers[0], centers[1])
    assert groups[second_group_id] == (centers[2], centers[4])
    assert model.tie_for(centers[3]) is None
    window.close()


def test_parameter_sections_are_compact_typed_column_grids(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    window.show_manual_fit(project, sample)
    assert window._manual_draft is not None
    model = ManualModelState(
        energy_shift=ManualParameterIntent(0.0),
        elastic_area=ManualParameterIntent(0.7),
        lorentzians=(
            ManualLorentzianState(
                area=ManualParameterIntent(0.2),
                center=ManualParameterIntent(-0.1),
                fwhm=ManualParameterIntent(0.5),
            ),
            ManualLorentzianState(
                area=ManualParameterIntent(0.3),
                center=ManualParameterIntent(0.1),
                fwhm=ManualParameterIntent(0.8),
            ),
        ),
        background=BackgroundModel.LINEAR,
        b0=ManualParameterIntent(0.1),
        b1=ManualParameterIntent(0.0, free=False),
    )
    window._manual_draft = window._manual_draft.with_model(0, model)
    window._refresh_manual_fit()
    application.processEvents()

    fixed_sections = window.manual_fit_editor.sections.findChildren(
        QFrame,
        "manualFunctionSection",
    )
    fixed_by_role = {item.property("functionRole"): item for item in fixed_sections}
    elastic_layout = fixed_by_role["δ"].layout()
    background_layout = fixed_by_role["Background"].layout()
    lorentzian_sections = window.manual_fit_editor.sections.findChildren(
        QFrame,
        "manualLorentzianSection",
    )
    lorentzian = lorentzian_sections[-1] if lorentzian_sections else None
    assert isinstance(elastic_layout, QGridLayout)
    assert isinstance(background_layout, QGridLayout)
    assert lorentzian is not None
    lorentzian_layout = lorentzian.layout()
    assert isinstance(lorentzian_layout, QGridLayout)

    background_references = tuple(
        reference
        for reference in model.parameter_references()
        if reference.family in {ParameterFamily.OFFSET, ParameterFamily.SLOPE}
    )
    assert len(background_references) == 2
    assert all(
        window.manual_fit_editor.parameter_controls[reference].chain is None
        for reference in background_references
    )
    assert not any(
        button.property("chainControl")
        for button in fixed_by_role["Background"].findChildren(QToolButton)
    )

    def header_text(layout: QGridLayout, column: int) -> str:
        item = layout.itemAtPosition(1, column)
        assert item is not None
        label = item.widget()
        assert isinstance(label, QLabel)
        return label.text()

    assert [header_text(elastic_layout, column) for column in range(1, 3)] == [
        "Area",
        "Center",
    ]
    assert [header_text(lorentzian_layout, column) for column in range(1, 4)] == [
        "Area",
        "Center",
        "FWHM",
    ]
    assert [header_text(background_layout, column) for column in range(1, 3)] == [
        "Offset",
        "Slope",
    ]
    for layout in (elastic_layout, lorentzian_layout, background_layout):
        assert [layout.columnStretch(column) for column in range(1, 4)] == [1, 1, 1]
        assert layout.columnMinimumWidth(0) == 32
    assert fixed_by_role["δ"].property("parameterColumnCount") == 3
    assert fixed_by_role["Background"].property("parameterColumnCount") == 3
    assert lorentzian.property("parameterColumnCount") == 3
    assert elastic_layout.itemAtPosition(1, 3) is None
    assert elastic_layout.itemAtPosition(2, 3) is None
    assert background_layout.itemAtPosition(1, 3) is None
    assert background_layout.itemAtPosition(2, 3) is None
    assert lorentzian_layout.itemAtPosition(2, 1) is not None
    assert lorentzian_layout.itemAtPosition(3, 3) is not None
    visible_labels = {
        label.text() for label in window.manual_fit_editor.findChildren(QLabel)
    }
    assert "Current Value" not in visible_labels
    assert "Lower bound" not in visible_labels
    assert "Upper bound" not in visible_labels
    window.close()


def test_lorentzian_section_add_enters_direct_interaction_mode(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    window.show_manual_fit(project, sample)

    window.manual_fit_editor.lorentzian_add_button.click()

    assert window._pending_manual_interaction is not None
    assert (
        window._pending_manual_interaction.component_kind
        is ManualComponentKind.LORENTZIAN
    )
    assert window.dataset_view._manual_component_kind is ManualComponentKind.LORENTZIAN
    window.close()


def test_compact_numeric_text_preserves_exact_unrelated_values(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    window.show_manual_fit(project, sample)
    assert window._manual_draft is not None
    exact_area = 0.12345678901234566
    exact_center = 0.08837654123456789
    model = ManualModelState(
        lorentzians=(
            ManualLorentzianState(
                area=ManualParameterIntent(exact_area),
                center=ManualParameterIntent(exact_center),
                fwhm=ManualParameterIntent(0.5000000000000001),
            ),
        ),
    )
    window._manual_draft = window._manual_draft.with_model(0, model)
    window._refresh_manual_fit()
    rendered = window._manual_draft.setup(0).model
    assert rendered is not None
    area = next(
        reference
        for reference in rendered.parameter_references()
        if reference.family is ParameterFamily.AREA
    )
    center = next(
        reference
        for reference in rendered.parameter_references()
        if reference.family is ParameterFamily.CENTER
    )
    controls = window.manual_fit_editor.parameter_controls
    assert controls[area].current.text() == "0.12"
    assert controls[area].current.stored_value == exact_area
    assert controls[area].current.toolTip() == repr(exact_area)

    controls[center].current.setText("0.125")
    controls[center].current.textEdited.emit("0.125")
    controls[center].current.editingFinished.emit()

    updated = window._manual_draft.setup(0).model
    assert updated is not None
    assert updated.parameter_intent(area).current_value == exact_area
    assert updated.parameter_intent(center).current_value == 0.125
    window.close()


def test_chain_colors_follow_tie_group_membership_not_parameter_family(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    window.show_manual_fit(project, sample)
    assert window._manual_draft is not None
    model = ManualModelState(
        lorentzians=tuple(
            ManualLorentzianState(
                area=ManualParameterIntent(0.2 + index / 10),
                center=ManualParameterIntent(0.0),
                fwhm=ManualParameterIntent(0.5),
            )
            for index in range(5)
        ),
    )
    draft = window._manual_draft.with_model(0, model)
    areas = tuple(
        reference
        for reference in model.parameter_references()
        if reference.family is ParameterFamily.AREA
    )
    draft = create_parameter_tie(
        draft,
        0,
        areas[:2],
        source_member=areas[0],
        tie_group_id="areas-first",
    )
    draft = create_parameter_tie(
        draft,
        0,
        areas[2:4],
        source_member=areas[2],
        tie_group_id="areas-second",
    )
    window._manual_draft = draft
    window._refresh_manual_fit()
    assert _chain_control(window, areas[0]).property("tieColor") == "accent"
    assert _chain_control(window, areas[1]).property("tieColor") == "accent"
    assert _chain_control(window, areas[2]).property("tieColor") == "amber"
    assert _chain_control(window, areas[3]).property("tieColor") == "amber"
    assert _chain_control(window, areas[4]).property("tieColor") is None
    window.close()


@pytest.mark.parametrize(
    ("slope", "expected"),
    (
        (ManualParameterIntent(0.0, free=False), "B0"),
        (ManualParameterIntent(0.0, free=True), "B1"),
        (ManualParameterIntent(1.0e-30, free=False), "B1"),
    ),
)
def test_background_summary_uses_exact_fixed_zero_semantics(
    application: QApplication,
    slope: ManualParameterIntent,
    expected: str,
) -> None:
    window, project, sample = _window_with_sample(application)
    window.show_manual_fit(project, sample)
    assert window._manual_draft is not None
    window._manual_draft = window._manual_draft.with_model(
        0,
        ManualModelState(
            background=BackgroundModel.LINEAR,
            b0=ManualParameterIntent(0.1),
            b1=slope,
        ),
    )
    window._refresh_manual_fit()

    assert window.manual_fit_editor.model_label.text().endswith(expected)
    window.close()


def test_manual_fit_precedes_collapsed_generic_inspector_context(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    window.show_manual_fit(project, sample)
    layout = window._inspector_layout

    assert layout.indexOf(window.manual_fit_editor) < layout.indexOf(
        window.inspector_resolution_title,
    )
    assert layout.indexOf(window.inspector_resolution_button) < layout.indexOf(
        window.inspector_dataset_toggle,
    )
    assert layout.indexOf(window.inspector_dataset_toggle) < layout.indexOf(
        window.inspector_context_label,
    )
    assert layout.indexOf(window.inspector_context_label) < layout.indexOf(
        window.inspector_source_toggle,
    )
    assert not window.inspector_dataset_toggle.isHidden()
    assert not window.inspector_source_toggle.isHidden()
    assert not window.inspector_dataset_toggle.isChecked()
    assert not window.inspector_source_toggle.isChecked()
    assert window.inspector_context_label.isHidden()

    window.inspector_dataset_toggle.click()

    assert not window.inspector_context_label.isHidden()
    window.close()


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (4.4526, "4.45"),
        (1.47962, "1.48"),
        (12.345, "12.35"),
        (0.0, "0"),
        (0.5, "0.5"),
        (338285.0, "3.38e5"),
        (1234567.0, "1.23e6"),
        (0.000012345, "1.23e-5"),
    ),
)
def test_compact_number_display_uses_short_decimal_or_scientific_notation(
    value: float,
    expected: str,
) -> None:
    edit = CompactNumberEdit(value)

    assert edit.text() == expected
    assert edit.stored_value == value
    assert edit.precise_value(required=True) == value


def test_compact_number_focus_round_trips_exact_untouched_float(
    application: QApplication,
) -> None:
    exact = 0.12345678901234566
    edit = CompactNumberEdit(exact)

    QApplication.sendEvent(
        edit,
        QFocusEvent(QEvent.Type.FocusIn, Qt.FocusReason.OtherFocusReason),
    )
    assert edit.text() == repr(exact)
    assert edit.precise_value(required=True) == exact

    QApplication.sendEvent(
        edit,
        QFocusEvent(QEvent.Type.FocusOut, Qt.FocusReason.OtherFocusReason),
    )
    assert edit.text() == "0.12"
    assert edit.precise_value(required=True) == exact
    application.processEvents()


def test_manual_disclosures_share_compact_icon_treatment(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    window.show_manual_fit(project, sample)
    assert window._manual_draft is not None
    window._manual_draft = window._manual_draft.with_model(
        0,
        ManualModelState(
            lorentzians=(
                ManualLorentzianState(
                    area=ManualParameterIntent(0.2),
                    center=ManualParameterIntent(0.0),
                    fwhm=ManualParameterIntent(0.5),
                ),
            ),
            background=BackgroundModel.LINEAR,
            b0=ManualParameterIntent(0.1),
            b1=ManualParameterIntent(0.0, free=False),
        ),
    )
    window._refresh_manual_fit()
    application.processEvents()
    disclosures = [
        window.manual_fit_editor.add_button,
        window.manual_fit_editor.diagnostics_button,
        window.inspector_dataset_toggle,
        window.inspector_source_toggle,
        *(
            button
            for button in window.manual_fit_editor.findChildren(
                type(window.manual_fit_editor.add_button),
            )
            if button.property("compactDisclosure")
        ),
    ]

    assert disclosures
    for button in disclosures:
        assert button.property("compactDisclosure")
        assert 8 <= button.iconSize().width() <= 10
        assert 8 <= button.iconSize().height() <= 10
        assert button.minimumHeight() >= 22
        assert "▸" not in button.text()
        assert "▾" not in button.text()
    window.close()


def test_central_manual_fit_entry_targets_open_dataset_and_shows_active_state(
    application: QApplication,
) -> None:
    window, project, opened = _window_with_sample(application)
    selected_only = window.workspace.add_dataset(project, _sample())
    project_item = window.workspace.tree.topLevelItem(0)
    assert project_item is not None
    data_item = project_item.child(0)
    assert data_item is not None
    selected_item = None
    for index in range(data_item.childCount()):
        candidate = data_item.child(index)
        if candidate is not None and candidate.text(0) == selected_only.name:
            selected_item = candidate
            break
    assert selected_item is not None
    window.workspace.tree.setCurrentItem(selected_item)

    assert window.manual_fit_button.text() == "Manual Fit"
    assert not window.manual_fit_button.isHidden()
    assert window.manual_fit_button.isEnabled()
    assert not window.manual_fit_button.isChecked()
    window.manual_fit_button.click()

    assert window._manual_owner == (project, opened.workflow_dataset_id)
    assert window._open_dataset is opened
    assert window.manual_fit_button.isChecked()
    window.close_manual_fit()
    assert not window.manual_fit_button.isChecked()
    window.close()


def test_resolution_manual_fit_eligibility_follows_public_workflow_boundary(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    resolution = window.workspace.add_dataset(project, _resolution(sample.dataset))
    assert window.open_dataset(project, resolution)

    available, detail = window._manual_fit_capability(project, resolution)

    assert not available
    assert "require a Sample dataset" in detail
    assert not window.manual_fit_button.isHidden()
    assert not window.manual_fit_button.isEnabled()
    assert not window.manual_fit_action.isEnabled()
    window.close()


def test_inspector_splitter_resizes_and_parameter_columns_use_added_width(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    window.show()
    window.show_manual_fit(project, sample)
    assert window._manual_draft is not None
    window._manual_draft = window._manual_draft.with_model(
        0,
        ManualModelState(
            lorentzians=(
                ManualLorentzianState(
                    area=ManualParameterIntent(0.2),
                    center=ManualParameterIntent(0.0),
                    fwhm=ManualParameterIntent(0.5),
                ),
            ),
        ),
    )
    window._refresh_manual_fit()
    window.splitter.setSizes((220, 660, 280))
    application.processEvents()
    compact_width = window.inspector.width()
    compact_fields = tuple(
        controls.current.width()
        for controls in window.manual_fit_editor.parameter_controls.values()
    )

    window.splitter.setSizes((220, 500, 440))
    application.processEvents()
    wide_width = window.inspector.width()
    wide_fields = tuple(
        controls.current.width()
        for controls in window.manual_fit_editor.parameter_controls.values()
    )

    assert window.splitter.handleWidth() >= 8
    assert window.inspector.maximumWidth() > 480
    assert wide_width > compact_width
    assert all(
        wide > compact
        for compact, wide in zip(compact_fields, wide_fields, strict=True)
    )
    window.close_manual_fit()
    window.show_manual_fit(project, sample)
    application.processEvents()
    assert window.inspector.width() == wide_width

    window.splitter.setSizes((220, 700, 220))
    application.processEvents()
    assert window.inspector.width() < wide_width
    assert window.inspector.width() >= window.inspector.minimumWidth()
    assert window.central_workspace.width() >= window.central_workspace.minimumWidth()
    window.close()


def test_small_semantic_indicators_use_saturated_palette(
    application: QApplication,
) -> None:
    window = MainWindow()
    scheme = window._appearance_controller.current_scheme
    indicators = indicator_tokens_for(scheme)
    surface_tokens = tokens_for(scheme)

    assert window.workspace._analysis_state_colors[DatasetAnalysisState.REQUIRED] == (
        indicators.error
    )
    assert window.workspace._analysis_state_colors[DatasetAnalysisState.READY] == (
        indicators.warning
    )
    assert window.manual_fit_editor._icon_colors["accent"] == indicators.accent
    assert window.manual_fit_editor._icon_colors["amber"] == indicators.warning
    assert window.manual_fit_editor._icon_colors["violet"] == indicators.violet
    for color in (
        indicators.error,
        indicators.warning,
        indicators.accent,
        indicators.violet,
    ):
        assert QColor(color).saturation() >= 140
    assert indicators.warning != surface_tokens.warning
    window.close()


def test_manual_fit_panel_starts_without_redundant_inspector_heading(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    window.show_manual_fit(project, sample)

    assert window.inspector.findChild(QLabel, "inspectorTitle") is None
    first = window._inspector_layout.itemAt(0)
    assert first is not None
    assert first.widget() is window.manual_fit_editor
    window.close()


def test_add_component_focus_mode_and_escape_are_complete_and_nonmutating(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    window.show_manual_fit(project, sample)
    add_elastic = window.manual_fit_editor.findChild(QToolButton, "addElasticButton")
    assert add_elastic is not None

    add_elastic.click()

    view = window.dataset_view
    assert view.property("manualInteractionActive")
    assert not view.manual_interaction_instruction.isHidden()
    assert "Add δ" in view.manual_interaction_instruction.text()
    assert view.canvas.cursor().shape() is Qt.CursorShape.CrossCursor
    controls_effect = view.controls_container.graphicsEffect()
    assert controls_effect is not None
    assert controls_effect.isEnabled()
    assert view.spectrum_interaction_frame.property("manualInteractionActive")

    view._on_spectrum_key_press(cast(KeyEvent, SimpleNamespace(key="escape")))

    assert not view.property("manualInteractionActive")
    assert view.manual_interaction_instruction.isHidden()
    assert view._manual_component_kind is None
    assert not view._manual_geometry_artists
    assert window._pending_manual_interaction is None
    assert window._manual_draft is not None
    assert window._manual_draft.setup(0).model is None
    window.close()


def test_elastic_press_move_release_uses_geometry_until_workflow_completion(
    application: QApplication,
) -> None:
    window, _project, _sample_state = _window_with_manual_resolution(application)
    view = window.dataset_view
    add_elastic = window.manual_fit_editor.findChild(QToolButton, "addElasticButton")
    assert add_elastic is not None
    add_elastic.click()

    view._on_spectrum_button_press(_spectrum_event(window, 0.0, 2.0, pressed=True))
    assert view._manual_geometry_artists
    assert window._manual_preview is None
    assert view._manual_pending_evaluation is not None
    assert window._manual_draft is not None
    assert window._manual_draft.setup(0).model is None

    view._on_spectrum_mouse_motion(_spectrum_event(window, 0.1, 2.2))
    marker = view._manual_geometry_artists[0]
    assert tuple(np.asarray(marker.get_xdata(), dtype=float)) == (0.1,)
    assert tuple(np.asarray(marker.get_ydata(), dtype=float)) == (2.2,)
    assert window._manual_draft.setup(0).model is None

    view._on_spectrum_button_release(_spectrum_event(window, 0.1, 2.2))

    model = window._manual_draft.setup(0).model
    assert model is not None
    assert model.elastic_area is not None
    assert not view.property("manualInteractionActive")
    assert view.manual_interaction_instruction.isHidden()
    assert window._pending_manual_interaction is None
    window.close()


def test_lorentzian_geometry_is_symmetric_and_submits_raw_endpoints(
    application: QApplication,
) -> None:
    window, _project, _sample_state = _window_with_manual_resolution(application)
    view = window.dataset_view
    submitted: list[dict[str, float]] = []
    view.manual_component_completed.connect(
        lambda kind, hints: (
            submitted.append(dict(hints))
            if kind is ManualComponentKind.LORENTZIAN
            else None
        ),
    )

    for endpoint in (-0.3, 0.5):
        window._begin_manual_component_interaction(ManualComponentKind.LORENTZIAN)
        view._on_spectrum_button_press(
            _spectrum_event(window, 0.1, 1.5, pressed=True),
        )
        view._on_spectrum_mouse_motion(_spectrum_event(window, endpoint, 1.6))
        guide = view._manual_geometry_artists[-1]
        distance = abs(endpoint - 0.1)
        guide_x = tuple(np.asarray(guide.get_xdata(), dtype=float))
        assert guide_x == pytest.approx(
            (0.1 - distance, 0.1 + distance),
        )
        view._on_spectrum_button_release(_spectrum_event(window, endpoint, 1.6))

    assert [item["width_endpoint_energy"] for item in submitted] == [-0.3, 0.5]
    assert all("observed_fwhm" not in item for item in submitted)
    assert abs(submitted[0]["width_endpoint_energy"] - 0.1) == pytest.approx(
        abs(submitted[1]["width_endpoint_energy"] - 0.1),
    )
    assert window._manual_draft is not None
    model = window._manual_draft.setup(0).model
    assert model is not None
    assert len(model.lorentzians) == 2
    assert not view.property("manualInteractionActive")
    window.close()


def test_zero_width_lorentzian_rejection_exits_focus_without_partial_component(
    application: QApplication,
) -> None:
    window, _project, _sample_state = _window_with_manual_resolution(application)
    view = window.dataset_view
    window._begin_manual_component_interaction(ManualComponentKind.LORENTZIAN)
    view._on_spectrum_button_press(_spectrum_event(window, 0.2, 1.5, pressed=True))
    view._on_spectrum_button_release(_spectrum_event(window, 0.2, 1.5))

    assert window._manual_draft is not None
    assert window._manual_draft.setup(0).model is None
    assert window._pending_manual_interaction is None
    assert view._manual_component_kind is None
    assert view._manual_pending_evaluation is None
    assert not view.property("manualInteractionActive")
    assert not view._manual_geometry_artists
    window.close()


def test_background_one_gesture_previews_and_commits_full_domain_b1(
    application: QApplication,
) -> None:
    window, _project, _sample = _window_with_manual_resolution(application)
    view = window.dataset_view

    window._begin_manual_component_interaction(ManualComponentKind.BACKGROUND)
    view._on_spectrum_button_press(_spectrum_event(window, -1.0, 1.0, pressed=True))
    view._on_spectrum_mouse_motion(_spectrum_event(window, 0.8, 1.8))
    assert window._manual_draft is not None
    assert window._manual_draft.setup(0).model is None
    assert view._manual_pending_evaluation is not None
    assert window.dataset_view.dataset is not None
    display_energy = window.dataset_view.dataset.spectra[0].energy
    assert view._manual_pending_evaluation.energy == pytest.approx(display_energy)
    assert len(view._manual_geometry_artists) == 2
    view._on_spectrum_button_release(_spectrum_event(window, 0.8, 1.8))

    model = window._manual_draft.setup(0).model
    assert model is not None
    assert model.background is BackgroundModel.LINEAR
    assert not view.property("manualInteractionActive")
    assert view.manual_interaction_instruction.isHidden()
    assert window._pending_manual_interaction is None

    view.set_current_group(1)
    window._begin_manual_component_interaction(ManualComponentKind.BACKGROUND)
    view._on_spectrum_button_press(_spectrum_event(window, 0.2, -0.1, pressed=True))
    assert view._manual_pending_evaluation is not None
    view._on_spectrum_button_release(_spectrum_event(window, 0.2, -0.1))
    click_model = window._manual_draft.setup(1).model
    assert click_model is not None
    assert click_model.background is BackgroundModel.LINEAR
    assert click_model.b1 is not None
    assert click_model.b1.current_value == 0.0
    window.close()
