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
from PySide6.QtCore import QEvent, QLocale, QPoint, Qt, QTimer
from PySide6.QtGui import QColor, QFocusEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGridLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QScrollArea,
    QToolButton,
)

import ezqens.gui.main_window as main_window_module
from ezqens.domain import (
    DiagnosticSeverity,
    QBins,
    ReducedDataset,
    Spectrum,
    SpectrumRole,
)
from ezqens.fitting import (
    BACKGROUND_COMPONENT,
    ELASTIC_COMPONENT,
    BackgroundModel,
    ManualParameterIntent,
    ParameterFamily,
    ParameterReference,
)
from ezqens.gui import MainWindow, create_application
from ezqens.gui.dataset_view import OVERVIEW_ACTIVE_ALPHA, OVERVIEW_ACTIVE_COLOR
from ezqens.gui.manual_fit import (
    CompactNumberEdit,
    ManualFitLifecycle,
)
from ezqens.gui.theme import indicator_tokens_for, tokens_for
from ezqens.gui.workspace import DatasetAnalysisState, DatasetState, ProjectState
from ezqens.preprocessing import BoundarySide
from ezqens.workflow import (
    ManualCenterGroupState,
    ManualComponentKind,
    ManualFitDraft,
    ManualFitExecutionOutcome,
    ManualLorentzianState,
    ManualModelState,
    ManualParameterEdit,
    WorkflowDiagnostic,
    WorkflowDiagnosticCode,
    WorkflowProject,
    create_parameter_tie,
    manual_workflow_readiness,
    run_and_adopt_manual_fit,
    update_manual_parameter,
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


def _spectrum_pointer_event(
    window: MainWindow,
    x_value: float,
    y_value: float,
    *,
    pressed: bool = False,
    double_click: bool = False,
) -> MouseEvent:
    axes = window.dataset_view.spectrum_axes
    assert axes is not None
    x_pixel, y_pixel = axes.transData.transform((x_value, y_value))
    return cast(
        MouseEvent,
        SimpleNamespace(
            button=MouseButton.LEFT if pressed else None,
            inaxes=axes,
            xdata=x_value,
            ydata=y_value,
            x=x_pixel,
            y=y_pixel,
            dblclick=double_click,
        ),
    )


def _chain_control(window: MainWindow, reference: ParameterReference) -> QToolButton:
    controls = window.manual_fit_editor.parameter_controls[reference]
    assert controls.chain is not None
    return controls.chain


def _lifecycle(window: MainWindow) -> ManualFitLifecycle:
    return window._manual_fit_lifecycle


def _replace_text_and_tab(
    application: QApplication,
    edit: CompactNumberEdit,
    text: str,
) -> None:
    edit.setFocus()
    application.processEvents()
    edit.selectAll()
    QTest.keyClick(edit, Qt.Key.Key_Backspace)
    if text:
        QTest.keyClicks(edit, text)
    QTest.keyClick(edit, Qt.Key.Key_Tab)
    application.processEvents()


def _set_runnable_background(
    window: MainWindow,
    *,
    lower: float | None = 0.0,
    upper: float | None = 3.0,
    free: bool = True,
    bounds_enabled: bool = True,
) -> ParameterReference:
    assert window._manual_draft is not None
    window._manual_draft = window._manual_draft.with_model(
        window.dataset_view.current_group_index,
        ManualModelState(
            background=BackgroundModel.LINEAR,
            b0=ManualParameterIntent(
                0.5,
                lower,
                upper,
                free,
                bounds_enabled,
            ),
            b1=ManualParameterIntent(0.0, free=False),
        ),
    )
    window._refresh_manual_fit()
    return ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.OFFSET)


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


def test_manual_fit_action_absorbs_checked_before_opening_current_sample(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    window.manual_fit_action.setCheckable(True)

    window.manual_fit_action.trigger()

    assert window.manual_fit_action.isChecked()
    assert window._manual_owner == (project, sample.workflow_dataset_id)
    assert window._manual_draft is not None
    assert not window.manual_fit_editor.isHidden()
    window.close()


def test_inspector_resolution_button_absorbs_checked_and_uses_open_sample(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, project, sample = _window_with_sample(application)
    resolution = window.workspace.add_dataset(project, _resolution(sample.dataset))
    monkeypatch.setattr(
        QInputDialog,
        "getItem",
        lambda *_args, **_kwargs: (resolution.name, True),
    )
    button = window.inspector_resolution_button
    button.setCheckable(True)
    window.show()
    window.set_inspector_visible(True)
    application.processEvents()

    QTest.mouseClick(button, Qt.MouseButton.LeftButton)

    assert button.isChecked()
    workflow = window._workflow_project_for(project)
    assert len(workflow.resolution_associations) == 1
    association = workflow.resolution_associations[0]
    assert association.sample_id == sample.workflow_dataset_id
    assert association.resolution_id == resolution.workflow_dataset_id
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


def test_saved_batch_boundary_uses_existing_fit_invalidation_lifecycle(
    application: QApplication,
) -> None:
    window, _project, _sample = _window_with_manual_resolution(application)
    _set_runnable_background(window)
    window.manual_fit_editor.run_button.click()
    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    assert window._manual_fit_result is not None
    window.dataset_view.set_current_group(1)
    _set_runnable_background(window)
    window.manual_fit_editor.run_button.click()
    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    assert window._manual_fit_result is not None
    session = window._active_manual_session
    assert session is not None
    assert session.execution_state(0).lifecycle is ManualFitLifecycle.CURRENT
    assert session.execution_state(1).lifecycle is ManualFitLifecycle.CURRENT
    window.dataset_view.set_current_group(0)

    window.enter_mask_task()
    assert window._mask_draft is not None
    assert window._mask_draft.set_auto_boundary(
        0,
        side=BoundarySide.LEFT,
        energy=-1.0,
    )
    assert window._mask_draft.set_auto_boundary(
        0,
        side=BoundarySide.RIGHT,
        energy=1.0,
    )
    assert window._mask_draft.apply_boundary_to_all_groups(0)
    window._refresh_mask_preview()

    assert window.save_mask_task()

    assert _lifecycle(window) is ManualFitLifecycle.NEEDS_FIT
    assert window._manual_fit_result is None
    assert session.execution_state(0).lifecycle is ManualFitLifecycle.NEEDS_FIT
    assert session.execution_state(1).lifecycle is ManualFitLifecycle.NEEDS_FIT
    assert session.execution_state(0).fit_result is None
    assert session.execution_state(1).fit_result is None
    assert window.dataset_view.residual_axes is None
    window.close()


def test_saved_group_local_mask_invalidates_only_that_group_result(
    application: QApplication,
) -> None:
    window, _project, _sample_state = _window_with_manual_resolution(application)
    _set_runnable_background(window)
    window.manual_fit_editor.run_button.click()
    result_a = window._manual_fit_result
    assert result_a is not None
    window.dataset_view.set_current_group(1)
    _set_runnable_background(window)
    window.manual_fit_editor.run_button.click()
    result_b = window._manual_fit_result
    assert result_b is not None
    session = window._active_manual_session
    assert session is not None

    window.dataset_view.set_current_group(0)
    window.enter_mask_task()
    assert window._mask_draft is not None
    assert window._mask_draft.exclude_points(
        0,
        np.array([True, False, False, False, False]),
    )
    window._refresh_mask_preview()
    assert window.save_mask_task()

    assert session.execution_state(0).lifecycle is ManualFitLifecycle.NEEDS_FIT
    assert session.execution_state(0).fit_result is None
    assert session.execution_state(1).lifecycle is ManualFitLifecycle.CURRENT
    assert session.execution_state(1).fit_result is result_b
    window.dataset_view.set_current_group(1)
    assert window._manual_fit_result is result_b
    assert window.dataset_view._manual_fit_result is result_b
    assert window.dataset_view.residual_axes is not None
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

    edited_center = "0.12567891234"
    controls[center].current.setText(edited_center)
    assert controls[center].current.hasAcceptableInput()
    controls[center].current.textEdited.emit(edited_center)
    controls[center].current.editingFinished.emit()

    updated = window._manual_draft.setup(0).model
    assert updated is not None
    assert updated.parameter_intent(area).current_value == exact_area
    assert updated.parameter_intent(center).current_value == 0.12567891234
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


@pytest.mark.parametrize(
    ("exact", "unfocused", "focused"),
    (
        (1.234567, "1.23", "1.235"),
        (0.00001234567, "1.23e-5", "1.235e-5"),
    ),
)
def test_compact_number_focus_is_concise_and_lossless(
    application: QApplication,
    exact: float,
    unfocused: str,
    focused: str,
) -> None:
    edit = CompactNumberEdit(exact)
    assert edit.text() == unfocused

    QApplication.sendEvent(
        edit,
        QFocusEvent(QEvent.Type.FocusIn, Qt.FocusReason.OtherFocusReason),
    )
    assert edit.text() == focused
    assert edit.precise_value(required=True) == exact

    QApplication.sendEvent(
        edit,
        QFocusEvent(QEvent.Type.FocusOut, Qt.FocusReason.OtherFocusReason),
    )
    assert edit.text() == unfocused
    assert edit.precise_value(required=True) == exact
    application.processEvents()


def test_manual_current_and_bounds_accept_dot_decimal_input(
    application: QApplication,
) -> None:
    previous_locale = QLocale()
    QLocale.setDefault(QLocale(QLocale.Language.Swedish, QLocale.Country.Sweden))
    try:
        locale_probe = CompactNumberEdit(0.0)
        locale_probe.setText("1.23")
        assert locale_probe.hasAcceptableInput()
    finally:
        QLocale.setDefault(previous_locale)

    window, project, sample = _window_with_sample(application)
    window.show_manual_fit(project, sample)
    assert window._manual_draft is not None
    window._manual_draft = window._manual_draft.with_model(
        0,
        ManualModelState(
            lorentzians=(
                ManualLorentzianState(
                    area=ManualParameterIntent(0.5),
                    center=ManualParameterIntent(0.0),
                    fwhm=ManualParameterIntent(0.4),
                ),
            ),
        ),
    )
    window._refresh_manual_fit()
    model = window._manual_draft.setup(0).model
    assert model is not None
    area = next(
        reference
        for reference in model.parameter_references()
        if reference.family is ParameterFamily.AREA
    )

    for field_name, text in (
        ("current", "1.23"),
        ("lower", "0.05"),
        ("upper", "2.75"),
    ):
        edit = getattr(window.manual_fit_editor.parameter_controls[area], field_name)
        edit.setText(text)
        assert edit.hasAcceptableInput()
        edit.textEdited.emit(text)
        edit.editingFinished.emit()

    model = window._manual_draft.setup(0).model
    assert model is not None
    intent = model.parameter_intent(area)
    assert intent.current_value == 1.23
    assert intent.user_lower_limit == 0.05
    assert intent.user_upper_limit == 2.75
    window.close()


def test_real_tab_commits_only_user_edits_and_allows_blank_optional_bounds(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, _project, _sample = _window_with_manual_resolution(application)
    assert window._manual_draft is not None
    exact = 1.234567
    window._manual_draft = window._manual_draft.with_model(
        0,
        ManualModelState(
            background=BackgroundModel.LINEAR,
            b0=ManualParameterIntent(exact, 0.1, 2.0),
            b1=ManualParameterIntent(0.0, free=False),
        ),
    )
    offset = ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.OFFSET)
    window._manual_fit_lifecycle = ManualFitLifecycle.CURRENT
    window._refresh_manual_fit()
    calls: list[tuple[ParameterReference, ManualParameterEdit]] = []
    actual_update = update_manual_parameter

    def recording_update(
        draft: ManualFitDraft,
        group_index: int,
        reference: ParameterReference,
        edit: ManualParameterEdit,
    ) -> ManualFitDraft:
        calls.append((reference, edit))
        return actual_update(draft, group_index, reference, edit)

    monkeypatch.setattr(
        main_window_module,
        "update_manual_parameter",
        recording_update,
    )
    window.show()
    application.processEvents()
    controls = window.manual_fit_editor.parameter_controls[offset]

    controls.current.setFocus()
    application.processEvents()
    assert controls.current.text() == "1.235"
    QTest.keyClick(controls.current, Qt.Key.Key_Tab)
    application.processEvents()
    assert calls == []
    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    assert controls.current.text() == "1.23"
    model = window._manual_draft.setup(0).model
    assert model is not None
    assert model.parameter_intent(offset).current_value == exact

    controls.lower.setFocus()
    application.processEvents()
    QTest.keyClick(controls.lower, Qt.Key.Key_Tab)
    application.processEvents()
    assert calls == []
    assert _lifecycle(window) is ManualFitLifecycle.CURRENT

    _replace_text_and_tab(application, controls.lower, "")
    assert len(calls) == 1
    model = window._manual_draft.setup(0).model
    assert model is not None
    intent = model.parameter_intent(offset)
    assert intent.user_lower_limit is None
    assert intent.user_upper_limit == 2.0
    assert intent.user_bounds_enabled

    controls = window.manual_fit_editor.parameter_controls[offset]
    _replace_text_and_tab(application, controls.upper, "")
    assert len(calls) == 2
    model = window._manual_draft.setup(0).model
    assert model is not None
    intent = model.parameter_intent(offset)
    assert intent.user_lower_limit is None
    assert intent.user_upper_limit is None
    assert intent.user_bounds_enabled

    controls = window.manual_fit_editor.parameter_controls[offset]
    _replace_text_and_tab(application, controls.current, "")
    assert len(calls) == 2
    model = window._manual_draft.setup(0).model
    assert model is not None
    assert model.parameter_intent(offset).current_value == exact
    assert controls.current.text() == "1.23"

    _replace_text_and_tab(application, controls.current, "1.23")
    assert len(calls) == 3
    model = window._manual_draft.setup(0).model
    assert model is not None
    assert model.parameter_intent(offset).current_value == 1.23
    assert _lifecycle(window) is ManualFitLifecycle.NEEDS_FIT
    window.close()


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


def test_populated_three_slot_parameter_grid_sets_usable_inspector_minimum(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    window.show()
    window.show_manual_fit(project, sample)
    assert window._manual_draft is not None
    lorentzian = ManualLorentzianState(
        area=ManualParameterIntent(0.2),
        center=ManualParameterIntent(0.0),
        fwhm=ManualParameterIntent(0.5),
    )
    window._manual_draft = window._manual_draft.with_model(
        0,
        ManualModelState(
            energy_shift=ManualParameterIntent(0.0),
            elastic_area=ManualParameterIntent(0.4),
            lorentzians=(lorentzian,),
            background=BackgroundModel.LINEAR,
            b0=ManualParameterIntent(0.1),
            b1=ManualParameterIntent(0.0),
        ),
    )
    window._refresh_manual_fit()
    application.processEvents()
    populated_minimum = window.inspector.minimumWidth()
    assert populated_minimum > 220

    window.splitter.setSizes((220, 900, 1))
    application.processEvents()
    assert window.inspector.width() >= populated_minimum
    inspector_right = window.inspector.contentsRect().right()
    for entry in window.manual_fit_editor.parameter_controls.values():
        widgets = [
            entry.current,
            entry.lower,
            entry.upper,
            entry.fixed,
            entry.bounds_enabled,
        ]
        if entry.chain is not None:
            widgets.append(entry.chain)
        assert all(
            widget.mapTo(window.inspector, widget.rect().topRight()).x()
            <= inspector_right
            for widget in widgets
        )

    control_map = window.manual_fit_editor.parameter_controls
    elastic_area = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA)
    elastic_center = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER)
    lorentzian_area = ParameterReference(lorentzian.identity, ParameterFamily.AREA)
    lorentzian_center = ParameterReference(
        lorentzian.identity,
        ParameterFamily.CENTER,
    )
    fwhm = ParameterReference(lorentzian.identity, ParameterFamily.FWHM)
    narrow_widths = tuple(item.current.width() for item in control_map.values())
    sections = window.manual_fit_editor.sections
    assert (
        abs(
            control_map[elastic_area].current.mapTo(sections, QPoint()).x()
            - control_map[lorentzian_area].current.mapTo(sections, QPoint()).x()
        )
        <= 4
    )
    assert (
        abs(
            control_map[elastic_center].current.mapTo(sections, QPoint()).x()
            - control_map[lorentzian_center].current.mapTo(sections, QPoint()).x()
        )
        <= 4
    )
    assert (
        control_map[fwhm]
        .bounds_enabled.mapTo(
            window.inspector,
            control_map[fwhm].bounds_enabled.rect().topRight(),
        )
        .x()
        <= inspector_right
    )

    window.splitter.setSizes((220, 450, 520))
    application.processEvents()
    wide_widths = tuple(item.current.width() for item in control_map.values())
    assert all(
        wide >= narrow for narrow, wide in zip(narrow_widths, wide_widths, strict=True)
    )
    assert (
        abs(
            control_map[elastic_area].current.mapTo(sections, QPoint()).x()
            - control_map[lorentzian_area].current.mapTo(sections, QPoint()).x()
        )
        <= 4
    )
    assert (
        abs(
            control_map[elastic_center].current.mapTo(sections, QPoint()).x()
            - control_map[lorentzian_center].current.mapTo(sections, QPoint()).x()
        )
        <= 4
    )
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
    assert not hasattr(view, "_manual_geometry_artists")
    assert window._pending_manual_interaction is None
    assert window._manual_draft is not None
    assert window._manual_draft.setup(0).model is None
    window.close()


def test_elastic_press_move_release_uses_only_workflow_scientific_preview(
    application: QApplication,
) -> None:
    window, _project, _sample_state = _window_with_manual_resolution(application)
    view = window.dataset_view
    add_elastic = window.manual_fit_editor.findChild(QToolButton, "addElasticButton")
    assert add_elastic is not None
    add_elastic.click()

    view._on_spectrum_button_press(_spectrum_event(window, 0.0, 2.0, pressed=True))
    assert window._manual_preview is None
    assert view._manual_pending_evaluation is not None
    assert view.spectrum_axes is not None
    assert not any(line.get_zorder() >= 8 for line in view.spectrum_axes.lines)
    assert window._manual_draft is not None
    assert window._manual_draft.setup(0).model is None

    view._on_spectrum_mouse_motion(_spectrum_event(window, 0.1, 2.2))
    evaluation = view._manual_pending_evaluation
    assert evaluation is not None
    assert any(
        np.array_equal(line.get_ydata(), evaluation.total)
        for line in view.spectrum_axes.lines
    )
    assert not any(line.get_zorder() >= 8 for line in view.spectrum_axes.lines)
    assert window._manual_draft.setup(0).model is None

    view._on_spectrum_button_release(_spectrum_event(window, 0.1, 2.2))

    model = window._manual_draft.setup(0).model
    assert model is not None
    assert model.elastic_area is not None
    assert not view.property("manualInteractionActive")
    assert view.manual_interaction_instruction.isHidden()
    assert window._pending_manual_interaction is None
    window.close()


def test_lorentzian_drag_uses_only_updating_scientific_curve_and_raw_endpoint(
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

    window._begin_manual_component_interaction(ManualComponentKind.LORENTZIAN)
    view._on_spectrum_button_press(
        _spectrum_event(window, 0.1, 1.5, pressed=True),
    )
    assert view._spectrum_zoom_rectangle is None
    assert view._manual_pending_evaluation is None
    assert not hasattr(view, "_manual_geometry_artists")
    assert view.spectrum_axes is not None
    assert not any(line.get_zorder() >= 8 for line in view.spectrum_axes.lines)

    view._on_spectrum_mouse_motion(_spectrum_event(window, 0.3, 1.6))
    first_evaluation = view._manual_pending_evaluation
    assert first_evaluation is not None
    first_total = np.array(first_evaluation.total, copy=True)
    assert any(
        np.array_equal(line.get_ydata(), first_total)
        for line in view.spectrum_axes.lines
    )
    assert not any(line.get_zorder() >= 8 for line in view.spectrum_axes.lines)

    view._on_spectrum_mouse_motion(_spectrum_event(window, 1.8, 1.6))
    final_evaluation = view._manual_pending_evaluation
    assert final_evaluation is not None
    assert not np.array_equal(final_evaluation.total, first_total)
    provisional_total = np.array(final_evaluation.total, copy=True)
    assert any(
        np.array_equal(line.get_ydata(), provisional_total)
        for line in view.spectrum_axes.lines
    )
    assert not any(line.get_zorder() >= 8 for line in view.spectrum_axes.lines)
    view._on_spectrum_button_release(_spectrum_event(window, 1.8, 1.6))

    assert [item["width_endpoint_energy"] for item in submitted] == [1.8]
    assert all("observed_fwhm" not in item for item in submitted)
    assert window._manual_draft is not None
    model = window._manual_draft.setup(0).model
    assert model is not None
    assert len(model.lorentzians) == 1
    assert window._manual_preview is not None
    np.testing.assert_allclose(
        window._manual_preview.display_evaluation.total,
        provisional_total,
    )
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
    assert not hasattr(view, "_manual_geometry_artists")
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
    assert view.spectrum_axes is not None
    assert not any(line.get_zorder() >= 8 for line in view.spectrum_axes.lines)
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


def test_run_fit_adopts_values_seeds_next_run_and_uses_result_plot(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, _project, _sample = _window_with_manual_resolution(application)
    offset = _set_runnable_background(window)
    starts: list[float] = []
    fitting_states: list[tuple[ManualFitLifecycle, bool]] = []
    actual_run = run_and_adopt_manual_fit

    def recording_run(
        project: WorkflowProject,
        draft: ManualFitDraft,
        group_index: int,
    ) -> ManualFitExecutionOutcome:
        model = draft.setup(group_index).model
        assert model is not None
        starts.append(model.parameter_intent(offset).current_value)
        fitting_states.append(
            (_lifecycle(window), window.manual_fit_editor.run_button.isEnabled())
        )
        window._run_manual_fit()
        assert len(starts) == len(fitting_states)
        return actual_run(project, draft, group_index)

    monkeypatch.setattr(
        main_window_module,
        "run_and_adopt_manual_fit",
        recording_run,
    )
    assert window.manual_fit_editor.run_button.isEnabled()
    assert window.manual_fit_editor.status_label.text() == "Ready · Not yet fit"

    window.manual_fit_editor.run_button.click()

    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    assert window.manual_fit_editor.status_label.text() == "Fit current"
    assert window._manual_fit_result is not None
    first_result = window._manual_fit_result
    assert window.dataset_view._manual_fit_result is window._manual_fit_result
    assert window._manual_draft is not None
    adopted = window._manual_draft.setup(0).model
    assert adopted is not None
    fitted_value = window._manual_fit_result.parameter_by_reference(offset).value
    adopted_offset = adopted.parameter_intent(offset)
    assert adopted_offset.current_value == pytest.approx(fitted_value)
    assert adopted_offset.user_lower_limit == 0.0
    assert adopted_offset.user_upper_limit == 3.0
    assert adopted_offset.free
    assert adopted_offset.user_bounds_enabled
    rendered = window.manual_fit_editor.parameter_controls[offset].current
    assert rendered.stored_value == pytest.approx(fitted_value)
    fitted_total = window._manual_fit_result.evaluation.total
    spectrum_axes = window.dataset_view.spectrum_axes
    assert spectrum_axes is not None
    assert any(
        np.array_equal(line.get_ydata(), fitted_total) for line in spectrum_axes.lines
    )

    window.manual_fit_editor.run_button.click()

    assert starts[0] == 0.5
    assert starts[1] == pytest.approx(fitted_value)
    assert fitting_states == [
        (ManualFitLifecycle.FITTING, False),
        (ManualFitLifecycle.FITTING, False),
    ]
    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    assert window._manual_fit_result is not first_result
    assert not window.manual_fit_editor.result_section.isHidden()
    assert window.dataset_view.residual_axes is not None
    window.close()


def test_run_fit_readiness_warning_and_structured_failure_behavior(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked, project, _sample = _window_with_sample(application)
    blocked.show_manual_fit(project, blocked._open_dataset)
    _set_runnable_background(blocked)
    assert not blocked.manual_fit_editor.run_button.isEnabled()
    assert blocked.manual_fit_editor.status_label.text().startswith("Blocked")
    blocked.close()

    window, project, _sample = _window_with_manual_resolution(application)
    _set_runnable_background(window)
    assert window._manual_draft is not None
    model = window._manual_draft.setup(0).model
    assert model is not None
    readiness = manual_workflow_readiness(
        window._workflow_project_for(project),
        window._manual_draft,
        0,
    )
    warning = WorkflowDiagnostic(
        WorkflowDiagnosticCode.MANUAL_FIT_EXECUTION_FAILED,
        DiagnosticSeverity.WARNING,
        "non-blocking test warning",
        group_index=0,
    )
    window.manual_fit_editor.set_state(
        model,
        (),
        replace(readiness, workflow_diagnostics=(warning,)),
    )
    assert window.manual_fit_editor.run_button.isEnabled()

    failure = WorkflowDiagnostic(
        WorkflowDiagnosticCode.MANUAL_FIT_EXECUTION_FAILED,
        DiagnosticSeverity.ERROR,
        "structured execution failure",
        group_index=0,
    )
    monkeypatch.setattr(
        main_window_module,
        "run_and_adopt_manual_fit",
        lambda *args, **kwargs: ManualFitExecutionOutcome(
            None,
            None,
            (failure,),
        ),
    )
    original = window._manual_draft
    window.manual_fit_editor.run_button.click()

    assert window._manual_draft is original
    assert window._manual_fit_result is None
    assert _lifecycle(window) is ManualFitLifecycle.READY
    assert (
        "structured execution failure"
        in window.manual_fit_editor.diagnostics_label.text()
    )
    assert window.manual_fit_editor.result_section.isHidden()
    assert window.dataset_view.residual_axes is None
    assert window.manual_fit_editor.run_button.isEnabled()
    window.close()


@pytest.mark.parametrize(
    "control_name",
    ("current", "lower", "upper", "bounds_enabled", "fixed"),
)
def test_parameter_changes_mark_current_fit_stale(
    application: QApplication,
    control_name: str,
) -> None:
    window, _project, _sample = _window_with_manual_resolution(application)
    offset = _set_runnable_background(window)
    window.manual_fit_editor.run_button.click()
    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    controls = window.manual_fit_editor.parameter_controls[offset]

    if control_name in {"bounds_enabled", "fixed"}:
        getattr(controls, control_name).click()
    else:
        edit = getattr(controls, control_name)
        text = {"current": "0.6", "lower": "0.1", "upper": "2.5"}[control_name]
        edit.setText(text)
        edit.textEdited.emit(text)
        edit.editingFinished.emit()

    assert _lifecycle(window) is ManualFitLifecycle.NEEDS_FIT
    assert window.manual_fit_editor.status_label.text() == "Needs fit"
    assert window._manual_fit_result is None
    assert window.dataset_view._manual_fit_result is None
    assert window._manual_preview is not None
    window.close()


def test_bounds_toggle_preserves_blank_one_sided_and_dormant_inverted_values(
    application: QApplication,
) -> None:
    window, _project, _sample = _window_with_manual_resolution(application)
    offset = _set_runnable_background(
        window,
        lower=0.1,
        upper=0.5,
        bounds_enabled=False,
    )
    controls = window.manual_fit_editor.parameter_controls[offset]
    assert controls.bounds_enabled.text() == "×"
    assert controls.lower.text() == "0.1"
    assert controls.upper.text() == "0.5"

    controls.bounds_enabled.click()
    model = window._manual_draft.setup(0).model  # type: ignore[union-attr]
    assert model is not None
    assert model.parameter_intent(offset).user_bounds_enabled
    assert model.parameter_intent(offset).user_lower_limit == 0.1
    assert model.parameter_intent(offset).user_upper_limit == 0.5

    window.manual_fit_editor.parameter_controls[offset].bounds_enabled.click()
    lower = window.manual_fit_editor.parameter_controls[offset].lower
    lower.setText("")
    lower.textEdited.emit("")
    lower.editingFinished.emit()
    model = window._manual_draft.setup(0).model  # type: ignore[union-attr]
    assert model is not None
    assert not model.parameter_intent(offset).user_bounds_enabled
    assert model.parameter_intent(offset).user_lower_limit is None
    assert model.parameter_intent(offset).user_upper_limit == 0.5

    upper = window.manual_fit_editor.parameter_controls[offset].upper
    upper.setText("")
    upper.textEdited.emit("")
    upper.editingFinished.emit()
    controls = window.manual_fit_editor.parameter_controls[offset]
    controls.bounds_enabled.click()
    model = window._manual_draft.setup(0).model  # type: ignore[union-attr]
    assert model is not None
    assert model.parameter_intent(offset).user_bounds_enabled
    assert model.parameter_intent(offset).user_lower_limit is None
    assert model.parameter_intent(offset).user_upper_limit is None

    controls = window.manual_fit_editor.parameter_controls[offset]
    controls.bounds_enabled.click()
    for name, text in (("lower", "0.8"), ("upper", "0.2")):
        edit = getattr(window.manual_fit_editor.parameter_controls[offset], name)
        edit.setText(text)
        edit.textEdited.emit(text)
        edit.editingFinished.emit()
    model = window._manual_draft.setup(0).model  # type: ignore[union-attr]
    assert model is not None
    intent = model.parameter_intent(offset)
    assert not intent.user_bounds_enabled
    assert intent.user_lower_limit == 0.8
    assert intent.user_upper_limit == 0.2
    assert window.manual_fit_editor.run_button.isEnabled()

    window.manual_fit_editor.parameter_controls[offset].bounds_enabled.click()
    model = window._manual_draft.setup(0).model  # type: ignore[union-attr]
    assert model is not None
    assert model.parameter_intent(offset).user_bounds_enabled
    assert not window.manual_fit_editor.run_button.isEnabled()
    assert window.manual_fit_editor.status_label.text().startswith("Blocked")
    window.close()


def test_tied_numeric_edits_refresh_every_member_once_and_mark_fit_stale(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, _project, _sample = _window_with_manual_resolution(application)
    assert window._manual_draft is not None
    window._manual_draft = window._manual_draft.with_model(
        0,
        ManualModelState(
            lorentzians=tuple(
                ManualLorentzianState(
                    area=ManualParameterIntent(0.4, free=False),
                    center=ManualParameterIntent(0.15, -0.5, 0.5),
                    fwhm=ManualParameterIntent(0.4, free=False),
                )
                for _ in range(2)
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
    window._manual_fit_lifecycle = ManualFitLifecycle.CURRENT
    calls: list[tuple[ParameterReference, ManualParameterEdit]] = []
    actual_update = update_manual_parameter

    def recording_update(
        draft: ManualFitDraft,
        group_index: int,
        reference: ParameterReference,
        edit: ManualParameterEdit,
    ) -> ManualFitDraft:
        calls.append((reference, edit))
        return actual_update(draft, group_index, reference, edit)

    monkeypatch.setattr(
        main_window_module,
        "update_manual_parameter",
        recording_update,
    )
    window.show()
    application.processEvents()
    stale_sibling = window.manual_fit_editor.parameter_controls[centers[1]].current

    current = window.manual_fit_editor.parameter_controls[centers[0]].current
    _replace_text_and_tab(application, current, "0.20")

    assert len(calls) == 1
    assert _lifecycle(window) is ManualFitLifecycle.NEEDS_FIT
    assert (
        window.manual_fit_editor.parameter_controls[centers[1]].current
        is not stale_sibling
    )
    for reference in centers:
        controls = window.manual_fit_editor.parameter_controls[reference]
        assert controls.current.text() == "0.2"
        assert controls.current.stored_value == 0.2

    for field_name, text, expected in (
        ("lower", "-0.25", -0.25),
        ("upper", "0.75", 0.75),
    ):
        edit = getattr(
            window.manual_fit_editor.parameter_controls[centers[0]],
            field_name,
        )
        _replace_text_and_tab(application, edit, text)
        for reference in centers:
            refreshed = getattr(
                window.manual_fit_editor.parameter_controls[reference],
                field_name,
            )
            assert refreshed.stored_value == expected
            assert refreshed.text() == text

    assert len(calls) == 3
    model = window._manual_draft.setup(0).model
    assert model is not None
    for reference in centers:
        intent = model.parameter_intent(reference)
        assert intent.current_value == 0.2
        assert intent.user_lower_limit == -0.25
        assert intent.user_upper_limit == 0.75
    window.close()


def test_tied_bounds_state_is_shared_and_untie_copies_it(
    application: QApplication,
) -> None:
    window, _project, _sample = _window_with_manual_resolution(application)
    assert window._manual_draft is not None
    window._manual_draft = window._manual_draft.with_model(
        0,
        ManualModelState(
            lorentzians=tuple(
                ManualLorentzianState(
                    area=ManualParameterIntent(0.3, 0.1, 0.8),
                    center=ManualParameterIntent(0.0),
                    fwhm=ManualParameterIntent(0.4),
                )
                for _ in range(2)
            ),
        ),
    )
    window._refresh_manual_fit()
    model = window._manual_draft.setup(0).model
    assert model is not None
    areas = tuple(
        reference
        for reference in model.parameter_references()
        if reference.family is ParameterFamily.AREA
    )
    _chain_control(window, areas[0]).click()
    _chain_control(window, areas[1]).click()
    window._manual_fit_lifecycle = ManualFitLifecycle.CURRENT

    window.manual_fit_editor.parameter_controls[areas[0]].bounds_enabled.click()

    model = window._manual_draft.setup(0).model
    assert model is not None
    assert not model.parameter_intent(areas[0]).user_bounds_enabled
    assert not model.parameter_intent(areas[1]).user_bounds_enabled
    assert not window.manual_fit_editor.parameter_controls[
        areas[1]
    ].bounds_enabled.isChecked()
    assert _lifecycle(window) is ManualFitLifecycle.NEEDS_FIT

    _chain_control(window, areas[1]).click()
    model = window._manual_draft.setup(0).model
    assert model is not None
    assert model.tie_for(areas[1]) is None
    assert not model.parameter_intent(areas[1]).user_bounds_enabled
    assert model.parameter_intent(areas[1]).user_lower_limit == 0.1
    assert model.parameter_intent(areas[1]).user_upper_limit == 0.8
    window.close()


def test_navigation_restores_current_fit_and_scientific_changes_invalidate_it(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_manual_resolution(application)
    _set_runnable_background(window)
    window.manual_fit_editor.run_button.click()
    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    assert window.dataset_view.residual_axes is not None
    assert not window.manual_fit_editor.result_section.isHidden()
    result = window._manual_fit_result
    assert result is not None

    window.dataset_view.set_current_group(1)
    assert _lifecycle(window) is ManualFitLifecycle.READY
    assert window._manual_fit_result is None
    assert window.dataset_view.residual_axes is None
    assert window.manual_fit_editor.result_section.isHidden()

    window.dataset_view.set_current_group(0)
    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    assert window._manual_fit_result is result
    assert window.dataset_view._manual_fit_result is result
    assert window.dataset_view.residual_axes is not None
    window._remove_manual_component(BACKGROUND_COMPONENT)
    assert _lifecycle(window) is ManualFitLifecycle.NEEDS_FIT

    _set_runnable_background(window)
    window.manual_fit_editor.run_button.click()
    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    replacement = window.workspace.add_dataset(project, _resolution(sample.dataset))
    assert window.apply_resolution_for_sample(
        project,
        sample,
        replacement,
        replace_confirmed=True,
    )
    assert _lifecycle(window) is ManualFitLifecycle.NEEDS_FIT
    assert window._manual_fit_result is None
    window.close()


def test_two_groups_retain_results_and_invalidate_only_edited_group(
    application: QApplication,
) -> None:
    window, _project, _sample_state = _window_with_manual_resolution(application)
    offset = _set_runnable_background(window)
    window.manual_fit_editor.run_button.click()
    result_a = window._manual_fit_result
    assert result_a is not None
    assert result_a.provenance.group_index == 0
    fitted_a = window._manual_draft
    assert fitted_a is not None
    adopted_a = fitted_a.setup(0).model
    assert adopted_a is not None

    window.dataset_view.set_current_group(1)
    _set_runnable_background(window)
    window.manual_fit_editor.run_button.click()
    result_b = window._manual_fit_result
    assert result_b is not None
    assert result_b is not result_a
    assert result_b.provenance.group_index == 1
    session = window._active_manual_session
    assert session is not None
    assert session.execution_state(0).fit_result is result_a
    assert session.execution_state(1).fit_result is result_b

    window.dataset_view.set_current_group(0)
    rendered_group_one_results: list[object | None] = []
    callback_id = window.dataset_view.canvas.mpl_connect(
        "draw_event",
        lambda _event: (
            rendered_group_one_results.append(window.dataset_view._manual_fit_result)
            if window.dataset_view.current_group_index == 1
            else None
        ),
    )
    window.dataset_view.set_current_group(1)
    application.processEvents()
    window.dataset_view.canvas.mpl_disconnect(callback_id)
    assert all(item is not result_a for item in rendered_group_one_results)
    assert any(item is result_b for item in rendered_group_one_results)

    for group_index, expected in ((0, result_a), (1, result_b), (0, result_a)):
        draft_before = window._manual_draft
        window.dataset_view.set_current_group(group_index)
        application.processEvents()
        assert window._manual_draft is draft_before
        assert _lifecycle(window) is ManualFitLifecycle.CURRENT
        assert window._manual_fit_result is expected
        assert window.dataset_view._manual_fit_result is expected
        assert window.dataset_view.standardized_residual_line is not None
        np.testing.assert_array_equal(
            window.dataset_view.standardized_residual_line.get_ydata(),
            expected.standardized_residuals,
        )

    assert window._manual_draft is not None
    current_a = window._manual_draft.setup(0).model
    assert current_a is not None
    assert current_a is adopted_a
    intent = current_a.parameter_intent(offset)
    window._update_manual_parameter(
        offset,
        ManualParameterEdit(
            intent.current_value + 0.01,
            intent.user_lower_limit,
            intent.user_upper_limit,
            intent.free,
            intent.user_bounds_enabled,
        ),
    )
    assert session.execution_state(0).lifecycle is ManualFitLifecycle.NEEDS_FIT
    assert session.execution_state(0).fit_result is None
    assert session.execution_state(1).lifecycle is ManualFitLifecycle.CURRENT
    assert session.execution_state(1).fit_result is result_b

    window.manual_fit_editor.run_button.click()
    assert session.execution_state(0).lifecycle is ManualFitLifecycle.CURRENT
    assert window._manual_draft.setup(0).model is not adopted_a
    window._remove_manual_component(BACKGROUND_COMPONENT)
    assert session.execution_state(0).lifecycle is ManualFitLifecycle.NEEDS_FIT
    assert session.execution_state(1).lifecycle is ManualFitLifecycle.CURRENT
    assert session.execution_state(1).fit_result is result_b

    window.dataset_view.set_current_group(1)
    assert window._manual_fit_result is result_b
    assert window.dataset_view._manual_fit_result is result_b
    assert window.dataset_view.residual_axes is not None
    window.close()


def test_current_fit_result_uses_public_arrays_identities_and_evidence(
    application: QApplication,
) -> None:
    window, _project, _sample = _window_with_manual_resolution(application)
    offset = _set_runnable_background(window)
    slope = ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.SLOPE)
    window.manual_fit_editor.run_button.click()
    assert window._manual_fit_result is not None
    result = window._manual_fit_result

    assert window.dataset_view.residual_axes is not None
    assert window.dataset_view.standardized_residual_line is not None
    assert np.array_equal(
        window.dataset_view.standardized_residual_line.get_xdata(),
        result.evaluation.energy,
    )
    assert np.array_equal(
        window.dataset_view.standardized_residual_line.get_ydata(),
        result.standardized_residuals,
    )
    assert window.dataset_view.spectrum_axes is not None
    assert window.dataset_view.residual_axes.get_shared_x_axes().joined(
        window.dataset_view.spectrum_axes,
        window.dataset_view.residual_axes,
    )
    assert not window.manual_fit_editor.result_section.isHidden()

    warning = WorkflowDiagnostic(
        WorkflowDiagnosticCode.MANUAL_FIT_EXECUTION_FAILED,
        DiagnosticSeverity.WARNING,
        "structured result warning",
        group_index=0,
    )
    parameters = tuple(
        replace(
            estimate,
            standard_error=(
                0.012345
                if offset in estimate.references
                else None
                if slope in estimate.references
                else estimate.standard_error
            ),
            active_lower_bound=offset in estimate.references,
        )
        for estimate in result.parameters
    )
    presented = replace(
        result,
        parameters=tuple(reversed(parameters)),
        covariance=None,
        correlation=None,
        diagnostics=replace(
            result.diagnostics,
            covariance_available=False,
        ),
    )
    assert window._manual_draft is not None
    before = window._manual_draft.setup(0).model
    assert before is not None
    before_bounds = before.parameter_intent(offset)
    window._manual_fit_result = presented
    window._manual_execution_diagnostics = (warning,)
    window.dataset_view.set_manual_fit_result(presented)
    window._refresh_manual_fit()

    rows = window.manual_fit_editor.result_parameter_rows
    assert set(rows) == set(before.parameter_references())
    for reference, row in rows.items():
        estimate = presented.parameter_by_reference(reference)
        assert float(row.value.text()) == pytest.approx(estimate.value, rel=5e-4)
    assert rows[offset].uncertainty.text() == "0.012"
    assert rows[slope].uncertainty.text() == "—"
    assert "lower bound" in rows[offset].details.text()
    assert "Covariance unavailable" in (
        window.manual_fit_editor.result_important_label.text()
    )
    assert "structured result warning" in (
        window.manual_fit_editor.result_important_label.text()
    )
    assert float(
        window.manual_fit_editor.result_statistics["chi_square"].text()
    ) == pytest.approx(
        presented.statistics.chi_square,
        rel=5e-4,
    )
    assert window.manual_fit_editor.result_statistics[
        "nominal_degrees_of_freedom"
    ].text() == str(presented.statistics.nominal_degrees_of_freedom)
    assert window.manual_fit_editor.result_summary_label.text().startswith("Converged")
    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    assert "Condition number:" in (
        window.manual_fit_editor.result_diagnostics_label.text()
    )
    assert "Residual RMS:" in window.manual_fit_editor.result_diagnostics_label.text()
    presented_model = window._manual_draft.setup(0).model
    assert presented_model is not None
    assert presented_model.parameter_intent(offset) == before_bounds
    assert "±" not in window.manual_fit_editor.parameter_controls[offset].current.text()

    window.show()
    application.processEvents()
    current = window.manual_fit_editor.parameter_controls[offset].current
    _replace_text_and_tab(application, current, "0.6")
    assert _lifecycle(window) is ManualFitLifecycle.NEEDS_FIT
    assert window._manual_fit_result is None
    assert window.dataset_view.residual_axes is None
    assert window.dataset_view.standardized_residual_line is None
    assert window.manual_fit_editor.result_section.isHidden()
    assert window.manual_fit_editor.result_parameter_rows == {}
    assert window._manual_preview is not None
    window.close()


def test_tied_fit_result_repeats_one_identity_mapped_estimate_per_member(
    application: QApplication,
) -> None:
    window, _project, _sample = _window_with_manual_resolution(application)
    assert window._manual_draft is not None
    window._manual_draft = window._manual_draft.with_model(
        0,
        ManualModelState(
            lorentzians=(
                ManualLorentzianState(
                    area=ManualParameterIntent(0.3, 0.0, 2.0),
                    center=ManualParameterIntent(-0.4, free=False),
                    fwhm=ManualParameterIntent(0.5, free=False),
                ),
                ManualLorentzianState(
                    area=ManualParameterIntent(0.3, 0.0, 2.0),
                    center=ManualParameterIntent(0.4, free=False),
                    fwhm=ManualParameterIntent(0.7, free=False),
                ),
            ),
        ),
    )
    window._refresh_manual_fit()
    model = window._manual_draft.setup(0).model
    assert model is not None
    areas = tuple(
        reference
        for reference in model.parameter_references()
        if reference.family is ParameterFamily.AREA
    )
    _chain_control(window, areas[0]).click()
    _chain_control(window, areas[1]).click()

    window.manual_fit_editor.run_button.click()

    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    assert window._manual_fit_result is not None
    first = window._manual_fit_result.parameter_by_reference(areas[0])
    second = window._manual_fit_result.parameter_by_reference(areas[1])
    assert first is second
    rows = window.manual_fit_editor.result_parameter_rows
    assert rows[areas[0]].value.text() == rows[areas[1]].value.text()
    assert rows[areas[0]].uncertainty.text() == rows[areas[1]].uncertainty.text()
    assert rows[areas[0]].details.text().startswith("Chain 1")
    assert rows[areas[1]].details.text().startswith("Chain 1")
    assert rows[areas[0]].name.text().startswith("L1 ")
    assert rows[areas[1]].name.text().startswith("L2 ")
    window.close()


def test_fit_result_defaults_collapsed_inside_one_scrollable_inspector(
    application: QApplication,
) -> None:
    window, _project, _sample = _window_with_manual_resolution(application)
    _set_runnable_background(window)
    window.resize(960, 420)
    window.show()
    window.manual_fit_editor.run_button.click()
    application.processEvents()

    editor = window.manual_fit_editor
    assert not editor.result_section.isHidden()
    assert not editor.result_toggle_button.isChecked()
    assert editor.result_content.isHidden()
    assert window.inspector_scroll_area.widget() is window.inspector_scroll_body
    assert window.inspector_scroll_body.findChildren(QScrollArea) == []
    parameter_heights = tuple(
        controls.current.height() for controls in editor.parameter_controls.values()
    )

    editor.result_toggle_button.click()
    application.processEvents()

    assert not editor.result_content.isHidden()
    assert window.inspector_scroll_area.verticalScrollBar().maximum() > 0
    assert (
        tuple(
            controls.current.height() for controls in editor.parameter_controls.values()
        )
        == parameter_heights
    )
    assert all(height >= 20 for height in parameter_heights)
    window.close()


def test_fit_result_redraw_leaves_bright_heatmap_highlight_unchanged(
    application: QApplication,
) -> None:
    window, _project, _sample = _window_with_manual_resolution(application)
    _set_runnable_background(window)
    view = window.dataset_view
    highlight = view.overview_active_highlight
    assert highlight is not None
    meshes = view.overview_meshes
    mesh_values = tuple(np.array(mesh.get_array(), copy=True) for mesh in meshes)
    frozen_style = (
        highlight.get_x(),
        highlight.get_width(),
        highlight.get_facecolor(),
        highlight.get_edgecolor(),
        highlight.get_alpha(),
        highlight.get_zorder(),
    )
    facecolor = np.asarray(highlight.get_facecolor(), dtype=float)
    edgecolor = np.asarray(highlight.get_edgecolor(), dtype=float)
    assert facecolor[:3] == pytest.approx((1.0, 1.0, 1.0))
    assert facecolor[-1] == pytest.approx(OVERVIEW_ACTIVE_ALPHA)
    assert edgecolor[:3] == pytest.approx((1.0, 1.0, 1.0))
    assert edgecolor[-1] == pytest.approx(1.0)
    assert OVERVIEW_ACTIVE_COLOR == "#ffffff"

    window.manual_fit_editor.run_button.click()
    result = window._manual_fit_result
    assert result is not None
    assert view.overview_active_highlight is highlight
    assert view.overview_meshes == meshes
    for mesh, values in zip(view.overview_meshes, mesh_values, strict=True):
        np.testing.assert_array_equal(mesh.get_array(), values)

    view.set_manual_fit_result(result)

    assert view.overview_active_highlight is highlight
    assert (
        highlight.get_x(),
        highlight.get_width(),
        highlight.get_facecolor(),
        highlight.get_edgecolor(),
        highlight.get_alpha(),
        highlight.get_zorder(),
    ) == frozen_style
    view.set_current_group(1)
    moved = view.overview_active_highlight
    assert moved is not None
    assert (moved.get_x(), moved.get_x() + moved.get_width()) == (
        view.overview_x_cell_bounds[1]
    )
    assert moved.get_facecolor() == highlight.get_facecolor()
    assert moved.get_alpha() == highlight.get_alpha()
    window.close()


def test_idle_spectrum_drag_zooms_live_in_either_direction_and_tiny_drag_is_safe(
    application: QApplication,
) -> None:
    window, _project, _sample = _window_with_sample(application)
    view = window.dataset_view
    view.canvas.draw()  # type: ignore[no-untyped-call]

    view._on_spectrum_button_press(
        _spectrum_pointer_event(window, -1.5, 1.2, pressed=True),
    )
    rectangle = view._spectrum_zoom_rectangle
    assert rectangle is not None
    view._on_spectrum_mouse_motion(_spectrum_pointer_event(window, 1.0, 2.8))
    assert rectangle.get_width() == pytest.approx(2.5)
    assert rectangle.get_height() == pytest.approx(1.6)
    view._on_spectrum_button_release(_spectrum_pointer_event(window, 1.0, 2.8))
    assert view._spectrum_zoom_rectangle is None
    assert view.spectrum_axes is not None
    assert view.spectrum_axes.get_xlim() == pytest.approx((-1.5, 1.0))
    assert view.spectrum_axes.get_ylim() == pytest.approx((1.2, 2.8))

    view.reset_view()
    view.canvas.draw()  # type: ignore[no-untyped-call]
    view._on_spectrum_button_press(
        _spectrum_pointer_event(window, 1.3, 2.7, pressed=True),
    )
    view._on_spectrum_mouse_motion(_spectrum_pointer_event(window, -1.2, 1.1))
    view._on_spectrum_button_release(_spectrum_pointer_event(window, -1.2, 1.1))
    assert view.spectrum_axes is not None
    assert view.spectrum_axes.get_xlim() == pytest.approx((-1.2, 1.3))
    assert view.spectrum_axes.get_ylim() == pytest.approx((1.1, 2.7))

    limits_before_tiny_drag = (
        view.spectrum_axes.get_xlim(),
        view.spectrum_axes.get_ylim(),
    )
    view._on_spectrum_button_press(
        _spectrum_pointer_event(window, 0.0, 2.0, pressed=True),
    )
    view._on_spectrum_button_release(
        _spectrum_pointer_event(window, 0.00001, 2.00001),
    )
    assert view.spectrum_axes.get_xlim() == limits_before_tiny_drag[0]
    assert view.spectrum_axes.get_ylim() == limits_before_tiny_drag[1]
    window.close()


def test_spectrum_zoom_is_display_only_respects_owners_and_resets(
    application: QApplication,
) -> None:
    window, _project, sample = _window_with_manual_resolution(application)
    _set_runnable_background(window)
    window.manual_fit_editor.run_button.click()
    view = window.dataset_view
    view.canvas.draw()  # type: ignore[no-untyped-call]
    draft = window._manual_draft
    result = window._manual_fit_result
    lifecycle = _lifecycle(window)
    selection = view.selection
    arrays = tuple(
        (spectrum.energy.copy(), spectrum.intensity.copy(), spectrum.uncertainty.copy())
        for spectrum in sample.dataset.spectra
    )

    view.begin_manual_component_interaction(ManualComponentKind.ELASTIC)
    view._on_spectrum_button_press(
        _spectrum_pointer_event(window, -1.0, 1.2, pressed=True),
    )
    assert view._spectrum_zoom_rectangle is None
    view.cancel_manual_component_interaction()
    view._on_spectrum_button_press(
        _spectrum_pointer_event(window, -1.0, 1.2, pressed=True),
    )
    assert view._spectrum_zoom_rectangle is not None
    view._cancel_spectrum_zoom()

    view.set_mask_tool("rectangle")
    view._on_spectrum_button_press(
        _spectrum_pointer_event(window, -1.0, 1.2, pressed=True),
    )
    assert view._spectrum_zoom_rectangle is None
    assert view._mask_preview_rectangle is not None
    view.set_mask_tool(None)
    view._on_spectrum_button_press(
        _spectrum_pointer_event(window, -1.0, 1.2, pressed=True),
    )
    view._on_spectrum_mouse_motion(_spectrum_pointer_event(window, 1.0, 2.8))
    view._on_spectrum_button_release(_spectrum_pointer_event(window, 1.0, 2.8))

    assert window._manual_draft is draft
    assert window._manual_fit_result is result
    assert _lifecycle(window) is lifecycle is ManualFitLifecycle.CURRENT
    assert view.selection is selection
    assert view.residual_axes is not None
    assert view.spectrum_axes is not None
    assert view.residual_axes.get_shared_x_axes().joined(
        view.spectrum_axes,
        view.residual_axes,
    )
    for spectrum, snapshot in zip(sample.dataset.spectra, arrays, strict=True):
        np.testing.assert_array_equal(spectrum.energy, snapshot[0])
        np.testing.assert_array_equal(spectrum.intensity, snapshot[1])
        np.testing.assert_array_equal(spectrum.uncertainty, snapshot[2])

    zoomed_limits = view.spectrum_axes.get_xlim()
    view._on_spectrum_button_press(
        _spectrum_pointer_event(
            window,
            0.0,
            2.0,
            pressed=True,
            double_click=True,
        ),
    )
    assert view.spectrum_axes is not None
    assert view.spectrum_axes.get_xlim() != zoomed_limits
    assert window._manual_draft is draft
    assert window._manual_fit_result is result
    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    assert view.residual_axes is not None
    window.close()


def test_manual_fit_collapse_preserves_current_state_and_cancels_only_provisional(
    application: QApplication,
) -> None:
    window, _project, _sample = _window_with_manual_resolution(application)
    offset = _set_runnable_background(window)
    window.manual_fit_editor.run_button.click()
    draft = window._manual_draft
    result = window._manual_fit_result
    assert draft is not None
    assert result is not None
    intent = draft.setup(0).model
    assert intent is not None
    stored_offset = intent.parameter_intent(offset)

    window._begin_manual_component_interaction(ManualComponentKind.LORENTZIAN)
    window.dataset_view._on_spectrum_button_press(
        _spectrum_event(window, 0.0, 1.5, pressed=True),
    )
    window.dataset_view._on_spectrum_mouse_motion(
        _spectrum_event(window, 1.8, 1.6),
    )
    assert window._pending_manual_interaction is not None
    assert window.dataset_view._manual_pending_evaluation is not None

    window.manual_fit_editor.collapse_button.click()

    assert window.manual_fit_editor.body.isHidden()
    assert not window.manual_fit_editor.isHidden()
    assert window._pending_manual_interaction is None
    assert window.dataset_view._manual_pending_evaluation is None
    assert window._manual_draft is draft
    assert window._manual_fit_result is result
    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    assert window.dataset_view.residual_axes is not None
    current_model = draft.setup(0).model
    assert current_model is not None
    assert current_model.parameter_intent(offset) == stored_offset

    window.manual_fit_editor.collapse_button.click()

    assert not window.manual_fit_editor.body.isHidden()
    assert window._manual_draft is draft
    assert window._manual_fit_result is result
    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    window.close()


def test_clear_model_confirms_current_group_only_and_keeps_manual_task_open(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, project, _sample = _window_with_manual_resolution(application)
    assert window._manual_draft is not None
    other_group_model = ManualModelState(
        background=BackgroundModel.LINEAR,
        b0=ManualParameterIntent(0.2),
        b1=ManualParameterIntent(0.0, free=False),
    )
    window._manual_draft = window._manual_draft.with_model(1, other_group_model)
    current_group_model = ManualModelState(
        energy_shift=ManualParameterIntent(0.0, free=False),
        elastic_area=ManualParameterIntent(0.5, free=False),
        lorentzians=(
            ManualLorentzianState(
                area=ManualParameterIntent(0.4, 0.0, 2.0),
                center=ManualParameterIntent(0.0, free=False),
                fwhm=ManualParameterIntent(0.6, free=False),
            ),
        ),
        background=BackgroundModel.LINEAR,
        b0=ManualParameterIntent(0.2, 0.0, 3.0),
        b1=ManualParameterIntent(0.0, free=False),
    )
    window._manual_draft = window._manual_draft.with_model(0, current_group_model)
    window._refresh_manual_fit()
    window.manual_fit_editor.run_button.click()
    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    assert window._manual_fit_result is not None
    workflow = window._workflow_project_for(project)
    associations = workflow.resolution_associations
    editor = window.manual_fit_editor
    assert editor.clear_model_button.text() == "Clear"
    assert editor.clear_model_button.isEnabled()
    assert editor.clear_model_button.menu() is None
    assert not hasattr(editor, "model_actions_button")

    confirmations: list[tuple[str, str]] = []

    def reject_clear(
        _parent: object,
        title: str,
        message: str,
        **_kwargs: object,
    ) -> bool:
        confirmations.append((title, message))
        return False

    monkeypatch.setattr(main_window_module, "confirm_dialog", reject_clear)
    before = window._manual_draft
    editor.clear_model_button.click()
    assert confirmations == [
        ("Clear Model", "Clear all functions from the current Group?")
    ]
    assert window._manual_draft is before
    assert window._manual_fit_result is not None

    monkeypatch.setattr(
        main_window_module,
        "confirm_dialog",
        lambda *_args, **_kwargs: True,
    )
    editor.clear_model_button.click()

    assert window._manual_draft is not None
    assert window._manual_draft.setup(0).model is None
    assert window._manual_draft.setup(1).model == other_group_model
    assert editor.model_label.text() == "Model    N/A"
    assert not editor.clear_model_button.isEnabled()
    assert not editor.isHidden()
    assert not editor.body.isHidden()
    assert window._manual_owner is not None
    assert window._workflow_project_for(project).resolution_associations == associations
    assert _lifecycle(window) is ManualFitLifecycle.NEEDS_FIT
    assert window._manual_fit_result is None
    assert window.dataset_view.residual_axes is None
    assert editor.result_section.isHidden()
    assert editor.parameter_controls == {}
    assert window._manual_preview is None
    assert window.dataset_view._manual_preview is None
    assert window.dataset_view._manual_pending_evaluation is None
    assert window.dataset_view._manual_fit_result is None
    assert window.dataset_view.spectrum_axes is not None
    assert window.dataset_view._manual_display_evaluation() is None
    assert all(
        line.get_zorder() < 3 for line in window.dataset_view.spectrum_axes.lines
    )

    window.dataset_view.set_current_group(1)
    assert window._manual_preview is not None
    window.dataset_view.set_current_group(0)
    assert window._manual_draft.setup(0).model is None
    assert window._manual_preview is None
    assert window.dataset_view.spectrum_axes is not None
    assert window.dataset_view._manual_display_evaluation() is None
    assert all(
        line.get_zorder() < 3 for line in window.dataset_view.spectrum_axes.lines
    )

    add_elastic = editor.findChild(QToolButton, "addElasticButton")
    assert add_elastic is not None
    add_elastic.click()
    window.dataset_view._on_spectrum_button_press(
        _spectrum_event(window, 0.0, 2.0, pressed=True),
    )
    window.dataset_view._on_spectrum_button_release(
        _spectrum_event(window, 0.0, 2.0),
    )
    assert window._manual_draft.setup(0).model is not None
    window.close()


def test_clear_model_converges_on_repeated_component_removal_lifecycle(
    application: QApplication,
) -> None:
    window, project, _sample = _window_with_manual_resolution(application)
    assert window._manual_draft is not None
    model = ManualModelState(
        energy_shift=ManualParameterIntent(0.0, free=False),
        elastic_area=ManualParameterIntent(0.5, free=False),
        lorentzians=(
            ManualLorentzianState(
                area=ManualParameterIntent(0.4, free=False),
                center=ManualParameterIntent(0.0, free=False),
                fwhm=ManualParameterIntent(0.6, free=False),
            ),
        ),
        background=BackgroundModel.LINEAR,
        b0=ManualParameterIntent(0.2, free=False),
        b1=ManualParameterIntent(0.0, free=False),
    )
    other_group_model = ManualModelState(
        background=BackgroundModel.LINEAR,
        b0=ManualParameterIntent(0.3, free=False),
        b1=ManualParameterIntent(0.0, free=False),
    )
    window._manual_draft = window._manual_draft.with_model(0, model)
    window._manual_draft = window._manual_draft.with_model(1, other_group_model)
    window._refresh_manual_fit()

    assert window.dataset_view.spectrum_axes is not None
    assert any(
        line.get_zorder() >= 3 for line in window.dataset_view.spectrum_axes.lines
    )
    identities = (
        ELASTIC_COMPONENT,
        model.lorentzians[0].identity,
        BACKGROUND_COMPONENT,
    )
    for identity in identities:
        window._remove_manual_component(identity)

    assert window._manual_draft.setup(0).model is None
    assert window._manual_draft.setup(1).model == other_group_model
    assert window.dataset_view.spectrum_axes is not None
    individual_state = (
        window.manual_fit_editor.model_label.text(),
        tuple(window.manual_fit_editor.parameter_controls),
        window._manual_preview,
        window._manual_fit_result,
        window._manual_fit_lifecycle,
        window.dataset_view._manual_display_evaluation(),
        window.dataset_view.residual_axes,
        tuple(
            line.get_zorder()
            for line in window.dataset_view.spectrum_axes.lines
            if line.get_zorder() >= 3
        ),
    )

    window._manual_draft = window._manual_draft.with_model(0, model)
    window._refresh_manual_fit()
    assert window.clear_manual_model(confirmed=True)
    assert window._manual_draft.setup(0).model is None
    assert window._manual_draft.setup(1).model == other_group_model
    assert window.dataset_view.spectrum_axes is not None
    clear_state = (
        window.manual_fit_editor.model_label.text(),
        tuple(window.manual_fit_editor.parameter_controls),
        window._manual_preview,
        window._manual_fit_result,
        window._manual_fit_lifecycle,
        window.dataset_view._manual_display_evaluation(),
        window.dataset_view.residual_axes,
        tuple(
            line.get_zorder()
            for line in window.dataset_view.spectrum_axes.lines
            if line.get_zorder() >= 3
        ),
    )
    assert clear_state == individual_state
    assert (
        window.workspace.dataset_analysis_state_for(project, _sample)
        is DatasetAnalysisState.PARTIALLY_FIT
    )

    window.dataset_view.set_current_group(1)
    assert window._manual_preview is not None
    window.dataset_view.set_current_group(0)
    assert window.dataset_view._manual_display_evaluation() is None
    assert window.dataset_view.spectrum_axes is not None
    assert all(
        line.get_zorder() < 3 for line in window.dataset_view.spectrum_axes.lines
    )

    window.dataset_view.set_current_group(1)
    assert window.clear_manual_model(confirmed=True)
    assert (
        window.workspace.dataset_analysis_state_for(project, _sample)
        is DatasetAnalysisState.READY
    )
    window.close()


def test_visible_clear_and_real_confirmation_remove_rendered_model(
    application: QApplication,
) -> None:
    window, _project, _sample = _window_with_manual_resolution(application)
    assert window._manual_draft is not None
    other_group_model = ManualModelState(
        background=BackgroundModel.LINEAR,
        b0=ManualParameterIntent(0.3, free=False),
        b1=ManualParameterIntent(0.0, free=False),
    )
    current_group_model = ManualModelState(
        energy_shift=ManualParameterIntent(0.0, free=False),
        elastic_area=ManualParameterIntent(0.5, free=False),
        lorentzians=(
            ManualLorentzianState(
                area=ManualParameterIntent(0.4, 0.0, 2.0),
                center=ManualParameterIntent(0.0, free=False),
                fwhm=ManualParameterIntent(0.6, free=False),
            ),
        ),
        background=BackgroundModel.LINEAR,
        b0=ManualParameterIntent(0.2, 0.0, 3.0),
        b1=ManualParameterIntent(0.0, free=False),
    )
    window._manual_draft = window._manual_draft.with_model(0, current_group_model)
    window._manual_draft = window._manual_draft.with_model(1, other_group_model)
    window._refresh_manual_fit()
    window.manual_fit_editor.run_button.click()
    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    assert window._manual_fit_result is not None
    assert window.dataset_view.residual_axes is not None
    assert window.dataset_view.spectrum_axes is not None
    assert any(
        line.get_zorder() >= 3 for line in window.dataset_view.spectrum_axes.lines
    )

    window.show()
    window.set_inspector_visible(True)
    application.processEvents()
    clear_button = window.manual_fit_editor.clear_model_button
    assert clear_button.isVisible()
    seen_dialogs: list[QDialog] = []

    def confirm_clear() -> None:
        dialog = application.activeModalWidget()
        assert isinstance(dialog, QDialog)
        assert dialog.windowTitle() == "Clear Model"
        seen_dialogs.append(dialog)
        confirm_button = next(
            button
            for button in dialog.findChildren(QPushButton)
            if button.text() == "Clear Model"
        )
        QTest.mouseClick(confirm_button, Qt.MouseButton.LeftButton)

    QTimer.singleShot(0, confirm_clear)
    QTest.mouseClick(clear_button, Qt.MouseButton.LeftButton)
    application.processEvents()
    window.dataset_view.canvas.draw()  # type: ignore[no-untyped-call]

    assert seen_dialogs
    assert window._manual_draft.setup(0).model is None
    assert window._manual_draft.setup(1).model == other_group_model
    assert window.manual_fit_editor.model_label.text() == "Model    N/A"
    assert window.manual_fit_editor.parameter_controls == {}
    assert window._manual_preview is None
    assert window._manual_fit_result is None
    assert _lifecycle(window) is ManualFitLifecycle.NEEDS_FIT
    assert window.dataset_view._manual_preview is None
    assert window.dataset_view._manual_pending_evaluation is None
    assert window.dataset_view._manual_fit_result is None
    assert window.dataset_view._manual_display_evaluation() is None
    assert window.dataset_view.residual_axes is None
    assert window.dataset_view.standardized_residual_line is None
    assert window.manual_fit_editor.result_section.isHidden()
    assert window.dataset_view.spectrum_axes is not None
    assert any(
        line.get_zorder() < 3 for line in window.dataset_view.spectrum_axes.lines
    )
    assert all(
        line.get_zorder() < 3 for line in window.dataset_view.spectrum_axes.lines
    )

    window.dataset_view.set_current_group(1)
    assert window._manual_preview is not None
    window.dataset_view.set_current_group(0)
    assert window._manual_preview is None
    assert window.dataset_view._manual_display_evaluation() is None
    assert window.dataset_view.spectrum_axes is not None
    assert all(
        line.get_zorder() < 3 for line in window.dataset_view.spectrum_axes.lines
    )
    window.close()


def test_reopening_exact_current_fitted_dataset_is_a_coherent_noop(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_manual_resolution(application)
    view = window.dataset_view
    view.set_current_group(1)
    _set_runnable_background(window)
    window.manual_fit_editor.run_button.click()
    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    assert window._manual_fit_result is not None
    view.canvas.draw()  # type: ignore[no-untyped-call]
    view._on_spectrum_button_press(
        _spectrum_pointer_event(window, -1.0, 1.2, pressed=True),
    )
    view._on_spectrum_mouse_motion(_spectrum_pointer_event(window, 1.0, 2.8))
    view._on_spectrum_button_release(_spectrum_pointer_event(window, 1.0, 2.8))

    draft = window._manual_draft
    result = window._manual_fit_result
    selection = view.selection
    spectrum_axes = view.spectrum_axes
    residual_axes = view.residual_axes
    x_limits = view._spectrum_x_limits
    y_limits = view._spectrum_y_limits

    assert window.open_dataset(project, sample)

    assert view.current_group_index == 1
    assert window._manual_draft is draft
    assert window._manual_fit_result is result
    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    assert view._manual_fit_result is result
    assert view.selection is selection
    assert view.spectrum_axes is spectrum_axes
    assert view.residual_axes is residual_axes
    assert view._spectrum_x_limits == x_limits
    assert view._spectrum_y_limits == y_limits
    assert not window.manual_fit_editor.result_section.isHidden()
    window.close()


def test_resolution_invalidation_requires_active_sample_and_changed_association(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    sample_a = window.workspace.add_dataset(project, _sample())
    sample_b = window.workspace.add_dataset(project, _sample())
    resolution_a = window.workspace.add_dataset(
        project,
        _resolution(sample_a.dataset),
    )
    resolution_b = window.workspace.add_dataset(
        project,
        _resolution(sample_b.dataset),
    )
    replacement_a = window.workspace.add_dataset(
        project,
        _resolution(sample_a.dataset),
    )
    window.open_dataset(project, sample_a)
    assert window.apply_resolution_for_sample(
        project,
        sample_a,
        resolution_a,
        replace_confirmed=True,
    )
    window.show_manual_fit(project, sample_a)
    _set_runnable_background(window)
    window.manual_fit_editor.run_button.click()
    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    result_a = window._manual_fit_result
    assert result_a is not None
    window.dataset_view.set_current_group(1)
    _set_runnable_background(window)
    window.manual_fit_editor.run_button.click()
    result_b = window._manual_fit_result
    assert result_b is not None
    session = window._active_manual_session
    assert session is not None
    window.dataset_view.set_current_group(0)
    window._begin_manual_component_interaction(ManualComponentKind.ELASTIC)
    pending = window._pending_manual_interaction
    assert pending is not None

    assert window.apply_resolution_for_sample(
        project,
        sample_b,
        resolution_b,
        replace_confirmed=True,
    )
    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    assert window._manual_fit_result is result_a
    assert window.dataset_view._manual_fit_result is result_a
    assert session.execution_state(1).fit_result is result_b
    assert window._pending_manual_interaction is pending

    assert window.apply_resolution_for_sample(
        project,
        sample_a,
        resolution_a,
        replace_confirmed=True,
    )
    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    assert window._manual_fit_result is result_a
    assert window.dataset_view._manual_fit_result is result_a
    assert session.execution_state(1).fit_result is result_b
    assert window._pending_manual_interaction is pending

    assert window.apply_resolution_for_sample(
        project,
        sample_a,
        replacement_a,
        replace_confirmed=True,
    )
    assert _lifecycle(window) is ManualFitLifecycle.NEEDS_FIT
    assert window._manual_fit_result is None
    assert window.dataset_view._manual_fit_result is None
    assert window.dataset_view.residual_axes is None
    assert window._pending_manual_interaction is None
    assert session.execution_state(0).lifecycle is ManualFitLifecycle.NEEDS_FIT
    assert session.execution_state(1).lifecycle is ManualFitLifecycle.NEEDS_FIT
    assert session.execution_state(0).fit_result is None
    assert session.execution_state(1).fit_result is None
    window.close()


@pytest.mark.parametrize("locked", [False, True])
def test_rectangle_y_zoom_cross_group_scope_follows_lock(
    application: QApplication,
    locked: bool,
) -> None:
    window, project, sample = _window_with_sample(application)
    view = window.dataset_view
    workflow = window._workflow_project_for(project)
    arrays = tuple(
        (item.energy.copy(), item.intensity.copy(), item.uncertainty.copy())
        for item in sample.dataset.spectra
    )
    assert view.spectrum_axes is not None
    default_y = view.spectrum_axes.get_ylim()
    view.set_y_range_locked(locked)
    view.canvas.draw()  # type: ignore[no-untyped-call]
    view._on_spectrum_button_press(
        _spectrum_pointer_event(window, -1.0, 1.2, pressed=True),
    )
    view._on_spectrum_mouse_motion(_spectrum_pointer_event(window, 1.0, 2.8))
    view._on_spectrum_button_release(_spectrum_pointer_event(window, 1.0, 2.8))
    assert view.spectrum_axes is not None
    zoomed_x = view.spectrum_axes.get_xlim()
    zoomed_y = view.spectrum_axes.get_ylim()

    view.set_current_group(1)

    assert view.spectrum_axes is not None
    assert view.spectrum_axes.get_xlim() == pytest.approx(zoomed_x)
    expected_y = zoomed_y if locked else default_y
    assert view.spectrum_axes.get_ylim() == pytest.approx(expected_y)
    assert window._workflow_project_for(project) is workflow
    assert window._open_dataset is sample
    for spectrum, snapshot in zip(sample.dataset.spectra, arrays, strict=True):
        np.testing.assert_array_equal(spectrum.energy, snapshot[0])
        np.testing.assert_array_equal(spectrum.intensity, snapshot[1])
        np.testing.assert_array_equal(spectrum.uncertainty, snapshot[2])
    window.close()


def test_unlocked_wheel_y_zoom_does_not_cross_groups(
    application: QApplication,
) -> None:
    window, _project, _sample = _window_with_sample(application)
    view = window.dataset_view
    axes = view.spectrum_axes
    assert axes is not None
    default_y = axes.get_ylim()
    view.set_y_range_locked(False)
    view.canvas.draw()  # type: ignore[no-untyped-call]
    x_pixel, y_pixel = axes.transData.transform((0.0, 2.0))
    view._on_scroll(
        cast(
            MouseEvent,
            SimpleNamespace(
                button="up",
                inaxes=axes,
                xdata=0.0,
                ydata=2.0,
                x=x_pixel,
                y=y_pixel,
            ),
        ),
    )
    assert axes.get_ylim() != pytest.approx(default_y)

    view.set_current_group(1)

    assert view.spectrum_axes is not None
    assert view.spectrum_axes.get_ylim() == pytest.approx(default_y)
    window.close()


def test_workspace_manual_model_progress_and_inspector_heading(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_sample(application)
    workspace = window.workspace

    def sample_state_tooltip() -> str:
        project_item = workspace.tree.topLevelItem(0)
        assert project_item is not None
        data_item = project_item.child(0)
        assert data_item is not None
        sample_item = data_item.child(project.datasets.index(sample))
        assert sample_item is not None
        return sample_item.toolTip(1)

    assert (
        workspace.dataset_analysis_state_for(project, sample)
        is DatasetAnalysisState.READY
    )
    assert "Ready for Fit" in sample_state_tooltip()
    window.show_manual_fit(project, sample)
    assert window.manual_fit_editor.title.text() == "Fitting Parameters"
    assert window.manual_fit_button.text() == "Manual Fit"
    assert window.manual_fit_action.text() == "Manual Fit…"
    assert (
        workspace.dataset_analysis_state_for(project, sample)
        is DatasetAnalysisState.READY
    )

    assert window._manual_draft is not None
    other_group_model = ManualModelState(
        background=BackgroundModel.LINEAR,
        b0=ManualParameterIntent(0.2),
        b1=ManualParameterIntent(0.0, free=False),
    )
    window._manual_draft = window._manual_draft.with_model(1, other_group_model)
    window._refresh_manual_fit()
    assert (
        workspace.dataset_analysis_state_for(project, sample)
        is DatasetAnalysisState.PARTIALLY_FIT
    )
    assert "Fitting has started" in sample_state_tooltip()
    view = window.dataset_view
    view.set_current_group(1)
    assert (
        workspace.dataset_analysis_state_for(project, sample)
        is DatasetAnalysisState.PARTIALLY_FIT
    )
    view.set_current_group(0)
    _set_runnable_background(window)
    assert window.clear_manual_model(confirmed=True)
    assert (
        workspace.dataset_analysis_state_for(project, sample)
        is DatasetAnalysisState.PARTIALLY_FIT
    )
    view.set_current_group(1)
    assert window.clear_manual_model(confirmed=True)
    assert (
        workspace.dataset_analysis_state_for(project, sample)
        is DatasetAnalysisState.READY
    )
    assert "Ready for Fit" in sample_state_tooltip()
    window.close()

    blocked_window = MainWindow()
    blocked_project = blocked_window.workspace.new_project()
    blocked_sample = blocked_window.workspace.add_dataset(
        blocked_project,
        replace(_sample(), q_bins=None),
    )
    blocked_window.open_dataset(blocked_project, blocked_sample)
    blocked_window.show_manual_fit(blocked_project, blocked_sample)
    assert blocked_window._manual_draft is not None
    blocked_window._manual_draft = blocked_window._manual_draft.with_model(
        0,
        other_group_model,
    )
    blocked_window._refresh_manual_fit()
    assert (
        blocked_window.workspace.dataset_analysis_state_for(
            blocked_project,
            blocked_sample,
        )
        is DatasetAnalysisState.REQUIRED
    )
    blocked_window.close()


def test_samples_retain_independent_manual_sessions_across_navigation(
    application: QApplication,
) -> None:
    window = MainWindow()
    project = window.workspace.new_project()
    sample_a = window.workspace.add_dataset(project, _sample())
    sample_b = window.workspace.add_dataset(project, _sample())
    resolution_a = window.workspace.add_dataset(
        project,
        _resolution(sample_a.dataset),
    )
    window.open_dataset(project, sample_a)
    assert window.apply_resolution_for_sample(
        project,
        sample_a,
        resolution_a,
        replace_confirmed=True,
    )
    window.show_manual_fit(project, sample_a)
    window.dataset_view.set_current_group(1)
    assert window._manual_draft is not None
    shared_center = "shared-center"
    window._manual_draft = window._manual_draft.with_model(
        1,
        ManualModelState(
            lorentzians=(
                ManualLorentzianState(
                    area=ManualParameterIntent(0.3, 0.05, 1.5, True, True),
                    center_group=shared_center,
                    fwhm=ManualParameterIntent(0.5, free=False),
                ),
                ManualLorentzianState(
                    area=ManualParameterIntent(0.3, 0.05, 1.5, True, True),
                    center_group=shared_center,
                    fwhm=ManualParameterIntent(0.8, free=False),
                ),
            ),
            center_groups=(
                ManualCenterGroupState(
                    shared_center,
                    ManualParameterIntent(0.0, free=False),
                ),
            ),
        ),
    )
    window._refresh_manual_fit()
    model = window._manual_draft.setup(1).model
    assert model is not None
    areas = tuple(
        reference
        for reference in model.parameter_references()
        if reference.family is ParameterFamily.AREA
    )
    _chain_control(window, areas[0]).click()
    _chain_control(window, areas[1]).click()
    window.manual_fit_editor.run_button.click()
    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    assert window._manual_fit_result is not None
    assert window.dataset_view.set_y_scale("log")
    assert window.dataset_view.y_scale == "log"
    assert window.dataset_view.y_range_locked
    window.manual_fit_editor.result_toggle_button.click()
    assert window.manual_fit_editor.result_toggle_button.isChecked()

    owner_a = (project, sample_a.workflow_dataset_id)
    session_a = window._manual_sessions[owner_a]
    draft_a = window._manual_draft
    result_a = window._manual_fit_result
    fitted_model_a = draft_a.setup(1).model
    associations = window._workflow_project_for(project).resolution_associations
    assert fitted_model_a is not None
    assert fitted_model_a.center_groups[0].group_id == shared_center
    assert fitted_model_a.parameter_ties
    assert (
        window.workspace.dataset_analysis_state_for(project, sample_a)
        is DatasetAnalysisState.PARTIALLY_FIT
    )

    assert window.open_dataset(project, sample_b)

    assert window._active_manual_session is None
    assert window.dataset_view.y_scale == "linear"
    assert window.manual_fit_editor.isHidden()
    assert session_a.draft is draft_a
    assert session_a.group_index == 1
    assert session_a.lifecycle is ManualFitLifecycle.CURRENT
    assert session_a.fit_result is result_a
    assert (
        window.workspace.dataset_analysis_state_for(project, sample_a)
        is DatasetAnalysisState.PARTIALLY_FIT
    )
    assert (
        window.workspace.dataset_analysis_state_for(project, sample_b)
        is DatasetAnalysisState.READY
    )

    window.show_manual_fit(project, sample_b)
    _set_runnable_background(window)
    owner_b = (project, sample_b.workflow_dataset_id)
    session_b = window._manual_sessions[owner_b]
    draft_b = window._manual_draft
    assert draft_b is not draft_a
    assert (
        window.workspace.dataset_analysis_state_for(project, sample_b)
        is DatasetAnalysisState.PARTIALLY_FIT
    )

    assert window.open_dataset(project, sample_a)

    assert window._active_manual_session is session_a
    assert window.dataset_view.y_scale == "log"
    assert window.dataset_view.y_range_locked
    assert window.dataset_view.current_group_index == 1
    assert window._manual_draft is draft_a
    assert window._manual_draft.setup(1).model is fitted_model_a
    assert window._manual_fit_result is result_a
    assert _lifecycle(window) is ManualFitLifecycle.CURRENT
    assert window.dataset_view._manual_fit_result is result_a
    assert window.dataset_view.residual_axes is not None
    assert window.dataset_view.standardized_residual_line is not None
    assert not window.manual_fit_editor.result_section.isHidden()
    assert window.manual_fit_editor.result_toggle_button.isChecked()
    assert not window.manual_fit_editor.result_content.isHidden()
    restored_model = window._manual_draft.setup(1).model
    assert restored_model is not None
    assert restored_model.parameter_ties == fitted_model_a.parameter_ties
    assert restored_model.center_groups == fitted_model_a.center_groups
    for reference in restored_model.parameter_references():
        assert restored_model.parameter_intent(reference) == (
            fitted_model_a.parameter_intent(reference)
        )
    assert window._workflow_project_for(project).resolution_associations == associations

    window._manual_draft = window._manual_draft.with_model(
        0,
        ManualModelState(
            background=BackgroundModel.LINEAR,
            b0=ManualParameterIntent(0.2),
            b1=ManualParameterIntent(0.0, free=False),
        ),
    )
    window._refresh_manual_fit()
    assert window.clear_manual_model(confirmed=True)
    assert window._manual_draft.setup(1).model is None
    assert window._manual_draft.setup(0).model is not None
    assert (
        window.workspace.dataset_analysis_state_for(project, sample_a)
        is DatasetAnalysisState.PARTIALLY_FIT
    )
    window.dataset_view.set_current_group(0)
    assert window.clear_manual_model(confirmed=True)
    assert (
        window.workspace.dataset_analysis_state_for(project, sample_a)
        is DatasetAnalysisState.READY
    )
    assert window._workflow_project_for(project).resolution_associations == associations

    assert window.open_dataset(project, sample_b)
    assert window._active_manual_session is session_b
    assert window.dataset_view.y_scale == "linear"
    assert window._manual_draft is draft_b
    assert window._manual_draft.setup(0).model is not None
    assert window._manual_draft.setup(1).model is None
    assert (
        window.workspace.dataset_analysis_state_for(project, sample_a)
        is DatasetAnalysisState.READY
    )
    assert (
        window.workspace.dataset_analysis_state_for(project, sample_b)
        is DatasetAnalysisState.PARTIALLY_FIT
    )
    window.close()


def test_log_masks_nonpositive_manual_curves_without_scientific_mutation(
    application: QApplication,
) -> None:
    window, project, sample = _window_with_manual_resolution(application)
    assert window._manual_draft is not None
    window._manual_draft = window._manual_draft.with_model(
        0,
        ManualModelState(
            background=BackgroundModel.LINEAR,
            b0=ManualParameterIntent(-0.5, free=False),
            b1=ManualParameterIntent(0.0, free=False),
        ),
    )
    window._refresh_manual_fit()
    draft = window._manual_draft
    preview = window._manual_preview
    lifecycle = _lifecycle(window)
    workflow = window._workflow_project_for(project)
    selection = window.dataset_view.selection
    arrays = tuple(
        (item.energy.copy(), item.intensity.copy(), item.uncertainty.copy())
        for item in sample.dataset.spectra
    )
    lock_state = window.dataset_view.y_range_locked
    assert preview is not None

    assert window.dataset_view.set_y_scale("log")

    assert window.dataset_view.y_scale == "log"
    assert window.dataset_view.y_range_locked is lock_state
    assert window._manual_draft is draft
    assert window._manual_preview is preview
    assert _lifecycle(window) is lifecycle
    assert window._manual_fit_result is None
    assert window._workflow_project_for(project) is workflow
    assert window.dataset_view.selection is selection
    assert window.dataset_view.spectrum_axes is not None
    model_lines = tuple(
        line
        for line in window.dataset_view.spectrum_axes.lines
        if line.get_zorder() >= 3
    )
    assert model_lines
    assert all(
        np.ma.getmaskarray(np.ma.asarray(line.get_ydata())).all()
        for line in model_lines
    )
    for spectrum, snapshot in zip(sample.dataset.spectra, arrays, strict=True):
        np.testing.assert_array_equal(spectrum.energy, snapshot[0])
        np.testing.assert_array_equal(spectrum.intensity, snapshot[1])
        np.testing.assert_array_equal(spectrum.uncertainty, snapshot[2])
    window.close()
