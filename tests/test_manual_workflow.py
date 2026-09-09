"""Application/workflow tests for Single-Q Manual Fit prerequisites."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import ezqens.fitting as fitting_module
import ezqens.workflow.manual as manual_workflow_module
from ezqens.domain import (
    DiagnosticSeverity,
    FractionalCoverage,
    FractionalCoverageOrigin,
    QBins,
    ReducedDataset,
    Spectrum,
    SpectrumRole,
)
from ezqens.fitting import (
    BACKGROUND_COMPONENT,
    ELASTIC_COMPONENT,
    BackgroundModel,
    ComponentFamily,
    ComponentIdentity,
    ElasticInteractionSeed,
    FitResult,
    FittingError,
    LorentzianInteractionSeed,
    ManualFitDiagnosticCode,
    ManualFitReadiness,
    ManualFitReadinessDiagnostic,
    ManualModelPreview,
    ManualParameterIntent,
    ParameterFamily,
    ParameterReference,
    SpectralModelDefinition,
)
from ezqens.preprocessing import FittingSelection, detect_edge_padding
from ezqens.resolution import PreparedResolution
from ezqens.workflow import (
    ManualCenterGroupState,
    ManualComponentKind,
    ManualLorentzianState,
    ManualModelState,
    ManualParameterEdit,
    ManualParameterTieState,
    PendingManualComponentPreview,
    ProjectDataset,
    WorkflowDiagnosticCode,
    WorkflowError,
    WorkflowProject,
    add_project_dataset,
    adopt_single_q_auto_fit_candidate,
    applied_resolution,
    apply_manual_setup_to_all_groups,
    apply_resolution,
    begin_component_interaction,
    clone_manual_setup_to_group,
    commit_fitting_selection,
    complete_background_interaction,
    complete_elastic_interaction,
    complete_lorentzian_interaction,
    create_parameter_tie,
    create_project,
    join_center_group,
    join_parameter_tie,
    manual_workflow_readiness,
    materialize_manual_setup,
    open_manual_fit_draft,
    preflight_apply_resolution,
    preview_manual_fit,
    preview_pending_background_interaction,
    preview_pending_elastic_interaction,
    preview_pending_lorentzian_interaction,
    remove_manual_component,
    remove_project_dataset,
    replace_project_dataset,
    rerole_project_dataset,
    resolve_manual_fit_context,
    run_and_adopt_manual_fit,
    run_manual_fit,
    run_single_q_auto_fit,
    untie_center_group_parameter,
    untie_parameter,
    update_manual_parameter,
)


def intent(
    value: float,
    lower: float | None = None,
    upper: float | None = None,
    *,
    free: bool = False,
    bounds_enabled: bool = True,
) -> ManualParameterIntent:
    return ManualParameterIntent(value, lower, upper, free, bounds_enabled)


def make_dataset(
    role: SpectrumRole,
    q_values: tuple[float, ...],
    *,
    energy_by_group: tuple[np.ndarray, ...] | None = None,
    source_reference: str | None = None,
) -> ReducedDataset:
    energies = energy_by_group or tuple(np.linspace(-0.5, 0.5, 101) for _ in q_values)
    spectra = []
    for index, (q_value, energy) in enumerate(zip(q_values, energies, strict=True)):
        if role is SpectrumRole.RESOLUTION:
            values = np.exp(-0.5 * np.square(energy / 0.04))
            uncertainty = np.full(energy.size, 1.0e-3)
        else:
            values = (
                1.0
                + 0.15 * np.exp(-0.5 * np.square(energy / 0.12))
                + 0.01 * index * energy
            )
            uncertainty = np.full(energy.size, 0.02)
        spectra.append(
            Spectrum(
                role=role,
                group_index=index,
                group_label=f"Q={q_value}",
                energy=energy,
                intensity=values,
                uncertainty=uncertainty,
                energy_unit="meV",
                intensity_unit="arb",
                uncertainty_unit="arb",
            )
        )
    return ReducedDataset(
        role=role,
        spectra=tuple(spectra),
        source_reference=source_reference,
        q_bins=QBins.from_q_values(q_values),
    )


def configured_project(
    *,
    q_values: tuple[float, ...] = (0.5, 0.75, 1.0),
    apply: bool = True,
    commit_selection: bool = True,
) -> tuple[WorkflowProject, ProjectDataset, ProjectDataset]:
    project = create_project("test", project_id="project-test")
    sample_data = make_dataset(SpectrumRole.SAMPLE, q_values)
    resolution_data = make_dataset(SpectrumRole.RESOLUTION, q_values)
    project, sample = add_project_dataset(
        project,
        sample_data,
        dataset_id="sample",
    )
    project, resolution = add_project_dataset(
        project,
        resolution_data,
        dataset_id="resolution",
    )
    if commit_selection:
        selection = FittingSelection.uniform(
            sample_data,
            detect_edge_padding(sample_data),
            lower_energy=-0.5,
            upper_energy=0.5,
        )
        project = commit_fitting_selection(project, sample, selection)
    if apply:
        project = apply_resolution(project, sample, resolution)
    return project, sample, resolution


def configured_project_with_distinct_group_coverage() -> tuple[
    WorkflowProject, ProjectDataset, ProjectDataset
]:
    q_values = (0.5, 0.75)
    sample_data = make_dataset(
        SpectrumRole.SAMPLE,
        q_values,
        energy_by_group=(
            np.linspace(-1.5, 1.5, 301),
            np.linspace(-0.2, 0.2, 81),
        ),
    )
    resolution_data = make_dataset(
        SpectrumRole.RESOLUTION,
        q_values,
        energy_by_group=(
            np.linspace(-0.5, 0.5, 201),
            np.linspace(-0.08, 0.08, 65),
        ),
    )
    project = create_project("coverage", project_id="project-coverage")
    project, sample = add_project_dataset(project, sample_data, dataset_id="sample")
    project, resolution = add_project_dataset(
        project,
        resolution_data,
        dataset_id="resolution",
    )
    selection = FittingSelection.uniform(
        sample_data,
        detect_edge_padding(sample_data),
        lower_energy=-1.5,
        upper_energy=1.5,
    )
    project = commit_fitting_selection(project, sample, selection)
    project = apply_resolution(project, sample, resolution)
    return project, sample, resolution


def test_single_q_autofit_adopts_typed_fitted_candidate_without_label_parsing() -> None:
    q_values = (0.5,)
    resolution_data = make_dataset(SpectrumRole.RESOLUTION, q_values)
    resolution_spectrum = resolution_data.spectra[0]
    normalized_resolution = resolution_spectrum.intensity / np.trapezoid(
        resolution_spectrum.intensity,
        resolution_spectrum.energy,
    )
    sample_data = ReducedDataset(
        role=SpectrumRole.SAMPLE,
        spectra=(
            replace(
                make_dataset(SpectrumRole.SAMPLE, q_values).spectra[0],
                intensity=2.0 * normalized_resolution,
            ),
        ),
        q_bins=QBins.from_q_values(q_values),
    )
    project = create_project("autofit", project_id="project-autofit")
    project, sample = add_project_dataset(project, sample_data, dataset_id="sample")
    project, resolution = add_project_dataset(
        project,
        resolution_data,
        dataset_id="resolution",
    )
    project = commit_fitting_selection(
        project,
        sample,
        FittingSelection.uniform(
            sample_data,
            detect_edge_padding(sample_data),
            lower_energy=-0.5,
            upper_energy=0.5,
        ),
    )
    project = apply_resolution(project, sample, resolution)
    draft = open_manual_fit_draft(project, sample)

    outcome = run_single_q_auto_fit(project, draft, 0)

    assert outcome.project_id == project.project_id
    assert outcome.sample_id == sample.dataset_id
    assert outcome.group_index == 0
    assert len(outcome.recommendation.candidate_results) == 9
    chosen = next(
        candidate
        for candidate in outcome.recommendation.candidate_results
        if candidate.success
        and candidate.candidate.background is BackgroundModel.LINEAR
        and candidate.candidate.lorentzian_count == 2
    )
    adoption = adopt_single_q_auto_fit_candidate(
        project,
        draft,
        outcome,
        chosen,
    )

    assert draft.setup(0).model is None
    assert adoption.success
    assert adoption.fit_result is chosen.fit
    assert adoption.adopted_draft is not None
    adopted = adoption.adopted_draft.setup(0).model
    assert adopted is not None
    assert chosen.fit is not None
    fitted = chosen.fit.fitted_model
    assert fitted is not None
    assert fitted.energy_shift is not None
    assert adopted.energy_shift is None
    assert len(adopted.center_groups) == 1
    shared_center = adopted.center_groups[0]
    assert shared_center.intent.current_value == fitted.energy_shift.initial_value
    assert shared_center.intent.user_lower_limit == fitted.energy_shift.lower_bound
    assert shared_center.intent.user_upper_limit == fitted.energy_shift.upper_bound
    assert shared_center.intent.free is fitted.energy_shift.free
    center_references = tuple(
        reference
        for reference in adopted.parameter_references()
        if reference.family is ParameterFamily.CENTER
    )
    assert adopted.center_group_members(shared_center.group_id) == center_references
    assert adopted.elastic_center_group == shared_center.group_id
    assert all(
        component.center is None and component.center_group == shared_center.group_id
        for component in adopted.lorentzians
    )
    assert adopted.elastic_area is not None
    assert adopted.elastic_area.user_lower_limit == 0.0
    assert tuple(item.identity for item in adopted.lorentzians) == tuple(
        item.identity for item in fitted.lorentzians
    )
    assert all(item.area.user_lower_limit == 0.0 for item in adopted.lorentzians)
    assert all(
        item.fwhm.user_lower_limit is not None and item.fwhm.user_lower_limit > 0.0
        for item in adopted.lorentzians
    )
    assert adopted.b0 is not None
    assert fitted.b0 is not None
    assert adopted.b0.current_value == fitted.b0.initial_value
    assert adopted.b0.free is fitted.b0.free
    assert adopted.b0.user_lower_limit is None
    assert adopted.b0.user_upper_limit is None
    assert adopted.b1 is not None
    assert fitted.b1 is not None
    assert adopted.b1.current_value == fitted.b1.initial_value
    assert adopted.b1.free is fitted.b1.free
    assert adopted.b1.user_lower_limit is None
    assert adopted.b1.user_upper_limit is None
    adopted_result = adoption.fit_result
    assert adopted_result is not None
    np.testing.assert_array_equal(
        adopted_result.evaluation.total,
        chosen.fit.evaluation.total,
    )
    np.testing.assert_array_equal(
        adopted_result.standardized_residuals,
        chosen.fit.standardized_residuals,
    )

    updated_center = replace(
        shared_center.intent,
        current_value=shared_center.intent.current_value + 0.01,
        user_lower_limit=-0.2,
        user_upper_limit=0.2,
    )
    edited_draft = update_manual_parameter(
        adoption.adopted_draft,
        0,
        center_references[0],
        ManualParameterEdit(
            current_value=updated_center.current_value,
            user_lower_limit=updated_center.user_lower_limit,
            user_upper_limit=updated_center.user_upper_limit,
            free=updated_center.free,
            user_bounds_enabled=updated_center.user_bounds_enabled,
        ),
    )
    edited = edited_draft.setup(0).model
    assert edited is not None
    assert all(
        edited.parameter_intent(reference) == updated_center
        for reference in center_references
    )
    detached_draft = untie_center_group_parameter(
        edited_draft,
        0,
        center_references[-1],
    )
    detached = detached_draft.setup(0).model
    assert detached is not None
    assert detached.center_group_for(center_references[-1]) is None
    assert detached.parameter_intent(center_references[-1]) == updated_center
    retied_draft = join_center_group(
        detached_draft,
        0,
        shared_center.group_id,
        center_references[-1],
    )
    retied = retied_draft.setup(0).model
    assert retied is not None
    assert retied.center_group_members(shared_center.group_id) == center_references

    prior_model = ManualModelState(
        background=BackgroundModel.CONSTANT,
        b0=ManualParameterIntent(0.25),
    )
    working_draft = draft.with_model(0, prior_model)

    def assert_stale_rejected(changed_project: WorkflowProject) -> None:
        rejected = adopt_single_q_auto_fit_candidate(
            changed_project,
            working_draft,
            outcome,
            chosen,
        )
        assert not rejected.success
        assert rejected.adopted_draft is None
        assert rejected.fit_result is None
        assert rejected.diagnostics[0].code is (
            WorkflowDiagnosticCode.AUTO_FIT_CONTEXT_CHANGED
        )
        assert working_draft.setup(0).model is prior_model

    changed_sample_data = replace(
        sample.dataset,
        spectra=(
            replace(
                sample.dataset.spectra[0],
                intensity=sample.dataset.spectra[0].intensity + 0.01,
            ),
        ),
    )
    changed_sample_project, changed_sample = replace_project_dataset(
        project,
        sample,
        changed_sample_data,
    )
    changed_sample_project = commit_fitting_selection(
        changed_sample_project,
        changed_sample,
        FittingSelection.uniform(
            changed_sample_data,
            detect_edge_padding(changed_sample_data),
            lower_energy=-0.5,
            upper_energy=0.5,
        ),
    )
    assert_stale_rejected(changed_sample_project)

    changed_selection = outcome.scientific_context.selection.with_group_range(
        0,
        lower_energy=-0.4,
        upper_energy=0.4,
    )
    changed_selection_project = commit_fitting_selection(
        project,
        sample,
        changed_selection,
    )
    assert_stale_rejected(changed_selection_project)

    changed_mask = np.array(
        outcome.scientific_context.selection.manual_exclusion_mask(0),
        copy=True,
    )
    retained_indices = np.flatnonzero(
        outcome.scientific_context.selection.retained_mask(0),
    )
    changed_mask[int(retained_indices[retained_indices.size // 2])] = True
    changed_mask_selection = (
        outcome.scientific_context.selection.with_group_manual_exclusion(
            0,
            changed_mask,
        )
    )
    changed_mask_project = commit_fitting_selection(
        project,
        sample,
        changed_mask_selection,
    )
    assert_stale_rejected(changed_mask_project)

    changed_resolution_data = replace(
        resolution.dataset,
        spectra=(
            replace(
                resolution.dataset.spectra[0],
                intensity=resolution.dataset.spectra[0].intensity * 1.01,
            ),
        ),
    )
    changed_resolution_project, _changed_resolution = replace_project_dataset(
        project,
        resolution,
        changed_resolution_data,
    )
    assert_stale_rejected(changed_resolution_project)

    wrong_group = adopt_single_q_auto_fit_candidate(
        project,
        working_draft,
        outcome,
        chosen,
        group_index=1,
    )
    assert not wrong_group.success
    assert wrong_group.adopted_draft is None
    assert wrong_group.diagnostics[0].code is (
        WorkflowDiagnosticCode.AUTO_FIT_CONTEXT_CHANGED
    )
    assert working_draft.setup(0).model is prior_model


def test_autofit_none_and_b0_adoption_use_editable_manual_backgrounds() -> None:
    project, sample, _ = configured_project(q_values=(0.5,))
    draft = open_manual_fit_draft(project, sample)
    outcome = run_single_q_auto_fit(project, draft, 0)

    none_candidate = next(
        candidate
        for candidate in outcome.recommendation.candidate_results
        if candidate.success
        and candidate.candidate.lorentzian_count == 0
        and candidate.candidate.background is BackgroundModel.NONE
    )
    none_adoption = adopt_single_q_auto_fit_candidate(
        project,
        draft,
        outcome,
        none_candidate,
    )
    assert none_adoption.success
    assert none_adoption.adopted_draft is not None
    none_model = none_adoption.adopted_draft.setup(0).model
    assert none_model is not None
    assert none_model.background is BackgroundModel.NONE
    assert none_model.b0 is None
    assert none_model.b1 is None
    assert all(
        reference.family not in (ParameterFamily.OFFSET, ParameterFamily.SLOPE)
        for reference in none_model.parameter_references()
    )

    b0_candidate = next(
        candidate
        for candidate in outcome.recommendation.candidate_results
        if candidate.success
        and candidate.candidate.lorentzian_count == 0
        and candidate.candidate.background is BackgroundModel.CONSTANT
    )
    assert b0_candidate.fit is not None
    fitted_b0 = b0_candidate.fit.fitted_model
    assert fitted_b0 is not None
    assert fitted_b0.background is BackgroundModel.CONSTANT
    assert fitted_b0.b0 is not None
    assert fitted_b0.b1 is None

    b0_adoption = adopt_single_q_auto_fit_candidate(
        project,
        draft,
        outcome,
        b0_candidate,
    )
    assert b0_adoption.success
    assert b0_adoption.adopted_draft is not None
    adopted_draft = b0_adoption.adopted_draft
    adopted_b0 = adopted_draft.setup(0).model
    assert adopted_b0 is not None
    assert adopted_b0.background is BackgroundModel.LINEAR
    assert adopted_b0.b0 is not None
    assert adopted_b0.b0.current_value == fitted_b0.b0.initial_value
    assert adopted_b0.b0.user_lower_limit is None
    assert adopted_b0.b0.user_upper_limit is None
    assert adopted_b0.b0.free is fitted_b0.b0.free
    assert adopted_b0.b1 == ManualParameterIntent(0.0, free=False)
    assert {
        reference.family
        for reference in adopted_b0.parameter_references()
        if reference.component == BACKGROUND_COMPONENT
    } == {ParameterFamily.OFFSET, ParameterFamily.SLOPE}

    materialized = materialize_manual_setup(project, adopted_draft, 0)
    assert materialized.fit_model.background is BackgroundModel.LINEAR
    assert materialized.fit_model.b0 is not None
    assert materialized.fit_model.b1 is not None
    assert materialized.fit_model.b1.initial_value == 0.0
    assert not materialized.fit_model.b1.free
    readiness = manual_workflow_readiness(project, adopted_draft, 0)
    assert readiness.runnable
    preview = preview_manual_fit(project, adopted_draft, 0)
    np.testing.assert_allclose(
        preview.retained_evaluation.total,
        b0_candidate.fit.evaluation.total,
        rtol=1.0e-13,
        atol=1.0e-13,
    )

    manual_result = run_manual_fit(project, adopted_draft, 0)
    assert manual_result.diagnostics.optimizer_success
    assert manual_result.fitted_model is not None
    assert manual_result.fitted_model.background is BackgroundModel.LINEAR
    assert manual_result.fitted_model.b1 is not None
    assert manual_result.fitted_model.b1.initial_value == 0.0
    assert not manual_result.fitted_model.b1.free
    assert manual_result.parameter("b1").value == 0.0
    assert not manual_result.parameter("b1").free
    assert manual_result.statistics.free_parameters == (
        b0_candidate.fit.statistics.free_parameters
    )


def lorentzian_identity(name: str) -> ComponentIdentity:
    return ComponentIdentity(ComponentFamily.LORENTZIAN, name)


def ref(
    identity: ComponentIdentity,
    family: ParameterFamily,
) -> ParameterReference:
    return ParameterReference(identity, family)


def fixed_manual_model() -> ManualModelState:
    first = lorentzian_identity("source-first")
    second = lorentzian_identity("source-second")
    area_tie = ManualParameterTieState(
        "source-area-tie",
        (
            ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA),
            ref(second, ParameterFamily.AREA),
        ),
        intent(0.4, 0.0),
    )
    return ManualModelState(
        elastic_area=intent(0.3, 0.0),
        lorentzians=(
            ManualLorentzianState(
                area=intent(0.2, 0.0),
                fwhm=intent(0.08, 1.0e-8),
                center_group="source-center",
                identity=first,
            ),
            ManualLorentzianState(
                area=intent(0.25, 0.0),
                fwhm=intent(0.18, 1.0e-8),
                center=intent(0.02, -0.1, 0.1),
                identity=second,
            ),
        ),
        center_groups=(
            ManualCenterGroupState(
                "source-center",
                intent(0.0, -0.1, 0.1),
            ),
        ),
        elastic_center_group="source-center",
        parameter_ties=(area_tie,),
    )


def test_resolution_is_never_implicitly_associated() -> None:
    project, sample, _ = configured_project(apply=False)

    assert applied_resolution(project, sample) is None
    resolved = resolve_manual_fit_context(project, sample, 0)

    assert not resolved.runnable
    assert resolved.diagnostics[0].code is (
        WorkflowDiagnosticCode.NO_APPLIED_RESOLUTION
    )


def test_apply_resolution_validates_exact_q_before_committing() -> None:
    project, sample, _ = configured_project(apply=False)
    bad_data = make_dataset(SpectrumRole.RESOLUTION, (0.5, 0.75))
    project, bad = add_project_dataset(project, bad_data, dataset_id="bad-resolution")

    with pytest.raises(WorkflowError) as caught:
        apply_resolution(project, sample, bad)

    assert caught.value.diagnostics[0].code is (
        WorkflowDiagnosticCode.Q_GROUP_COUNT_MISMATCH
    )
    assert caught.value.diagnostics[0].resolution_diagnostics[0].code == (
        "q_group_count_mismatch"
    )
    assert applied_resolution(project, sample) is None


def test_resolution_replacement_is_confirmation_first_and_transactional() -> None:
    project, sample, original = configured_project()
    bad_data = make_dataset(SpectrumRole.RESOLUTION, (0.5, 0.75))
    project, bad = add_project_dataset(project, bad_data, dataset_id="bad-resolution")
    preflight = preflight_apply_resolution(project, sample, bad)

    assert preflight.replacement_confirmation_required
    assert preflight.existing_resolution is not None
    assert preflight.existing_resolution.dataset_id == original.dataset_id
    with pytest.raises(WorkflowError) as unconfirmed:
        apply_resolution(project, sample, bad)
    assert unconfirmed.value.diagnostics[0].code is (
        WorkflowDiagnosticCode.RESOLUTION_REPLACEMENT_CONFIRMATION_REQUIRED
    )
    assert applied_resolution(project, sample) is original

    with pytest.raises(WorkflowError) as invalid:
        apply_resolution(project, sample, bad, replace_confirmed=True)
    assert invalid.value.diagnostics[0].code is (
        WorkflowDiagnosticCode.Q_GROUP_COUNT_MISMATCH
    )
    assert applied_resolution(project, sample) is original

    valid_data = make_dataset(SpectrumRole.RESOLUTION, (0.5, 0.75, 1.0))
    project, valid = add_project_dataset(project, valid_data, dataset_id="replacement")
    replaced = apply_resolution(
        project,
        sample,
        valid,
        replace_confirmed=True,
    )
    assert applied_resolution(replaced, sample) is valid


def test_apply_resolution_rejects_wrong_role_and_known_edge_mismatch() -> None:
    project, sample, resolution = configured_project(apply=False)
    project, wrong_role = add_project_dataset(
        project,
        make_dataset(SpectrumRole.SAMPLE, (0.5, 0.75, 1.0)),
        dataset_id="wrong-role",
    )
    with pytest.raises(WorkflowError) as wrong:
        apply_resolution(project, sample, wrong_role)
    assert wrong.value.diagnostics[0].code is (
        WorkflowDiagnosticCode.RESOLUTION_ROLE_REQUIRED
    )

    with pytest.raises(WorkflowError) as source_role:
        apply_resolution(project, resolution, resolution)
    assert source_role.value.diagnostics[0].code is (
        WorkflowDiagnosticCode.SAMPLE_ROLE_REQUIRED
    )

    sample_edges = QBins(
        q_values=np.array([0.5, 0.75, 1.0]),
        edges=np.array([0.35, 0.65, 0.85, 1.15]),
    )
    resolution_edges = QBins(
        q_values=np.array([0.5, 0.75, 1.0]),
        edges=np.array([0.30, 0.60, 0.90, 1.20]),
    )
    edged_sample_data = replace(sample.dataset, q_bins=sample_edges)
    project, edged_sample = replace_project_dataset(
        project,
        sample,
        edged_sample_data,
    )
    project, edged_resolution = add_project_dataset(
        project,
        replace(
            make_dataset(SpectrumRole.RESOLUTION, (0.5, 0.75, 1.0)),
            q_bins=resolution_edges,
        ),
        dataset_id="edge-mismatch",
    )
    with pytest.raises(WorkflowError) as edges:
        apply_resolution(project, edged_sample, edged_resolution)
    assert edges.value.diagnostics[0].code is WorkflowDiagnosticCode.Q_EDGE_MISMATCH


def test_cross_project_resolution_is_rejected() -> None:
    project, sample, _ = configured_project(apply=False)
    other = create_project("other", project_id="other-project")
    other, foreign = add_project_dataset(
        other,
        make_dataset(SpectrumRole.RESOLUTION, (0.5, 0.75, 1.0)),
        dataset_id="foreign",
    )

    with pytest.raises(WorkflowError) as caught:
        apply_resolution(project, sample, foreign)

    assert caught.value.diagnostics[0].code is (
        WorkflowDiagnosticCode.DATASET_PROJECT_MISMATCH
    )


def test_dataset_replacement_preserves_association_identity_and_rebinds_selection() -> (
    None
):
    project, sample, resolution = configured_project()
    replacement_data = replace(
        sample.dataset,
        source_reference="replacement.dat",
    )

    project, replacement = replace_project_dataset(
        project,
        sample,
        replacement_data,
    )
    resolved = resolve_manual_fit_context(project, sample, 0)

    assert replacement.dataset_id == sample.dataset_id
    assert applied_resolution(project, sample) is resolution
    assert resolved.runnable
    assert resolved.context is not None
    assert resolved.context.sample is replacement
    assert resolved.context.selection.dataset is replacement_data


def test_project_replacement_preserves_coverage_for_unchanged_xye() -> None:
    dataset = make_dataset(SpectrumRole.SAMPLE, (0.5, 0.75)).confirm_unrebinned_source()
    project = create_project("coverage", project_id="project-coverage-rebind")
    project, sample = add_project_dataset(project, dataset, dataset_id="sample")
    replacement_without_coverage = ReducedDataset(
        role=dataset.role,
        spectra=dataset.spectra,
        source_reference="metadata-only.dat",
        q_bins=QBins.from_q_values((0.55, 0.8)),
    )

    project, replacement = replace_project_dataset(
        project,
        sample,
        replacement_without_coverage,
    )

    assert replacement.dataset.fractional_coverage is dataset.fractional_coverage


def test_same_length_scientific_replacement_invalidates_inherited_coverage() -> None:
    dataset = make_dataset(SpectrumRole.SAMPLE, (0.5,)).confirm_unrebinned_source()
    project = create_project("coverage", project_id="project-coverage-change")
    project, sample = add_project_dataset(project, dataset, dataset_id="sample")
    changed_spectrum = replace(
        dataset.spectra[0],
        intensity=dataset.spectra[0].intensity + 0.01,
    )
    changed = replace(
        dataset,
        spectra=(changed_spectrum,),
        fractional_coverage=None,
    )

    _, replacement = replace_project_dataset(project, sample, changed)

    assert replacement.dataset.fractional_coverage is None


def test_spectrum_reordering_invalidates_inherited_coverage() -> None:
    dataset = make_dataset(SpectrumRole.SAMPLE, (0.5, 0.75)).confirm_unrebinned_source()
    project = create_project("coverage", project_id="project-coverage-order")
    project, sample = add_project_dataset(project, dataset, dataset_id="sample")
    reordered_spectra = (
        replace(dataset.spectra[1], group_index=0),
        replace(dataset.spectra[0], group_index=1),
    )
    reordered = replace(
        dataset,
        spectra=reordered_spectra,
        fractional_coverage=None,
    )

    _, replacement = replace_project_dataset(project, sample, reordered)

    assert replacement.dataset.fractional_coverage is None


def test_changed_xye_accepts_explicitly_replaced_aligned_coverage() -> None:
    dataset = make_dataset(SpectrumRole.SAMPLE, (0.5,)).confirm_unrebinned_source()
    project = create_project("coverage", project_id="project-coverage-explicit")
    project, sample = add_project_dataset(project, dataset, dataset_id="sample")
    changed_spectrum = replace(
        dataset.spectra[0],
        energy=dataset.spectra[0].energy + 0.001,
    )
    explicit_replacement = FractionalCoverage.aligned_with(
        (changed_spectrum,),
        values=(np.full(changed_spectrum.energy.size, 0.8),),
        origin=FractionalCoverageOrigin.EXPLICIT_SOURCE,
    )
    changed = replace(
        dataset,
        spectra=(changed_spectrum,),
        fractional_coverage=explicit_replacement,
    )

    _, replacement = replace_project_dataset(project, sample, changed)

    assert replacement.dataset.fractional_coverage is explicit_replacement


def test_adding_q_assignment_preserves_committed_selection() -> None:
    sample_data = replace(
        make_dataset(SpectrumRole.SAMPLE, (0.5, 0.75, 1.0)),
        q_bins=None,
    )
    project = create_project("q-assignment", project_id="project-q-assignment")
    project, sample = add_project_dataset(project, sample_data, dataset_id="sample")
    selection = FittingSelection.uniform(
        sample_data,
        detect_edge_padding(sample_data),
        lower_energy=-0.5,
        upper_energy=0.5,
    )
    project = commit_fitting_selection(project, sample, selection)
    replacement_data = replace(
        sample_data,
        q_bins=QBins.from_q_values((0.5, 0.75, 1.0)),
    )

    project, replacement = replace_project_dataset(
        project,
        sample,
        replacement_data,
    )

    assert len(project.fitting_selections) == 1
    assert project.fitting_selections[0].selection.dataset is replacement.dataset


def test_changing_q_values_and_edges_preserves_committed_selection() -> None:
    project, sample, _ = configured_project()
    replacement_data = replace(
        sample.dataset,
        q_bins=QBins(
            q_values=np.array([0.55, 0.85, 1.15]),
            edges=np.array([0.4, 0.7, 1.0, 1.3]),
        ),
    )

    project, replacement = replace_project_dataset(
        project,
        sample,
        replacement_data,
    )

    assert len(project.fitting_selections) == 1
    assert project.fitting_selections[0].selection.dataset is replacement.dataset


@pytest.mark.parametrize("changed_field", ("energy", "intensity", "uncertainty"))
def test_changed_measured_arrays_invalidate_committed_selection(
    changed_field: str,
) -> None:
    project, sample, resolution = configured_project()
    original_spectrum = sample.dataset.spectra[0]
    changed_values = np.array(getattr(original_spectrum, changed_field), copy=True)
    changed_values[changed_values.size // 2] += 0.001
    if changed_field == "energy":
        changed_spectrum = replace(original_spectrum, energy=changed_values)
    elif changed_field == "intensity":
        changed_spectrum = replace(original_spectrum, intensity=changed_values)
    else:
        changed_spectrum = replace(original_spectrum, uncertainty=changed_values)
    changed_dataset = replace(
        sample.dataset,
        spectra=(changed_spectrum, *sample.dataset.spectra[1:]),
    )

    project, replacement = replace_project_dataset(project, sample, changed_dataset)
    resolved = resolve_manual_fit_context(project, replacement, 0)

    assert replacement.dataset_id == sample.dataset_id
    assert applied_resolution(project, replacement) is resolution
    assert not project.fitting_selections
    assert not resolved.runnable
    assert resolved.diagnostics[0].code is (
        WorkflowDiagnosticCode.FITTING_SELECTION_UNAVAILABLE
    )


def test_removal_and_role_changes_invalidate_resolution_mappings() -> None:
    project, sample, resolution = configured_project()

    removed = remove_project_dataset(project, resolution)
    assert applied_resolution(removed, sample) is None

    project, sample, resolution = configured_project()
    reroled, _ = rerole_project_dataset(
        project,
        resolution,
        SpectrumRole.SAMPLE,
    )
    assert applied_resolution(reroled, sample) is None

    project, sample, _ = configured_project()
    reroled, current_sample = rerole_project_dataset(
        project,
        sample,
        SpectrumRole.RESOLUTION,
    )
    assert current_sample.dataset.role is SpectrumRole.RESOLUTION
    assert not reroled.resolution_associations


def test_context_reports_selection_group_and_preparation_blockers() -> None:
    no_selection, sample, resolution = configured_project(
        apply=False,
        commit_selection=False,
    )
    no_selection = apply_resolution(no_selection, sample, resolution)
    missing = resolve_manual_fit_context(no_selection, sample, 0)
    assert missing.diagnostics[0].code is (
        WorkflowDiagnosticCode.FITTING_SELECTION_UNAVAILABLE
    )

    project, sample, resolution = configured_project()
    invalid_group = resolve_manual_fit_context(project, sample, 99)
    assert invalid_group.diagnostics[0].code is WorkflowDiagnosticCode.INVALID_GROUP

    incompatible = replace(
        resolution.dataset,
        q_bins=QBins.from_q_values((0.5, 0.8, 1.0)),
    )
    project, _ = replace_project_dataset(project, resolution, incompatible)
    failed = resolve_manual_fit_context(project, sample, 0)
    assert failed.diagnostics[0].code is (
        WorkflowDiagnosticCode.RESOLUTION_PREPARATION_FAILED
    )
    assert failed.diagnostics[0].resolution_diagnostics[0].code == "q_value_mismatch"


def test_context_reports_sample_role_after_draft_dataset_is_reroled() -> None:
    project, sample, _ = configured_project()
    draft = open_manual_fit_draft(project, sample)
    project, current = rerole_project_dataset(
        project,
        sample,
        SpectrumRole.RESOLUTION,
    )

    resolved = resolve_manual_fit_context(project, current, 0)
    readiness = manual_workflow_readiness(project, draft, 0)

    assert resolved.diagnostics[0].code is WorkflowDiagnosticCode.SAMPLE_ROLE_REQUIRED
    assert not readiness.runnable
    assert readiness.workflow_diagnostics[0].code is (
        WorkflowDiagnosticCode.SAMPLE_ROLE_REQUIRED
    )


def test_empty_manual_draft_opens_without_resolution_and_is_not_runnable() -> None:
    project, sample, _ = configured_project(apply=False)
    draft = open_manual_fit_draft(project, sample)

    assert all(item.model is None for item in draft.setups)
    readiness = manual_workflow_readiness(project, draft, 0)
    assert not readiness.runnable
    assert readiness.workflow_diagnostics[0].code is (
        WorkflowDiagnosticCode.NO_APPLIED_RESOLUTION
    )


def test_background_completion_without_resolution_creates_b1() -> None:
    project, sample, _ = configured_project(apply=False)
    draft = open_manual_fit_draft(project, sample)

    pending = begin_component_interaction(
        draft,
        0,
        ManualComponentKind.BACKGROUND,
    )

    assert draft.setup(0).model is None
    completed = complete_background_interaction(
        draft,
        pending,
        first_energy=-0.2,
        first_height=0.1,
        second_energy=0.2,
        second_height=0.1,
    )
    model = completed.setup(0).model
    assert model is not None
    assert model.background is BackgroundModel.LINEAR
    assert model.b1 is not None
    assert model.b1.current_value == pytest.approx(0.0)
    assert model.b1.free


def test_committed_model_preview_failure_is_a_structured_workflow_error() -> None:
    project, sample, _ = configured_project()
    draft = open_manual_fit_draft(project, sample)
    pending = begin_component_interaction(
        draft,
        0,
        ManualComponentKind.BACKGROUND,
    )
    completed = complete_background_interaction(
        draft,
        pending,
        first_energy=-0.2,
        first_height=0.1,
        second_energy=0.2,
        second_height=0.1,
    )

    with pytest.raises(WorkflowError) as failed:
        preview_manual_fit(
            project,
            completed,
            0,
            display_energy=np.array([-0.2, np.nan, 0.2]),
        )

    assert failed.value.diagnostics[0].code is (
        WorkflowDiagnosticCode.MANUAL_PREVIEW_FAILED
    )
    assert completed.setup(0).model is not None


def test_elastic_and_lorentzian_completion_require_context_and_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, sample, _ = configured_project(apply=False)
    draft = open_manual_fit_draft(project, sample)
    elastic_pending = begin_component_interaction(
        draft,
        0,
        ManualComponentKind.ELASTIC,
    )
    with pytest.raises(WorkflowError) as missing:
        complete_elastic_interaction(
            project,
            draft,
            elastic_pending,
            component_peak_center=0.0,
            component_peak_height=1.0,
        )
    assert missing.value.diagnostics[0].code is (
        WorkflowDiagnosticCode.INTERACTION_CONTEXT_UNAVAILABLE
    )

    project, sample, _ = configured_project()
    draft = open_manual_fit_draft(project, sample)
    calls: list[str] = []

    def fake_elastic(*args: object, **kwargs: object) -> ElasticInteractionSeed:
        calls.append("elastic")
        return ElasticInteractionSeed(0.6, 0.01)

    def fake_lorentzian(*args: object, **kwargs: object) -> LorentzianInteractionSeed:
        calls.append("lorentzian")
        return LorentzianInteractionSeed(0.4, 0.1, 0.02, 0.11, 0.01)

    monkeypatch.setattr(
        manual_workflow_module,
        "initialize_elastic_from_interaction",
        fake_elastic,
    )
    monkeypatch.setattr(
        manual_workflow_module,
        "initialize_lorentzian_from_interaction",
        fake_lorentzian,
    )
    draft = complete_elastic_interaction(
        project,
        draft,
        begin_component_interaction(draft, 0, ManualComponentKind.ELASTIC),
        component_peak_center=0.0,
        component_peak_height=1.0,
    )
    draft = complete_lorentzian_interaction(
        project,
        draft,
        begin_component_interaction(draft, 0, ManualComponentKind.LORENTZIAN),
        component_peak_center=0.0,
        component_peak_height=1.0,
        width_endpoint_energy=0.05,
    )

    model = draft.setup(0).model
    assert calls == ["elastic", "lorentzian"]
    assert model is not None
    assert model.elastic_area is not None
    assert len(model.lorentzians) == 1
    assert model.lorentzians[0].center is not None


def test_lorentzian_completion_converts_one_sided_endpoint_to_full_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, sample, _ = configured_project()
    draft = open_manual_fit_draft(project, sample)
    observed_widths: list[float] = []

    def fake_lorentzian(*args: object, **kwargs: object) -> LorentzianInteractionSeed:
        observed_widths.append(cast(float, kwargs["observed_fwhm"]))
        return LorentzianInteractionSeed(0.4, 0.1, 0.02, 0.11, 0.01)

    monkeypatch.setattr(
        manual_workflow_module,
        "initialize_lorentzian_from_interaction",
        fake_lorentzian,
    )
    peak_center = 0.25
    for group_index, endpoint in enumerate((-0.25, 0.75)):
        draft = complete_lorentzian_interaction(
            project,
            draft,
            begin_component_interaction(
                draft,
                group_index,
                ManualComponentKind.LORENTZIAN,
            ),
            component_peak_center=peak_center,
            component_peak_height=1.0,
            width_endpoint_energy=endpoint,
        )

    assert observed_widths == pytest.approx([1.0, 1.0])
    left_model = draft.setup(0).model
    right_model = draft.setup(1).model
    assert left_model is not None
    assert right_model is not None
    assert left_model.lorentzians[0].area == right_model.lorentzians[0].area
    assert left_model.lorentzians[0].fwhm == right_model.lorentzians[0].fwhm
    assert left_model.lorentzians[0].center == right_model.lorentzians[0].center

    before_zero_width = tuple(observed_widths)
    with pytest.raises(WorkflowError) as zero_width:
        complete_lorentzian_interaction(
            project,
            draft,
            begin_component_interaction(
                draft,
                0,
                ManualComponentKind.LORENTZIAN,
            ),
            component_peak_center=peak_center,
            component_peak_height=1.0,
            width_endpoint_energy=peak_center,
        )
    assert zero_width.value.diagnostics[0].code is (
        WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION
    )
    assert "finite positive extent" in zero_width.value.diagnostics[0].message
    assert tuple(observed_widths) == before_zero_width


def test_pending_elastic_preview_matches_completion_without_mutating_draft() -> None:
    project, sample, _ = configured_project()
    draft = open_manual_fit_draft(project, sample)
    original = draft
    pending = begin_component_interaction(
        draft,
        0,
        ManualComponentKind.ELASTIC,
    )
    display_energy = sample.dataset.spectra[0].energy

    transient = preview_pending_elastic_interaction(
        project,
        draft,
        pending,
        component_peak_center=0.03,
        component_peak_height=1.2,
        display_energy=display_energy,
    )

    assert draft is original
    assert draft.setup(0).model is None
    assert transient.component_identity == pending.component_identity
    completed = complete_elastic_interaction(
        project,
        draft,
        pending,
        component_peak_center=0.03,
        component_peak_height=1.2,
    )
    committed = preview_manual_fit(
        project,
        completed,
        0,
        display_energy=display_energy,
    ).display_evaluation
    np.testing.assert_allclose(transient.evaluation.total, committed.total)
    np.testing.assert_allclose(
        transient.evaluation.component_curve(pending.component_identity),
        committed.component_curve(pending.component_identity),
    )


def test_pending_lorentzian_preview_matches_completion_and_rejects_zero_width() -> None:
    project, sample, _ = configured_project()
    draft = open_manual_fit_draft(project, sample)
    original = draft
    component_curves: list[np.ndarray] = []
    peak_center = 0.05
    for group_index, endpoint in enumerate((-0.15, 0.25)):
        pending = begin_component_interaction(
            draft,
            group_index,
            ManualComponentKind.LORENTZIAN,
        )
        display_energy = sample.dataset.spectra[group_index].energy
        transient = preview_pending_lorentzian_interaction(
            project,
            draft,
            pending,
            component_peak_center=peak_center,
            component_peak_height=0.8,
            width_endpoint_energy=endpoint,
            display_energy=display_energy,
        )
        assert draft is original
        assert draft.setup(group_index).model is None
        completed = complete_lorentzian_interaction(
            project,
            draft,
            pending,
            component_peak_center=peak_center,
            component_peak_height=0.8,
            width_endpoint_energy=endpoint,
        )
        committed = preview_manual_fit(
            project,
            completed,
            group_index,
            display_energy=display_energy,
        ).display_evaluation
        component_curve = transient.evaluation.component_curve(
            pending.component_identity
        )
        component_curves.append(component_curve)
        np.testing.assert_allclose(transient.evaluation.total, committed.total)
        np.testing.assert_allclose(
            component_curve,
            committed.component_curve(pending.component_identity),
        )

    np.testing.assert_allclose(component_curves[0], component_curves[1])
    zero = begin_component_interaction(draft, 0, ManualComponentKind.LORENTZIAN)
    with pytest.raises(WorkflowError) as invalid:
        preview_pending_lorentzian_interaction(
            project,
            draft,
            zero,
            component_peak_center=peak_center,
            component_peak_height=0.8,
            width_endpoint_energy=peak_center,
        )
    assert invalid.value.diagnostics[0].code is (
        WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION
    )
    assert draft is original
    assert draft.setup(0).model is None


def test_pending_background_preview_is_full_domain_b1_and_matches_completion() -> None:
    project, sample, _ = configured_project()
    draft = open_manual_fit_draft(project, sample)
    original = draft
    pending = begin_component_interaction(
        draft,
        0,
        ManualComponentKind.BACKGROUND,
    )
    display_energy = sample.dataset.spectra[0].energy
    transient = preview_pending_background_interaction(
        project,
        draft,
        pending,
        first_energy=-0.2,
        first_height=0.1,
        second_energy=0.2,
        second_height=0.3,
        display_energy=display_energy,
    )

    assert draft is original
    assert draft.setup(0).model is None
    expected = 0.2 + 0.5 * display_energy
    np.testing.assert_allclose(transient.evaluation.energy, display_energy)
    np.testing.assert_allclose(transient.evaluation.background, expected, atol=1e-15)
    np.testing.assert_allclose(transient.evaluation.total, expected, atol=1e-15)
    completed = complete_background_interaction(
        draft,
        pending,
        first_energy=-0.2,
        first_height=0.1,
        second_energy=0.2,
        second_height=0.3,
    )
    committed = preview_manual_fit(
        project,
        completed,
        0,
        display_energy=display_energy,
    ).display_evaluation
    np.testing.assert_allclose(transient.evaluation.total, committed.total)

    coincident = begin_component_interaction(
        draft,
        1,
        ManualComponentKind.BACKGROUND,
    )
    coincident_energy = sample.dataset.spectra[1].energy
    click_preview = preview_pending_background_interaction(
        project,
        draft,
        coincident,
        first_energy=0.1,
        first_height=-0.2,
        second_energy=0.1,
        second_height=-0.2,
        display_energy=coincident_energy,
    )
    np.testing.assert_allclose(click_preview.evaluation.background, -0.2)
    click_completed = complete_background_interaction(
        draft,
        coincident,
        first_energy=0.1,
        first_height=-0.2,
        second_energy=0.1,
        second_height=-0.2,
    )
    click_model = click_completed.setup(1).model
    assert click_model is not None
    assert click_model.background is BackgroundModel.LINEAR
    assert click_model.b1 is not None
    assert click_model.b1.current_value == 0.0


def test_pending_resolution_aware_previews_require_applied_resolution() -> None:
    project, sample, _ = configured_project(apply=False)
    draft = open_manual_fit_draft(project, sample)
    elastic = begin_component_interaction(draft, 0, ManualComponentKind.ELASTIC)
    lorentzian = begin_component_interaction(
        draft,
        0,
        ManualComponentKind.LORENTZIAN,
    )
    background = begin_component_interaction(
        draft,
        0,
        ManualComponentKind.BACKGROUND,
    )

    operations: tuple[Callable[[], PendingManualComponentPreview], ...] = (
        lambda: preview_pending_elastic_interaction(
            project,
            draft,
            elastic,
            component_peak_center=0.0,
            component_peak_height=1.0,
        ),
        lambda: preview_pending_lorentzian_interaction(
            project,
            draft,
            lorentzian,
            component_peak_center=0.0,
            component_peak_height=1.0,
            width_endpoint_energy=0.1,
        ),
        lambda: preview_pending_background_interaction(
            project,
            draft,
            background,
            first_energy=-0.1,
            first_height=0.0,
            second_energy=0.1,
            second_height=0.0,
        ),
    )
    for operation in operations:
        with pytest.raises(WorkflowError) as missing:
            operation()
        assert missing.value.diagnostics[0].code is (
            WorkflowDiagnosticCode.INTERACTION_CONTEXT_UNAVAILABLE
        )
    assert all(item.model is None for item in draft.setups)


def test_zero_height_interactions_preserve_zero_preview_and_adjust_fit_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, sample, _ = configured_project()
    draft = open_manual_fit_draft(project, sample)

    monkeypatch.setattr(
        manual_workflow_module,
        "initialize_elastic_from_interaction",
        lambda *args, **kwargs: ElasticInteractionSeed(0.0, 0.0),
    )
    elastic = complete_elastic_interaction(
        project,
        draft,
        begin_component_interaction(draft, 0, ManualComponentKind.ELASTIC),
        component_peak_center=0.0,
        component_peak_height=0.0,
    )
    elastic_model = elastic.setup(0).model
    assert elastic_model is not None
    elastic_reference = ParameterReference(
        ELASTIC_COMPONENT,
        ParameterFamily.AREA,
    )
    elastic_materialized = materialize_manual_setup(project, elastic, 0)
    elastic_area = elastic_materialized.parameter(elastic_reference)
    elastic_preview = preview_manual_fit(project, elastic, 0)

    assert elastic_model.parameter_intent(elastic_reference).current_value == 0.0
    assert elastic_area.preview_configuration.initial_value == 0.0
    assert elastic_area.fit_configuration.initial_value > 0.0
    assert elastic_area.fit_start_adjusted
    assert np.all(elastic_preview.display_evaluation.elastic == 0.0)

    monkeypatch.setattr(
        manual_workflow_module,
        "initialize_lorentzian_from_interaction",
        lambda *args, **kwargs: LorentzianInteractionSeed(
            0.0,
            0.1,
            0.0,
            0.12,
            0.02,
        ),
    )
    lorentzian = complete_lorentzian_interaction(
        project,
        draft,
        begin_component_interaction(draft, 1, ManualComponentKind.LORENTZIAN),
        component_peak_center=0.0,
        component_peak_height=0.0,
        width_endpoint_energy=0.06,
    )
    lorentzian_model = lorentzian.setup(1).model
    assert lorentzian_model is not None
    lorentzian_reference = ParameterReference(
        lorentzian_model.lorentzians[0].identity,
        ParameterFamily.AREA,
    )
    lorentzian_materialized = materialize_manual_setup(project, lorentzian, 1)
    lorentzian_area = lorentzian_materialized.parameter(lorentzian_reference)
    lorentzian_preview = preview_manual_fit(project, lorentzian, 1)

    assert lorentzian_model.parameter_intent(lorentzian_reference).current_value == 0.0
    assert lorentzian_area.preview_configuration.initial_value == 0.0
    assert lorentzian_area.fit_configuration.initial_value > 0.0
    assert lorentzian_area.fit_start_adjusted
    assert np.all(lorentzian_preview.display_evaluation.lorentzians[0] == 0.0)


def test_manual_intent_separates_preview_user_limits_and_intrinsic_fit_bounds() -> None:
    lorentzian = ManualLorentzianState(
        area=intent(0.2, 0.5, 1.0, free=True),
        fwhm=intent(0.1, free=False),
        center=intent(0.0, free=True),
        identity=lorentzian_identity("intent-lorentzian"),
    )
    model = ManualModelState(
        energy_shift=intent(0.0),
        elastic_area=intent(0.0, free=True),
        lorentzians=(lorentzian,),
    )
    project, sample, _ = configured_project()
    draft = open_manual_fit_draft(project, sample).with_model(0, model)
    materialized = materialize_manual_setup(project, draft, 0)
    preview = preview_manual_fit(project, draft, 0)
    elastic_area_reference = ParameterReference(
        ELASTIC_COMPONENT,
        ParameterFamily.AREA,
    )
    lorentzian_area_reference = ParameterReference(
        lorentzian.identity,
        ParameterFamily.AREA,
    )
    fwhm_reference = ParameterReference(
        lorentzian.identity,
        ParameterFamily.FWHM,
    )

    stored_area = model.parameter_intent(lorentzian_area_reference)
    area = materialized.parameter(lorentzian_area_reference)
    elastic_area = materialized.parameter(elastic_area_reference)
    fwhm = materialized.parameter(fwhm_reference)

    assert stored_area.current_value == 0.2
    assert stored_area.user_lower_limit == 0.5
    assert stored_area.user_upper_limit == 1.0
    assert area.preview_configuration.initial_value == 0.2
    assert np.all(np.isfinite(preview.display_evaluation.total))
    assert area.preview_configuration.lower_bound == 0.0
    assert area.fit_configuration.lower_bound == 0.5
    assert area.fit_configuration.initial_value > 0.5
    assert area.fit_start_adjusted
    assert model.parameter_intent(lorentzian_area_reference) is stored_area
    assert elastic_area.intent.user_lower_limit is None
    assert elastic_area.fit_configuration.lower_bound == 0.0
    assert elastic_area.fit_configuration.initial_value > 0.0
    assert fwhm.intent.user_lower_limit is None
    assert fwhm.fit_configuration.lower_bound > 0.0
    assert not fwhm.fit_configuration.free
    assert materialized.parameter(
        ParameterReference(lorentzian.identity, ParameterFamily.CENTER)
    ).fit_configuration.free


def test_parameter_edit_preserves_unrelated_identity_and_state() -> None:
    project, sample, _ = configured_project()
    draft = open_manual_fit_draft(project, sample).with_model(0, fixed_manual_model())
    model = draft.setup(0).model
    assert model is not None
    first, second = model.lorentzians

    updated = update_manual_parameter(
        draft,
        0,
        ref(second.identity, ParameterFamily.FWHM),
        ManualParameterEdit(0.22, 0.01, 0.5, True),
    )
    result = updated.setup(0).model

    assert result is not None
    assert tuple(item.identity for item in result.lorentzians) == (
        first.identity,
        second.identity,
    )
    assert result.lorentzians[0] == first
    assert result.lorentzians[1].fwhm.current_value == pytest.approx(0.22)
    assert result.center_groups == model.center_groups
    assert result.parameter_ties == model.parameter_ties


def test_parameter_edit_deletes_one_bound_without_disabling_bounds() -> None:
    project, sample, _ = configured_project()
    offset = ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.OFFSET)
    draft = open_manual_fit_draft(project, sample).with_model(
        0,
        ManualModelState(
            background=BackgroundModel.CONSTANT,
            b0=intent(0.2, -0.5, 0.5, free=True),
        ),
    )

    updated = update_manual_parameter(
        draft,
        0,
        offset,
        ManualParameterEdit(
            current_value=0.2,
            user_lower_limit=None,
            user_upper_limit=0.5,
            free=True,
            user_bounds_enabled=True,
        ),
    )

    model = updated.setup(0).model
    assert model is not None
    stored = model.parameter_intent(offset)
    assert stored.user_lower_limit is None
    assert stored.user_upper_limit == 0.5
    assert stored.user_bounds_enabled


def test_disabled_inverted_bounds_are_runnable_until_reenabled() -> None:
    project, sample, _ = configured_project()
    offset = ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.OFFSET)
    draft = open_manual_fit_draft(project, sample).with_model(
        0,
        ManualModelState(
            background=BackgroundModel.CONSTANT,
            b0=intent(
                0.3,
                0.8,
                0.2,
                free=True,
                bounds_enabled=False,
            ),
        ),
    )

    disabled = materialize_manual_setup(project, draft, 0).parameter(offset)
    assert manual_workflow_readiness(project, draft, 0).runnable
    assert disabled.intent.user_lower_limit == 0.8
    assert disabled.intent.user_upper_limit == 0.2
    assert not disabled.intent.user_bounds_enabled
    assert np.isneginf(disabled.fit_configuration.lower_bound)
    assert np.isposinf(disabled.fit_configuration.upper_bound)

    enabled = update_manual_parameter(
        draft,
        0,
        offset,
        ManualParameterEdit(0.3, 0.8, 0.2, True, True),
    )
    readiness = manual_workflow_readiness(project, enabled, 0)
    assert not readiness.runnable
    assert readiness.workflow_diagnostics[0].code is (
        WorkflowDiagnosticCode.MANUAL_MATERIALIZATION_FAILED
    )


def test_tie_untie_and_clone_preserve_bounds_enabled_state() -> None:
    project, sample, _ = configured_project()
    first = lorentzian_identity("bounds-first")
    second = lorentzian_identity("bounds-second")
    first_area = ref(first, ParameterFamily.AREA)
    second_area = ref(second, ParameterFamily.AREA)
    model = ManualModelState(
        lorentzians=(
            ManualLorentzianState(
                area=intent(0.2, 0.1, 0.5),
                fwhm=intent(0.1),
                center=intent(0.0),
                identity=first,
            ),
            ManualLorentzianState(
                area=intent(0.3),
                fwhm=intent(0.15),
                center=intent(0.0),
                identity=second,
            ),
        ),
    )
    draft = open_manual_fit_draft(project, sample).with_model(0, model)
    draft = create_parameter_tie(
        draft,
        0,
        (first_area,),
        source_member=first_area,
        tie_group_id="bounds-chain",
    )
    draft = update_manual_parameter(
        draft,
        0,
        first_area,
        ManualParameterEdit(0.25, 0.1, 0.5, False, False),
    )
    draft = join_parameter_tie(draft, 0, "bounds-chain", second_area)

    tied = draft.setup(0).model
    assert tied is not None
    assert tied.parameter_intent(first_area) is tied.parameter_intent(second_area)
    assert not tied.parameter_intent(first_area).user_bounds_enabled

    untied = untie_parameter(draft, 0, second_area)
    untied_model = untied.setup(0).model
    assert untied_model is not None
    departing = untied_model.parameter_intent(second_area)
    assert departing == ManualParameterIntent(0.25, 0.1, 0.5, False, False)
    assert departing is not untied_model.parameter_intent(first_area)

    applied = apply_manual_setup_to_all_groups(project, draft, 0)
    for setup in applied.setups:
        copied = setup.model
        assert copied is not None
        assert len(copied.parameter_ties) == 1
        assert not copied.parameter_ties[0].intent.user_bounds_enabled
        assert copied.parameter_ties[0].intent.user_lower_limit == 0.1
        assert copied.parameter_ties[0].intent.user_upper_limit == 0.5


def test_tie_create_join_untie_and_multiple_same_family_groups() -> None:
    first = lorentzian_identity("first")
    second = lorentzian_identity("second")
    third = lorentzian_identity("third")
    fourth = lorentzian_identity("fourth")
    model = ManualModelState(
        lorentzians=tuple(
            ManualLorentzianState(
                area=intent(value, 0.0),
                fwhm=intent(0.1 + index * 0.05, 1.0e-8),
                center=intent(index * 0.01, -0.1, 0.1),
                identity=identity,
            )
            for index, (identity, value) in enumerate(
                ((first, 0.2), (second, 0.3), (third, 0.4), (fourth, 0.5))
            )
        )
    )
    project, sample, _ = configured_project()
    draft = open_manual_fit_draft(project, sample).with_model(0, model)
    first_area = ref(first, ParameterFamily.AREA)
    second_area = ref(second, ParameterFamily.AREA)
    third_area = ref(third, ParameterFamily.AREA)
    fourth_area = ref(fourth, ParameterFamily.AREA)

    draft = create_parameter_tie(
        draft,
        0,
        (first_area,),
        source_member=first_area,
        tie_group_id="areas-a",
    )
    tied = draft.setup(0).model
    assert tied is not None
    assert tied.parameter_ties[0].members == (first_area,)
    draft = update_manual_parameter(
        draft,
        0,
        first_area,
        ManualParameterEdit(0.35, 0.1, 0.6, True),
    )
    singleton = draft.setup(0).model
    assert singleton is not None
    assert singleton.parameter_ties[0].intent == ManualParameterIntent(
        0.35,
        0.1,
        0.6,
        True,
    )
    draft = join_parameter_tie(draft, 0, "areas-a", second_area)
    draft = join_parameter_tie(draft, 0, "areas-a", third_area)
    joined = draft.setup(0).model
    assert joined is not None
    assert len(joined.parameter_ties[0].members) == 3
    assert joined.parameter_intent(third_area) is joined.parameter_ties[0].intent
    assert joined.parameter_intent(third_area) == ManualParameterIntent(
        0.35,
        0.1,
        0.6,
        True,
    )
    tied_materialization = materialize_manual_setup(project, draft, 0)
    assert tied_materialization.parameter(first_area) is (
        tied_materialization.parameter(third_area)
    )
    assert tied_materialization.parameter(first_area).fit_configuration.free
    assert (
        tied_materialization.parameter(first_area).fit_configuration.lower_bound == 0.1
    )

    draft = untie_parameter(draft, 0, third_area)
    two = draft.setup(0).model
    assert two is not None
    assert len(two.parameter_ties) == 1
    assert two.parameter_ties[0].members == (first_area, second_area)
    assert two.tie_for(third_area) is None
    assert two.parameter_intent(third_area) is not two.parameter_ties[0].intent
    assert two.parameter_intent(third_area) == ManualParameterIntent(
        0.35,
        0.1,
        0.6,
        True,
    )

    draft = untie_parameter(draft, 0, first_area)
    singleton = draft.setup(0).model
    assert singleton is not None
    assert singleton.parameter_ties[0].members == (second_area,)
    assert singleton.parameter_intent(first_area) is not (
        singleton.parameter_ties[0].intent
    )
    assert singleton.parameter_intent(first_area) == ManualParameterIntent(
        0.35,
        0.1,
        0.6,
        True,
    )
    draft = untie_parameter(draft, 0, second_area)
    independent = draft.setup(0).model
    assert independent is not None
    assert not independent.parameter_ties
    assert independent.parameter_intent(second_area) == ManualParameterIntent(
        0.35,
        0.1,
        0.6,
        True,
    )

    draft = create_parameter_tie(
        draft,
        0,
        (first_area,),
        source_member=first_area,
        tie_group_id="areas-a",
    )
    draft = join_parameter_tie(draft, 0, "areas-a", second_area)
    draft = create_parameter_tie(
        draft,
        0,
        (third_area,),
        source_member=third_area,
        tie_group_id="areas-b",
    )
    draft = join_parameter_tie(draft, 0, "areas-b", fourth_area)
    multiple = draft.setup(0).model
    assert multiple is not None
    assert {item.group_id for item in multiple.parameter_ties} == {
        "areas-a",
        "areas-b",
    }
    assert all(
        group.members[0].family is ParameterFamily.AREA
        for group in multiple.parameter_ties
    )


def test_empty_manual_parameter_tie_state_is_invalid() -> None:
    with pytest.raises(ValueError, match="at least one member"):
        ManualParameterTieState("empty", (), intent(0.1))


def test_singleton_elastic_center_tie_owns_materialized_state() -> None:
    project, sample, _ = configured_project()
    elastic_center = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER)
    chain_intent = intent(0.02, -0.05, 0.05, free=True)
    model = ManualModelState(
        energy_shift=intent(0.0, -0.1, 0.1),
        elastic_area=intent(0.5, 0.0),
        parameter_ties=(
            ManualParameterTieState(
                "elastic-center-chain",
                (elastic_center,),
                chain_intent,
            ),
        ),
    )
    draft = open_manual_fit_draft(project, sample).with_model(0, model)

    materialized = materialize_manual_setup(project, draft, 0)
    center = materialized.parameter(elastic_center)
    readiness = manual_workflow_readiness(project, draft, 0)
    preview = preview_manual_fit(project, draft, 0)
    result = run_manual_fit(project, draft, 0)

    assert center.fit_configuration.initial_value == pytest.approx(0.02)
    assert center.fit_configuration.lower_bound == pytest.approx(-0.05)
    assert center.fit_configuration.upper_bound == pytest.approx(0.05)
    assert center.fit_configuration.free
    assert materialized.fit_model.energy_shift is center.fit_configuration
    assert materialized.fit_model.parameter_ties[0].parameter is (
        center.fit_configuration
    )
    assert readiness.runnable
    assert np.all(np.isfinite(preview.display_evaluation.total))
    assert result.parameter_by_reference(elastic_center).references == (elastic_center,)
    assert result.configuration.parameter_ties[0].members == (elastic_center,)


def test_clone_and_apply_preserve_singleton_and_multiple_tie_groups() -> None:
    project, sample, _ = configured_project()
    identities = tuple(lorentzian_identity(f"topology-{index}") for index in range(3))
    references = tuple(ref(identity, ParameterFamily.AREA) for identity in identities)
    model = ManualModelState(
        lorentzians=tuple(
            ManualLorentzianState(
                area=intent(0.2 + 0.1 * index, 0.0),
                fwhm=intent(0.1 + 0.05 * index, 1.0e-8),
                center=intent(0.01 * index, -0.1, 0.1),
                identity=identity,
            )
            for index, identity in enumerate(identities)
        ),
        parameter_ties=(
            ManualParameterTieState(
                "singleton-area",
                (references[0],),
                intent(0.25, 0.0),
            ),
            ManualParameterTieState(
                "shared-area",
                references[1:],
                intent(0.35, 0.0),
            ),
        ),
    )
    draft = open_manual_fit_draft(project, sample).with_model(0, model)

    cloned = clone_manual_setup_to_group(project, draft, 0, 1)
    applied = apply_manual_setup_to_all_groups(project, cloned, 0)

    for group_index in (1, 2):
        target = applied.setup(group_index).model
        assert target is not None
        assert tuple(len(group.members) for group in target.parameter_ties) == (1, 2)
        assert {group.family for group in target.parameter_ties} == {
            ParameterFamily.AREA
        }
        assert {
            member.component
            for group in target.parameter_ties
            for member in group.members
        } == {component.identity for component in target.lorentzians}
        assert {group.group_id for group in target.parameter_ties}.isdisjoint(
            {group.group_id for group in model.parameter_ties}
        )


def test_component_removal_keeps_ties_valid_and_can_restore_empty_draft() -> None:
    first = lorentzian_identity("first")
    second = lorentzian_identity("second")
    members = (ref(first, ParameterFamily.AREA), ref(second, ParameterFamily.AREA))
    model = ManualModelState(
        lorentzians=(
            ManualLorentzianState(
                intent(0.2, 0.0),
                intent(0.1, 1.0e-8),
                center=intent(0.0, -0.1, 0.1),
                identity=first,
            ),
            ManualLorentzianState(
                intent(0.3, 0.0),
                intent(0.2, 1.0e-8),
                center=intent(0.01, -0.1, 0.1),
                identity=second,
            ),
        ),
        parameter_ties=(
            ManualParameterTieState(
                "areas",
                members,
                intent(0.25, 0.1, 0.8, free=True),
            ),
        ),
    )
    project, sample, _ = configured_project()
    draft = open_manual_fit_draft(project, sample).with_model(0, model)

    draft = remove_manual_component(draft, 0, first)
    remaining = draft.setup(0).model
    assert remaining is not None
    assert [item.identity for item in remaining.lorentzians] == [second]
    assert len(remaining.parameter_ties) == 1
    assert remaining.parameter_ties[0].group_id == "areas"
    assert remaining.parameter_ties[0].members == (members[1],)
    assert remaining.parameter_intent(members[1]) == ManualParameterIntent(
        0.25,
        0.1,
        0.8,
        True,
    )
    assert set(remaining.parameter_references()) == {
        ref(second, ParameterFamily.AREA),
        ref(second, ParameterFamily.FWHM),
        ref(second, ParameterFamily.CENTER),
    }

    draft = remove_manual_component(draft, 0, second)
    assert draft.setup(0).model is None


def test_group_clone_regenerates_local_identities_and_preserves_semantics() -> None:
    project, sample, _ = configured_project()
    source_model = fixed_manual_model()
    draft = open_manual_fit_draft(project, sample).with_model(0, source_model)

    cloned = clone_manual_setup_to_group(project, draft, 0, 1)
    target = cloned.setup(1).model

    assert target is not None
    assert target.background is source_model.background
    assert [item.area.current_value for item in target.lorentzians] == [
        item.area.current_value for item in source_model.lorentzians
    ]
    assert {item.identity for item in target.lorentzians}.isdisjoint(
        {item.identity for item in source_model.lorentzians}
    )
    assert target.center_groups[0].group_id != (source_model.center_groups[0].group_id)
    assert target.parameter_ties[0].group_id != (
        source_model.parameter_ties[0].group_id
    )
    assert target.parameter_ties[0].intent is not source_model.parameter_ties[0].intent
    assert target.parameter_ties[0].intent == source_model.parameter_ties[0].intent


def test_group_clone_preserves_intent_and_rematerializes_target_center_coverage() -> (
    None
):
    project, sample, _ = configured_project_with_distinct_group_coverage()
    source_component = ManualLorentzianState(
        area=intent(0.2),
        fwhm=intent(0.1),
        center=intent(0.0),
        identity=lorentzian_identity("coverage-source"),
    )
    draft = open_manual_fit_draft(project, sample).with_model(
        0,
        ManualModelState(lorentzians=(source_component,)),
    )
    cloned = clone_manual_setup_to_group(project, draft, 0, 1)
    target_model = cloned.setup(1).model
    assert target_model is not None
    source_reference = ParameterReference(
        source_component.identity,
        ParameterFamily.CENTER,
    )
    target_reference = ParameterReference(
        target_model.lorentzians[0].identity,
        ParameterFamily.CENTER,
    )
    source_center = materialize_manual_setup(project, cloned, 0).parameter(
        source_reference
    )
    target_center = materialize_manual_setup(project, cloned, 1).parameter(
        target_reference
    )

    assert source_center.intent.current_value == target_center.intent.current_value
    assert source_center.intent.user_lower_limit is None
    assert target_center.intent.user_lower_limit is None
    source_coverage = (
        source_center.preview_configuration.lower_bound,
        source_center.preview_configuration.upper_bound,
    )
    target_coverage = (
        target_center.preview_configuration.lower_bound,
        target_center.preview_configuration.upper_bound,
    )
    assert source_coverage != target_coverage

    common_half_width = 0.5 * min(source_coverage[1], target_coverage[1])
    edited = update_manual_parameter(
        draft,
        0,
        source_reference,
        ManualParameterEdit(
            current_value=0.0,
            user_lower_limit=-common_half_width,
            user_upper_limit=common_half_width,
            free=True,
        ),
    )
    copied = clone_manual_setup_to_group(project, edited, 0, 1)
    copied_target = copied.setup(1).model
    assert copied_target is not None
    copied_target_reference = ParameterReference(
        copied_target.lorentzians[0].identity,
        ParameterFamily.CENTER,
    )
    copied_intent = copied_target.parameter_intent(copied_target_reference)
    copied_center = materialize_manual_setup(project, copied, 1).parameter(
        copied_target_reference
    )

    assert copied_intent.current_value == 0.0
    assert copied_intent.user_lower_limit == -common_half_width
    assert copied_intent.user_upper_limit == common_half_width
    assert copied_center.preview_configuration.lower_bound == target_coverage[0]
    assert copied_center.preview_configuration.upper_bound == target_coverage[1]

    if source_coverage[1] > target_coverage[1]:
        incompatible_lower = 0.5 * (source_coverage[1] + target_coverage[1])
        incompatible_upper = 0.5 * (source_coverage[1] + incompatible_lower)
    else:
        incompatible_upper = 0.5 * (source_coverage[0] + target_coverage[0])
        incompatible_lower = 0.5 * (source_coverage[0] + incompatible_upper)
    incompatible = update_manual_parameter(
        draft,
        0,
        source_reference,
        ManualParameterEdit(
            current_value=0.0,
            user_lower_limit=incompatible_lower,
            user_upper_limit=incompatible_upper,
            free=True,
        ),
    )
    assert manual_workflow_readiness(project, incompatible, 0).runnable

    with pytest.raises(WorkflowError) as clone_error:
        clone_manual_setup_to_group(project, incompatible, 0, 1)
    assert clone_error.value.diagnostics[0].code is (
        WorkflowDiagnosticCode.TARGET_SETUP_INVALID
    )
    with pytest.raises(WorkflowError) as apply_error:
        apply_manual_setup_to_all_groups(project, incompatible, 0)
    assert any(
        item.code is WorkflowDiagnosticCode.TARGET_SETUP_INVALID
        and item.group_index == 1
        for item in apply_error.value.diagnostics
    )
    assert incompatible.setup(1).model is None


def test_clone_and_apply_never_execute_fit_or_autofit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, sample, _ = configured_project()
    draft = open_manual_fit_draft(project, sample).with_model(0, fixed_manual_model())

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("clone/apply must not execute fitting or AutoFit")

    monkeypatch.setattr(manual_workflow_module, "fit_single_q", forbidden)
    monkeypatch.setattr(fitting_module, "recommend_standard_candidates", forbidden)

    cloned = clone_manual_setup_to_group(project, draft, 0, 1)
    applied = apply_manual_setup_to_all_groups(project, cloned, 0)

    assert all(setup.model is not None for setup in applied.setups)


def test_apply_setup_to_all_is_independent_and_all_or_nothing() -> None:
    project, sample, _ = configured_project()
    source_model = fixed_manual_model()
    draft = open_manual_fit_draft(project, sample).with_model(0, source_model)

    applied = apply_manual_setup_to_all_groups(project, draft, 0)

    identities = [
        {item.identity for item in cast(ManualModelState, setup.model).lorentzians}
        for setup in applied.setups
    ]
    assert identities[0].isdisjoint(identities[1])
    assert identities[1].isdisjoint(identities[2])
    assert applied.setup(0).model is source_model

    selection = next(
        item.selection
        for item in project.fitting_selections
        if item.sample_id == sample.dataset_id
    )
    one_point = selection.with_group_range(
        1,
        lower_energy=0.0,
        upper_energy=0.0,
    )
    blocked_project = commit_fitting_selection(project, sample, one_point)
    original = open_manual_fit_draft(blocked_project, sample).with_model(
        0,
        source_model,
    )
    with pytest.raises(WorkflowError) as caught:
        apply_manual_setup_to_all_groups(blocked_project, original, 0)
    assert any(
        item.code is WorkflowDiagnosticCode.TARGET_SETUP_INVALID
        and item.group_index == 1
        for item in caught.value.diagnostics
    )
    assert original.setup(1).model is None
    assert original.setup(2).model is None


def test_thin_readiness_preview_and_run_delegate_to_frozen_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, sample, _ = configured_project()
    model = ManualModelState(
        background=BackgroundModel.CONSTANT,
        b0=intent(1.0),
    )
    draft = open_manual_fit_draft(project, sample).with_model(0, model)
    calls: list[str] = []
    preview_sentinel = cast(ManualModelPreview, SimpleNamespace())
    fit_sentinel = cast(FitResult, SimpleNamespace())

    def fake_readiness(*args: object, **kwargs: object) -> ManualFitReadiness:
        calls.append("readiness")
        return ManualFitReadiness(True)

    def fake_preview(*args: object, **kwargs: object) -> ManualModelPreview:
        calls.append("preview")
        return preview_sentinel

    def fake_fit(*args: object, **kwargs: object) -> FitResult:
        calls.append("fit")
        return fit_sentinel

    monkeypatch.setattr(
        manual_workflow_module,
        "manual_fit_readiness",
        fake_readiness,
    )
    monkeypatch.setattr(
        manual_workflow_module,
        "preview_manual_model",
        fake_preview,
    )
    monkeypatch.setattr(manual_workflow_module, "fit_single_q", fake_fit)

    readiness = manual_workflow_readiness(project, draft, 0)
    preview = preview_manual_fit(project, draft, 0)
    result = run_manual_fit(project, draft, 0)

    assert readiness.runnable
    assert preview is preview_sentinel
    assert result is fit_sentinel
    assert calls == ["readiness", "preview", "fit"]


def test_run_and_adopt_updates_values_preserves_intent_and_seeds_next_run() -> None:
    project, sample, _ = configured_project()
    original_model = ManualModelState(
        background=BackgroundModel.LINEAR,
        b0=intent(0.5, 0.0, 2.0, free=True),
        b1=intent(0.0, -1.0, 1.0),
    )
    original = open_manual_fit_draft(project, sample).with_model(0, original_model)
    background_offset = ParameterReference(
        BACKGROUND_COMPONENT,
        ParameterFamily.OFFSET,
    )
    background_slope = ParameterReference(
        BACKGROUND_COMPONENT,
        ParameterFamily.SLOPE,
    )

    first = run_and_adopt_manual_fit(project, original, 0)

    assert first.success
    assert first.fit_result is not None
    assert first.adopted_draft is not None
    assert original.setup(0).model is original_model
    assert original_model.b0 == intent(0.5, 0.0, 2.0, free=True)
    adopted_model = first.adopted_draft.setup(0).model
    assert adopted_model is not None
    adopted_offset = adopted_model.parameter_intent(background_offset)
    fitted_offset = first.fit_result.parameter_by_reference(background_offset)
    assert adopted_offset.current_value == pytest.approx(fitted_offset.value)
    assert adopted_offset.user_lower_limit == 0.0
    assert adopted_offset.user_upper_limit == 2.0
    assert adopted_offset.free
    adopted_slope = adopted_model.parameter_intent(background_slope)
    assert adopted_slope.current_value == pytest.approx(
        first.fit_result.parameter_by_reference(background_slope).value
    )
    assert adopted_slope.user_lower_limit == -1.0
    assert adopted_slope.user_upper_limit == 1.0
    assert not adopted_slope.free

    second = run_and_adopt_manual_fit(project, first.adopted_draft, 0)

    assert second.success
    assert second.fit_result is not None
    assert second.fit_result.configuration.b0 is not None
    assert second.fit_result.configuration.b0.initial_value == pytest.approx(
        fitted_offset.value
    )


def test_run_and_adopt_accepts_active_bound_without_changing_user_limits() -> None:
    project, sample, _ = configured_project()
    model = ManualModelState(
        background=BackgroundModel.CONSTANT,
        b0=intent(0.5, 0.0, 0.8, free=True),
    )
    draft = open_manual_fit_draft(project, sample).with_model(0, model)
    background_offset = ParameterReference(
        BACKGROUND_COMPONENT,
        ParameterFamily.OFFSET,
    )

    outcome = run_and_adopt_manual_fit(project, draft, 0)

    assert outcome.success
    assert outcome.fit_result is not None
    assert outcome.adopted_draft is not None
    estimate = outcome.fit_result.parameter_by_reference(background_offset)
    assert estimate.active_upper_bound
    adopted_model = outcome.adopted_draft.setup(0).model
    assert adopted_model is not None
    adopted = adopted_model.parameter_intent(background_offset)
    assert adopted.current_value == pytest.approx(estimate.value)
    assert adopted.user_lower_limit == 0.0
    assert adopted.user_upper_limit == 0.8
    assert adopted.free
    rematerialized = materialize_manual_setup(
        project,
        outcome.adopted_draft,
        0,
    ).parameter(background_offset)
    assert (
        rematerialized.fit_configuration.lower_bound
        <= (rematerialized.fit_configuration.initial_value)
        <= rematerialized.fit_configuration.upper_bound
    )
    assert rematerialized.intent.current_value == pytest.approx(estimate.value)


def test_run_and_adopt_preserves_disabled_dormant_user_bounds() -> None:
    project, sample, _ = configured_project()
    model = ManualModelState(
        background=BackgroundModel.CONSTANT,
        b0=intent(
            0.5,
            0.0,
            0.8,
            free=True,
            bounds_enabled=False,
        ),
    )
    draft = open_manual_fit_draft(project, sample).with_model(0, model)
    background_offset = ParameterReference(
        BACKGROUND_COMPONENT,
        ParameterFamily.OFFSET,
    )

    outcome = run_and_adopt_manual_fit(project, draft, 0)

    assert outcome.success
    assert outcome.fit_result is not None
    assert outcome.adopted_draft is not None
    estimate = outcome.fit_result.parameter_by_reference(background_offset)
    assert estimate.value > 0.8
    assert not estimate.active_upper_bound
    assert np.isposinf(estimate.upper_bound)
    adopted_model = outcome.adopted_draft.setup(0).model
    assert adopted_model is not None
    adopted = adopted_model.parameter_intent(background_offset)
    assert adopted.current_value == pytest.approx(estimate.value)
    assert adopted.user_lower_limit == 0.0
    assert adopted.user_upper_limit == 0.8
    assert adopted.free
    assert not adopted.user_bounds_enabled


def test_run_and_adopt_uses_reference_identity_after_fwhm_reordering() -> None:
    project, sample, _ = configured_project()
    broad = lorentzian_identity("submitted-broad")
    narrow = lorentzian_identity("submitted-narrow")
    model = ManualModelState(
        lorentzians=(
            ManualLorentzianState(
                area=intent(0.2, 0.0),
                fwhm=intent(0.3, 1.0e-8),
                center=intent(-0.02, -0.1, 0.1),
                identity=broad,
            ),
            ManualLorentzianState(
                area=intent(0.1, 0.0),
                fwhm=intent(0.08, 1.0e-8),
                center=intent(0.03, -0.1, 0.1),
                identity=narrow,
            ),
        ),
        background=BackgroundModel.CONSTANT,
        b0=intent(0.5, 0.0, 2.0, free=True),
    )
    draft = open_manual_fit_draft(project, sample).with_model(0, model)

    outcome = run_and_adopt_manual_fit(project, draft, 0)

    assert outcome.success
    assert outcome.fit_result is not None
    assert outcome.adopted_draft is not None
    assert outcome.fit_result.fitted_model is not None
    assert tuple(
        component.identity for component in outcome.fit_result.fitted_model.lorentzians
    ) == (narrow, broad)
    adopted = outcome.adopted_draft.setup(0).model
    assert adopted is not None
    assert tuple(component.identity for component in adopted.lorentzians) == (
        broad,
        narrow,
    )
    for reference in model.parameter_references():
        assert adopted.parameter_intent(reference).current_value == pytest.approx(
            outcome.fit_result.parameter_by_reference(reference).value
        )


def test_run_and_adopt_preserves_singleton_and_multiple_tie_topology() -> None:
    project, sample, _ = configured_project()
    identities = tuple(lorentzian_identity(f"adopt-tie-{index}") for index in range(3))
    area_references = tuple(
        ref(identity, ParameterFamily.AREA) for identity in identities
    )
    model = ManualModelState(
        lorentzians=tuple(
            ManualLorentzianState(
                area=intent(0.15 + 0.05 * index, 0.0),
                fwhm=intent(0.08 + 0.04 * index, 1.0e-8),
                center=intent(0.0, -0.1, 0.1),
                identity=identity,
            )
            for index, identity in enumerate(identities)
        ),
        background=BackgroundModel.CONSTANT,
        b0=intent(0.5, 0.0, 2.0, free=True),
        parameter_ties=(
            ManualParameterTieState(
                "shared-areas",
                area_references[:2],
                intent(0.25, 0.0),
            ),
            ManualParameterTieState(
                "singleton-area",
                area_references[2:],
                intent(0.4, 0.0),
            ),
        ),
    )
    draft = open_manual_fit_draft(project, sample).with_model(0, model)

    outcome = run_and_adopt_manual_fit(project, draft, 0)

    assert outcome.success
    assert outcome.fit_result is not None
    assert outcome.adopted_draft is not None
    adopted = outcome.adopted_draft.setup(0).model
    assert adopted is not None
    assert tuple(
        (group.group_id, group.members) for group in adopted.parameter_ties
    ) == tuple((group.group_id, group.members) for group in model.parameter_ties)
    assert adopted.parameter_intent(area_references[0]) is adopted.parameter_intent(
        area_references[1]
    )
    assert adopted.parameter_intent(area_references[2]) is not adopted.parameter_intent(
        area_references[0]
    )
    assert adopted.parameter_intent(area_references[0]).current_value == pytest.approx(
        outcome.fit_result.parameter_by_reference(area_references[0]).value
    )
    assert adopted.parameter_intent(area_references[2]).current_value == pytest.approx(
        outcome.fit_result.parameter_by_reference(area_references[2]).value
    )


def test_run_and_adopt_accepts_success_without_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, sample, _ = configured_project()
    draft = open_manual_fit_draft(project, sample).with_model(
        0,
        ManualModelState(
            background=BackgroundModel.CONSTANT,
            b0=intent(0.5, 0.0, 2.0, free=True),
        ),
    )
    fit_result = run_manual_fit(project, draft, 0)
    without_covariance = replace(
        fit_result,
        covariance=None,
        correlation=None,
        diagnostics=replace(
            fit_result.diagnostics,
            covariance_available=False,
        ),
    )
    monkeypatch.setattr(
        manual_workflow_module,
        "fit_single_q",
        lambda *args, **kwargs: without_covariance,
    )

    outcome = run_and_adopt_manual_fit(project, draft, 0)

    assert outcome.success
    assert outcome.fit_result is without_covariance
    assert outcome.adopted_draft is not None


def test_run_and_adopt_contains_optimizer_and_fitting_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, sample, _ = configured_project()
    model = ManualModelState(
        background=BackgroundModel.CONSTANT,
        b0=intent(0.5, 0.0, 2.0, free=True),
    )
    draft = open_manual_fit_draft(project, sample).with_model(0, model)
    fit_result = run_manual_fit(project, draft, 0)
    nonconverged = replace(
        fit_result,
        diagnostics=replace(fit_result.diagnostics, optimizer_success=False),
    )
    monkeypatch.setattr(
        manual_workflow_module,
        "fit_single_q",
        lambda *args, **kwargs: nonconverged,
    )

    failed = run_and_adopt_manual_fit(project, draft, 0)

    assert not failed.success
    assert failed.fit_result is nonconverged
    assert failed.adopted_draft is None
    assert failed.diagnostics[0].code is (
        WorkflowDiagnosticCode.MANUAL_FIT_DID_NOT_CONVERGE
    )
    assert draft.setup(0).model is model

    def raise_fitting_error(*args: object, **kwargs: object) -> FitResult:
        raise FittingError("controlled fitting failure")

    monkeypatch.setattr(
        manual_workflow_module,
        "fit_single_q",
        raise_fitting_error,
    )
    errored = run_and_adopt_manual_fit(project, draft, 0)
    assert not errored.success
    assert errored.fit_result is None
    assert errored.adopted_draft is None
    assert errored.diagnostics[0].code is (
        WorkflowDiagnosticCode.MANUAL_FIT_EXECUTION_FAILED
    )
    assert draft.setup(0).model is model


def test_run_and_adopt_rejects_malformed_reference_coverage_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, sample, _ = configured_project()
    model = ManualModelState(
        background=BackgroundModel.CONSTANT,
        b0=intent(0.5, 0.0, 2.0, free=True),
    )
    draft = open_manual_fit_draft(project, sample).with_model(0, model)
    fit_result = run_manual_fit(project, draft, 0)
    malformed = replace(
        fit_result,
        parameters=(replace(fit_result.parameters[0], references=()),),
    )
    monkeypatch.setattr(
        manual_workflow_module,
        "fit_single_q",
        lambda *args, **kwargs: malformed,
    )

    outcome = run_and_adopt_manual_fit(project, draft, 0)

    assert not outcome.success
    assert outcome.fit_result is malformed
    assert outcome.adopted_draft is None
    assert outcome.diagnostics[0].code is (
        WorkflowDiagnosticCode.MANUAL_FIT_ADOPTION_FAILED
    )
    assert "exact parameter-reference coverage" in outcome.diagnostics[0].message
    assert draft.setup(0).model is model


def test_run_and_adopt_revalidates_complete_adopted_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, sample, _ = configured_project()
    model = ManualModelState(
        background=BackgroundModel.CONSTANT,
        b0=intent(0.5, 0.0, 2.0, free=True),
    )
    draft = open_manual_fit_draft(project, sample).with_model(0, model)
    actual_readiness = fitting_module.manual_fit_readiness
    calls = 0

    def staged_readiness(
        prepared_resolution: PreparedResolution,
        selection: FittingSelection,
        group_index: int,
        fit_model: SpectralModelDefinition,
    ) -> ManualFitReadiness:
        nonlocal calls
        calls += 1
        if calls == 1:
            return actual_readiness(
                prepared_resolution,
                selection,
                group_index,
                fit_model,
            )
        return ManualFitReadiness(
            False,
            (
                ManualFitReadinessDiagnostic(
                    ManualFitDiagnosticCode.INVALID_PARAMETER_CONFIGURATION,
                    DiagnosticSeverity.ERROR,
                    "forced adopted-state validation failure",
                ),
            ),
        )

    monkeypatch.setattr(
        manual_workflow_module,
        "manual_fit_readiness",
        staged_readiness,
    )

    outcome = run_and_adopt_manual_fit(project, draft, 0)

    assert calls == 2
    assert not outcome.success
    assert outcome.fit_result is not None
    assert outcome.adopted_draft is None
    assert outcome.diagnostics[0].code is (
        WorkflowDiagnosticCode.MANUAL_FIT_ADOPTION_FAILED
    )
    assert "forced adopted-state validation failure" in outcome.diagnostics[0].message
    assert draft.setup(0).model is model
