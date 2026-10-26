"""Focused tests for anchor-outward independent Multi-Q execution."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

import ezqens.batch.core as batch_core
import ezqens.workflow.multi_q as multi_q_workflow
from ezqens.batch import (
    MultiQBranchResult,
    MultiQExecutionStatus,
    MultiQFitOutcome,
    MultiQFitStatus,
    execute_multi_q_branch,
)
from ezqens.convolution import build_convolution_plan
from ezqens.derived import DerivedPointStatus, derive_unified_qens
from ezqens.domain import QBins, ReducedDataset, Spectrum, SpectrumRole
from ezqens.fitting import (
    BACKGROUND_COMPONENT,
    ELASTIC_COMPONENT,
    AutoFitRecommendation,
    BackgroundModel,
    CandidateFitResult,
    ComponentFamily,
    ComponentIdentity,
    FitContextBinding,
    FitResult,
    FittingError,
    LorentzianComponent,
    ManualParameterIntent,
    ParameterConfiguration,
    ParameterFamily,
    ParameterReference,
    SpectralModelDefinition,
    StandardModelCandidate,
    evaluate_spectral_model,
    fit_single_q,
    fit_standard_candidate,
)
from ezqens.preprocessing import FittingSelection
from ezqens.resolution import (
    PreparedResolution,
    ResolutionAcceptance,
    ResolutionAcceptanceDecision,
    ResolutionAcceptanceSource,
    ResolutionAcceptanceWarning,
    ResolutionSupport,
    prepare_measured_resolution,
)
from ezqens.workflow import (
    AutoFitCandidateBranchResult,
    CenterTieIdentity,
    CenterTieKind,
    FittingWorkspaceState,
    IdentityMark,
    ManualCenterGroupState,
    ManualFitContext,
    ManualFitDraft,
    ManualLorentzianState,
    ManualModelMaterialization,
    ManualModelState,
    ManualParameterEdit,
    ManualParameterTieState,
    MethodTransferOptions,
    MethodTransferOverrides,
    MultiQConstraintTransferPolicy,
    ProjectDataset,
    QResult,
    ResultDisplayStatus,
    ResultOrigin,
    ResultSnapshot,
    ResultWorkingCopy,
    SavedFitResult,
    SelectedAutoFitMultiQResult,
    SelectedAutoFitProgressEvent,
    SingleQAutoFitOutcome,
    UnifiedResult,
    WorkflowProject,
    active_manual_draft,
    active_transferable_constraints,
    add_project_dataset,
    adopt_selected_auto_fit_branch,
    adopt_single_q_auto_fit_candidate,
    append_saved_result,
    apply_fitting_method,
    apply_manual_setup_to_all_groups,
    apply_resolution,
    auto_fit_all_q,
    capture_fitting_method,
    commit_fitting_selection,
    continue_selected_auto_fit_candidates,
    create_parameter_tie,
    create_project,
    derive_result_in_context,
    draft_edit_items,
    fit_all_q_from_current,
    fit_all_q_from_current_with_adoption,
    fit_evidence_is_current,
    fresh_analysis,
    has_unsaved_result_changes,
    join_parameter_tie,
    load_saved_result,
    materialize_manual_model,
    open_manual_fit_draft,
    remove_saved_result,
    rename_saved_result,
    replace_current_q_result,
    replace_project_dataset,
    resolve_manual_fit_context,
    result_q_display_status,
    revert_saved_result,
    run_and_adopt_manual_fit,
    run_manual_fit,
    run_single_q_auto_fit,
    save_current_result,
    save_current_result_as,
    save_working_method,
    set_best_saved_result,
    set_current_result,
    set_working_method,
    swap_current_lorentzian_identities,
    unsaved_result_items,
    update_manual_parameter,
    update_working_draft,
)
from ezqens.workflow.manual_state import replace_parameter_intent
from ezqens.workflow.results import _issue_result_snapshot as _issue_snapshot

FloatArray = npt.NDArray[np.float64]


def _issue_result_snapshot(
    result: FitResult | MultiQBranchResult,
    working_states: tuple[ManualModelState | None, ...],
    origin: ResultOrigin,
    candidate: StandardModelCandidate | None = None,
    *,
    sample_identity: tuple[str, str] = ("batch-project", "sample"),
) -> ResultSnapshot:
    return _issue_snapshot(
        result, working_states, origin, candidate, sample_identity=sample_identity
    )


def _gaussian(energy: FloatArray, sigma: float) -> FloatArray:
    return np.asarray(
        np.exp(-0.5 * np.square(energy / sigma)) / (sigma * math.sqrt(2.0 * math.pi)),
        dtype=np.float64,
    )


def _parameter(
    value: float,
    lower: float = -math.inf,
    upper: float = math.inf,
    *,
    free: bool = False,
) -> ParameterConfiguration:
    return ParameterConfiguration(value, lower, upper, free)


L1 = ComponentIdentity(ComponentFamily.LORENTZIAN, "branch_l1")
L2 = ComponentIdentity(ComponentFamily.LORENTZIAN, "branch_l2")


def _model(
    *,
    elastic: ParameterConfiguration | None = None,
    free: bool = False,
) -> SpectralModelDefinition:
    return SpectralModelDefinition(
        energy_shift=_parameter(0.01, -0.2, 0.2, free=free),
        elastic_area=elastic or _parameter(0.8, 0.0, 2.0, free=free),
        lorentzians=(
            LorentzianComponent(
                area=_parameter(0.35, 0.0, 2.0, free=free),
                fwhm=_parameter(0.09, 1.0e-6, 1.0, free=free),
                identity=L1,
            ),
            LorentzianComponent(
                area=_parameter(0.25, 0.0, 2.0, free=free),
                fwhm=_parameter(0.32, 1.0e-6, 1.0, free=free),
                identity=L2,
            ),
        ),
        background=BackgroundModel.CONSTANT,
        b0=_parameter(0.02, -1.0, 1.0, free=free),
    )


def _manual_model(
    *,
    background: BackgroundModel = BackgroundModel.LINEAR,
    slope_free: bool = True,
) -> ManualModelState:
    center = ManualParameterIntent(0.01, -0.2, 0.2)
    return ManualModelState(
        energy_shift=center,
        elastic_area=ManualParameterIntent(0.8, 0.0, 2.0),
        lorentzians=(
            ManualLorentzianState(
                area=ManualParameterIntent(0.35, 0.0, 2.0),
                fwhm=ManualParameterIntent(0.09, 1.0e-6, 1.0),
                identity=L1,
            ),
            ManualLorentzianState(
                area=ManualParameterIntent(0.25, 0.0, 2.0),
                fwhm=ManualParameterIntent(0.32, 1.0e-6, 1.0),
                identity=L2,
            ),
        ),
        background=background,
        b0=(
            ManualParameterIntent(0.02, -1.0, 1.0)
            if background is not BackgroundModel.NONE
            else None
        ),
        b1=(
            ManualParameterIntent(0.0, free=slope_free)
            if background is BackgroundModel.LINEAR
            else None
        ),
    )


def _dataset(
    role: SpectrumRole,
    energies: FloatArray,
    intensities: Sequence[FloatArray],
    q_values: FloatArray,
) -> ReducedDataset:
    return ReducedDataset(
        role=role,
        spectra=tuple(
            Spectrum(
                role=role,
                group_index=index,
                group_label=f"{role.value}-{index}",
                energy=energies,
                intensity=intensity,
                uncertainty=np.full(energies.size, 0.01),
                energy_unit="meV",
                intensity_unit="arb",
                uncertainty_unit="arb",
            )
            for index, intensity in enumerate(intensities)
        ),
        q_bins=QBins.from_q_values(q_values),
    )


@pytest.fixture(scope="module")
def multi_q_problem() -> tuple[
    PreparedResolution,
    FittingSelection,
    tuple[SpectralModelDefinition, ...],
    FitResult,
]:
    group_count = 5
    sample_energy = np.linspace(-0.65, 0.65, 81)
    resolution_energy = np.linspace(-0.25, 0.25, 61)
    resolution_values = _gaussian(resolution_energy, 0.035)
    q_values = np.linspace(0.5, 1.5, group_count)
    placeholder = _dataset(
        SpectrumRole.SAMPLE,
        sample_energy,
        tuple(np.ones(sample_energy.size) for _ in range(group_count)),
        q_values,
    )
    resolution = _dataset(
        SpectrumRole.RESOLUTION,
        resolution_energy,
        tuple(resolution_values for _ in range(group_count)),
        q_values,
    )
    provisional = prepare_measured_resolution(placeholder, resolution)
    truth = _model()
    intensities = tuple(
        evaluate_spectral_model(
            build_convolution_plan(provisional, index),
            truth,
            sample_energy,
        ).total
        for index in range(group_count)
    )
    sample = _dataset(
        SpectrumRole.SAMPLE,
        sample_energy,
        intensities,
        q_values,
    )
    prepared = prepare_measured_resolution(sample, resolution)
    selection = FittingSelection.uniform(
        sample,
        prepared.sample_padding,
        lower_energy=float(sample_energy[0]),
        upper_energy=float(sample_energy[-1]),
    )
    configurations = tuple(_model(free=True) for _ in range(group_count))
    anchor = fit_single_q(prepared, selection, 2, configurations[2])
    return prepared, selection, configurations, anchor


def _manual_context(
    prepared: PreparedResolution,
    selection: FittingSelection,
    group_index: int = 2,
) -> ManualFitContext:
    return ManualFitContext(
        project_id="test-project",
        sample=ProjectDataset(
            "test-project",
            "sample",
            prepared.sample_dataset,
        ),
        group_index=group_index,
        group_identity=prepared.sample_dataset.spectra[group_index].group_label,
        selection=selection,
        resolution=ProjectDataset(
            "test-project",
            "resolution",
            prepared.resolution_dataset,
        ),
        prepared_resolution=prepared,
    )


def _workflow_problem(
    prepared: PreparedResolution,
    selection: FittingSelection,
    *,
    background: BackgroundModel = BackgroundModel.CONSTANT,
    slope_free: bool = True,
) -> tuple[WorkflowProject, ProjectDataset, ManualFitDraft]:
    project = create_project("batch workflow", project_id="batch-project")
    project, sample = add_project_dataset(
        project,
        prepared.sample_dataset,
        dataset_id="sample",
    )
    project, resolution = add_project_dataset(
        project,
        prepared.resolution_dataset,
        dataset_id="resolution",
    )
    project = commit_fitting_selection(project, sample, selection)
    project = apply_resolution(project, sample, resolution)
    draft = open_manual_fit_draft(project, sample)
    for group_index in range(len(draft.setups)):
        draft = draft.with_model(
            group_index,
            _manual_model(background=background, slope_free=slope_free),
        )
    return project, sample, draft


def _single_q_auto_outcome(
    context: ManualFitContext,
    candidates: Sequence[CandidateFitResult],
    *,
    most_recommended: CandidateFitResult | None,
) -> SingleQAutoFitOutcome:
    spectrum = context.selection.dataset.spectra[context.group_index]
    retained = context.selection.retained_mask(context.group_index)
    recommendation = cast(
        AutoFitRecommendation,
        SimpleNamespace(
            candidate_results=tuple(candidates),
            most_recommended=most_recommended,
        ),
    )
    return SingleQAutoFitOutcome(
        scientific_context=context,
        recommendation=recommendation,
        measured_energy=spectrum.energy[retained],
        measured_intensity=spectrum.intensity[retained],
        measured_uncertainty=spectrum.uncertainty[retained],
    )


@pytest.fixture(scope="module")
def selected_auto_problem(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> tuple[
    WorkflowProject,
    ProjectDataset,
    ManualFitDraft,
    SingleQAutoFitOutcome,
    tuple[CandidateFitResult, CandidateFitResult],
]:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, sample, draft = _workflow_problem(prepared, selection)
    resolved = resolve_manual_fit_context(project, sample, 2)
    assert resolved.context is not None
    context = resolved.context
    one_lorentzian = StandardModelCandidate(1, BackgroundModel.NONE)
    one_fit = fit_standard_candidate(
        context.prepared_resolution,
        context.selection,
        context.group_index,
        one_lorentzian,
        max_nfev=800,
    )
    two_lorentzian = StandardModelCandidate(2, BackgroundModel.CONSTANT)
    two_fit = run_manual_fit(project, draft, context.group_index, max_nfev=800)
    evidence = (
        CandidateFitResult(two_lorentzian, two_fit),
        CandidateFitResult(one_lorentzian, one_fit),
    )
    outcome = _single_q_auto_outcome(
        context,
        evidence,
        most_recommended=evidence[0],
    )
    empty = replace(
        draft,
        setups=tuple(replace(setup, model=None) for setup in draft.setups),
    )
    return project, sample, empty, outcome, evidence


def _workflow_with_nan_resolution_uncertainty(
    prepared: PreparedResolution,
    selection: FittingSelection,
) -> tuple[WorkflowProject, ProjectDataset, ManualFitDraft]:
    resolution_spectra = list(prepared.resolution_dataset.spectra)
    source = resolution_spectra[2]
    uncertainty = np.array(source.uncertainty, copy=True)
    uncertainty[uncertainty.size // 2] = np.nan
    resolution_spectra[2] = replace(source, uncertainty=uncertainty)
    resolution_dataset = replace(
        prepared.resolution_dataset,
        spectra=tuple(resolution_spectra),
    )
    nan_prepared = prepare_measured_resolution(
        prepared.sample_dataset,
        resolution_dataset,
    )
    rebound_selection = FittingSelection(
        dataset=prepared.sample_dataset,
        padding=nan_prepared.sample_padding,
        ranges=selection.ranges,
        manual_exclusion_masks=selection.manual_exclusion_masks,
        manual_auto_reinclusion_masks=selection.manual_auto_reinclusion_masks,
    )
    return _workflow_problem(nan_prepared, rebound_selection)


def _result_with_values(
    anchor: FitResult,
    group_index: int,
    submitted: SpectralModelDefinition,
    values: dict[ParameterReference, float] | None = None,
    *,
    successful: bool = True,
    q_value: float | None = None,
    prepared_resolution: PreparedResolution | None = None,
    selection: FittingSelection | None = None,
) -> FitResult:
    values = values or {}
    parameters = tuple(
        replace(
            parameter,
            lower_bound=submitted.parameter_configuration(
                parameter.references[0]
            ).lower_bound,
            upper_bound=submitted.parameter_configuration(
                parameter.references[0]
            ).upper_bound,
            free=submitted.parameter_configuration(parameter.references[0]).free,
            value=next(
                (
                    values[reference]
                    for reference in parameter.references
                    if reference in values
                ),
                parameter.value,
            ),
        )
        for parameter in anchor.parameters
    )
    spectrum = anchor.provenance
    binding = anchor.context_binding
    if prepared_resolution is not None or selection is not None:
        if prepared_resolution is None or selection is None:
            raise ValueError(
                "prepared_resolution and selection must be supplied together"
            )
        binding = FitContextBinding(prepared_resolution, selection, group_index)
    elif binding is not None:
        binding = replace(binding, group_index=group_index)
    return replace(
        anchor,
        configuration=submitted,
        parameters=parameters,
        diagnostics=replace(
            anchor.diagnostics,
            optimizer_success=successful,
        ),
        provenance=replace(
            spectrum,
            group_index=group_index,
            group_label=f"sample-{group_index}",
            q_value=(0.5 + 0.25 * group_index if q_value is None else q_value),
        ),
        fitted_model=submitted,
        context_binding=binding,
    )


def _elastic_value(value: float) -> dict[ParameterReference, float]:
    return {
        ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA): value,
    }


def _recording_fit(
    anchor: FitResult,
    calls: list[tuple[int, SpectralModelDefinition]],
    result_values: dict[int, float] | None = None,
    failures: set[int] | None = None,
) -> Callable[..., FitResult]:
    result_values = result_values or {}
    failures = failures or set()

    def fit(
        _prepared: PreparedResolution,
        _selection: FittingSelection,
        group_index: int,
        model: SpectralModelDefinition,
        *,
        max_nfev: int,
    ) -> FitResult:
        assert max_nfev == 2500
        calls.append((group_index, model))
        if group_index in failures:
            raise FittingError(f"failed group {group_index}")
        return _result_with_values(
            anchor,
            group_index,
            model,
            _elastic_value(result_values.get(group_index, 0.8)),
            q_value=_prepared.q_value(group_index),
            prepared_resolution=_prepared,
            selection=_selection,
        )

    return fit


def _with_q_values(
    prepared: PreparedResolution,
    selection: FittingSelection,
    q_values: npt.ArrayLike,
) -> tuple[PreparedResolution, FittingSelection]:
    q_bins = QBins.from_q_values(q_values)
    sample = prepared.sample_dataset.assign_q_bins(q_bins)
    resolution = prepared.resolution_dataset.assign_q_bins(q_bins)
    reassigned = prepare_measured_resolution(sample, resolution)
    rebound_selection = FittingSelection(
        dataset=sample,
        padding=reassigned.sample_padding,
        ranges=selection.ranges,
        manual_exclusion_masks=selection.manual_exclusion_masks,
        manual_auto_reinclusion_masks=selection.manual_auto_reinclusion_masks,
    )
    return reassigned, rebound_selection


def test_middle_anchor_uses_independent_outward_success_chains(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem
    anchor = _result_with_values(anchor, 2, anchor.configuration, _elastic_value(1.0))
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(
        batch_core,
        "fit_single_q",
        _recording_fit(anchor, calls, {1: 1.1, 0: 1.2, 3: 1.3, 4: 1.4}),
    )

    result = execute_multi_q_branch(
        prepared,
        selection,
        configurations,
        anchor_group_index=2,
        anchor_fit=anchor,
    )

    assert [group for group, _model in calls] == [1, 0, 3, 4]
    starts = [
        model.elastic_area.initial_value
        for _group, model in calls
        if model.elastic_area is not None
    ]
    assert starts == [1.0, 1.1, 1.0, 1.3]
    assert [item.seed_group_index for item in result.outcomes] == [1, 2, None, 2, 3]
    retained_anchor = result.outcome(2).fit_result
    assert retained_anchor is not None
    assert retained_anchor.parameters == anchor.parameters
    assert retained_anchor.diagnostics == anchor.diagnostics
    assert retained_anchor.provenance == anchor.provenance
    assert result.status is MultiQExecutionStatus.COMPLETED


def test_branch_progress_publishes_authoritative_terminal_outcomes_in_order(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, configurations, base_anchor = multi_q_problem
    anchor = _result_with_values(
        base_anchor,
        2,
        base_anchor.configuration,
        _elastic_value(1.0),
    )
    target_configurations: list[SpectralModelDefinition | None] = list(configurations)
    target_configurations[1] = None
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(
        batch_core,
        "fit_single_q",
        _recording_fit(anchor, calls, {4: 1.4}, {3}),
    )
    events: list[MultiQFitOutcome] = []

    def record(outcome: MultiQFitOutcome) -> bool:
        events.append(outcome)
        return True

    result = execute_multi_q_branch(
        prepared,
        selection,
        target_configurations,
        anchor_group_index=2,
        anchor_fit=anchor,
        fit_excluded_groups=(0,),
        derived_result_excluded_groups=(4,),
        progress_callback=record,
    )

    assert [(item.group_index, item.status) for item in events] == [
        (2, MultiQFitStatus.SUCCESS),
        (0, MultiQFitStatus.EXCLUDED),
        (1, MultiQFitStatus.BLOCKED),
        (3, MultiQFitStatus.FAILED),
        (4, MultiQFitStatus.SUCCESS),
    ]
    assert all(item.status is not MultiQFitStatus.NOT_RUN for item in events)
    assert len({item.group_index for item in events}) == len(events)
    assert all(item is result.outcome(item.group_index) for item in events)
    assert events[-1].derived_result_excluded
    assert [group for group, _model in calls] == [3, 4]


def test_branch_start_progress_follows_execution_and_precedes_blocked_outcome(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem
    target_configurations: list[SpectralModelDefinition | None] = list(configurations)
    target_configurations[1] = None
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(batch_core, "fit_single_q", _recording_fit(anchor, calls))
    started: list[int] = []
    terminal: list[MultiQFitOutcome] = []
    event_order: list[tuple[str, int]] = []

    def record_started(group_index: int) -> object:
        started.append(group_index)
        event_order.append(("started", group_index))
        return object()

    def record_terminal(outcome: MultiQFitOutcome) -> None:
        terminal.append(outcome)
        event_order.append(("terminal", outcome.group_index))

    result = execute_multi_q_branch(
        prepared,
        selection,
        target_configurations,
        anchor_group_index=2,
        anchor_fit=anchor,
        fit_excluded_groups=(0,),
        group_started_callback=record_started,
        progress_callback=record_terminal,
    )

    assert started == [1, 3, 4]
    assert event_order == [
        ("terminal", 2),
        ("terminal", 0),
        ("started", 1),
        ("terminal", 1),
        ("started", 3),
        ("terminal", 3),
        ("started", 4),
        ("terminal", 4),
    ]
    assert result.outcome(1).status is MultiQFitStatus.BLOCKED
    assert all(item is result.outcome(item.group_index) for item in terminal)


def test_branch_start_progress_omits_cancelled_and_not_run_targets(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(batch_core, "fit_single_q", _recording_fit(anchor, calls))
    checks = 0
    started: list[int] = []

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks == 2

    result = execute_multi_q_branch(
        prepared,
        selection,
        configurations,
        anchor_group_index=2,
        anchor_fit=anchor,
        cancel_requested=cancelled,
        group_started_callback=started.append,
    )

    assert result.status is MultiQExecutionStatus.CANCELLED
    assert started == [1]
    assert [group for group, _model in calls] == [1]
    assert all(
        outcome.group_index not in started
        for outcome in result.outcomes
        if outcome.status is MultiQFitStatus.NOT_RUN
    )


def test_branch_progress_omits_targets_not_processed_after_cancellation(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem
    events: list[MultiQFitOutcome] = []

    result = execute_multi_q_branch(
        prepared,
        selection,
        configurations,
        anchor_group_index=2,
        anchor_fit=anchor,
        cancel_requested=lambda: True,
        progress_callback=events.append,
    )

    assert result.status is MultiQExecutionStatus.CANCELLED
    assert events == [result.outcome(2)]
    assert all(
        item.status is MultiQFitStatus.NOT_RUN
        for index, item in enumerate(result.outcomes)
        if index != 2
    )


def test_anchor_progress_exception_does_not_stop_branch_or_later_notifications(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(
        batch_core,
        "fit_single_q",
        _recording_fit(anchor, calls),
    )
    attempted: list[MultiQFitOutcome] = []

    def fail_on_anchor(outcome: MultiQFitOutcome) -> None:
        attempted.append(outcome)
        if outcome.group_index == 2:
            raise RuntimeError("observer failed on anchor")

    result = execute_multi_q_branch(
        prepared,
        selection,
        configurations,
        anchor_group_index=2,
        anchor_fit=anchor,
        progress_callback=fail_on_anchor,
    )

    assert result.status is MultiQExecutionStatus.COMPLETED
    assert [item.group_index for item in attempted] == [2, 1, 0, 3, 4]
    assert all(item.status is MultiQFitStatus.SUCCESS for item in result.outcomes)
    assert [group for group, _model in calls] == [1, 0, 3, 4]


def test_target_progress_exception_retains_target_and_continues_notifications(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(
        batch_core,
        "fit_single_q",
        _recording_fit(anchor, calls),
    )
    attempted: list[MultiQFitOutcome] = []

    def fail_on_first_target(outcome: MultiQFitOutcome) -> None:
        attempted.append(outcome)
        if outcome.group_index == 1:
            raise ValueError("observer failed on target")

    result = execute_multi_q_branch(
        prepared,
        selection,
        configurations,
        anchor_group_index=2,
        anchor_fit=anchor,
        progress_callback=fail_on_first_target,
    )

    assert result.status is MultiQExecutionStatus.COMPLETED
    assert [item.group_index for item in attempted] == [2, 1, 0, 3, 4]
    assert attempted[1] is result.outcome(1)
    assert result.outcome(1).status is MultiQFitStatus.SUCCESS
    assert all(item.status is MultiQFitStatus.SUCCESS for item in result.outcomes)


def test_start_progress_exception_does_not_change_execution_or_later_notifications(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(batch_core, "fit_single_q", _recording_fit(anchor, calls))
    attempted: list[int] = []

    def failing_observer(group_index: int) -> None:
        attempted.append(group_index)
        if group_index == 1:
            raise RuntimeError("start observer failed")

    result = execute_multi_q_branch(
        prepared,
        selection,
        configurations,
        anchor_group_index=2,
        anchor_fit=anchor,
        group_started_callback=failing_observer,
    )

    assert attempted == [1, 0, 3, 4]
    assert [group for group, _model in calls] == attempted
    assert result.status is MultiQExecutionStatus.COMPLETED
    assert all(item.status is MultiQFitStatus.SUCCESS for item in result.outcomes)


def test_start_progress_base_exception_propagates_before_target_work(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(batch_core, "fit_single_q", _recording_fit(anchor, calls))

    def interrupt(_group_index: int) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute_multi_q_branch(
            prepared,
            selection,
            configurations,
            anchor_group_index=2,
            anchor_fit=anchor,
            group_started_callback=interrupt,
        )

    assert calls == []


def test_selected_autofit_progress_event_rejects_not_run_outcome() -> None:
    with pytest.raises(ValueError, match="cannot publish NOT_RUN"):
        SelectedAutoFitProgressEvent(
            StandardModelCandidate(0, BackgroundModel.NONE),
            MultiQFitOutcome(0, MultiQFitStatus.NOT_RUN),
        )


def test_nonmonotonic_group_order_traverses_by_physical_q_without_reordering_results(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, configurations, _old_anchor = multi_q_problem
    q_values = np.asarray([1.2, 0.4, 1.0, 1.6, 0.7])
    prepared, selection = _with_q_values(prepared, selection, q_values)
    anchor = fit_single_q(prepared, selection, 2, configurations[2])
    anchor = _result_with_values(anchor, 2, anchor.configuration, _elastic_value(1.0))
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(
        batch_core,
        "fit_single_q",
        _recording_fit(anchor, calls, {4: 0.9, 1: 0.8, 0: 1.1, 3: 1.2}),
    )

    result = execute_multi_q_branch(
        prepared,
        selection,
        configurations,
        anchor_group_index=2,
        anchor_fit=anchor,
    )

    assert [group for group, _model in calls] == [4, 1, 0, 3]
    starts = {
        group: model.elastic_area.initial_value
        for group, model in calls
        if model.elastic_area is not None
    }
    assert starts == {4: 1.0, 1: 0.9, 0: 1.0, 3: 1.1}
    assert tuple(outcome.group_index for outcome in result.outcomes) == (0, 1, 2, 3, 4)
    preserved_q_bins = prepared.sample_dataset.q_bins
    assert preserved_q_bins is not None
    np.testing.assert_array_equal(preserved_q_bins.q_values, q_values)


def test_duplicate_representative_q_values_cannot_define_silent_index_traversal(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, configurations, _old_anchor = multi_q_problem
    prepared, selection = _with_q_values(
        prepared,
        selection,
        [0.5, 0.75, 1.0, 1.0, 1.5],
    )
    anchor = fit_single_q(prepared, selection, 2, configurations[2])

    with pytest.raises(ValueError, match="unique representative Q values"):
        execute_multi_q_branch(
            prepared,
            selection,
            configurations,
            anchor_group_index=2,
            anchor_fit=anchor,
        )


def test_current_context_manual_and_auto_results_are_valid_anchors(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, _configurations, manual_anchor = multi_q_problem
    auto_anchor = fit_standard_candidate(
        prepared,
        selection,
        2,
        StandardModelCandidate(0, BackgroundModel.CONSTANT),
    )

    for anchor in (manual_anchor, auto_anchor):
        configurations = tuple(anchor.configuration for _ in prepared.spectra)
        result = execute_multi_q_branch(
            prepared,
            selection,
            configurations,
            anchor_group_index=2,
            anchor_fit=anchor,
            cancel_requested=lambda: True,
        )

        assert result.outcome(2).fit_result is anchor
        assert result.status is MultiQExecutionStatus.CANCELLED


def test_dataset_switch_cannot_reuse_foreign_current_result(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem
    foreign_sample = replace(
        prepared.sample_dataset,
        source_reference="different-dataset-with-identical-values.dat",
    )
    foreign_prepared = prepare_measured_resolution(
        foreign_sample,
        prepared.resolution_dataset,
    )
    foreign_selection = replace(selection, dataset=foreign_sample)

    with pytest.raises(ValueError, match="active anchor scientific context"):
        execute_multi_q_branch(
            foreign_prepared,
            foreign_selection,
            configurations,
            anchor_group_index=2,
            anchor_fit=anchor,
        )


def test_changed_q_assignment_rejects_stale_anchor_result(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem
    reassigned, rebound_selection = _with_q_values(
        prepared,
        selection,
        [0.55, 0.8, 1.05, 1.3, 1.55],
    )

    with pytest.raises(ValueError, match="active anchor scientific context"):
        execute_multi_q_branch(
            reassigned,
            rebound_selection,
            configurations,
            anchor_group_index=2,
            anchor_fit=anchor,
        )


def test_changed_retained_selection_rejects_stale_anchor_result(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem
    changed_selection = selection.with_group_range(
        2,
        lower_energy=-0.4,
        upper_energy=0.4,
    )

    with pytest.raises(ValueError, match="active anchor scientific context"):
        execute_multi_q_branch(
            prepared,
            changed_selection,
            configurations,
            anchor_group_index=2,
            anchor_fit=anchor,
        )


def test_changed_prepared_resolution_rejects_stale_anchor_result(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem
    replacement = prepare_measured_resolution(
        prepared.sample_dataset,
        prepared.resolution_dataset,
    )
    assert replacement is not prepared

    with pytest.raises(ValueError, match="active anchor scientific context"):
        execute_multi_q_branch(
            replacement,
            selection,
            configurations,
            anchor_group_index=2,
            anchor_fit=anchor,
        )


@pytest.mark.parametrize(
    ("anchor_index", "expected_order"),
    [(0, [1, 2, 3, 4]), (4, [3, 2, 1, 0])],
)
def test_first_or_last_anchor_executes_only_one_side(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
    anchor_index: int,
    expected_order: list[int],
) -> None:
    prepared, selection, configurations, base_anchor = multi_q_problem
    anchor = _result_with_values(
        base_anchor,
        anchor_index,
        base_anchor.configuration,
    )
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(batch_core, "fit_single_q", _recording_fit(anchor, calls))

    result = execute_multi_q_branch(
        prepared,
        selection,
        configurations,
        anchor_group_index=anchor_index,
        anchor_fit=anchor,
    )

    assert [group for group, _model in calls] == expected_order
    assert result.outcome(anchor_index).fit_result is anchor


def test_failure_continues_from_most_recent_success_without_cross_side_seed(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, configurations, base_anchor = multi_q_problem
    anchor = _result_with_values(
        base_anchor,
        2,
        base_anchor.configuration,
        _elastic_value(0.7),
    )
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(
        batch_core,
        "fit_single_q",
        _recording_fit(anchor, calls, {0: 0.6, 3: 1.1}, {1}),
    )

    result = execute_multi_q_branch(
        prepared,
        selection,
        configurations,
        anchor_group_index=2,
        anchor_fit=anchor,
    )

    starts = {
        group: model.elastic_area.initial_value
        for group, model in calls
        if model.elastic_area is not None
    }
    assert starts == {1: 0.7, 0: 0.7, 3: 0.7, 4: 1.1}
    assert result.outcome(1).status is MultiQFitStatus.FAILED
    assert result.outcome(0).seed_group_index == 2
    assert result.outcome(3).seed_group_index == 2


def test_blocked_target_does_not_seed_next_target(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem
    one_point = selection.with_group_range(
        1,
        lower_energy=0.0,
        upper_energy=0.0,
    )
    anchor = fit_single_q(prepared, one_point, 2, configurations[2])
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(batch_core, "fit_single_q", _recording_fit(anchor, calls))

    result = execute_multi_q_branch(
        prepared,
        one_point,
        configurations,
        anchor_group_index=2,
        anchor_fit=anchor,
    )

    assert result.outcome(1).status is MultiQFitStatus.BLOCKED
    assert result.outcome(0).status is MultiQFitStatus.SUCCESS
    assert result.outcome(0).seed_group_index == 2
    assert 1 not in [group for group, _model in calls]


@pytest.mark.parametrize("blocker", ["missing", "topology"])
def test_pre_seed_blocker_has_no_seed_group_index(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    blocker: str,
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem
    targets: list[SpectralModelDefinition | None] = list(configurations)
    targets[1] = (
        None if blocker == "missing" else replace(configurations[1], elastic_area=None)
    )

    result = execute_multi_q_branch(
        prepared,
        selection,
        targets,
        anchor_group_index=2,
        anchor_fit=anchor,
    )

    assert result.outcome(1).status is MultiQFitStatus.BLOCKED
    assert result.outcome(1).seed_group_index is None


def test_unsuccessful_fit_is_retained_and_skipped_as_next_seed(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem
    anchor = _result_with_values(
        anchor,
        2,
        anchor.configuration,
        _elastic_value(0.7),
    )
    calls: list[tuple[int, SpectralModelDefinition]] = []

    def nonconverged_then_success(
        current_prepared: PreparedResolution,
        current_selection: FittingSelection,
        group_index: int,
        model: SpectralModelDefinition,
        *,
        max_nfev: int,
    ) -> FitResult:
        assert max_nfev == 2500
        calls.append((group_index, model))
        return _result_with_values(
            anchor,
            group_index,
            model,
            _elastic_value(9.0 if group_index == 1 else 0.6),
            successful=group_index != 1,
            q_value=current_prepared.q_value(group_index),
            prepared_resolution=current_prepared,
            selection=current_selection,
        )

    monkeypatch.setattr(batch_core, "fit_single_q", nonconverged_then_success)
    result = execute_multi_q_branch(
        prepared,
        selection,
        configurations,
        anchor_group_index=2,
        anchor_fit=anchor,
    )

    failed = result.outcome(1)
    assert failed.status is MultiQFitStatus.FAILED
    assert failed.fit_result is not None
    assert not failed.fit_result.diagnostics.optimizer_success
    assert result.outcome(0).status is MultiQFitStatus.SUCCESS
    assert result.outcome(0).seed_group_index == 2
    group_zero_submission = next(model for group, model in calls if group == 0)
    assert group_zero_submission.elastic_area is not None
    assert group_zero_submission.elastic_area.initial_value == pytest.approx(0.7)


def test_target_local_bounds_and_free_state_remain_authoritative(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, configurations, base_anchor = multi_q_problem
    anchor = _result_with_values(
        base_anchor,
        2,
        base_anchor.configuration,
        _elastic_value(1.4),
    )
    target = replace(
        configurations[1],
        elastic_area=_parameter(0.3, 0.2, 0.4, free=True),
    )
    fixed_target = replace(
        configurations[0],
        elastic_area=_parameter(0.25, 0.0, 2.0),
    )
    targets = list(configurations)
    targets[1] = target
    targets[0] = fixed_target
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(batch_core, "fit_single_q", _recording_fit(anchor, calls))

    execute_multi_q_branch(
        prepared,
        selection,
        targets,
        anchor_group_index=2,
        anchor_fit=anchor,
    )

    submitted = next(model for group, model in calls if group == 1)
    assert submitted.elastic_area == ParameterConfiguration(0.4, 0.2, 0.4, True)
    fixed_submission = next(model for group, model in calls if group == 0)
    assert fixed_submission.elastic_area == ParameterConfiguration(
        0.25,
        0.0,
        2.0,
        False,
    )


def test_free_parameters_use_previous_result_not_target_current_value(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, configurations, base_anchor = multi_q_problem
    anchor = _result_with_values(
        base_anchor,
        2,
        base_anchor.configuration,
        _elastic_value(0.37),
    )
    targets = list(configurations)
    targets[1] = replace(
        targets[1],
        elastic_area=_parameter(0.22, 0.2, 0.4, free=True),
    )
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(batch_core, "fit_single_q", _recording_fit(anchor, calls))

    execute_multi_q_branch(
        prepared,
        selection,
        targets,
        anchor_group_index=2,
        anchor_fit=anchor,
    )

    submitted = next(model for group, model in calls if group == 1)
    assert submitted.elastic_area is not None
    assert submitted.elastic_area.initial_value == pytest.approx(0.37)
    assert submitted.elastic_area.initial_value != pytest.approx(0.22)


def test_out_of_bounds_result_seed_uses_existing_interior_start_projection(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, configurations, base_anchor = multi_q_problem
    anchor = _result_with_values(
        base_anchor,
        2,
        base_anchor.configuration,
        _elastic_value(1.4),
    )
    targets: list[SpectralModelDefinition | None] = [None] * len(configurations)
    targets[2] = anchor.configuration
    targets[1] = replace(
        _model(),
        elastic_area=_parameter(0.22, 0.2, 0.4, free=True),
    )

    result = execute_multi_q_branch(
        prepared,
        selection,
        targets,
        anchor_group_index=2,
        anchor_fit=anchor,
        fit_excluded_groups={0, 3, 4},
        max_nfev=800,
    )

    fitted = result.outcome(1).fit_result
    assert fitted is not None
    parameter_index = next(
        index
        for index, parameter in enumerate(fitted.parameters)
        if ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA)
        in parameter.references
    )
    submitted_start = fitted.diagnostics.alternative_starts[0].start_parameter_values[
        parameter_index
    ]
    assert 0.2 < submitted_start < 0.4
    assert submitted_start == pytest.approx(0.4, abs=1.0e-9)


def test_optional_target_center_relationships_do_not_change_branch_composition(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem
    first = configurations[1].lorentzians[0]
    independent = replace(
        configurations[1],
        lorentzians=(
            replace(first, center=_parameter(0.02, -0.2, 0.2, free=True)),
            configurations[1].lorentzians[1],
        ),
    )
    targets = list(configurations)
    targets[1] = independent
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(batch_core, "fit_single_q", _recording_fit(anchor, calls))

    result = execute_multi_q_branch(
        prepared,
        selection,
        targets,
        anchor_group_index=2,
        anchor_fit=anchor,
    )

    assert result.outcome(1).status is MultiQFitStatus.SUCCESS
    submitted = next(model for group, model in calls if group == 1)
    assert submitted.lorentzians[0].center is not None


def test_many_to_one_seed_mapping_blocks_when_source_values_disagree(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, configurations, _anchor = multi_q_problem
    first = configurations[2].lorentzians[0]
    independent_anchor_model = replace(
        configurations[2],
        lorentzians=(
            replace(first, center=_parameter(0.02, -0.2, 0.2, free=True)),
            configurations[2].lorentzians[1],
        ),
    )
    anchor = fit_single_q(prepared, selection, 2, independent_anchor_model)
    anchor = _result_with_values(
        anchor,
        2,
        independent_anchor_model,
        {
            ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER): 0.01,
            ParameterReference(L1, ParameterFamily.CENTER): 0.05,
            ParameterReference(L2, ParameterFamily.CENTER): 0.01,
        },
    )

    result = execute_multi_q_branch(
        prepared,
        selection,
        configurations,
        anchor_group_index=2,
        anchor_fit=anchor,
    )

    blocked = result.outcome(1)
    assert blocked.status is MultiQFitStatus.BLOCKED
    assert blocked.seed_group_index is None
    assert "unambiguously" in blocked.diagnostics[0].message


def test_lorentzian_values_propagate_by_identity_without_width_rematching(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, configurations, base_anchor = multi_q_problem
    anchor = _result_with_values(
        base_anchor,
        2,
        base_anchor.configuration,
        {
            ParameterReference(L1, ParameterFamily.FWHM): 0.4,
            ParameterReference(L2, ParameterFamily.FWHM): 0.1,
            ParameterReference(L1, ParameterFamily.AREA): 0.2,
            ParameterReference(L2, ParameterFamily.AREA): 0.7,
        },
    )
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(batch_core, "fit_single_q", _recording_fit(anchor, calls))

    execute_multi_q_branch(
        prepared,
        selection,
        configurations,
        anchor_group_index=2,
        anchor_fit=anchor,
    )

    submitted = calls[0][1]
    assert tuple(component.identity for component in submitted.lorentzians) == (L1, L2)
    assert tuple(
        component.fwhm.initial_value for component in submitted.lorentzians
    ) == (
        0.4,
        0.1,
    )
    assert tuple(
        component.area.initial_value for component in submitted.lorentzians
    ) == (
        0.2,
        0.7,
    )


def test_each_success_is_an_independent_fit_result_and_origin_is_not_an_input(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem

    result = execute_multi_q_branch(
        prepared,
        selection,
        configurations,
        anchor_group_index=2,
        anchor_fit=anchor,
    )

    fits = [item.fit_result for item in result.outcomes]
    assert all(item.status is MultiQFitStatus.SUCCESS for item in result.outcomes)
    assert len({id(fit) for fit in fits}) == len(fits)
    assert fits[2] is anchor


def test_cancellation_preserves_completed_and_leaves_remaining_not_run(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(batch_core, "fit_single_q", _recording_fit(anchor, calls))
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks == 2

    result = execute_multi_q_branch(
        prepared,
        selection,
        configurations,
        anchor_group_index=2,
        anchor_fit=anchor,
        cancel_requested=cancelled,
    )

    assert result.status is MultiQExecutionStatus.CANCELLED
    assert result.outcome(1).status is MultiQFitStatus.SUCCESS
    assert result.outcome(2).status is MultiQFitStatus.SUCCESS
    assert result.outcome(0).status is MultiQFitStatus.NOT_RUN
    assert result.outcome(3).status is MultiQFitStatus.NOT_RUN
    assert result.outcome(4).status is MultiQFitStatus.NOT_RUN
    assert [group for group, _model in calls] == [1]


def test_fit_and_derived_exclusions_are_independent_and_do_not_seed(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(batch_core, "fit_single_q", _recording_fit(anchor, calls))

    result = execute_multi_q_branch(
        prepared,
        selection,
        configurations,
        anchor_group_index=2,
        anchor_fit=anchor,
        fit_excluded_groups={1},
        derived_result_excluded_groups={0, 3},
    )

    assert [group for group, _model in calls] == [0, 3, 4]
    assert result.outcome(1).status is MultiQFitStatus.EXCLUDED
    assert result.outcome(1).seed_group_index is None
    assert result.outcome(0).seed_group_index == 2
    assert result.outcome(0).derived_result_excluded
    assert result.outcome(3).status is MultiQFitStatus.SUCCESS
    assert result.outcome(3).derived_result_excluded


def test_method_composition_is_mandatory_but_optional_state_obeys_defaults_and_override(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    source = _manual_model(slope_free=False)
    area = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA)
    source = replace_parameter_intent(
        source,
        area,
        ManualParameterIntent(7.0, 2.0, 9.0, free=True),
    )
    source = replace(
        source,
        parameter_ties=(
            ManualParameterTieState(
                "areas",
                (
                    area,
                    ParameterReference(L1, ParameterFamily.AREA),
                ),
                ManualParameterIntent(7.0, 2.0, 9.0, free=True),
            ),
        ),
    )
    defaults = MethodTransferOptions(
        user_bounds=True,
        parameter_relationships=True,
        resolution=True,
        q_bins=True,
        fitting_selection=True,
    )
    context = _manual_context(prepared, selection)
    method = capture_fitting_method(
        source,
        context=context,
        transfer_defaults=defaults,
    )

    assert method.composition.elastic_present
    assert method.composition.lorentzian_identities == (L1, L2)
    assert method.composition.background is BackgroundModel.LINEAR
    assert not hasattr(method.parameter(area), "current_value")
    assert method.parameter(area).fixed_value is None

    applied = apply_fitting_method(
        method,
        None,
        target_resolution_dataset_id="other-resolution",
        overrides=MethodTransferOverrides(free_fixed=False),
    )
    assert applied.model.parameter_intent(area).current_value == pytest.approx(1.0)
    assert applied.model.parameter_intent(area).user_lower_limit == pytest.approx(2.0)
    assert applied.model.tie_for(area) is not None
    assert applied.resolution_dataset_id == "resolution"
    assert applied.q_bins is prepared.sample_dataset.q_bins
    assert applied.fitting_selection is selection

    slope = ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.SLOPE)
    assert applied.model.parameter_intent(slope).free
    fixed_slope = apply_fitting_method(
        method,
        None,
        overrides=MethodTransferOverrides(free_fixed=True),
    )
    assert fixed_slope.model.parameter_intent(slope).current_value == 0.0
    assert not fixed_slope.model.parameter_intent(slope).free


def test_transfer_off_retains_existing_target_or_uses_defaults_not_source_values() -> (
    None
):
    source = _manual_model()
    area = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA)
    source = replace_parameter_intent(
        source,
        area,
        ManualParameterIntent(8.0, 4.0, 10.0, free=False),
    )
    method = capture_fitting_method(source)
    target = replace_parameter_intent(
        _manual_model(),
        area,
        ManualParameterIntent(0.4, 0.1, 0.7, free=True),
    )

    retained = apply_fitting_method(method, target).model.parameter_intent(area)
    defaulted = apply_fitting_method(method, None).model.parameter_intent(area)

    assert retained == ManualParameterIntent(0.4, 0.1, 0.7, free=True)
    assert defaulted.current_value == pytest.approx(1.0)
    assert defaulted.current_value != pytest.approx(8.0)
    assert defaulted.free


def test_method_constraints_are_revalidated_against_target_local_coverage(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    source = _manual_model()
    center = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER)
    source = replace_parameter_intent(
        source,
        center,
        ManualParameterIntent(5.5, 5.0, 6.0),
    )
    method = capture_fitting_method(
        source,
        transfer_defaults=MethodTransferOptions(user_bounds=True),
    )
    applied = apply_fitting_method(method, None)

    with pytest.raises(ValueError, match="no intersection"):
        materialize_manual_model(applied.model, prepared, selection, 0)


def test_working_and_saved_method_result_lifecycles_are_independent(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    _prepared, _selection, _configurations, anchor = multi_q_problem
    first_method = capture_fitting_method(_manual_model())
    edited_method = replace(
        first_method,
        transfer_defaults=MethodTransferOptions(user_bounds=True),
    )
    state = save_working_method(
        set_working_method(FittingWorkspaceState(), first_method),
        "saved method",
    )
    state = set_working_method(state, edited_method)
    anchor_snapshot = _issue_result_snapshot(
        anchor,
        tuple(
            _manual_model() if index == anchor.provenance.group_index else None
            for index in range(len(_prepared.spectra))
        ),
        ResultOrigin.MANUAL_FIT,
    )
    state = save_current_result(
        set_current_result(state, anchor_snapshot), "saved result"
    )
    replacement_result = replace(anchor, fitted_model=anchor.configuration)
    replacement_snapshot = _issue_result_snapshot(
        replacement_result,
        anchor_snapshot.working_states,
        ResultOrigin.MANUAL_FIT,
    )
    state = set_current_result(state, replacement_snapshot)

    assert state.saved_methods[0].method is first_method
    assert state.working_method is edited_method
    group = anchor.provenance.group_index
    assert state.saved_results[0].result.point(group).fit_result is anchor
    assert state.current_result is not None
    assert state.current_result.point(group).fit_result is replacement_result
    assert state.current_saved_result_index == 0
    assert state.saved_results[0].modified


def _s5_constraint_model() -> ManualModelState:
    third = ComponentIdentity(ComponentFamily.LORENTZIAN, "s5_l3")
    base = _manual_model()
    first, second = base.lorentzians
    disabled_area = ManualParameterIntent(
        second.area.current_value,
        0.1,
        0.9,
        user_bounds_enabled=False,
    )
    first = replace(first, center_group="shared-center")
    second = replace(
        second,
        area=disabled_area,
        center=ManualParameterIntent(-0.04),
    )
    third_component = ManualLorentzianState(
        area=ManualParameterIntent(0.15),
        fwhm=ManualParameterIntent(0.5),
        center=ManualParameterIntent(0.08),
        identity=third,
    )
    center_members = (
        ParameterReference(second.identity, ParameterFamily.CENTER),
        ParameterReference(third, ParameterFamily.CENTER),
    )
    noncenter_members = (
        ParameterReference(first.identity, ParameterFamily.FWHM),
        ParameterReference(second.identity, ParameterFamily.FWHM),
    )
    return replace(
        base,
        lorentzians=(first, second, third_component),
        b0=ManualParameterIntent(-0.03, -1.0, 1.0, free=False),
        center_groups=(
            ManualCenterGroupState(
                "shared-center",
                ManualParameterIntent(0.01, -0.2, 0.2),
            ),
        ),
        elastic_center_group="shared-center",
        parameter_ties=(
            ManualParameterTieState(
                "center-chain",
                center_members,
                ManualParameterIntent(0.02),
            ),
            ManualParameterTieState(
                "width-chain",
                noncenter_members,
                ManualParameterIntent(0.2),
            ),
        ),
    )


def test_active_transferable_constraints_expose_only_effective_s5_state() -> None:
    model = _s5_constraint_model()
    active = active_transferable_constraints(model)
    disabled_area = ParameterReference(L2, ParameterFamily.AREA)
    fixed_offset = ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.OFFSET)

    bounds = {item.reference: item for item in active.bounds}
    elastic_area = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA)
    assert disabled_area not in bounds
    assert bounds[elastic_area].lower == 0.0
    assert bounds[elastic_area].upper == 2.0
    assert {item.reference for item in active.fixed} == {fixed_offset}
    assert active.fixed[0].value == pytest.approx(-0.03)

    ties = {item.identity: item.members for item in active.center_ties}
    group_identity = CenterTieIdentity(CenterTieKind.CENTER_GROUP, "shared-center")
    chain_identity = CenterTieIdentity(CenterTieKind.PARAMETER_TIE, "center-chain")
    assert ties[group_identity] == (
        ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER),
        ParameterReference(L1, ParameterFamily.CENTER),
    )
    assert ties[chain_identity] == (
        ParameterReference(L2, ParameterFamily.CENTER),
        ParameterReference(
            ComponentIdentity(ComponentFamily.LORENTZIAN, "s5_l3"),
            ParameterFamily.CENTER,
        ),
    )
    assert all(item.identity.group_id != "width-chain" for item in active.center_ties)


def test_active_transferable_constraints_derive_legacy_tie_from_structure() -> None:
    active = active_transferable_constraints(_manual_model())
    legacy = next(
        item
        for item in active.center_ties
        if item.identity.kind is CenterTieKind.LEGACY_SHARED_CENTER
    )
    assert legacy.identity.group_id is None
    assert legacy.members == (
        ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER),
        ParameterReference(L1, ParameterFamily.CENTER),
        ParameterReference(L2, ParameterFamily.CENTER),
    )


def test_s5_transfer_policy_selects_active_constraints_atomically() -> None:
    model = _s5_constraint_model()
    bounds_reference = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA)
    fixed_reference = ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.OFFSET)
    tie_identity = CenterTieIdentity(CenterTieKind.PARAMETER_TIE, "center-chain")

    policy = MultiQConstraintTransferPolicy(
        model,
        bounds=frozenset((bounds_reference,)),
        fixed=frozenset((fixed_reference,)),
        center_ties=frozenset((tie_identity,)),
    )
    assert policy.bounds == frozenset((bounds_reference,))
    assert policy.fixed == frozenset((fixed_reference,))
    assert policy.center_ties == frozenset((tie_identity,))
    assert MultiQConstraintTransferPolicy(model) == MultiQConstraintTransferPolicy(
        model
    )

    inactive_bounds = ParameterReference(L2, ParameterFamily.AREA)
    with pytest.raises(ValueError, match="inactive or unknown reference"):
        MultiQConstraintTransferPolicy(model, bounds=frozenset((inactive_bounds,)))
    with pytest.raises(ValueError, match="inactive or unknown reference"):
        MultiQConstraintTransferPolicy(
            model,
            fixed=frozenset(
                (ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.SLOPE),)
            ),
        )
    with pytest.raises(ValueError, match="inactive or unknown tie"):
        MultiQConstraintTransferPolicy(
            model,
            center_ties=frozenset(
                (CenterTieIdentity(CenterTieKind.CENTER_GROUP, "missing"),)
            ),
        )
    with pytest.raises(ValueError, match="CenterTieIdentity"):
        MultiQConstraintTransferPolicy(
            model,
            center_ties=frozenset(
                (ParameterReference(L2, ParameterFamily.CENTER),)  # type: ignore[arg-type]
            ),
        )


def test_current_result_save_and_save_as_lifecycle(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    _prepared, _selection, _configurations, anchor = multi_q_problem
    first_rerun = replace(anchor, fitted_model=anchor.configuration)
    second_rerun = replace(anchor, evaluation=anchor.evaluation)

    anchor_snapshot = _issue_result_snapshot(
        anchor,
        tuple(
            _manual_model() if index == anchor.provenance.group_index else None
            for index in range(len(_prepared.spectra))
        ),
        ResultOrigin.MANUAL_FIT,
    )
    first_snapshot = _issue_result_snapshot(
        first_rerun, anchor_snapshot.working_states, ResultOrigin.MANUAL_FIT
    )
    second_snapshot = _issue_result_snapshot(
        second_rerun, anchor_snapshot.working_states, ResultOrigin.MANUAL_FIT
    )
    state = set_current_result(FittingWorkspaceState(), anchor_snapshot)
    assert state.saved_results == ()
    assert state.associated_saved_result is None
    with pytest.raises(ValueError, match="use Save As"):
        save_current_result(state)

    state = save_current_result_as(state, "first snapshot")
    assert state.current_saved_result_index == 0
    assert state.associated_saved_result is state.saved_results[0]
    group = anchor.provenance.group_index
    assert state.saved_results[0].result.point(group).fit_result is anchor

    state = set_current_result(
        state,
        first_snapshot,
    )
    assert state.current_saved_result_index == 0
    assert state.saved_results[0].result.point(group).fit_result is anchor
    state = save_current_result(state)
    assert len(state.saved_results) == 1
    assert state.saved_results[0].name == "first snapshot"
    assert state.saved_results[0].result.point(group).fit_result is first_rerun

    state = save_current_result_as(state, "second snapshot")
    assert state.current_saved_result_index == 1
    assert len(state.saved_results) == 2
    assert state.saved_results[0].result.point(group).fit_result is first_rerun
    assert state.saved_results[1].result.point(group).fit_result is first_rerun

    state = set_current_result(
        state,
        second_snapshot,
    )
    state = save_current_result(state)
    assert len(state.saved_results) == 2
    assert state.saved_results[0].result.point(group).fit_result is first_rerun
    assert state.saved_results[1].result.point(group).fit_result is second_rerun

    state = set_current_result(state, anchor_snapshot)
    assert state.current_result is not None
    assert state.current_result.point(group).fit_result is anchor
    assert state.current_saved_result_index == 1
    assert state.saved_results[1].modified
    assert tuple(
        item.result.point(group).fit_result for item in state.saved_results
    ) == (
        first_rerun,
        second_rerun,
    )


def test_result_snapshot_requires_exact_successful_q_state_scope(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, _selection, _configurations, anchor = multi_q_problem
    group = anchor.provenance.group_index
    states = tuple(
        _manual_model() if index == group else None
        for index in range(len(prepared.spectra))
    )
    issued = _issue_result_snapshot(anchor, states, ResultOrigin.MANUAL_FIT)
    with pytest.raises(ValueError, match="issued by execution workflow"):
        ResultSnapshot(issued.result)
    with pytest.raises(ValueError, match="issued by execution workflow"):
        replace(issued, result=issued.result)
    forged = object.__new__(ResultSnapshot)
    for name in ("result",):
        object.__setattr__(forged, name, getattr(issued, name))
    with pytest.raises(ValueError, match="created at execution"):
        set_current_result(FittingWorkspaceState(), forged)
    with pytest.raises(ValueError, match="active item"):
        FittingWorkspaceState(active_item=cast(ResultWorkingCopy, forged))
    with pytest.raises(ValueError, match="ResultSnapshot"):
        append_saved_result(FittingWorkspaceState(), "forged", forged)
    with pytest.raises(ValueError, match="only its successful Q"):
        _issue_result_snapshot(anchor, (None,) * len(states), ResultOrigin.MANUAL_FIT)
    with pytest.raises(ValueError, match="only its successful Q"):
        _issue_result_snapshot(
            anchor,
            tuple(_manual_model() for _ in states),
            ResultOrigin.MANUAL_FIT,
        )

    outcomes = tuple(
        MultiQFitOutcome(index, MultiQFitStatus.SUCCESS, anchor)
        if index == group
        else MultiQFitOutcome(index, MultiQFitStatus.NOT_RUN)
        for index in range(len(states))
    )
    branch = MultiQBranchResult(group, MultiQExecutionStatus.CANCELLED, outcomes)
    _issue_result_snapshot(branch, states, ResultOrigin.MANUAL_FIT_ALL_Q)
    with pytest.raises(ValueError, match="exactly at successful Qs"):
        _issue_result_snapshot(
            branch,
            tuple(_manual_model() for _ in states),
            ResultOrigin.MANUAL_FIT_ALL_Q,
        )
    with pytest.raises(ValueError, match="exactly at successful Qs"):
        _issue_result_snapshot(
            branch,
            (None,) * len(states),
            ResultOrigin.MANUAL_FIT_ALL_Q,
        )


def test_workspace_append_remove_preserves_current_and_entry_identity(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, _selection, _configurations, anchor = multi_q_problem
    current = _issue_result_snapshot(
        anchor,
        tuple(
            _manual_model() if index == anchor.provenance.group_index else None
            for index in range(len(prepared.spectra))
        ),
        ResultOrigin.MANUAL_FIT,
    )
    other = _issue_result_snapshot(
        replace(anchor, fitted_model=anchor.configuration),
        current.working_states,
        ResultOrigin.MANUAL_FIT,
    )
    state = save_current_result_as(
        set_current_result(FittingWorkspaceState(), current), "associated"
    )
    associated = state.saved_results[0]
    state = append_saved_result(state, "first", other)
    first = state.saved_results[1]
    state = append_saved_result(state, "second", current)
    second = state.saved_results[2]
    assert state.current is current
    assert state.current_saved_result_index == 0
    assert state.saved_results == (associated, first, second)

    state = remove_saved_result(state, first)
    assert state.saved_results[0] is associated
    assert state.saved_results[1] is second
    assert state.current is current
    assert state.current_saved_result_index == 0
    with pytest.raises(ValueError, match="not in this workspace"):
        remove_saved_result(state, first)
    with pytest.raises(ValueError, match="issued by the workflow"):
        replace(second)
    same_content_other_entry = append_saved_result(
        FittingWorkspaceState(), "second", current
    ).saved_results[0]
    with pytest.raises(ValueError, match="not in this workspace"):
        remove_saved_result(state, same_content_other_entry)

    state = remove_saved_result(state, associated)
    assert state.current is None
    assert state.active_item is None
    assert state.current_saved_result_index is None
    assert state.saved_results == (second,)
    state = save_current_result_as(set_current_result(state, current), "latest")
    assert state.current_saved_result_index == 1
    state = remove_saved_result(state, second)
    assert state.current_saved_result_index == 0
    assert state.saved_results[0].name == "latest"


def test_manual_run_snapshot_is_unchanged_by_later_draft_edit(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)
    execution = run_and_adopt_manual_fit(project, draft, 2, max_nfev=800)
    assert execution.success
    assert execution.snapshot is not None
    assert execution.adopted_draft is not None
    snapshot = execution.snapshot
    model = snapshot.working_states[2]
    assert model is not None
    reference = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA)
    intent = model.parameter_intent(reference)
    edited = update_manual_parameter(
        execution.adopted_draft,
        2,
        reference,
        ManualParameterEdit(
            intent.current_value + 0.01,
            intent.user_lower_limit,
            intent.user_upper_limit,
            intent.free,
            intent.user_bounds_enabled,
        ),
    )
    assert edited.setup(2).model is not model
    assert snapshot.working_states[2] is model
    assert model.parameter_intent(reference) == intent
    state = save_current_result_as(
        set_current_result(FittingWorkspaceState(), snapshot), "before edit"
    )
    assert state.saved_results[0].snapshot is snapshot


def test_fit_all_q_rejects_missing_current_result_without_execution(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)

    def unexpected_fit(*_args: object, **_kwargs: object) -> FitResult:
        raise AssertionError("missing Current must not execute a fit")

    monkeypatch.setattr(batch_core, "fit_single_q", unexpected_fit)
    with pytest.raises(ValueError, match="current successful Single-Q Result"):
        fit_all_q_from_current(
            project,
            draft,
            anchor_group_index=2,
            max_nfev=800,
        )


def test_fit_all_q_rejects_unsuccessful_current_result_without_execution(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)
    anchor = run_manual_fit(project, draft, 2, max_nfev=800)
    unsuccessful = replace(
        anchor,
        diagnostics=replace(anchor.diagnostics, optimizer_success=False),
    )

    def unexpected_fit(*_args: object, **_kwargs: object) -> FitResult:
        raise AssertionError("unsuccessful Current must not execute a fit")

    monkeypatch.setattr(batch_core, "fit_single_q", unexpected_fit)
    with pytest.raises(ValueError, match="successful current Result"):
        fit_all_q_from_current(
            project,
            draft,
            anchor_group_index=2,
            current_fit=unsuccessful,
        )


def test_fit_all_q_retains_anchor_and_existing_sequential_seed_chains(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)
    anchor = run_manual_fit(project, draft, 2, max_nfev=800)
    anchor = _result_with_values(
        anchor,
        2,
        anchor.configuration,
        _elastic_value(1.0),
    )
    source = draft.setup(2).model
    assert source is not None
    policy = MultiQConstraintTransferPolicy(source)
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(
        batch_core,
        "fit_single_q",
        _recording_fit(anchor, calls, {3: 1.3}, {1}),
    )

    result = fit_all_q_from_current(
        project,
        draft,
        anchor_group_index=2,
        current_fit=anchor,
        transfer_policy=policy,
    )

    retained_anchor = result.outcome(2).fit_result
    assert retained_anchor is not None
    assert retained_anchor.parameters == anchor.parameters
    assert retained_anchor.diagnostics == anchor.diagnostics
    assert retained_anchor.provenance == anchor.provenance
    assert [group for group, _model in calls] == [1, 0, 3, 4]
    starts = {
        group: model.elastic_area.initial_value
        for group, model in calls
        if model.elastic_area is not None
    }
    assert starts == {1: 1.0, 0: 1.0, 3: 1.0, 4: 1.3}
    assert result.outcome(1).status is MultiQFitStatus.FAILED
    assert result.outcome(0).seed_group_index == 2
    assert result.outcome(3).seed_group_index == 2
    assert result.outcome(4).seed_group_index == 3


def test_fit_all_q_exposes_start_and_terminal_progress_without_changing_results(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)
    anchor = run_manual_fit(project, draft, 2, max_nfev=800)
    source = draft.setup(2).model
    assert source is not None
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(batch_core, "fit_single_q", _recording_fit(anchor, calls))
    started: list[int] = []
    terminal: list[MultiQFitOutcome] = []

    result = fit_all_q_from_current(
        project,
        draft,
        anchor_group_index=2,
        current_fit=anchor,
        transfer_policy=MultiQConstraintTransferPolicy(source),
        fit_excluded_groups=(0,),
        group_started_callback=started.append,
        progress_callback=terminal.append,
    )

    assert started == [1, 3, 4]
    assert [(item.group_index, item.status) for item in terminal] == [
        (2, MultiQFitStatus.SUCCESS),
        (0, MultiQFitStatus.EXCLUDED),
        (1, MultiQFitStatus.SUCCESS),
        (3, MultiQFitStatus.SUCCESS),
        (4, MultiQFitStatus.SUCCESS),
    ]
    assert all(item is result.outcome(item.group_index) for item in terminal)
    assert [group for group, _model in calls] == started


def _s5b_source_and_target_models() -> tuple[ManualModelState, ManualModelState]:
    base = _manual_model(background=BackgroundModel.LINEAR)
    first, second = base.lorentzians
    source = replace(
        base,
        elastic_area=ManualParameterIntent(0.8, 0.0, 2.0),
        lorentzians=(
            replace(
                first,
                area=ManualParameterIntent(0.35, 0.2, 0.4),
                center_group="selected-center",
            ),
            replace(second, center=ManualParameterIntent(0.04)),
        ),
        b0=ManualParameterIntent(-0.03, -1.0, 1.0, free=False),
        center_groups=(
            ManualCenterGroupState(
                "selected-center",
                ManualParameterIntent(0.01, -0.2, 0.2),
            ),
        ),
        elastic_center_group="selected-center",
        parameter_ties=(
            ManualParameterTieState(
                "unselected-center",
                (ParameterReference(L2, ParameterFamily.CENTER),),
                ManualParameterIntent(0.04),
            ),
            ManualParameterTieState(
                "noncenter-width",
                (
                    ParameterReference(L1, ParameterFamily.FWHM),
                    ParameterReference(L2, ParameterFamily.FWHM),
                ),
                ManualParameterIntent(0.2, free=False),
            ),
        ),
    )
    target = replace(
        base,
        energy_shift=ManualParameterIntent(0.0),
        elastic_area=ManualParameterIntent(0.4, 0.1, 0.5),
        lorentzians=(
            replace(
                first,
                area=ManualParameterIntent(0.3, -1.0, 0.5),
                fwhm=ManualParameterIntent(0.7, 0.05, 1.0),
            ),
            replace(second, fwhm=ManualParameterIntent(0.6, 0.05, 1.0)),
        ),
        b0=ManualParameterIntent(0.2, -0.5, 0.5, free=True),
        center_groups=(),
        elastic_center_group=None,
        parameter_ties=(),
    )
    return source, target


def test_fit_all_q_applies_only_selected_s5_constraints(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)
    source, target = _s5b_source_and_target_models()
    draft = draft.with_model(2, source).with_model(1, target)
    anchor = run_manual_fit(project, draft, 2, max_nfev=800)
    selected_bounds = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA)
    selected_fixed = ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.OFFSET)
    selected_tie = CenterTieIdentity(
        CenterTieKind.CENTER_GROUP,
        "selected-center",
    )
    policy = MultiQConstraintTransferPolicy(
        source,
        bounds=frozenset((selected_bounds,)),
        fixed=frozenset((selected_fixed,)),
        center_ties=frozenset((selected_tie,)),
    )
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(batch_core, "fit_single_q", _recording_fit(anchor, calls))

    fit_all_q_from_current(
        project,
        draft,
        anchor_group_index=2,
        current_fit=anchor,
        transfer_policy=policy,
    )

    submitted = next(model for group, model in calls if group == 1)
    assert submitted.elastic_area is not None
    assert submitted.elastic_area.lower_bound == pytest.approx(0.0)
    assert submitted.elastic_area.upper_bound == pytest.approx(2.0)
    first_component = next(
        item for item in submitted.lorentzians if item.identity == L1
    )
    assert first_component.area.lower_bound == pytest.approx(0.0)
    assert first_component.area.upper_bound == pytest.approx(0.5)
    assert submitted.b0 is not None
    assert submitted.b0.initial_value == pytest.approx(-0.03)
    assert not submitted.b0.free
    assert first_component.fwhm.free
    assert first_component.fwhm.initial_value == pytest.approx(
        anchor.parameter_by_reference(
            ParameterReference(L1, ParameterFamily.FWHM)
        ).value
    )
    assert submitted.elastic_center_group == "selected-center"
    assert first_component.center_group == "selected-center"
    assert all(
        group.group_id != "unselected-center" for group in submitted.parameter_ties
    )
    assert all(
        group.group_id != "noncenter-width" for group in submitted.parameter_ties
    )


def test_fit_all_q_transfers_selected_general_center_tie_atomically(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)
    base = _manual_model(background=BackgroundModel.LINEAR)
    first, second = base.lorentzians
    members = (
        ParameterReference(L1, ParameterFamily.CENTER),
        ParameterReference(L2, ParameterFamily.CENTER),
    )
    source = replace(
        base,
        lorentzians=(
            replace(first, center=ManualParameterIntent(0.01)),
            replace(second, center=ManualParameterIntent(0.01)),
        ),
        center_groups=(
            ManualCenterGroupState("unselected-elastic", ManualParameterIntent(0.01)),
        ),
        elastic_center_group="unselected-elastic",
        parameter_ties=(
            ManualParameterTieState(
                "selected-center-chain",
                members,
                ManualParameterIntent(0.01),
            ),
        ),
    )
    draft = draft.with_model(2, source)
    anchor = run_manual_fit(project, draft, 2, max_nfev=800)
    policy = MultiQConstraintTransferPolicy(
        source,
        center_ties=frozenset(
            (
                CenterTieIdentity(
                    CenterTieKind.PARAMETER_TIE,
                    "selected-center-chain",
                ),
            )
        ),
    )
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(batch_core, "fit_single_q", _recording_fit(anchor, calls))

    fit_all_q_from_current(
        project,
        draft,
        anchor_group_index=2,
        current_fit=anchor,
        transfer_policy=policy,
    )

    submitted = next(model for group, model in calls if group == 1)
    transferred = next(
        group
        for group in submitted.parameter_ties
        if group.group_id == "selected-center-chain"
    )
    assert transferred.members == members
    assert all(
        group.group_id != "unselected-elastic" for group in submitted.center_groups
    )


def _s5b_center_tie_models(
    source_intent: ManualParameterIntent,
    target_first: ManualParameterIntent,
    target_second: ManualParameterIntent,
    *,
    reverse_members: bool = False,
) -> tuple[
    ManualModelState,
    ManualModelState,
    tuple[ParameterReference, ParameterReference],
]:
    base = _manual_model(background=BackgroundModel.LINEAR)
    first, second = base.lorentzians
    members = (
        ParameterReference(L1, ParameterFamily.CENTER),
        ParameterReference(L2, ParameterFamily.CENTER),
    )
    stored_members = tuple(reversed(members)) if reverse_members else members
    source = replace(
        base,
        lorentzians=(
            replace(first, center=source_intent),
            replace(second, center=source_intent),
        ),
        parameter_ties=(
            ManualParameterTieState(
                "selected-center-chain",
                stored_members,
                source_intent,
            ),
        ),
    )
    target = replace(
        base,
        lorentzians=(
            replace(first, center=target_first),
            replace(second, center=target_second),
        ),
        parameter_ties=(),
    )
    return source, target, members


def _run_s5b_center_tie_transfer(
    problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
    source: ManualModelState,
    target: ManualModelState,
    *,
    bounds: frozenset[ParameterReference] = frozenset(),
    fixed: frozenset[ParameterReference] = frozenset(),
) -> tuple[MultiQBranchResult, list[tuple[int, SpectralModelDefinition]]]:
    prepared, selection, _configurations, _anchor = problem
    project, _sample, draft = _workflow_problem(prepared, selection)
    draft = draft.with_model(2, source).with_model(1, target)
    anchor = run_manual_fit(project, draft, 2, max_nfev=800)
    policy = MultiQConstraintTransferPolicy(
        source,
        bounds=bounds,
        fixed=fixed,
        center_ties=frozenset(
            (
                CenterTieIdentity(
                    CenterTieKind.PARAMETER_TIE,
                    "selected-center-chain",
                ),
            )
        ),
    )
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(batch_core, "fit_single_q", _recording_fit(anchor, calls))
    result = fit_all_q_from_current(
        project,
        draft,
        anchor_group_index=2,
        current_fit=anchor,
        transfer_policy=policy,
    )
    return result, calls


def test_s5_center_tie_only_blocks_conflicting_target_free_fixed_state(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target, _members = _s5b_center_tie_models(
        ManualParameterIntent(0.01),
        ManualParameterIntent(0.1, -0.3, 0.3, free=False),
        ManualParameterIntent(0.2, -0.3, 0.3, free=True),
    )

    result, calls = _run_s5b_center_tie_transfer(
        multi_q_problem,
        monkeypatch,
        source,
        target,
    )

    assert result.outcome(1).status is MultiQFitStatus.BLOCKED
    assert all(group_index != 1 for group_index, _model in calls)


def test_s5_center_tie_only_blocks_conflicting_enabled_target_bounds(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target, _members = _s5b_center_tie_models(
        ManualParameterIntent(0.01),
        ManualParameterIntent(0.1, -0.3, 0.3),
        ManualParameterIntent(0.1, -0.1, 0.2),
    )

    result, calls = _run_s5b_center_tie_transfer(
        multi_q_problem,
        monkeypatch,
        source,
        target,
    )

    assert result.outcome(1).status is MultiQFitStatus.BLOCKED
    assert all(group_index != 1 for group_index, _model in calls)


@pytest.mark.parametrize("reverse_members", [False, True])
def test_s5_free_center_tie_uses_predecessor_seed_despite_different_currents(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
    reverse_members: bool,
) -> None:
    source, target, members = _s5b_center_tie_models(
        ManualParameterIntent(0.01, -0.3, 0.3),
        ManualParameterIntent(0.1, -0.3, 0.3),
        ManualParameterIntent(0.2, -0.3, 0.3),
        reverse_members=reverse_members,
    )

    result, calls = _run_s5b_center_tie_transfer(
        multi_q_problem,
        monkeypatch,
        source,
        target,
    )

    assert result.outcome(1).status is MultiQFitStatus.SUCCESS
    anchor = result.outcome(2).fit_result
    assert anchor is not None
    predecessor_value = anchor.parameter_by_reference(members[0]).value
    submitted = next(model for group, model in calls if group == 1)
    assert all(
        submitted.parameter_configuration(member).initial_value
        == pytest.approx(predecessor_value)
        for member in members
    )


def test_s5_center_tie_only_blocks_unequal_fixed_target_values(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target, _members = _s5b_center_tie_models(
        ManualParameterIntent(0.01),
        ManualParameterIntent(0.1, -0.3, 0.3, free=False),
        ManualParameterIntent(0.2, -0.3, 0.3, free=False),
    )

    result, calls = _run_s5b_center_tie_transfer(
        multi_q_problem,
        monkeypatch,
        source,
        target,
    )

    assert result.outcome(1).status is MultiQFitStatus.BLOCKED
    assert all(group_index != 1 for group_index, _model in calls)


def test_s5_center_tie_only_preserves_compatible_target_intent(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_intent = ManualParameterIntent(0.12, -0.3, 0.3, free=False)
    source, target, members = _s5b_center_tie_models(
        ManualParameterIntent(0.01),
        target_intent,
        target_intent,
    )

    result, calls = _run_s5b_center_tie_transfer(
        multi_q_problem,
        monkeypatch,
        source,
        target,
    )

    assert result.outcome(1).status is MultiQFitStatus.SUCCESS
    submitted = next(model for group, model in calls if group == 1)
    for member in members:
        parameter = submitted.parameter_configuration(member)
        assert parameter.initial_value == pytest.approx(0.12)
        assert parameter.lower_bound == pytest.approx(-0.3)
        assert parameter.upper_bound == pytest.approx(0.3)
        assert not parameter.free


def test_s5_center_tie_with_selected_fixed_resolves_target_state(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_intent = ManualParameterIntent(0.07, -0.3, 0.3, free=False)
    source, target, members = _s5b_center_tie_models(
        source_intent,
        ManualParameterIntent(0.1, -0.3, 0.3, free=False),
        ManualParameterIntent(0.2, -0.3, 0.3, free=True),
    )

    result, calls = _run_s5b_center_tie_transfer(
        multi_q_problem,
        monkeypatch,
        source,
        target,
        fixed=frozenset(members),
    )

    assert result.outcome(1).status is MultiQFitStatus.SUCCESS
    submitted = next(model for group, model in calls if group == 1)
    for member in members:
        parameter = submitted.parameter_configuration(member)
        assert parameter.initial_value == pytest.approx(0.07)
        assert not parameter.free


def test_s5_center_tie_with_selected_bounds_resolves_target_state(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_intent = ManualParameterIntent(0.1, -0.08, 0.16)
    source, target, members = _s5b_center_tie_models(
        source_intent,
        ManualParameterIntent(0.1, -0.3, 0.3),
        ManualParameterIntent(0.1, -0.1, 0.2),
    )

    result, calls = _run_s5b_center_tie_transfer(
        multi_q_problem,
        monkeypatch,
        source,
        target,
        bounds=frozenset(members),
    )

    assert result.outcome(1).status is MultiQFitStatus.SUCCESS
    submitted = next(model for group, model in calls if group == 1)
    for member in members:
        parameter = submitted.parameter_configuration(member)
        assert parameter.lower_bound == pytest.approx(-0.08)
        assert parameter.upper_bound == pytest.approx(0.16)
        assert parameter.free


@pytest.mark.parametrize("reverse_members", [False, True])
def test_s5_center_tie_member_order_does_not_select_an_intent(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
    reverse_members: bool,
) -> None:
    target_intent = ManualParameterIntent(0.13, -0.25, 0.35, free=False)
    source, target, members = _s5b_center_tie_models(
        ManualParameterIntent(-0.02),
        target_intent,
        target_intent,
        reverse_members=reverse_members,
    )

    result, calls = _run_s5b_center_tie_transfer(
        multi_q_problem,
        monkeypatch,
        source,
        target,
    )

    assert result.outcome(1).status is MultiQFitStatus.SUCCESS
    submitted = next(model for group, model in calls if group == 1)
    assert {
        (
            submitted.parameter_configuration(member).initial_value,
            submitted.parameter_configuration(member).lower_bound,
            submitted.parameter_configuration(member).upper_bound,
            submitted.parameter_configuration(member).free,
        )
        for member in members
    } == {(0.13, -0.25, 0.35, False)}


def test_fit_all_q_legacy_method_arguments_are_explicitly_separate(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)
    anchor = run_manual_fit(project, draft, 2, max_nfev=800)
    source = draft.setup(2).model
    assert source is not None

    with pytest.raises(ValueError, match="MultiQConstraintTransferPolicy"):
        fit_all_q_from_current(
            project,
            draft,
            anchor_group_index=2,
            current_fit=anchor,
            method=capture_fitting_method(source),
        )


def test_fresh_selected_autofit_accepts_reprepared_aligned_nan_uncertainty(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, sample, draft = _workflow_with_nan_resolution_uncertainty(
        prepared,
        selection,
    )
    outcome = run_single_q_auto_fit(project, draft, 2, max_nfev=800)
    refreshed = resolve_manual_fit_context(project, sample, 2)
    assert refreshed.context is not None
    assert (
        refreshed.context.prepared_resolution
        is not outcome.scientific_context.prepared_resolution
    )
    assert np.array_equal(
        refreshed.context.prepared_resolution.spectra[2].normalized_uncertainty,
        outcome.scientific_context.prepared_resolution.spectra[
            2
        ].normalized_uncertainty,
        equal_nan=True,
    )
    selected = tuple(
        item.candidate
        for item in outcome.recommendation.candidate_results
        if item.success
    )[:2]
    assert len(selected) == 2

    result = continue_selected_auto_fit_candidates(
        project,
        draft,
        outcome,
        selected,
        max_nfev=800,
    )

    assert result.status is MultiQExecutionStatus.COMPLETED
    assert result.selected_candidates == selected
    assert len(result.branches) == 2


def test_prepared_context_nan_equality_is_aligned_and_not_nan_to_finite(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, _selection, _configurations, _anchor = multi_q_problem
    base = prepared.spectra[0].normalized_uncertainty
    aligned_nan = np.array(base, copy=True)
    aligned_nan[0] = np.nan
    finite = np.array(aligned_nan, copy=True)
    finite[0] = 1.0

    def context_view(uncertainty: FloatArray) -> PreparedResolution:
        spectra = tuple(
            SimpleNamespace(
                accepted_mask=item.accepted_mask,
                normalized_intensity=item.normalized_intensity,
                normalized_uncertainty=(
                    uncertainty if index == 0 else item.normalized_uncertainty
                ),
            )
            for index, item in enumerate(prepared.spectra)
        )
        return cast(
            PreparedResolution,
            SimpleNamespace(
                sample_dataset=prepared.sample_dataset,
                resolution_dataset=prepared.resolution_dataset,
                diagnostics=prepared.diagnostics,
                padding_comparisons=prepared.padding_comparisons,
                sample_padding=prepared.sample_padding,
                resolution_padding=prepared.resolution_padding,
                spectra=spectra,
                acceptance_provenance=prepared.acceptance_provenance,
            ),
        )

    first = context_view(aligned_nan)
    second = context_view(np.array(aligned_nan, copy=True))
    changed = context_view(finite)

    assert multi_q_workflow._prepared_resolution_context_equal(first, second)
    assert not multi_q_workflow._prepared_resolution_context_equal(first, changed)


def test_selected_auto_candidates_create_ordered_independent_branches(
    selected_auto_problem: tuple[
        WorkflowProject,
        ProjectDataset,
        ManualFitDraft,
        SingleQAutoFitOutcome,
        tuple[CandidateFitResult, CandidateFitResult],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, _sample, draft, outcome, evidence = selected_auto_problem
    most_recommended, other_successful = evidence

    def unexpected_auto_rerun(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("selected branches must reuse one AutoFit evaluation")

    progress: list[SelectedAutoFitProgressEvent] = []

    monkeypatch.setattr(
        "ezqens.workflow.multi_q.run_single_q_auto_fit",
        unexpected_auto_rerun,
    )
    result = continue_selected_auto_fit_candidates(
        project,
        draft,
        outcome,
        (other_successful.candidate, most_recommended.candidate),
        progress_callback=progress.append,
        max_nfev=800,
    )

    assert result.status is MultiQExecutionStatus.COMPLETED
    assert result.selected_candidates == (
        other_successful.candidate,
        most_recommended.candidate,
    )
    assert all(len(item.working_states) == 5 for item in result.branches)
    assert result.branches[0].working_states is not result.branches[1].working_states
    assert [item.candidate for item in progress] == [
        *([other_successful.candidate] * 5),
        *([most_recommended.candidate] * 5),
    ]
    assert [item.outcome.group_index for item in progress] == [
        2,
        1,
        0,
        3,
        4,
        2,
        1,
        0,
        3,
        4,
    ]
    assert all(item.outcome.status is MultiQFitStatus.SUCCESS for item in progress)
    for item, original in zip(
        result.branches,
        (other_successful, most_recommended),
        strict=True,
    ):
        assert item.anchor_evidence is original
        assert item.anchor_evidence.fit is not None
        assert original.fit is not None
        assert item.anchor_evidence.fit.parameters == original.fit.parameters
        assert item.anchor_evidence.fit.diagnostics == original.fit.diagnostics
        branch = item.branch_result
        anchor = branch.outcome(branch.anchor_group_index).fit_result
        assert anchor is not None
        assert anchor.configuration is item.anchor_evidence.fit.configuration
        assert anchor.evaluation is item.anchor_evidence.fit.evaluation
        assert all(
            outcome.status is MultiQFitStatus.SUCCESS for outcome in branch.outcomes
        )
        expected_identities = {
            component.identity
            for component in item.anchor_evidence.fit.configuration.lorentzians
        }
        assert all(
            group.fit_result is not None
            and {
                component.identity
                for component in group.fit_result.configuration.lorentzians
            }
            == expected_identities
            for group in branch.outcomes
        )
    for group_index in (0, 1, 3, 4):
        first = result.branches[0].branch_result.outcome(group_index).fit_result
        second = result.branches[1].branch_result.outcome(group_index).fit_result
        assert first is not None and second is not None
        assert first is not second


def test_selected_branch_adopts_owned_manual_states_and_ignores_stale_target_setup(
    selected_auto_problem: tuple[
        WorkflowProject,
        ProjectDataset,
        ManualFitDraft,
        SingleQAutoFitOutcome,
        tuple[CandidateFitResult, CandidateFitResult],
    ],
) -> None:
    project, sample, draft, evaluation, evidence = selected_auto_problem
    candidate = evidence[0].candidate
    stale = ManualModelState(
        background=BackgroundModel.CONSTANT,
        b0=ManualParameterIntent(8.0, free=False),
    )
    draft = draft.with_model(1, stale)
    selected = continue_selected_auto_fit_candidates(
        project,
        draft,
        evaluation,
        (candidate,),
        fit_excluded_groups=(0, 3, 4),
        max_nfev=800,
    ).branches[0]
    assert selected.anchor_evidence is evidence[0]
    assert selected.evaluation is evaluation
    assert not hasattr(selected, "snapshot")
    assert selected.working_states[1] is not stale
    assert selected.branch_result.outcome(1).status is MultiQFitStatus.SUCCESS
    target_fit = selected.branch_result.outcome(1).fit_result
    assert target_fit is not None
    assert target_fit.configuration.b0 is not None
    assert target_fit.configuration.b0.free

    adopted = adopt_selected_auto_fit_branch(
        project,
        draft,
        evaluation,
        selected,
        center_group_id="selected-center",
    )
    assert adopted.snapshot is not None
    assert all(
        adopted.snapshot.result.point(item.group_index).outcome is item
        for item in adopted.branch_result.outcomes
    )
    assert adopted.snapshot.origin is ResultOrigin.AUTO_FIT_BRANCH
    with pytest.raises(ValueError, match="snapshot"):
        replace(adopted, snapshot=cast(ResultSnapshot, None))
    assert adopted.snapshot.candidate == candidate
    original_auto_fit = evidence[0].fit
    assert original_auto_fit is not None
    with pytest.raises(ValueError, match="issued by execution workflow"):
        ResultSnapshot(adopted.snapshot.result)
    unissued_auto = object.__new__(ResultSnapshot)
    object.__setattr__(unissued_auto, "result", adopted.snapshot.result)
    with pytest.raises(ValueError, match="created at execution"):
        set_current_result(FittingWorkspaceState(), unissued_auto)
    with pytest.raises(ValueError, match="ResultSnapshot"):
        append_saved_result(FittingWorkspaceState(), "unconverted", unissued_auto)
    with pytest.raises(ValueError, match="issued by execution workflow"):
        replace(adopted.snapshot, result=cast(UnifiedResult, selected.branch_result))
    anchor_state = adopted.adopted_draft.setup(2).model
    target_state = adopted.adopted_draft.setup(1).model
    assert anchor_state is not None and target_state is not None
    single = adopt_single_q_auto_fit_candidate(
        project,
        draft,
        evaluation,
        evidence[0],
        center_group_id="selected-center",
    )
    assert single.success and single.adopted_draft is not None
    single_state = single.adopted_draft.setup(2).model
    assert single_state is not None
    assert anchor_state.background is single_state.background
    assert anchor_state.b0 == single_state.b0
    assert tuple((item.area, item.fwhm) for item in anchor_state.lorentzians) == tuple(
        (item.area, item.fwhm) for item in single_state.lorentzians
    )
    assert single.snapshot is not None
    assert all(
        state is None
        for index, state in enumerate(single.snapshot.working_states)
        if index != 2
    )
    assert anchor_state.center_groups[0].group_id == "selected-center"
    assert target_state.center_groups[0].group_id == "selected-center"
    assert anchor_state.elastic_area is not None
    assert anchor_state.elastic_area.user_lower_limit == 0.0
    assert anchor_state.center_groups[0].intent.user_lower_limit is not None
    assert anchor_state.center_groups[0].intent.user_upper_limit is not None
    assert anchor_state.lorentzians[0].fwhm.user_lower_limit is not None
    assert target_state.elastic_area is not None
    assert target_state.elastic_area.user_lower_limit is None
    assert target_state.center_groups[0].intent.user_lower_limit is None
    assert target_state.lorentzians[0].fwhm.user_lower_limit is None
    assert target_state.background is BackgroundModel.LINEAR
    assert target_state.b1 == ManualParameterIntent(0.0, free=False)
    retained_target = selected.working_states[1]
    assert retained_target is not None
    substituted_target = replace(
        retained_target,
        elastic_area=ManualParameterIntent(
            1.0,
            user_upper_limit=100.0,
        ),
    )
    retained_states = list(selected.working_states)
    retained_states[1] = substituted_target
    with pytest.raises(ValueError, match="issued by execution"):
        replace(selected, working_states=tuple(retained_states))
    with pytest.raises(ValueError, match="issued by execution"):
        AutoFitCandidateBranchResult(
            selected.anchor_evidence,
            selected.branch_result,
            evaluation,
            tuple(retained_states),
        )
    forged = object.__new__(AutoFitCandidateBranchResult)
    for name in ("anchor_evidence", "branch_result", "evaluation", "working_states"):
        object.__setattr__(forged, name, getattr(selected, name))
    with pytest.raises(ValueError, match="not issued by AutoFit execution"):
        adopt_selected_auto_fit_branch(
            project,
            draft,
            evaluation,
            forged,
            center_group_id="selected-center",
        )
    assert all(
        (state is not None) == (outcome.status is MultiQFitStatus.SUCCESS)
        for state, outcome in zip(
            adopted.snapshot.working_states,
            adopted.branch_result.outcomes,
            strict=True,
        )
    )
    converted_anchor = adopted.branch_result.outcome(2).fit_result
    assert converted_anchor is not None
    assert converted_anchor is not evidence[0].fit
    continued = fit_all_q_from_current(
        project,
        adopted.adopted_draft,
        anchor_group_index=2,
        current_fit=converted_anchor,
        fit_excluded_groups=(0, 3, 4),
        max_nfev=800,
    )
    retained_anchor = continued.outcome(2).fit_result
    assert retained_anchor is not None
    assert retained_anchor.evaluation is converted_anchor.evaluation
    assert retained_anchor.parameters is converted_anchor.parameters
    assert continued.outcome(1).status is MultiQFitStatus.SUCCESS

    foreign = replace(evaluation, measured_energy=evaluation.measured_energy.copy())
    with pytest.raises(ValueError, match="not owned"):
        adopt_selected_auto_fit_branch(
            project, draft, foreign, selected, center_group_id="selected-center"
        )
    changed_selection = evaluation.scientific_context.selection.with_group_range(
        2, lower_energy=-0.5, upper_energy=0.5
    )
    changed_project = commit_fitting_selection(project, sample, changed_selection)
    with pytest.raises(ValueError, match="no longer matches"):
        adopt_selected_auto_fit_branch(
            changed_project,
            draft,
            evaluation,
            selected,
            center_group_id="selected-center",
        )


def test_autofit_single_q_adoption_can_anchor_manual_fit_all_q(
    selected_auto_problem: tuple[
        WorkflowProject,
        ProjectDataset,
        ManualFitDraft,
        SingleQAutoFitOutcome,
        tuple[CandidateFitResult, CandidateFitResult],
    ],
) -> None:
    project, _sample, draft, evaluation, evidence = selected_auto_problem
    adopted = adopt_single_q_auto_fit_candidate(project, draft, evaluation, evidence[0])
    assert adopted.success and adopted.fit_result is not None
    assert adopted.adopted_draft is not None
    branch = fit_all_q_from_current(
        project,
        adopted.adopted_draft,
        anchor_group_index=2,
        current_fit=adopted.fit_result,
        fit_excluded_groups=(0, 3, 4),
        max_nfev=800,
    )
    retained_anchor = branch.outcome(2).fit_result
    assert retained_anchor is not None
    assert retained_anchor.evaluation is adopted.fit_result.evaluation
    assert retained_anchor.parameters is adopted.fit_result.parameters
    assert branch.outcome(1).status is MultiQFitStatus.SUCCESS


def test_selected_branch_adoption_keeps_excluded_target_setup(
    selected_auto_problem: tuple[
        WorkflowProject,
        ProjectDataset,
        ManualFitDraft,
        SingleQAutoFitOutcome,
        tuple[CandidateFitResult, CandidateFitResult],
    ],
) -> None:
    project, _sample, draft, evaluation, evidence = selected_auto_problem
    existing = ManualModelState(
        background=BackgroundModel.CONSTANT,
        b0=ManualParameterIntent(0.7, free=False),
    )
    draft = draft.with_model(1, existing)
    selected = continue_selected_auto_fit_candidates(
        project,
        draft,
        evaluation,
        (evidence[0].candidate,),
        fit_excluded_groups=(0, 1, 3, 4),
        max_nfev=800,
    ).branches[0]
    assert selected.branch_result.outcome(1).status is MultiQFitStatus.EXCLUDED
    adopted = adopt_selected_auto_fit_branch(
        project,
        draft,
        evaluation,
        selected,
        center_group_id="selected-center",
    )
    assert adopted.adopted_draft.setup(1).model is existing
    assert adopted.snapshot is not None
    assert adopted.snapshot.working_states[1] is None


def test_selected_branch_adoption_keeps_failed_target_setup(
    selected_auto_problem: tuple[
        WorkflowProject,
        ProjectDataset,
        ManualFitDraft,
        SingleQAutoFitOutcome,
        tuple[CandidateFitResult, CandidateFitResult],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, _sample, draft, evaluation, evidence = selected_auto_problem
    existing = ManualModelState(
        background=BackgroundModel.CONSTANT,
        b0=ManualParameterIntent(-0.2, free=False),
    )
    draft = draft.with_model(1, existing)

    def fail_target(
        _prepared: PreparedResolution,
        _selection: FittingSelection,
        _group_index: int,
        _model: SpectralModelDefinition,
        *,
        max_nfev: int,
    ) -> FitResult:
        assert max_nfev == 800
        raise FittingError("synthetic target failure")

    monkeypatch.setattr(batch_core, "fit_single_q", fail_target)
    selected = continue_selected_auto_fit_candidates(
        project,
        draft,
        evaluation,
        (evidence[0].candidate,),
        fit_excluded_groups=(0, 3, 4),
        max_nfev=800,
    ).branches[0]
    assert selected.branch_result.outcome(1).status is MultiQFitStatus.FAILED
    adopted = adopt_selected_auto_fit_branch(
        project,
        draft,
        evaluation,
        selected,
        center_group_id="selected-center",
    )
    assert adopted.adopted_draft.setup(1).model is existing
    assert adopted.snapshot is not None
    assert adopted.snapshot.working_states[1] is None


def test_unified_result_single_q_replacement_preserves_other_fitted_qs(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem
    states = tuple(
        _manual_model(background=BackgroundModel.CONSTANT) if index == 2 else None
        for index in range(len(prepared.spectra))
    )
    first = _issue_result_snapshot(anchor, states, ResultOrigin.MANUAL_FIT)
    state = set_current_result(FittingWorkspaceState(), first)
    target = fit_single_q(prepared, selection, 1, configurations[1])
    target_model = _manual_model(background=BackgroundModel.CONSTANT)
    second = _issue_result_snapshot(
        target,
        tuple(target_model if index == 1 else None for index in range(len(states))),
        ResultOrigin.MANUAL_FIT,
    )
    state = replace_current_q_result(state, second)
    current = state.current
    assert current is not None
    assert current.result.fitted_count == 2
    assert current.result.total_count == 5
    assert current.result.point(2).fit_result is anchor
    assert current.result.point(1).fit_result is target
    assert current.result.point(1).working_state is target_model
    assert dict(current.result.status_counts)[MultiQFitStatus.SUCCESS] == 2

    project, _sample, draft = _workflow_problem(prepared, selection)
    draft = draft.with_model(1, target_model).with_model(2, states[2])
    context = resolve_manual_fit_context(project, project.datasets[0], 1).context
    assert context is not None
    assert (
        result_q_display_status(state, draft, 1, context)
        is ResultDisplayStatus.FITTED_CURRENT
    )
    edited = draft.with_model(1, replace(target_model, b0=ManualParameterIntent(0.2)))
    assert (
        result_q_display_status(state, edited, 1, context)
        is ResultDisplayStatus.NEEDS_FIT
    )
    assert has_unsaved_result_changes(state, draft)
    saved = save_current_result_as(state, "baseline")
    matching = draft
    for group_index in (0, 3, 4):
        matching = matching.with_model(group_index, None)
    assert not has_unsaved_result_changes(saved, matching)
    changed = replace(
        context,
        selection=context.selection.with_group_range(
            1, lower_energy=-0.4, upper_energy=0.4
        ),
    )
    assert (
        result_q_display_status(saved, draft, 1, changed)
        is ResultDisplayStatus.NEEDS_FIT
    )


@pytest.fixture
def two_fitted_qs(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> tuple[WorkflowProject, ProjectDataset, ManualFitDraft, FittingWorkspaceState]:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, sample, draft = _workflow_problem(prepared, selection)
    for index in (0, 3, 4):
        draft = draft.with_model(index, None)
    state = FittingWorkspaceState()
    for index in (2, 1):
        executed = run_and_adopt_manual_fit(project, draft, index)
        assert executed.success and executed.snapshot is not None
        assert executed.adopted_draft is not None
        draft = executed.adopted_draft
        state = replace_current_q_result(state, executed.snapshot)
    return project, sample, draft, state


def _contexts_for(
    project: WorkflowProject, sample: ProjectDataset
) -> tuple[ManualFitContext | None, ...]:
    return tuple(
        resolve_manual_fit_context(project, sample, index).context
        for index in range(len(sample.dataset.spectra))
    )


def test_dataset_energy_relabel_marks_fit_evidence_stale_without_modifying_result(
    two_fitted_qs: tuple[
        WorkflowProject, ProjectDataset, ManualFitDraft, FittingWorkspaceState
    ],
) -> None:
    project, sample, draft, state = two_fitted_qs
    snapshot = state.current
    assert snapshot is not None
    changed = sample.dataset.assign_units(
        energy_unit="unknown",
        intensity_unit=sample.dataset.intensity_unit,
        uncertainty_unit=sample.dataset.uncertainty_unit,
    )
    for index in (1, 2):
        context = _contexts_for(project, sample)[index]
        assert context is not None
        fit = snapshot.result.point(index).fit_result
        assert fit is not None and fit_evidence_is_current(fit, context, index)
        changed_context = replace(context, sample=replace(sample, dataset=changed))
        assert not fit_evidence_is_current(fit, changed_context, index)
        assert (
            result_q_display_status(state, draft, index, changed_context)
            is ResultDisplayStatus.NEEDS_FIT
        )
        assert snapshot.result.point(index).fit_result is fit
    assert state.current is snapshot


def test_per_q_validity_range_change_preserves_other_q_and_anchor(
    two_fitted_qs: tuple[
        WorkflowProject, ProjectDataset, ManualFitDraft, FittingWorkspaceState
    ],
) -> None:
    project, sample, draft, state = two_fitted_qs
    contexts = _contexts_for(project, sample)
    original = contexts[1]
    assert original is not None and state.current is not None
    changed = original.selection.with_group_range(
        1, lower_energy=-0.4, upper_energy=0.4
    )
    project = commit_fitting_selection(project, sample, changed)
    contexts = _contexts_for(project, sample)
    assert (
        result_q_display_status(state, draft, 1, contexts[1])
        is ResultDisplayStatus.NEEDS_FIT
    )
    assert (
        result_q_display_status(state, draft, 2, contexts[2])
        is ResultDisplayStatus.FITTED_CURRENT
    )
    q_bins = sample.dataset.q_bins
    assert q_bins is not None
    derived = derive_result_in_context(state.current, q_bins, contexts)
    assert derived.point(1).derivation_status is DerivedPointStatus.BLOCKED
    assert derived.point(2).derivation_status is DerivedPointStatus.AVAILABLE
    anchor = state.current.result.point(2).fit_result
    assert anchor is not None
    continued = fit_all_q_from_current(
        project, draft, anchor_group_index=2, current_fit=anchor
    )
    retained = continued.outcome(2).fit_result
    assert retained is not None
    np.testing.assert_array_equal(retained.evaluation.total, anchor.evaluation.total)
    with pytest.raises(ValueError, match="active context"):
        fit_all_q_from_current(
            project,
            draft,
            anchor_group_index=1,
            current_fit=state.current.result.point(1).fit_result,
        )


def test_per_q_merge_load_and_working_copy_restore_after_q_replacement(
    two_fitted_qs: tuple[
        WorkflowProject, ProjectDataset, ManualFitDraft, FittingWorkspaceState
    ],
) -> None:
    project, sample, draft, state = two_fitted_qs
    assert state.current is not None
    fit_q2 = state.current.result.point(2).fit_result
    assert fit_q2 is not None
    state = set_current_result(
        FittingWorkspaceState(),
        _issue_result_snapshot(
            fit_q2,
            tuple(draft.setup(2).model if index == 2 else None for index in range(5)),
            ResultOrigin.MANUAL_FIT,
        ),
    )
    state = save_current_result_as(state, "before Q replacement")
    state = append_saved_result(state, "second", state.saved_results[0].snapshot)
    old_current = state.current
    assert old_current is not None
    edited = draft.with_model(0, _manual_model())
    context = _contexts_for(project, sample)[2]
    assert context is not None
    state, draft = load_saved_result(state, edited, state.saved_results[1], context)
    cached = state.saved_results[0]
    assert cached.modified_copy is not None
    assert len(state.saved_results) == 2
    q_bins = sample.dataset.q_bins
    assert q_bins is not None
    new_q = QBins.from_q_values(q_bins.q_values + 0.0125)
    project, sample = replace_project_dataset(
        project, sample, sample.dataset.assign_q_bins(new_q)
    )
    resolution = project.datasets[1]
    project, _resolution = replace_project_dataset(
        project, resolution, resolution.dataset.assign_q_bins(new_q)
    )
    draft = draft.with_model(1, _manual_model(background=BackgroundModel.CONSTANT))
    executed = run_and_adopt_manual_fit(project, draft, 1)
    assert executed.success and executed.snapshot is not None
    assert executed.adopted_draft is not None
    draft = executed.adopted_draft
    state = replace_current_q_result(state, executed.snapshot)
    assert state.current is not None
    assert state.current.result.point(2) is old_current.result.point(2)
    state = save_current_result_as(state, "merged")
    contexts = _contexts_for(project, sample)
    context = contexts[1]
    assert context is not None
    loaded, restored = load_saved_result(state, draft, state.saved_results[-1], context)
    assert (
        result_q_display_status(loaded, restored, 1, contexts[1])
        is ResultDisplayStatus.FITTED_CURRENT
    )
    assert (
        result_q_display_status(loaded, restored, 2, contexts[2])
        is ResultDisplayStatus.NEEDS_FIT
    )
    assert loaded.current is not None
    derived = derive_result_in_context(loaded.current, new_q, contexts)
    assert derived.point(1).derivation_status is DerivedPointStatus.AVAILABLE
    assert derived.point(2).derivation_status is DerivedPointStatus.BLOCKED
    loaded, restored = load_saved_result(loaded, restored, cached, context)
    assert loaded.current is old_current
    assert restored.setup(0).model is edited.setup(0).model
    assert (
        result_q_display_status(loaded, restored, 2, contexts[2])
        is ResultDisplayStatus.NEEDS_FIT
    )


def test_load_retains_evidence_after_resolution_change_and_rejects_other_sample(
    two_fitted_qs: tuple[
        WorkflowProject, ProjectDataset, ManualFitDraft, FittingWorkspaceState
    ],
) -> None:
    project, sample, draft, state = two_fitted_qs
    state = save_current_result_as(state, "original")
    entry = state.saved_results[0]
    assert entry.snapshot is not None
    resolution = project.datasets[1]
    replaced = replace(
        resolution.dataset,
        spectra=tuple(
            replace(spectrum, intensity=spectrum.intensity * 1.02)
            for spectrum in resolution.dataset.spectra
        ),
    )
    project, _resolution = replace_project_dataset(project, resolution, replaced)
    contexts = _contexts_for(project, sample)
    context = contexts[2]
    assert context is not None
    loaded, restored = load_saved_result(state, draft, entry, context)
    assert loaded.current is entry.snapshot and loaded.current is not None
    for index in (1, 2):
        assert restored.setup(index).model is entry.snapshot.working_states[index]
        assert loaded.current.result.point(index).fit_result is not None
        assert (
            result_q_display_status(loaded, restored, index, contexts[index])
            is ResultDisplayStatus.NEEDS_FIT
        )
    q_bins = sample.dataset.q_bins
    assert q_bins is not None
    derived = derive_result_in_context(loaded.current, q_bins, contexts)
    assert all(
        derived.point(index).derivation_status is DerivedPointStatus.BLOCKED
        for index in (1, 2)
    )
    foreign_context = replace(
        context, sample=replace(sample, dataset_id="another-sample")
    )
    foreign_draft = replace(draft, sample_id="another-sample")
    with pytest.raises(ValueError, match="another Sample"):
        load_saved_result(state, foreign_draft, entry, foreign_context)
    assert entry.snapshot is not None
    foreign = _issue_result_snapshot(
        cast(FitResult, entry.snapshot.result.point(2).fit_result),
        tuple(
            entry.snapshot.working_states[index] if index == 2 else None
            for index in range(5)
        ),
        ResultOrigin.MANUAL_FIT,
        sample_identity=(draft.project_id, "another-sample"),
    )
    with pytest.raises(ValueError, match="another Sample"):
        replace_current_q_result(state, foreign)


def test_value_equal_recreated_context_is_current(
    two_fitted_qs: tuple[
        WorkflowProject, ProjectDataset, ManualFitDraft, FittingWorkspaceState
    ],
) -> None:
    project, sample, draft, state = two_fitted_qs
    assert state.current is not None
    old_context = _contexts_for(project, sample)[2]
    assert old_context is not None
    copies = []
    for entry in project.datasets:
        dataset = replace(
            entry.dataset,
            spectra=tuple(
                replace(
                    spectrum,
                    energy=spectrum.energy.copy(),
                    intensity=spectrum.intensity.copy(),
                    uncertainty=spectrum.uncertainty.copy(),
                )
                for spectrum in entry.dataset.spectra
            ),
        )
        project, replacement = replace_project_dataset(project, entry, dataset)
        copies.append(replacement)
    sample = copies[0]
    contexts = _contexts_for(project, sample)
    context = contexts[2]
    assert context is not None
    assert context.selection is not old_context.selection
    assert context.prepared_resolution is not old_context.prepared_resolution
    assert context.sample.dataset is not old_context.sample.dataset
    fit = state.current.result.point(2).fit_result
    assert fit is not None and fit_evidence_is_current(fit, context, 2)
    assert (
        result_q_display_status(state, draft, 2, context)
        is ResultDisplayStatus.FITTED_CURRENT
    )
    q_bins = sample.dataset.q_bins
    assert q_bins is not None
    derived = derive_result_in_context(state.current, q_bins, contexts)
    assert derived.point(1).derivation_status is DerivedPointStatus.AVAILABLE
    assert derived.point(2).derivation_status is DerivedPointStatus.AVAILABLE


@pytest.mark.parametrize(
    "changed_input",
    ("energy", "intensity", "uncertainty", "mask", "q", "kernel", "acceptance"),
)
def test_scientific_input_changes_affect_only_their_q(
    two_fitted_qs: tuple[
        WorkflowProject, ProjectDataset, ManualFitDraft, FittingWorkspaceState
    ],
    changed_input: str,
) -> None:
    project, sample, draft, state = two_fitted_qs
    assert state.current is not None
    original = _contexts_for(project, sample)[1]
    assert original is not None
    measured = original.sample.dataset
    resolution = original.resolution.dataset
    if changed_input in ("energy", "intensity", "uncertainty"):
        spectra = list(measured.spectra)
        spectrum = spectra[1]
        values = {
            "energy": spectrum.energy,
            "intensity": spectrum.intensity,
            "uncertainty": spectrum.uncertainty,
        }[changed_input].copy()
        values[len(values) // 3] += 1e-6
        if changed_input == "energy":
            spectra[1] = replace(spectrum, energy=values)
        elif changed_input == "intensity":
            spectra[1] = replace(spectrum, intensity=values)
        else:
            spectra[1] = replace(spectrum, uncertainty=values)
        measured = replace(measured, spectra=tuple(spectra))
    elif changed_input == "q":
        q_bins = measured.q_bins
        assert q_bins is not None
        values = q_bins.q_values.copy()
        values[1] += 0.0125
        q_bins = QBins.from_q_values(values)
        measured = measured.assign_q_bins(q_bins)
        resolution = resolution.assign_q_bins(q_bins)
    elif changed_input == "kernel":
        spectra = list(resolution.spectra)
        spectrum = spectra[1]
        spectra[1] = replace(spectrum, intensity=spectrum.intensity * 1.02)
        resolution = replace(resolution, spectra=tuple(spectra))
    prepared = prepare_measured_resolution(
        measured,
        resolution,
        acceptance_decisions={
            1: ResolutionAcceptance(ResolutionAcceptanceDecision.KEEP, confirmed=True)
        }
        if changed_input == "acceptance"
        else None,
    )
    selection = replace(
        original.selection, dataset=measured, padding=prepared.sample_padding
    )
    if changed_input == "mask":
        mask = selection.manual_exclusion_mask(1).copy()
        mask[len(mask) // 3] = True
        selection = selection.with_group_manual_exclusion(1, mask)
    contexts = tuple(
        replace(
            original,
            sample=replace(original.sample, dataset=measured),
            resolution=replace(original.resolution, dataset=resolution),
            selection=selection,
            prepared_resolution=prepared,
            group_index=index,
            group_identity=measured.spectra[index].group_label,
        )
        for index in range(5)
    )
    assert (
        result_q_display_status(state, draft, 1, contexts[1])
        is ResultDisplayStatus.NEEDS_FIT
    )
    assert (
        result_q_display_status(state, draft, 2, contexts[2])
        is ResultDisplayStatus.FITTED_CURRENT
    )
    q_bins = measured.q_bins
    assert q_bins is not None
    derived = derive_result_in_context(state.current, q_bins, contexts)
    assert derived.point(1).derivation_status is DerivedPointStatus.BLOCKED
    assert derived.point(2).derivation_status is DerivedPointStatus.AVAILABLE


def test_unified_workspace_working_copy_rename_and_best_by_identity(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, _configurations, anchor = multi_q_problem
    project, sample, draft = _workflow_problem(prepared, selection)
    fitted = draft.setup(2).model
    assert fitted is not None
    draft = draft.with_model(2, fitted)
    snapshot = _issue_result_snapshot(
        anchor,
        tuple(fitted if index == 2 else None for index in range(len(draft.setups))),
        ResultOrigin.MANUAL_FIT,
    )
    state = save_current_result_as(
        set_current_result(FittingWorkspaceState(), snapshot), "A"
    )
    first = state.saved_results[0]
    with pytest.raises(ValueError, match="issued by the workflow"):
        SavedFitResult("forged", snapshot)
    with pytest.raises(ValueError, match="issued by the workflow"):
        replace(first, name="forged")
    state = append_saved_result(state, "A", snapshot)
    second = state.saved_results[1]
    state = set_best_saved_result(state, second)
    assert state.best_saved_result is not None
    assert state.best_saved_result.snapshot is snapshot
    best = state.best_saved_result
    state = rename_saved_result(state, best, "renamed")
    assert state.saved_results[0] is first
    assert state.saved_results[1].name == "renamed"
    assert state.best_saved_result is state.saved_results[1]
    edited = draft.with_model(1, _manual_model())
    assert has_unsaved_result_changes(state, edited)
    context = resolve_manual_fit_context(project, sample, 2).context
    assert context is not None
    loaded, restored = load_saved_result(state, edited, state.saved_results[1], context)
    assert loaded.current is snapshot
    assert loaded.current_saved_result_index == 1
    assert len(loaded.saved_results) == 2
    cached = loaded.saved_results[0]
    assert cached.name == "A"
    assert cached.modified_copy is not None
    assert cached.modified_copy.draft_states[1] is edited.setup(1).model
    assert restored.setup(1).model is None
    loaded, restored = load_saved_result(loaded, restored, cached, context)
    assert restored.setup(1).model is edited.setup(1).model
    loaded = remove_saved_result(loaded, loaded.saved_results[1])
    assert loaded.current_saved_result_index == 0
    assert loaded.best_saved_result is None
    newer = _issue_result_snapshot(
        anchor,
        tuple(fitted if index == 2 else None for index in range(len(draft.setups))),
        ResultOrigin.MANUAL_FIT,
    )
    saved_after_fit = save_current_result(set_current_result(loaded, newer))
    assert saved_after_fit.associated_saved_result is not None
    assert saved_after_fit.associated_saved_result.snapshot is newer
    assert saved_after_fit.associated_saved_result.draft_states == newer.working_states
    assert saved_after_fit.associated_saved_result.modified_copy is None


def test_draft_only_untitled_is_separate_from_saved_results(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, _configurations, anchor = multi_q_problem
    project, sample, draft = _workflow_problem(prepared, selection)
    fitted = draft.setup(2).model
    assert fitted is not None
    snapshot = _issue_result_snapshot(
        anchor,
        tuple(fitted if index == 2 else None for index in range(len(draft.setups))),
        ResultOrigin.MANUAL_FIT,
    )
    state = append_saved_result(FittingWorkspaceState(), "fitted", snapshot)
    edited = draft.with_model(1, _manual_model())
    state = update_working_draft(state, edited)
    untitled = state.untitled
    assert untitled is not None and not untitled.has_result
    assert untitled.has_draft_edits
    assert state.active_item is untitled
    assert len(state.saved_results) == 1
    assert unsaved_result_items(state) == ()
    assert draft_edit_items(state) == (untitled,)
    context = resolve_manual_fit_context(project, sample, 2).context
    assert context is not None
    loaded, restored = load_saved_result(state, edited, state.saved_results[0], context)
    assert loaded.current is snapshot
    assert loaded.untitled is None
    assert len(loaded.saved_results) == 1
    assert restored.setup(1).model is None
    assert not has_unsaved_result_changes(loaded, restored)
    saved = save_current_result(loaded)
    assert saved.associated_saved_result is not None
    assert saved.associated_saved_result.modified_copy is None


@pytest.fixture
def workspace_problem(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> tuple[ManualFitDraft, ResultSnapshot]:
    prepared, selection, _configurations, anchor = multi_q_problem
    _project, _sample, draft = _workflow_problem(prepared, selection)
    for index in (0, 1, 3, 4):
        draft = draft.with_model(index, None)
    snapshot = _issue_result_snapshot(
        anchor,
        tuple(setup.model for setup in draft.setups),
        ResultOrigin.MANUAL_FIT,
    )
    return draft, snapshot


def test_open_switch_keeps_result_and_draft_copies_on_each_saved_entry(
    workspace_problem: tuple[ManualFitDraft, ResultSnapshot],
) -> None:
    draft, snapshot = workspace_problem
    state = save_current_result_as(
        set_current_result(FittingWorkspaceState(), snapshot), "first"
    )
    state = append_saved_result(state, "second", snapshot)
    newer = _issue_result_snapshot(
        cast(FitResult, snapshot.result.point(2).fit_result),
        snapshot.working_states,
        ResultOrigin.MANUAL_FIT,
    )
    state = set_current_result(state, newer)
    edited = draft.with_model(1, _manual_model())
    state, second_draft = load_saved_result(state, edited, state.saved_results[1])
    first = state.saved_results[0]
    copy = first.modified_copy
    assert copy is not None and copy.snapshot is newer
    assert first.modified and first.has_draft_edits
    assert copy.draft_states[1] is edited.setup(1).model
    assert len(state.saved_results) == 2
    assert state.current is snapshot and second_draft.setup(1).model is None
    assert state.active_item is state.saved_results[1]

    state, restored = load_saved_result(state, second_draft, first)
    assert state.current is newer
    assert state.saved_results[0].modified_copy is copy
    assert restored.setup(1).model is edited.setup(1).model
    assert active_manual_draft(state, second_draft) == restored
    assert unsaved_result_items(state) == (state.saved_results[0],)


@pytest.mark.parametrize("operation", ["open", "fresh"])
def test_untitled_result_requires_explicit_discard_consent(
    workspace_problem: tuple[ManualFitDraft, ResultSnapshot], operation: str
) -> None:
    draft, snapshot = workspace_problem
    state = append_saved_result(FittingWorkspaceState(), "saved", snapshot)
    state = set_current_result(state, snapshot)
    untitled = state.untitled
    assert untitled is not None and untitled.has_result and untitled.modified
    assert state.active_item is untitled
    assert unsaved_result_items(state) == (untitled,)
    with pytest.raises(ValueError, match="discard consent"):
        if operation == "open":
            load_saved_result(state, draft, state.saved_results[0])
        else:
            fresh_analysis(state, draft)
    assert state.untitled is untitled and state.current is snapshot
    if operation == "open":
        updated, restored = load_saved_result(
            state, draft, state.saved_results[0], discard_untitled=True
        )
        assert updated.untitled is None
        assert updated.active_item is updated.saved_results[0]
        assert updated.current is snapshot
        assert restored.setup(2).model is snapshot.working_states[2]
    else:
        updated, restored = fresh_analysis(state, draft, discard_untitled=True)
        assert updated.untitled is not None
        assert updated.active_item is updated.untitled
        assert not updated.untitled.has_result
        assert updated.current is None
        assert all(setup.model is None for setup in restored.setups)
    assert len(updated.saved_results) == 1
    assert unsaved_result_items(updated) == ()


def test_save_as_moves_untitled_and_resets_modified_saved_source(
    workspace_problem: tuple[ManualFitDraft, ResultSnapshot],
) -> None:
    draft, snapshot = workspace_problem
    state = set_current_result(FittingWorkspaceState(), snapshot)
    assert state.untitled is not None
    with pytest.raises(ValueError, match="requires a name"):
        save_current_result(state)
    state = save_current_result(state, "first")
    assert state.untitled is None
    assert state.active_item is state.saved_results[0]
    assert not state.saved_results[0].modified
    assert state.saved_results[0].modified_copy is None
    newer = _issue_result_snapshot(
        cast(FitResult, snapshot.result.point(2).fit_result),
        snapshot.working_states,
        ResultOrigin.MANUAL_FIT,
    )
    edited = draft.with_model(1, _manual_model())
    state = set_current_result(state, newer, draft=edited)
    assert state.saved_results[0].modified
    state = save_current_result_as(state, "second")
    assert tuple(entry.name for entry in state.saved_results) == ("first", "second")
    assert state.saved_results[0].snapshot is snapshot
    assert state.saved_results[0].modified_copy is None
    assert not state.saved_results[0].has_draft_edits
    assert state.active_item is state.saved_results[1]
    assert state.current is newer
    assert not state.saved_results[1].modified
    assert state.saved_results[1].has_draft_edits
    assert state.saved_results[1].working_copy.draft_states[1] is edited.setup(1).model
    # Unfitted parameters are cached outside, never written into fitted evidence.
    assert newer.working_states[1] is None


def test_save_overwrites_result_but_does_not_save_unfitted_parameters_into_result(
    workspace_problem: tuple[ManualFitDraft, ResultSnapshot],
) -> None:
    draft, snapshot = workspace_problem
    state = save_current_result_as(
        set_current_result(FittingWorkspaceState(), snapshot), "saved"
    )
    state = set_best_saved_result(state, state.saved_results[0])
    newer = _issue_result_snapshot(
        cast(FitResult, snapshot.result.point(2).fit_result),
        snapshot.working_states,
        ResultOrigin.MANUAL_FIT,
    )
    state = set_current_result(state, newer)
    edited = draft.with_model(1, _manual_model())
    state = update_working_draft(state, edited)
    state = save_current_result(state)
    entry = state.saved_results[0]
    assert state.active_item is entry and state.current is newer
    assert entry.best and entry.snapshot is newer and not entry.modified
    assert entry.modified_copy is None
    assert entry.has_draft_edits
    assert unsaved_result_items(state) == ()
    assert draft_edit_items(state) == (entry,)
    assert entry.working_copy.draft_states[1] is edited.setup(1).model
    assert entry.snapshot.working_states[1] is None


@pytest.mark.parametrize("modified", [False, True])
@pytest.mark.parametrize("transition", ["revert", "save_as"])
def test_saved_draft_baseline_survives_revert_save_as_and_repeated_save(
    workspace_problem: tuple[ManualFitDraft, ResultSnapshot],
    modified: bool,
    transition: str,
) -> None:
    draft, snapshot = workspace_problem
    baseline = draft.with_model(1, _manual_model())
    baseline_states = tuple(setup.model for setup in baseline.setups)
    state = save_current_result_as(
        set_current_result(FittingWorkspaceState(), snapshot, draft=baseline), "A"
    )
    for _ in range(2):
        state = save_current_result(state)
        entry = state.saved_results[0]
        assert entry.draft_states == baseline_states
        assert entry.modified_copy is None
        assert entry.snapshot is snapshot
        assert entry.snapshot.working_states[1] is None

    working_draft = baseline
    if modified:
        working_draft = baseline.with_model(
            1, _manual_model(background=BackgroundModel.LINEAR)
        ).with_model(0, _manual_model())
        state = update_working_draft(state, working_draft)
        assert state.saved_results[0].modified_copy is not None
    entry = state.saved_results[0]
    if transition == "revert":
        state, restored = revert_saved_result(state, working_draft, entry)
        assert restored.setup(1).model is baseline.setup(1).model
        assert restored.setup(0).model is None
        assert state.active_item is state.saved_results[0]
    else:
        state = save_current_result_as(state, "B")
        assert state.active_item is state.saved_results[1]
        assert state.saved_results[1].draft_states == tuple(
            setup.model for setup in working_draft.setups
        )
        assert state.saved_results[1].snapshot is snapshot
        assert state.saved_results[1].modified_copy is None
        if not modified:
            assert state.saved_results[0] is entry
    assert state.saved_results[0].draft_states == baseline_states
    assert state.saved_results[0].modified_copy is None
    assert state.saved_results[0].snapshot is snapshot
    assert snapshot.result.point(1).fit_result is None
    assert snapshot.working_states[1] is None


def test_save_updates_draft_baseline_and_revert_keeps_latest_saved_setup(
    workspace_problem: tuple[ManualFitDraft, ResultSnapshot],
) -> None:
    draft, snapshot = workspace_problem
    state = save_current_result_as(
        set_current_result(FittingWorkspaceState(), snapshot), "A"
    )
    baseline = draft.with_model(1, _manual_model())
    state = update_working_draft(state, baseline)
    for _ in range(2):
        state = save_current_result(state)
        assert state.saved_results[0].draft_states is not None
        assert state.saved_results[0].draft_states[1] is baseline.setup(1).model
        assert state.saved_results[0].modified_copy is None
    edited = baseline.with_model(1, None)
    state = update_working_draft(state, edited)
    state, restored = revert_saved_result(state, edited, state.saved_results[0])
    assert restored.setup(1).model is baseline.setup(1).model
    assert state.current is snapshot
    assert snapshot.working_states[1] is None


@pytest.mark.parametrize("active_history", [False, True])
def test_regrouping_allows_fresh_analysis_with_historical_saved_result(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    active_history: bool,
) -> None:
    prepared, selection, _configurations, anchor = multi_q_problem
    project, sample, draft = _workflow_problem(prepared, selection)
    snapshot = _issue_result_snapshot(
        anchor,
        tuple(draft.setup(2).model if index == 2 else None for index in range(5)),
        ResultOrigin.MANUAL_FIT,
    )
    state = append_saved_result(FittingWorkspaceState(), "five Qs", snapshot)
    if active_history:
        state, draft = load_saved_result(state, draft, state.saved_results[0])
    historical = state.saved_results[0]
    q_bins = sample.dataset.q_bins
    assert q_bins is not None
    project, sample = replace_project_dataset(
        project,
        sample,
        replace(
            sample.dataset,
            spectra=sample.dataset.spectra[:3],
            q_bins=QBins.from_q_values(q_bins.q_values[:3]),
        ),
    )
    regrouped = open_manual_fit_draft(project, sample)
    assert len(regrouped.setups) == 3
    state, empty = fresh_analysis(state, regrouped)
    assert state.untitled is not None and len(state.untitled.draft_states) == 3
    assert all(setup.model is None for setup in empty.setups)
    edited = empty.with_model(0, _manual_model())
    state = update_working_draft(state, edited)
    assert state.active_working_copy is not None
    assert state.active_working_copy.draft_states[0] is edited.setup(0).model
    assert state.saved_results == (historical,)
    assert historical.snapshot.result.total_count == 5
    with pytest.raises(ValueError, match="Q topology differs"):
        load_saved_result(state, edited, historical)
    assert state.active_item is state.untitled
    with pytest.raises(ValueError, match="another Sample"):
        update_working_draft(state, replace(edited, sample_id="different-sample"))
    with pytest.raises(ValueError, match="match its Result Sample and Q groups"):
        set_current_result(FittingWorkspaceState(), snapshot, draft=regrouped)


def test_regrouped_draft_update_without_active_item_keeps_historical_entry(
    workspace_problem: tuple[ManualFitDraft, ResultSnapshot],
) -> None:
    draft, snapshot = workspace_problem
    state = append_saved_result(FittingWorkspaceState(), "five Qs", snapshot)
    historical = state.saved_results[0]
    regrouped = replace(draft, setups=draft.setups[:3])
    state = update_working_draft(state, regrouped)
    assert state.untitled is not None and len(state.untitled.draft_states) == 3
    assert state.saved_results == (historical,)
    with pytest.raises(ValueError, match="Q topology differs"):
        load_saved_result(state, regrouped, historical)


def test_revert_saved_result_restores_active_and_leaves_other_active_work_alone(
    workspace_problem: tuple[ManualFitDraft, ResultSnapshot],
) -> None:
    draft, snapshot = workspace_problem
    state = save_current_result_as(
        set_current_result(FittingWorkspaceState(), snapshot), "saved"
    )
    newer = _issue_result_snapshot(
        cast(FitResult, snapshot.result.point(2).fit_result),
        snapshot.working_states,
        ResultOrigin.MANUAL_FIT,
    )
    edited = draft.with_model(1, _manual_model())
    state = set_current_result(state, newer, draft=edited)
    state, restored = revert_saved_result(state, edited, state.saved_results[0])
    assert state.current is snapshot
    assert restored.setup(1).model is None
    assert state.saved_results[0].modified_copy is None
    assert not has_unsaved_result_changes(state, restored)

    state = set_current_result(state, newer, draft=edited)
    state, empty = fresh_analysis(state, edited)
    untitled = state.untitled
    assert untitled is not None
    state, unchanged = revert_saved_result(state, empty, state.saved_results[0])
    assert state.active_item is untitled and unchanged is empty
    assert state.saved_results[0].modified_copy is None


def test_unsaved_and_draft_queries_cover_inactive_modified_entries_and_untitled(
    workspace_problem: tuple[ManualFitDraft, ResultSnapshot],
) -> None:
    draft, snapshot = workspace_problem
    state = save_current_result_as(
        set_current_result(FittingWorkspaceState(), snapshot), "saved"
    )
    edited = draft.with_model(1, _manual_model())
    state = update_working_draft(state, edited)
    entry = state.saved_results[0]
    assert entry.has_result and not entry.modified and entry.has_draft_edits
    state, empty = fresh_analysis(state, edited)
    assert state.saved_results[0] is entry
    assert state.untitled is not None and not state.untitled.has_result
    assert unsaved_result_items(state) == (entry,)
    assert draft_edit_items(state) == (entry,)
    state = set_current_result(state, snapshot)
    untitled = state.untitled
    assert untitled is not None
    assert unsaved_result_items(state) == (untitled, entry)
    assert draft_edit_items(state) == (entry,)
    assert has_unsaved_result_changes(state)
    state = remove_saved_result(state, entry)
    assert state.active_item is untitled and state.current is snapshot
    assert unsaved_result_items(state) == (untitled,)
    assert draft_edit_items(state) == ()
    assert all(setup.model is None for setup in empty.setups)


def test_rename_best_and_append_preserve_active_modified_working_copy(
    workspace_problem: tuple[ManualFitDraft, ResultSnapshot],
) -> None:
    draft, snapshot = workspace_problem
    state = save_current_result_as(
        set_current_result(FittingWorkspaceState(), snapshot), "first"
    )
    edited = draft.with_model(1, _manual_model())
    state = update_working_draft(state, edited)
    copy = state.active_working_copy
    active = state.active_item
    state = append_saved_result(state, "second", snapshot)
    assert state.active_item is active and state.active_working_copy is copy
    state = rename_saved_result(state, state.saved_results[0], "renamed")
    assert state.active_item is state.saved_results[0]
    assert state.active_working_copy is copy
    state = set_best_saved_result(state, state.saved_results[0])
    assert state.active_item is state.best_saved_result
    assert state.active_working_copy is copy
    state = set_best_saved_result(state, state.saved_results[1])
    assert state.active_item is state.saved_results[0]
    assert state.active_working_copy is copy
    assert tuple(entry.name for entry in state.saved_results) == ("renamed", "second")
    assert tuple(entry.best for entry in state.saved_results) == (False, True)
    state = remove_saved_result(state, state.saved_results[0])
    assert state.active_item is None and state.current is None
    assert state.untitled is None
    assert all(
        setup.model is None for setup in active_manual_draft(state, edited).setups
    )


def test_working_copy_issuance_and_per_dataset_retention(
    workspace_problem: tuple[ManualFitDraft, ResultSnapshot],
) -> None:
    draft, snapshot = workspace_problem
    first = set_current_result(FittingWorkspaceState(), snapshot)
    copy = first.untitled
    assert copy is not None
    with pytest.raises(ValueError, match="issued by the workflow"):
        ResultWorkingCopy(snapshot, snapshot.working_states, copy.sample_identity)
    with pytest.raises(ValueError, match="issued by the workflow"):
        replace(copy, snapshot=None)
    second_draft = replace(draft, sample_id="different-sample")
    second, empty = fresh_analysis(FittingWorkspaceState(), second_draft)
    second = update_working_draft(second, empty.with_model(1, _manual_model()))
    assert second.current is None
    assert active_manual_draft(second, second_draft).setup(1).model is not None
    assert first.current is snapshot and first.untitled is copy
    assert active_manual_draft(first, draft).setup(2).model is draft.setup(2).model
    with pytest.raises(ValueError, match="another Sample"):
        active_manual_draft(first, second_draft)


def test_open_saved_working_copies_preserves_per_q_display_validity(
    two_fitted_qs: tuple[
        WorkflowProject, ProjectDataset, ManualFitDraft, FittingWorkspaceState
    ],
) -> None:
    project, sample, draft, state = two_fitted_qs
    state = save_current_result_as(state, "first")
    assert state.current is not None
    state = append_saved_result(state, "second", state.current)
    model = draft.setup(1).model
    assert model is not None
    edited = draft.with_model(
        1, replace(model, elastic_area=ManualParameterIntent(0.75))
    )
    state = update_working_draft(state, edited)
    contexts = _contexts_for(project, sample)
    assert (
        result_q_display_status(state, edited, 1, contexts[1])
        is ResultDisplayStatus.NEEDS_FIT
    )
    assert (
        result_q_display_status(state, edited, 2, contexts[2])
        is ResultDisplayStatus.FITTED_CURRENT
    )
    state, clean_draft = load_saved_result(state, edited, state.saved_results[1])
    assert all(
        result_q_display_status(state, clean_draft, index, contexts[index])
        is ResultDisplayStatus.FITTED_CURRENT
        for index in (1, 2)
    )
    state, restored = load_saved_result(state, clean_draft, state.saved_results[0])
    assert restored.setup(1).model is edited.setup(1).model
    assert (
        result_q_display_status(state, restored, 1, contexts[1])
        is ResultDisplayStatus.NEEDS_FIT
    )
    assert (
        result_q_display_status(state, restored, 2, contexts[2])
        is ResultDisplayStatus.FITTED_CURRENT
    )


def test_unified_identity_swap_is_exact_and_derived_series_follow_identity(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, _configurations, anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)
    model = draft.setup(2).model
    assert model is not None
    snapshot = _issue_result_snapshot(
        anchor,
        tuple(model if index == 2 else None for index in range(len(draft.setups))),
        ResultOrigin.MANUAL_FIT,
    )
    state = save_current_result_as(
        set_current_result(FittingWorkspaceState(), snapshot), "before swap"
    )
    state, draft = swap_current_lorentzian_identities(state, draft, 2, L1, L2)
    assert state.current is not None
    assert state.saved_results[0].snapshot is snapshot
    assert has_unsaved_result_changes(state, draft)
    point = state.current.result.point(2)
    swapped = point.fit_result
    assert swapped is not None
    assert swapped.fitted_model is not None
    swapped_draft = draft.setup(2).model
    assert swapped_draft is not None
    assert point.identity_marks == (IdentityMark.USER_SWAP,)
    assert tuple(item.identity for item in swapped.fitted_model.lorentzians) == (
        L2,
        L1,
    )
    assert tuple(item.identity for item in swapped_draft.lorentzians) == (
        L2,
        L1,
    )
    assert swapped.statistics is anchor.statistics
    np.testing.assert_array_equal(swapped.evaluation.total, anchor.evaluation.total)
    np.testing.assert_array_equal(swapped.raw_residuals, anchor.raw_residuals)
    np.testing.assert_array_equal(swapped.covariance, anchor.covariance)
    for before, after in ((L1, L2), (L2, L1)):
        for family in (ParameterFamily.AREA, ParameterFamily.FWHM):
            original = anchor.parameter_by_reference(ParameterReference(before, family))
            relabeled = swapped.parameter_by_reference(
                ParameterReference(after, family)
            )
            assert relabeled.value == original.value
            assert relabeled.standard_error == original.standard_error
            assert relabeled.free == original.free
        np.testing.assert_array_equal(
            swapped.evaluation.component_curve(after),
            anchor.evaluation.component_curve(before),
        )
    assert (
        swapped.parameter_by_reference(
            ParameterReference(L2, ParameterFamily.FWHM)
        ).value
        == anchor.parameter_by_reference(
            ParameterReference(L1, ParameterFamily.FWHM)
        ).value
    )
    q_bins = prepared.sample_dataset.q_bins
    assert q_bins is not None
    derived = derive_unified_qens(
        state.current.result, q_bins, context_valid=(False, False, True, False, False)
    )
    assert derived.point(2).derivation_status is DerivedPointStatus.AVAILABLE
    assert derived.lorentzian_series(L2)[2] is not None
    assert derived.lorentzian_series(L1)[2] is not None
    assert derived.point(1).derivation_status is DerivedPointStatus.NOT_APPLICABLE


def test_unified_derivation_links_identity_across_mixed_compositions(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem
    q_bins = prepared.sample_dataset.q_bins
    assert q_bins is not None
    two_state = _manual_model(background=BackgroundModel.CONSTANT)
    one_state = replace(two_state, lorentzians=two_state.lorentzians[:1])
    one_config = replace(
        configurations[1], lorentzians=configurations[1].lorentzians[:1]
    )
    one_fit = fit_single_q(prepared, selection, 1, one_config)
    first = _issue_result_snapshot(
        anchor,
        tuple(two_state if index == 2 else None for index in range(5)),
        ResultOrigin.MANUAL_FIT,
    )
    second = _issue_result_snapshot(
        one_fit,
        tuple(one_state if index == 1 else None for index in range(5)),
        ResultOrigin.MANUAL_FIT,
    )
    state = replace_current_q_result(
        set_current_result(FittingWorkspaceState(), first), second
    )
    assert state.current is not None
    result = state.current.result
    derived = derive_unified_qens(
        result, q_bins, context_valid=(False, True, True, False, False)
    )
    assert derived.point(1).derivation_status is DerivedPointStatus.AVAILABLE
    assert derived.point(2).derivation_status is DerivedPointStatus.AVAILABLE
    assert derived.lorentzian_series(L1)[1] is not None
    assert derived.lorentzian_series(L1)[2] is not None
    assert derived.lorentzian_series(L2)[1] is None
    assert derived.lorentzian_series(L2)[2] is not None
    assert derived.eisf_series[1] is not None
    assert derived.eisf_series[2] is not None
    assert len(derived.eisf_series[1].quasielastic_areas) == 1
    assert len(derived.eisf_series[2].quasielastic_areas) == 2

    stale = derive_unified_qens(
        result, q_bins, context_valid=(False, False, True, False, False)
    )
    assert stale.point(1).derivation_status is DerivedPointStatus.BLOCKED
    excluded_points = list(result.points)
    excluded_points[2] = replace(
        excluded_points[2],
        outcome=replace(excluded_points[2].outcome, derived_result_excluded=True),
    )
    excluded = derive_unified_qens(
        replace(result, points=tuple(excluded_points)),
        q_bins,
        context_valid=(False, True, True, False, False),
    )
    assert excluded.point(2).derivation_status is DerivedPointStatus.EXCLUDED
    assert excluded.lorentzian_series(L2)[2] is None


def test_autofit_adoption_identity_inherit_width_pair_and_fresh_runs(
    selected_auto_problem: tuple[
        WorkflowProject,
        ProjectDataset,
        ManualFitDraft,
        SingleQAutoFitOutcome,
        tuple[CandidateFitResult, CandidateFitResult],
    ],
) -> None:
    project, _sample, draft, evaluation, evidence = selected_auto_problem
    adopted_by_candidate: dict[StandardModelCandidate, ResultSnapshot] = {}
    for candidate in evidence:
        first = adopt_single_q_auto_fit_candidate(
            project, draft, evaluation, candidate, center_group_id="group"
        )
        independent = adopt_single_q_auto_fit_candidate(
            project, draft, evaluation, candidate, center_group_id="group"
        )
        assert first.success and independent.success
        assert first.snapshot is not None and independent.snapshot is not None
        adopted_by_candidate[candidate.candidate] = first.snapshot
        first_fit = first.fit_result
        independent_fit = independent.fit_result
        assert first_fit is not None and independent_fit is not None
        first_model = first_fit.fitted_model
        independent_model = independent_fit.fitted_model
        assert first_model is not None and independent_model is not None
        first_ids = {item.identity for item in first_model.lorentzians}
        independent_ids = {item.identity for item in independent_model.lorentzians}
        assert first_ids.isdisjoint(independent_ids)
        assert all(
            not item.component_id.startswith("standard_lorentzian_")
            for item in first_ids
        )

        paired = adopt_single_q_auto_fit_candidate(
            project,
            draft,
            evaluation,
            candidate,
            center_group_id="group",
            current=first.snapshot,
        )
        assert paired.success and paired.fit_result is not None
        assert paired.snapshot is not None
        paired_model = paired.fit_result.fitted_model
        paired_fit = paired.fit_result
        assert paired_model is not None
        assert {item.identity for item in paired_model.lorentzians} == first_ids
        if len(first_ids) == 2:
            assert paired.snapshot.result.point(2).identity_marks == (
                IdentityMark.PAIRED_BY_WIDTH,
            )

            def width(fit: FitResult, identity: ComponentIdentity) -> float:
                return fit.parameter_by_reference(
                    ParameterReference(identity, ParameterFamily.FWHM)
                ).value

            for old, new in zip(
                sorted(
                    first_model.lorentzians,
                    key=lambda item: width(first_fit, item.identity),
                ),
                sorted(
                    paired_model.lorentzians,
                    key=lambda item: width(paired_fit, item.identity),
                ),
                strict=True,
            ):
                assert old.identity == new.identity
        else:
            assert paired.snapshot.result.point(2).identity_marks == ()

    two, one = evidence
    cross_composition = adopt_single_q_auto_fit_candidate(
        project,
        draft,
        evaluation,
        one,
        center_group_id="group",
        current=adopted_by_candidate[two.candidate],
    )
    assert cross_composition.success and cross_composition.fit_result is not None
    assert cross_composition.snapshot is not None
    old_fit = adopted_by_candidate[two.candidate].result.point(2).fit_result
    assert old_fit is not None and old_fit.fitted_model is not None
    assert cross_composition.fit_result.fitted_model is not None
    assert {item.identity for item in old_fit.fitted_model.lorentzians}.isdisjoint(
        item.identity for item in cross_composition.fit_result.fitted_model.lorentzians
    )
    assert cross_composition.snapshot.result.point(2).identity_marks == ()

    branch = continue_selected_auto_fit_candidates(
        project, draft, evaluation, (evidence[0].candidate,), max_nfev=800
    ).branches[0]
    adopted_branch = adopt_selected_auto_fit_branch(
        project, draft, evaluation, branch, center_group_id="branch-group"
    )
    ids_by_q = []
    for outcome in adopted_branch.branch_result.outcomes:
        fit = outcome.fit_result
        if fit is not None:
            assert fit.fitted_model is not None
            ids_by_q.append({item.identity for item in fit.fitted_model.lorentzians})
    assert ids_by_q and all(ids == ids_by_q[0] for ids in ids_by_q)
    assert ids_by_q[0].isdisjoint(first_ids)
    assert adopted_branch.snapshot is not None
    branch_current = set_current_result(
        FittingWorkspaceState(), adopted_branch.snapshot
    )
    single = adopt_single_q_auto_fit_candidate(
        project,
        draft,
        evaluation,
        evidence[1],
        center_group_id="post-branch-single",
        current=branch_current.current,
    )
    assert single.success and single.snapshot is not None
    merged = replace_current_q_result(branch_current, single.snapshot)
    assert merged.current is not None
    assert merged.current.result.point(2).fit_result is single.fit_result
    assert any(
        point.fit_result is not None
        for index, point in enumerate(adopted_branch.snapshot.result.points)
        if index != 2
    )
    for index in (0, 1, 3, 4):
        assert merged.current.result.point(
            index
        ) is adopted_branch.snapshot.result.point(index)


@pytest.mark.parametrize("tied_side", ("previous", "new", "both"))
def test_exact_width_ties_adopt_fresh_identities(
    selected_auto_problem: tuple[
        WorkflowProject,
        ProjectDataset,
        ManualFitDraft,
        SingleQAutoFitOutcome,
        tuple[CandidateFitResult, CandidateFitResult],
    ],
    tied_side: str,
) -> None:
    project, _sample, draft, evaluation, evidence = selected_auto_problem
    candidate = evidence[0]
    assert candidate.fit is not None
    context = evaluation.scientific_context
    config = replace(
        candidate.fit.configuration,
        lorentzians=tuple(
            replace(
                component, fwhm=replace(component.fwhm, initial_value=0.2, free=False)
            )
            for component in candidate.fit.configuration.lorentzians
        ),
    )
    tied_fit = fit_single_q(context.prepared_resolution, context.selection, 2, config)
    tied = CandidateFitResult(candidate.candidate, tied_fit)
    tied_evaluation = _single_q_auto_outcome(context, (tied,), most_recommended=tied)
    previous = adopt_single_q_auto_fit_candidate(
        project,
        draft,
        tied_evaluation if tied_side in ("previous", "both") else evaluation,
        tied if tied_side in ("previous", "both") else candidate,
        center_group_id="exact-tie",
    )
    assert previous.success and previous.snapshot is not None
    current = adopt_single_q_auto_fit_candidate(
        project,
        draft,
        tied_evaluation if tied_side in ("new", "both") else evaluation,
        tied if tied_side in ("new", "both") else candidate,
        current=previous.snapshot,
        center_group_id="exact-tie",
    )
    assert current.success and current.snapshot is not None
    old_fit = previous.fit_result
    new_fit = current.fit_result
    assert old_fit is not None and old_fit.fitted_model is not None
    assert new_fit is not None and new_fit.fitted_model is not None
    assert {item.identity for item in old_fit.fitted_model.lorentzians}.isdisjoint(
        item.identity for item in new_fit.fitted_model.lorentzians
    )
    assert current.snapshot.result.point(2).identity_marks == ()


def test_three_lorentzian_swap_preserves_tie_and_exact_evidence(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, _configs, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)
    model = draft.setup(2).model
    assert model is not None
    third = ComponentIdentity(ComponentFamily.LORENTZIAN, "third")
    model = replace(
        model,
        lorentzians=(
            *model.lorentzians,
            ManualLorentzianState(
                area=ManualParameterIntent(0.1, 0.0, 2.0),
                fwhm=ManualParameterIntent(0.6, 1e-6, 1.0),
                identity=third,
            ),
        ),
    )
    draft = draft.with_model(2, model)
    draft = create_parameter_tie(
        draft,
        2,
        (ParameterReference(L1, ParameterFamily.AREA),),
        source_member=ParameterReference(L1, ParameterFamily.AREA),
        tie_group_id="area-chain",
    )
    draft = join_parameter_tie(
        draft, 2, "area-chain", ParameterReference(third, ParameterFamily.AREA)
    )
    executed = run_and_adopt_manual_fit(project, draft, 2)
    assert executed.success and executed.snapshot is not None
    assert executed.adopted_draft is not None and executed.fit_result is not None
    before = executed.fit_result
    state, swapped_draft = swap_current_lorentzian_identities(
        set_current_result(FittingWorkspaceState(), executed.snapshot),
        executed.adopted_draft,
        2,
        L1,
        L2,
    )
    assert state.current is not None
    after = state.current.result.point(2).fit_result
    assert after is not None and after.fitted_model is not None
    mapping = {L1: L2, L2: L1, third: third}
    for parameter in before.parameters:
        reference = parameter.references[0]
        relabeled = replace(
            reference, component=mapping.get(reference.component, reference.component)
        )
        actual = after.parameter_by_reference(relabeled)
        assert actual.value == parameter.value
        assert actual.standard_error == parameter.standard_error
        assert actual.free == parameter.free
    np.testing.assert_array_equal(after.covariance, before.covariance)
    np.testing.assert_array_equal(after.correlation, before.correlation)
    np.testing.assert_array_equal(after.raw_residuals, before.raw_residuals)
    np.testing.assert_array_equal(
        after.standardized_residuals, before.standardized_residuals
    )
    np.testing.assert_array_equal(after.evaluation.total, before.evaluation.total)
    for old, new in mapping.items():
        np.testing.assert_array_equal(
            after.evaluation.component_curve(new),
            before.evaluation.component_curve(old),
        )
    assert after.statistics is before.statistics
    assert after.diagnostics is before.diagnostics
    assert set(after.fitted_model.parameter_ties[0].members) == {
        ParameterReference(L2, ParameterFamily.AREA),
        ParameterReference(third, ParameterFamily.AREA),
    }
    working = swapped_draft.setup(2).model
    assert working is not None
    assert set(working.parameter_ties[0].members) == set(
        after.fitted_model.parameter_ties[0].members
    )


def test_unified_non_fitted_qs_contain_no_nested_fit_result(
    selected_auto_problem: tuple[
        WorkflowProject,
        ProjectDataset,
        ManualFitDraft,
        SingleQAutoFitOutcome,
        tuple[CandidateFitResult, CandidateFitResult],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, _sample, draft, evaluation, evidence = selected_auto_problem
    original = fit_single_q

    def unsuccessful(
        prepared: PreparedResolution,
        selection: FittingSelection,
        group: int,
        model: SpectralModelDefinition,
        *,
        max_nfev: int,
    ) -> FitResult:
        fit = original(prepared, selection, group, model, max_nfev=max_nfev)
        return replace(
            fit,
            diagnostics=replace(
                fit.diagnostics,
                optimizer_success=False,
                optimizer_status=0,
                optimizer_message="maximum evaluations reached",
            ),
        )

    def materialize_with_blocker(
        model: ManualModelState,
        prepared: PreparedResolution,
        selection: FittingSelection,
        group_index: int,
    ) -> ManualModelMaterialization:
        if group_index == 3:
            raise ValueError("target model unavailable")
        return materialize_manual_model(model, prepared, selection, group_index)

    def contains_fit_result(value: object) -> bool:
        if isinstance(value, FitResult):
            return True
        if is_dataclass(value) and not isinstance(value, type):
            return any(
                contains_fit_result(getattr(value, item.name)) for item in fields(value)
            )
        if isinstance(value, dict):
            return any(
                contains_fit_result(item) for pair in value.items() for item in pair
            )
        if isinstance(value, (tuple, list)):
            return any(contains_fit_result(item) for item in value)
        return False

    monkeypatch.setattr(batch_core, "fit_single_q", unsuccessful)
    monkeypatch.setattr(
        multi_q_workflow, "materialize_manual_model", materialize_with_blocker
    )
    selected = continue_selected_auto_fit_candidates(
        project, draft, evaluation, (evidence[0].candidate,), fit_excluded_groups=(0,)
    ).branches[0]
    adopted = adopt_selected_auto_fit_branch(
        project, draft, evaluation, selected, center_group_id="failed-evidence"
    )
    for point in adopted.snapshot.result.points:
        if point.status is MultiQFitStatus.SUCCESS:
            assert point.fit_result is not None
            assert point.working_state is not None
            assert point.outcome is adopted.branch_result.outcome(
                point.outcome.group_index
            )
        else:
            assert point.fit_result is None and point.working_state is None
            assert point.outcome.fit_result is None
            assert not contains_fit_result(point)
    failed = adopted.snapshot.result.point(1)
    assert failed.status is MultiQFitStatus.FAILED
    original_failure = selected.branch_result.outcome(1)
    assert original_failure.fit_result is not None
    assert (
        original_failure.fit_result.configuration.background is BackgroundModel.CONSTANT
    )
    assert adopted.branch_result.outcome(1) is original_failure
    assert failed.outcome.error_type == original_failure.error_type
    assert failed.outcome.error_message == original_failure.error_message
    assert failed.outcome.diagnostics == original_failure.diagnostics
    assert (
        failed.optimizer_status
        == original_failure.fit_result.diagnostics.optimizer_status
        == 0
    )
    assert failed.optimizer_message == "maximum evaluations reached"
    for projected in (
        QResult(original_failure),
        replace(failed, outcome=original_failure),
    ):
        assert not contains_fit_result(projected)
        assert projected.optimizer_status == failed.optimizer_status
        assert projected.optimizer_message == failed.optimizer_message
    blocked = adopted.snapshot.result.point(3)
    assert blocked.status is MultiQFitStatus.BLOCKED
    assert blocked.outcome.diagnostics
    assert blocked.outcome.diagnostics == selected.branch_result.outcome(3).diagnostics
    context = evaluation.scientific_context
    anchor = evidence[0].fit
    assert anchor is not None
    anchor = replace(
        anchor,
        context_binding=FitContextBinding(
            context.prepared_resolution, context.selection, 2
        ),
    )
    for cancelled in (False, True):

        def cancel(value: bool = cancelled) -> bool:
            return value

        branch = execute_multi_q_branch(
            context.prepared_resolution,
            context.selection,
            (None,) * 5,
            anchor_group_index=2,
            anchor_fit=anchor,
            fit_excluded_groups=(0,),
            cancel_requested=cancel,
        )
        for outcome in branch.outcomes:
            if outcome.status is not MultiQFitStatus.SUCCESS:
                projected = QResult(outcome)
                assert projected.fit_result is None
                assert projected.outcome.fit_result is None
                assert not contains_fit_result(projected)


def test_selected_auto_candidates_reject_invalid_evidence_before_execution(
    selected_auto_problem: tuple[
        WorkflowProject,
        ProjectDataset,
        ManualFitDraft,
        SingleQAutoFitOutcome,
        tuple[CandidateFitResult, CandidateFitResult],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, _sample, draft, outcome, evidence = selected_auto_problem
    first, second = evidence

    def unexpected_branch(*_args: object, **_kwargs: object) -> MultiQBranchResult:
        raise AssertionError("all selections must validate before branch execution")

    monkeypatch.setattr(
        "ezqens.workflow.multi_q.execute_multi_q_branch",
        unexpected_branch,
    )
    with pytest.raises(ValueError, match="duplicate AutoFit candidate"):
        continue_selected_auto_fit_candidates(
            project,
            draft,
            outcome,
            (first.candidate, first.candidate),
        )

    absent = StandardModelCandidate(3, BackgroundModel.NONE)
    with pytest.raises(ValueError, match="absent from the anchor evaluation"):
        continue_selected_auto_fit_candidates(
            project,
            draft,
            outcome,
            (first.candidate, absent),
        )

    failed_candidate = StandardModelCandidate(0, BackgroundModel.NONE)
    failed = CandidateFitResult(
        failed_candidate,
        None,
        error_type="FittingError",
        error_message="candidate failed",
    )
    failed_outcome = replace(
        outcome,
        recommendation=cast(
            AutoFitRecommendation,
            SimpleNamespace(
                candidate_results=(*evidence, failed),
                most_recommended=first,
            ),
        ),
    )
    with pytest.raises(ValueError, match="no successful anchor FitResult"):
        continue_selected_auto_fit_candidates(
            project,
            draft,
            failed_outcome,
            (failed_candidate,),
        )

    assert second.fit is not None
    binding = second.fit.context_binding
    assert binding is not None
    stale_selection = replace(binding.selection)
    stale_fit = replace(
        second.fit,
        context_binding=FitContextBinding(
            binding.prepared_resolution,
            stale_selection,
            binding.group_index,
        ),
    )
    stale = replace(second, fit=stale_fit)
    stale_outcome = replace(
        outcome,
        recommendation=cast(
            AutoFitRecommendation,
            SimpleNamespace(
                candidate_results=(first, stale),
                most_recommended=first,
            ),
        ),
    )
    with pytest.raises(ValueError, match="does not belong to the active context"):
        continue_selected_auto_fit_candidates(
            project,
            draft,
            stale_outcome,
            (second.candidate,),
        )

    old_q_bins = binding.prepared_resolution.sample_dataset.q_bins
    assert old_q_bins is not None
    foreign_prepared, foreign_selection = _with_q_values(
        binding.prepared_resolution,
        binding.selection,
        old_q_bins.q_values,
    )
    foreign_fit = replace(
        second.fit,
        context_binding=FitContextBinding(
            foreign_prepared,
            foreign_selection,
            binding.group_index,
        ),
    )
    foreign = replace(second, fit=foreign_fit)
    foreign_outcome = replace(
        outcome,
        recommendation=cast(
            AutoFitRecommendation,
            SimpleNamespace(
                candidate_results=(first, foreign),
                most_recommended=first,
            ),
        ),
    )
    with pytest.raises(ValueError, match="does not belong to the active context"):
        continue_selected_auto_fit_candidates(
            project,
            draft,
            foreign_outcome,
            (second.candidate,),
        )


def test_selected_auto_candidate_cancellation_preserves_completed_branch(
    selected_auto_problem: tuple[
        WorkflowProject,
        ProjectDataset,
        ManualFitDraft,
        SingleQAutoFitOutcome,
        tuple[CandidateFitResult, CandidateFitResult],
    ],
) -> None:
    project, _sample, draft, outcome, evidence = selected_auto_problem
    checks = 0
    progress: list[SelectedAutoFitProgressEvent] = []

    def cancel_after_first_branch() -> bool:
        nonlocal checks
        checks += 1
        return checks > 4

    result = continue_selected_auto_fit_candidates(
        project,
        draft,
        outcome,
        (evidence[1].candidate, evidence[0].candidate),
        cancel_requested=cancel_after_first_branch,
        progress_callback=progress.append,
        max_nfev=800,
    )

    assert result.status is MultiQExecutionStatus.CANCELLED
    assert len(result.branches) == 1
    assert result.branches[0].candidate == evidence[1].candidate
    assert result.branches[0].branch_result.status is MultiQExecutionStatus.COMPLETED
    assert all(
        item.status is not MultiQFitStatus.NOT_RUN
        for item in result.branches[0].branch_result.outcomes
    )
    assert len(progress) == 5
    assert all(item.candidate == evidence[1].candidate for item in progress)
    assert [item.outcome.group_index for item in progress] == [2, 1, 0, 3, 4]


def test_selected_auto_progress_exception_does_not_stop_later_candidates(
    selected_auto_problem: tuple[
        WorkflowProject,
        ProjectDataset,
        ManualFitDraft,
        SingleQAutoFitOutcome,
        tuple[CandidateFitResult, CandidateFitResult],
    ],
) -> None:
    project, _sample, draft, outcome, evidence = selected_auto_problem
    selected = (evidence[1].candidate, evidence[0].candidate)
    attempted: list[SelectedAutoFitProgressEvent] = []

    def failing_observer(event: SelectedAutoFitProgressEvent) -> None:
        attempted.append(event)
        raise RuntimeError("selected progress observer failed")

    result = continue_selected_auto_fit_candidates(
        project,
        draft,
        outcome,
        selected,
        progress_callback=failing_observer,
        max_nfev=800,
    )

    assert result.status is MultiQExecutionStatus.COMPLETED
    assert tuple(item.candidate for item in result.branches) == selected
    assert [item.candidate for item in attempted] == [
        *([selected[0]] * 5),
        *([selected[1]] * 5),
    ]


def test_cancelled_selected_auto_result_requires_an_ordered_terminal_boundary(
    selected_auto_problem: tuple[
        WorkflowProject,
        ProjectDataset,
        ManualFitDraft,
        SingleQAutoFitOutcome,
        tuple[CandidateFitResult, CandidateFitResult],
    ],
) -> None:
    project, _sample, draft, outcome, evidence = selected_auto_problem
    selected = (evidence[1].candidate, evidence[0].candidate)
    completed = continue_selected_auto_fit_candidates(
        project,
        draft,
        outcome,
        selected,
        max_nfev=800,
    )
    cancelled_first = continue_selected_auto_fit_candidates(
        project,
        draft,
        outcome,
        selected[:1],
        cancel_requested=lambda: True,
        max_nfev=800,
    ).branches[0]
    cancelled_second = continue_selected_auto_fit_candidates(
        project,
        draft,
        outcome,
        selected[1:],
        cancel_requested=lambda: True,
        max_nfev=800,
    ).branches[0]

    between_branches = SelectedAutoFitMultiQResult(
        MultiQExecutionStatus.CANCELLED,
        selected,
        completed.branches[:1],
    )
    with_partial_final = SelectedAutoFitMultiQResult(
        MultiQExecutionStatus.CANCELLED,
        selected,
        (completed.branches[0], cancelled_second),
    )
    assert len(between_branches.branches) == 1
    assert with_partial_final.branches[-1].branch_result.status is (
        MultiQExecutionStatus.CANCELLED
    )

    with pytest.raises(ValueError, match="final retained branch"):
        SelectedAutoFitMultiQResult(
            MultiQExecutionStatus.CANCELLED,
            selected,
            (cancelled_first, completed.branches[1]),
        )
    with pytest.raises(ValueError, match="final retained branch"):
        SelectedAutoFitMultiQResult(
            MultiQExecutionStatus.CANCELLED,
            selected,
            (cancelled_first, cancelled_second),
        )
    with pytest.raises(ValueError, match="selection order"):
        SelectedAutoFitMultiQResult(
            MultiQExecutionStatus.CANCELLED,
            selected,
            (completed.branches[1],),
        )


def test_fresh_auto_all_q_uses_most_recommended_direct_b0_anchor_only(
    selected_auto_problem: tuple[
        WorkflowProject,
        ProjectDataset,
        ManualFitDraft,
        SingleQAutoFitOutcome,
        tuple[CandidateFitResult, CandidateFitResult],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, _sample, draft, outcome, evidence = selected_auto_problem
    candidate, other_successful = evidence
    assert candidate.fit is not None
    calls = 0

    def fake_auto(*_args: object, **_kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            recommendation=SimpleNamespace(
                most_recommended=candidate,
                candidate_results=(candidate, other_successful),
            ),
            scientific_context=outcome.scientific_context,
        )

    monkeypatch.setattr("ezqens.workflow.multi_q.run_single_q_auto_fit", fake_auto)
    result = auto_fit_all_q(
        project,
        draft,
        anchor_group_index=2,
        max_nfev=800,
    )

    assert calls == 1
    retained_anchor = result.outcome(2).fit_result
    assert retained_anchor is not None
    assert retained_anchor.parameters == candidate.fit.parameters
    assert retained_anchor.diagnostics == candidate.fit.diagnostics
    assert retained_anchor.configuration.background is BackgroundModel.CONSTANT
    assert len(retained_anchor.configuration.lorentzians) == 2


def test_applied_b0_current_fit_retains_editable_fixed_zero_b1_branch(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(
        prepared,
        selection,
        background=BackgroundModel.LINEAR,
        slope_free=False,
    )
    anchor = run_manual_fit(project, draft, 2, max_nfev=800)

    result = fit_all_q_from_current(
        project,
        draft,
        anchor_group_index=2,
        current_fit=anchor,
        max_nfev=800,
    )

    retained_anchor = result.outcome(2).fit_result
    assert retained_anchor is not None
    assert retained_anchor.parameters == anchor.parameters
    assert retained_anchor.diagnostics == anchor.diagnostics
    assert anchor.configuration.background is BackgroundModel.LINEAR
    assert anchor.configuration.b1 is not None
    assert anchor.configuration.b1.initial_value == 0.0
    assert not anchor.configuration.b1.free


def test_topology_incompatible_current_result_is_rejected_without_refit(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, editable_draft = _workflow_problem(
        prepared,
        selection,
        background=BackgroundModel.LINEAR,
        slope_free=False,
    )
    direct_b0_draft = editable_draft
    for group_index in range(len(direct_b0_draft.setups)):
        direct_b0_draft = direct_b0_draft.with_model(
            group_index,
            _manual_model(background=BackgroundModel.CONSTANT),
        )
    auto_candidate_result = run_manual_fit(
        project,
        direct_b0_draft,
        2,
        max_nfev=800,
    )

    def unexpected_fit(*_args: object, **_kwargs: object) -> FitResult:
        raise AssertionError("incompatible Current must not be refit")

    monkeypatch.setattr(batch_core, "fit_single_q", unexpected_fit)
    with pytest.raises(ValueError, match="topology is incompatible"):
        fit_all_q_from_current(
            project,
            editable_draft,
            anchor_group_index=2,
            current_fit=auto_candidate_result,
            max_nfev=800,
        )


def test_fresh_auto_all_q_does_not_fall_back_to_best_supported(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)
    context = _manual_context(prepared, selection)

    def fake_auto(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            recommendation=SimpleNamespace(
                most_recommended=None,
                best_supported_candidate=object(),
            ),
            scientific_context=context,
        )

    monkeypatch.setattr("ezqens.workflow.multi_q.run_single_q_auto_fit", fake_auto)
    with pytest.raises(ValueError, match="Most Recommended"):
        auto_fit_all_q(project, draft, anchor_group_index=2)


def test_current_fit_composition_materializes_an_initially_empty_branch(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, configured = _workflow_problem(prepared, selection)
    anchor = run_manual_fit(project, configured, 2, max_nfev=800)
    empty = replace(
        configured,
        setups=tuple(
            setup if setup.group_index == 2 else replace(setup, model=None)
            for setup in configured.setups
        ),
    )

    result = fit_all_q_from_current(
        project,
        empty,
        anchor_group_index=2,
        current_fit=anchor,
        max_nfev=800,
    )

    anchor_identities = {
        component.identity for component in anchor.configuration.lorentzians
    }
    assert all(outcome.status is MultiQFitStatus.SUCCESS for outcome in result.outcomes)
    assert all(
        outcome.fit_result is not None
        and {
            component.identity
            for component in outcome.fit_result.configuration.lorentzians
        }
        == anchor_identities
        for outcome in result.outcomes
    )


@pytest.mark.parametrize("background", [BackgroundModel.NONE, BackgroundModel.CONSTANT])
@pytest.mark.parametrize("transfer_all", [False, True])
def test_s5_empty_target_preserves_independent_centers_after_manual_adoption(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    background: BackgroundModel,
    transfer_all: bool,
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)
    source = _manual_model(background=background)
    source = replace(
        source,
        lorentzians=(
            replace(source.lorentzians[0], center=ManualParameterIntent(0.06)),
        ),
        b0=(
            ManualParameterIntent(0.02, -1.0, 1.0, free=False)
            if background is BackgroundModel.CONSTANT
            else None
        ),
        b1=None,
    )
    draft = draft.with_model(2, source).with_model(1, None)
    execution = run_and_adopt_manual_fit(project, draft, 2, max_nfev=800)
    assert execution.success
    assert execution.fit_result is not None
    assert execution.adopted_draft is not None
    anchor_model = execution.adopted_draft.setup(2).model
    assert anchor_model is not None
    active = active_transferable_constraints(anchor_model)
    policy = MultiQConstraintTransferPolicy(
        anchor_model,
        bounds=(
            frozenset(item.reference for item in active.bounds)
            if transfer_all
            else frozenset()
        ),
        fixed=(
            frozenset(item.reference for item in active.fixed)
            if transfer_all
            else frozenset()
        ),
        center_ties=(
            frozenset(item.identity for item in active.center_ties)
            if transfer_all
            else frozenset()
        ),
    )
    assert not active.center_ties

    branch = fit_all_q_from_current(
        project,
        execution.adopted_draft,
        anchor_group_index=2,
        current_fit=execution.fit_result,
        transfer_policy=policy,
        fit_excluded_groups=(0, 3, 4),
        max_nfev=800,
    )
    target = branch.outcome(1)
    assert target.status is MultiQFitStatus.SUCCESS
    assert target.fit_result is not None
    elastic_center = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER)
    lorentzian_center = ParameterReference(L1, ParameterFamily.CENTER)
    assert target.fit_result.parameter_by_reference(
        elastic_center
    ) is not target.fit_result.parameter_by_reference(lorentzian_center)
    assert target.fit_result.configuration.energy_shift is not None
    assert target.fit_result.configuration.lorentzians[0].center is not None
    assert target.fit_result.configuration.elastic_area is not None
    assert target.fit_result.configuration.elastic_area.upper_bound == (
        2.0 if transfer_all else math.inf
    )
    if background is BackgroundModel.CONSTANT:
        assert target.fit_result.configuration.b0 is not None
        assert target.fit_result.configuration.b0.free is not transfer_all
    for reference in (elastic_center, lorentzian_center):
        assert target.fit_result.configuration.parameter_configuration(
            reference
        ).initial_value == pytest.approx(
            execution.fit_result.parameter_by_reference(reference).value
        )


@pytest.mark.parametrize("transfer_tie", [False, True])
def test_s5_empty_target_applies_only_selected_shared_center_group(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    transfer_tie: bool,
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)
    source = _manual_model(background=BackgroundModel.CONSTANT)
    source = replace(
        source,
        energy_shift=None,
        lorentzians=(replace(source.lorentzians[0], center_group="anchor-center"),),
        center_groups=(
            ManualCenterGroupState("anchor-center", ManualParameterIntent(0.01)),
        ),
        elastic_center_group="anchor-center",
        b1=None,
    )
    draft = draft.with_model(2, source).with_model(1, None)
    execution = run_and_adopt_manual_fit(project, draft, 2, max_nfev=800)
    assert execution.success
    assert execution.fit_result is not None
    assert execution.adopted_draft is not None
    anchor_model = execution.adopted_draft.setup(2).model
    assert anchor_model is not None
    active = active_transferable_constraints(anchor_model)
    center_ties = (
        frozenset(item.identity for item in active.center_ties)
        if transfer_tie
        else frozenset()
    )
    assert len(active.center_ties) == 1

    branch = fit_all_q_from_current(
        project,
        execution.adopted_draft,
        anchor_group_index=2,
        current_fit=execution.fit_result,
        transfer_policy=MultiQConstraintTransferPolicy(
            anchor_model,
            center_ties=center_ties,
        ),
        fit_excluded_groups=(0, 3, 4),
        max_nfev=800,
    )
    target = branch.outcome(1)
    assert target.status is MultiQFitStatus.SUCCESS
    assert target.fit_result is not None
    elastic_center = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER)
    lorentzian_center = ParameterReference(L1, ParameterFamily.CENTER)
    assert (
        target.fit_result.parameter_by_reference(elastic_center)
        is target.fit_result.parameter_by_reference(lorentzian_center)
    ) is transfer_tie
    assert (
        target.fit_result.configuration.elastic_center_group == "anchor-center"
    ) is transfer_tie


@pytest.mark.parametrize("background", [BackgroundModel.NONE, BackgroundModel.CONSTANT])
@pytest.mark.parametrize("transfer_all", [False, True])
def test_s5_empty_target_applies_only_selected_manual_center_bond(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    background: BackgroundModel,
    transfer_all: bool,
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)
    source = _manual_model(background=background)
    source = replace(
        source,
        lorentzians=(
            replace(source.lorentzians[0], center=ManualParameterIntent(0.06)),
        ),
        b1=None,
    )
    elastic_center = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER)
    lorentzian_center = ParameterReference(L1, ParameterFamily.CENTER)
    draft = draft.with_model(2, source).with_model(1, None)
    draft = create_parameter_tie(
        draft,
        2,
        (elastic_center,),
        source_member=elastic_center,
        tie_group_id="center-bond",
    )
    draft = join_parameter_tie(draft, 2, "center-bond", lorentzian_center)
    execution = run_and_adopt_manual_fit(project, draft, 2, max_nfev=800)
    assert execution.success
    assert execution.fit_result is not None
    assert execution.adopted_draft is not None
    anchor_model = execution.adopted_draft.setup(2).model
    assert anchor_model is not None
    active = active_transferable_constraints(anchor_model)
    assert len(active.center_ties) == 1
    assert active.center_ties[0].identity == CenterTieIdentity(
        CenterTieKind.PARAMETER_TIE, "center-bond"
    )
    policy = MultiQConstraintTransferPolicy(
        anchor_model,
        bounds=(
            frozenset(item.reference for item in active.bounds)
            if transfer_all
            else frozenset()
        ),
        fixed=(
            frozenset(item.reference for item in active.fixed)
            if transfer_all
            else frozenset()
        ),
        center_ties=(
            frozenset(item.identity for item in active.center_ties)
            if transfer_all
            else frozenset()
        ),
    )

    branch = fit_all_q_from_current(
        project,
        execution.adopted_draft,
        anchor_group_index=2,
        current_fit=execution.fit_result,
        transfer_policy=policy,
        fit_excluded_groups=(0, 3, 4),
        max_nfev=800,
    )
    target = branch.outcome(1)
    assert target.status is MultiQFitStatus.SUCCESS
    assert target.fit_result is not None
    assert (
        target.fit_result.parameter_by_reference(elastic_center)
        is target.fit_result.parameter_by_reference(lorentzian_center)
    ) is transfer_all
    assert {
        group.group_id for group in target.fit_result.configuration.parameter_ties
    } == ({"center-bond"} if transfer_all else set())


def test_s5_selected_center_bond_cannot_detach_populated_legacy_center(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)
    source = _manual_model(background=BackgroundModel.CONSTANT)
    source = replace(
        source,
        lorentzians=(
            replace(source.lorentzians[0], center=ManualParameterIntent(0.06)),
        ),
        b1=None,
    )
    populated_target = replace(
        source,
        lorentzians=(replace(source.lorentzians[0], center=None),),
    )
    elastic_center = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER)
    lorentzian_center = ParameterReference(L1, ParameterFamily.CENTER)
    draft = draft.with_model(2, source).with_model(1, populated_target)
    draft = create_parameter_tie(
        draft,
        2,
        (elastic_center,),
        source_member=elastic_center,
        tie_group_id="center-bond",
    )
    draft = join_parameter_tie(draft, 2, "center-bond", lorentzian_center)
    execution = run_and_adopt_manual_fit(project, draft, 2, max_nfev=800)
    assert execution.success
    assert execution.fit_result is not None
    assert execution.adopted_draft is not None
    anchor_model = execution.adopted_draft.setup(2).model
    assert anchor_model is not None
    tie = CenterTieIdentity(CenterTieKind.PARAMETER_TIE, "center-bond")

    baseline = fit_all_q_from_current(
        project,
        execution.adopted_draft,
        anchor_group_index=2,
        current_fit=execution.fit_result,
        transfer_policy=MultiQConstraintTransferPolicy(anchor_model),
        fit_excluded_groups=(0, 3, 4),
        max_nfev=800,
    )
    assert baseline.outcome(1).status is MultiQFitStatus.SUCCESS
    selected = fit_all_q_from_current(
        project,
        execution.adopted_draft,
        anchor_group_index=2,
        current_fit=execution.fit_result,
        transfer_policy=MultiQConstraintTransferPolicy(
            anchor_model,
            center_ties=frozenset((tie,)),
        ),
        fit_excluded_groups=(0, 3, 4),
        max_nfev=800,
    )
    assert selected.outcome(1).status is MultiQFitStatus.BLOCKED
    assert selected.outcome(1).fit_result is None


@pytest.mark.parametrize("background", [BackgroundModel.NONE, BackgroundModel.CONSTANT])
def test_s5_success_adopts_fresh_independent_target_for_next_manual_run(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    background: BackgroundModel,
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, sample, draft = _workflow_problem(prepared, selection)
    source = _manual_model(background=background)
    source = replace(
        source,
        lorentzians=(
            replace(source.lorentzians[0], center=ManualParameterIntent(0.06)),
        ),
        b1=None,
    )
    draft = draft.with_model(2, source).with_model(1, None)
    execution = run_and_adopt_manual_fit(project, draft, 2, max_nfev=800)
    assert execution.success
    assert execution.fit_result is not None
    assert execution.adopted_draft is not None
    assert resolve_manual_fit_context(project, sample, 1).context is not None
    assert execution.adopted_draft.setup(1).model is None

    continued = fit_all_q_from_current_with_adoption(
        project,
        execution.adopted_draft,
        anchor_group_index=2,
        current_fit=execution.fit_result,
        fit_excluded_groups=(0, 3, 4),
        max_nfev=800,
    )
    target = continued.branch_result.outcome(1)
    assert target.status is MultiQFitStatus.SUCCESS
    assert target.fit_result is not None
    adopted = continued.adopted_draft.setup(1).model
    assert adopted is not None
    assert execution.adopted_draft.setup(1).model is None
    assert adopted.lorentzians[0].identity == L1
    assert adopted.lorentzians[0].center is not None
    assert not adopted.parameter_ties
    for reference in adopted.parameter_references():
        assert adopted.parameter_intent(reference).current_value == pytest.approx(
            target.fit_result.parameter_by_reference(reference).value
        )
    elastic_center = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER)
    lorentzian_center = ParameterReference(L1, ParameterFamily.CENTER)
    assert target.fit_result.parameter_by_reference(
        elastic_center
    ) is not target.fit_result.parameter_by_reference(lorentzian_center)
    retained_parameters = target.fit_result.parameters
    rerun = run_manual_fit(project, continued.adopted_draft, 1, max_nfev=800)
    assert rerun.diagnostics.optimizer_success
    assert target.fit_result.parameters == retained_parameters
    elastic_area = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA)
    original_value = target.fit_result.parameter_by_reference(elastic_area).value
    intent = adopted.parameter_intent(elastic_area)
    edited = update_manual_parameter(
        continued.adopted_draft,
        1,
        elastic_area,
        ManualParameterEdit(
            original_value + 0.01,
            intent.user_lower_limit,
            intent.user_upper_limit,
            intent.free,
            intent.user_bounds_enabled,
        ),
    )
    assert edited.setup(1).model is not adopted
    assert (
        target.fit_result.parameter_by_reference(elastic_area).value == original_value
    )


def test_s5_success_adopts_selected_center_tie_and_constraints(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)
    source = _manual_model(background=BackgroundModel.CONSTANT)
    source = replace(
        source,
        lorentzians=(
            replace(source.lorentzians[0], center=ManualParameterIntent(0.06)),
        ),
        b0=ManualParameterIntent(0.02, -0.2, 0.2, free=False),
        b1=None,
    )
    elastic_center = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER)
    lorentzian_center = ParameterReference(L1, ParameterFamily.CENTER)
    elastic_area = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA)
    offset = ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.OFFSET)
    draft = draft.with_model(2, source).with_model(1, None)
    draft = create_parameter_tie(
        draft,
        2,
        (elastic_center,),
        source_member=elastic_center,
        tie_group_id="center-bond",
    )
    draft = join_parameter_tie(draft, 2, "center-bond", lorentzian_center)
    execution = run_and_adopt_manual_fit(project, draft, 2, max_nfev=800)
    assert execution.success
    assert execution.fit_result is not None
    assert execution.adopted_draft is not None
    anchor_model = execution.adopted_draft.setup(2).model
    assert anchor_model is not None
    active = active_transferable_constraints(anchor_model)
    policy = MultiQConstraintTransferPolicy(
        anchor_model,
        bounds=frozenset(item.reference for item in active.bounds),
        fixed=frozenset(item.reference for item in active.fixed),
        center_ties=frozenset(item.identity for item in active.center_ties),
    )
    assert elastic_area in policy.bounds
    assert offset in policy.fixed

    continued = fit_all_q_from_current_with_adoption(
        project,
        execution.adopted_draft,
        anchor_group_index=2,
        current_fit=execution.fit_result,
        transfer_policy=policy,
        fit_excluded_groups=(0, 3, 4),
        max_nfev=800,
    )
    target = continued.branch_result.outcome(1)
    assert target.status is MultiQFitStatus.SUCCESS
    assert target.fit_result is not None
    adopted = continued.adopted_draft.setup(1).model
    assert adopted is not None
    assert adopted.parameter_ties[0].group_id == "center-bond"
    assert adopted.parameter_ties[0].members == (elastic_center, lorentzian_center)
    assert adopted.parameter_intent(elastic_area).user_lower_limit == 0.0
    assert adopted.parameter_intent(elastic_area).user_upper_limit == 2.0
    assert adopted.parameter_intent(elastic_area).user_bounds_enabled
    assert adopted.parameter_intent(offset).user_lower_limit == -0.2
    assert adopted.parameter_intent(offset).user_upper_limit == 0.2
    assert not adopted.parameter_intent(offset).free
    assert adopted.parameter_intent(offset).current_value == 0.02
    assert adopted.parameter_intent(elastic_center).current_value == pytest.approx(
        target.fit_result.parameter_by_reference(elastic_center).value
    )
    assert target.fit_result.parameter_by_reference(
        elastic_center
    ) is target.fit_result.parameter_by_reference(lorentzian_center)


def test_s5_success_adopts_selected_center_group(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)
    source = _manual_model(background=BackgroundModel.CONSTANT)
    source = replace(
        source,
        energy_shift=None,
        lorentzians=(replace(source.lorentzians[0], center_group="anchor-center"),),
        center_groups=(
            ManualCenterGroupState("anchor-center", ManualParameterIntent(0.01)),
        ),
        elastic_center_group="anchor-center",
        b1=None,
    )
    draft = draft.with_model(2, source).with_model(1, None)
    execution = run_and_adopt_manual_fit(project, draft, 2, max_nfev=800)
    assert execution.success
    assert execution.fit_result is not None
    assert execution.adopted_draft is not None
    anchor_model = execution.adopted_draft.setup(2).model
    assert anchor_model is not None
    active = active_transferable_constraints(anchor_model)
    assert len(active.center_ties) == 1
    continued = fit_all_q_from_current_with_adoption(
        project,
        execution.adopted_draft,
        anchor_group_index=2,
        current_fit=execution.fit_result,
        transfer_policy=MultiQConstraintTransferPolicy(
            anchor_model,
            center_ties=frozenset(item.identity for item in active.center_ties),
        ),
        fit_excluded_groups=(0, 3, 4),
        max_nfev=800,
    )
    target = continued.branch_result.outcome(1)
    assert target.status is MultiQFitStatus.SUCCESS
    assert target.fit_result is not None
    adopted = continued.adopted_draft.setup(1).model
    assert adopted is not None
    assert adopted.elastic_center_group == "anchor-center"
    assert adopted.lorentzians[0].center_group == "anchor-center"
    assert adopted.center_groups[0].intent.current_value == pytest.approx(
        target.fit_result.parameter_by_reference(
            ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER)
        ).value
    )


def test_s5_success_preserves_untransferred_target_local_manual_intent(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)
    source = _manual_model(background=BackgroundModel.CONSTANT)
    source = replace(
        source,
        lorentzians=(
            replace(source.lorentzians[0], center=ManualParameterIntent(0.06)),
        ),
        b1=None,
    )
    area = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA)
    offset = ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.OFFSET)
    target_model = replace_parameter_intent(
        source,
        area,
        ManualParameterIntent(
            0.65,
            1.5,
            0.5,
            user_bounds_enabled=False,
        ),
    )
    target_model = replace_parameter_intent(
        target_model,
        offset,
        ManualParameterIntent(0.02, -0.3, 0.3, free=False),
    )
    draft = draft.with_model(2, source).with_model(1, target_model)
    execution = run_and_adopt_manual_fit(project, draft, 2, max_nfev=800)
    assert execution.success
    assert execution.fit_result is not None
    assert execution.adopted_draft is not None
    anchor_model = execution.adopted_draft.setup(2).model
    assert anchor_model is not None
    continued = fit_all_q_from_current_with_adoption(
        project,
        execution.adopted_draft,
        anchor_group_index=2,
        current_fit=execution.fit_result,
        transfer_policy=MultiQConstraintTransferPolicy(anchor_model),
        fit_excluded_groups=(0, 3, 4),
        max_nfev=800,
    )
    fitted = continued.branch_result.outcome(1).fit_result
    assert fitted is not None
    adopted = continued.adopted_draft.setup(1).model
    assert adopted is not None
    adopted_area = adopted.parameter_intent(area)
    assert adopted_area.user_lower_limit == 1.5
    assert adopted_area.user_upper_limit == 0.5
    assert not adopted_area.user_bounds_enabled
    assert adopted_area.current_value == pytest.approx(
        fitted.parameter_by_reference(area).value
    )
    adopted_offset = adopted.parameter_intent(offset)
    assert adopted_offset.user_lower_limit == -0.3
    assert adopted_offset.user_upper_limit == 0.3
    assert not adopted_offset.free
    assert adopted_offset.current_value == 0.02
    assert execution.adopted_draft.setup(1).model is target_model


def test_s5_non_success_targets_keep_prior_manual_setups(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)
    draft = draft.with_model(1, None)
    blocked_model = draft.setup(0).model
    excluded_model = draft.setup(3).model
    not_run_model = draft.setup(4).model
    assert blocked_model is not None
    assert excluded_model is not None
    assert not_run_model is not None
    invalid = ManualParameterIntent(0.35, 0.8, 0.2)
    draft = draft.with_model(
        0,
        replace_parameter_intent(
            blocked_model,
            ParameterReference(L1, ParameterFamily.AREA),
            invalid,
        ),
    )
    anchor = run_manual_fit(project, draft, 2, max_nfev=800)

    def fail_target(
        _prepared: PreparedResolution,
        _selection: FittingSelection,
        group_index: int,
        _model: SpectralModelDefinition,
        *,
        max_nfev: int,
    ) -> FitResult:
        assert group_index == 1
        assert max_nfev == 800
        raise FittingError("synthetic target failure")

    monkeypatch.setattr(batch_core, "fit_single_q", fail_target)
    checks = 0

    def cancel_before_higher() -> bool:
        nonlocal checks
        checks += 1
        return checks == 3

    continued = fit_all_q_from_current_with_adoption(
        project,
        draft,
        anchor_group_index=2,
        current_fit=anchor,
        fit_excluded_groups=(3,),
        cancel_requested=cancel_before_higher,
        max_nfev=800,
    )
    branch = continued.branch_result
    assert branch.status is MultiQExecutionStatus.CANCELLED
    assert [item.status for item in branch.outcomes] == [
        MultiQFitStatus.BLOCKED,
        MultiQFitStatus.FAILED,
        MultiQFitStatus.SUCCESS,
        MultiQFitStatus.EXCLUDED,
        MultiQFitStatus.NOT_RUN,
    ]
    assert continued.adopted_draft is draft
    assert continued.adopted_draft.setup(1).model is None
    for index in (0, 3, 4):
        assert continued.adopted_draft.setup(index).model is draft.setup(index).model


def test_ambiguous_reversed_setup_clone_blocks_without_positional_mapping(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)
    anchor_only = replace(
        draft,
        setups=tuple(
            setup if setup.group_index == 2 else replace(setup, model=None)
            for setup in draft.setups
        ),
    )
    cloned = apply_manual_setup_to_all_groups(project, anchor_only, 2)
    lower_model = cloned.setup(1).model
    anchor_model = cloned.setup(2).model
    assert lower_model is not None
    assert anchor_model is not None
    assert {item.identity for item in lower_model.lorentzians} != {
        item.identity for item in anchor_model.lorentzians
    }
    reversed_components = tuple(reversed(lower_model.lorentzians))
    tied_reference = ParameterReference(
        reversed_components[0].identity,
        ParameterFamily.AREA,
    )
    local_fixed = ManualParameterIntent(
        current_value=0.47,
        user_lower_limit=0.4,
        user_upper_limit=0.6,
        free=False,
    )
    lower_model = replace(
        lower_model,
        lorentzians=(
            reversed_components[0],
            replace(
                reversed_components[1],
                fwhm=ManualParameterIntent(0.22, 0.1, 0.5, free=True),
            ),
        ),
        parameter_ties=(
            ManualParameterTieState(
                "local-area-chain",
                (tied_reference,),
                local_fixed,
            ),
        ),
    )
    cloned = cloned.with_model(1, lower_model)
    anchor = run_manual_fit(project, cloned, 2, max_nfev=800)

    result = fit_all_q_from_current(
        project,
        cloned,
        anchor_group_index=2,
        current_fit=anchor,
        max_nfev=800,
    )

    assert result.outcome(1).status is MultiQFitStatus.BLOCKED
    assert result.outcome(1).fit_result is None


def test_reversed_storage_order_with_branch_identities_preserves_target_state() -> None:
    source = _manual_model()
    method = capture_fitting_method(source)
    target = replace(source, lorentzians=tuple(reversed(source.lorentzians)))
    distinctive = ManualParameterIntent(
        current_value=0.61,
        user_lower_limit=0.4,
        user_upper_limit=0.8,
        free=False,
    )
    target = replace_parameter_intent(
        target,
        ParameterReference(L2, ParameterFamily.AREA),
        distinctive,
    )

    applied = apply_fitting_method(method, target)

    assert tuple(item.identity for item in applied.model.lorentzians) == (L2, L1)
    assert (
        applied.model.parameter_intent(ParameterReference(L2, ParameterFamily.AREA))
        == distinctive
    )


def test_reversed_lorentzian_storage_order_is_the_same_branch_identity(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem
    targets = list(configurations)
    targets[1] = replace(
        configurations[1],
        lorentzians=tuple(reversed(configurations[1].lorentzians)),
    )
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(batch_core, "fit_single_q", _recording_fit(anchor, calls))

    result = execute_multi_q_branch(
        prepared,
        selection,
        targets,
        anchor_group_index=2,
        anchor_fit=anchor,
    )

    assert result.outcome(1).status is MultiQFitStatus.SUCCESS
    submitted = next(model for group, model in calls if group == 1)
    assert tuple(item.identity for item in submitted.lorentzians) == (L2, L1)


def test_ambiguous_unmatched_components_raise_when_target_state_must_be_preserved() -> (
    None
):
    method = capture_fitting_method(_manual_model())
    target_l1 = ComponentIdentity(ComponentFamily.LORENTZIAN, "target-l1")
    target_l2 = ComponentIdentity(ComponentFamily.LORENTZIAN, "target-l2")
    target = replace(
        _manual_model(),
        lorentzians=(
            replace(_manual_model().lorentzians[0], identity=target_l1),
            replace(_manual_model().lorentzians[1], identity=target_l2),
        ),
    )
    distinctive = ManualParameterIntent(
        current_value=0.47,
        user_lower_limit=0.9,
        user_upper_limit=0.2,
        free=False,
        user_bounds_enabled=False,
    )
    target = replace(
        target,
        parameter_ties=(
            ManualParameterTieState(
                "target-area-chain",
                (
                    ParameterReference(target_l1, ParameterFamily.AREA),
                    ParameterReference(target_l2, ParameterFamily.AREA),
                ),
                distinctive,
            ),
        ),
    )

    with pytest.raises(ValueError, match="no trustworthy component correspondence"):
        apply_fitting_method(method, target)


def test_one_unmatched_component_has_unique_correspondence() -> None:
    method = capture_fitting_method(_manual_model())
    target_l2 = ComponentIdentity(ComponentFamily.LORENTZIAN, "target-l2")
    source = _manual_model()
    distinctive = ManualParameterIntent(0.47, 0.4, 0.8, free=False)
    target = replace(
        source,
        lorentzians=(
            replace(source.lorentzians[1], identity=target_l2, area=distinctive),
            source.lorentzians[0],
        ),
    )

    applied = apply_fitting_method(method, target)

    assert tuple(item.identity for item in applied.model.lorentzians) == (L2, L1)
    assert (
        applied.model.parameter_intent(ParameterReference(L2, ParameterFamily.AREA))
        == distinctive
    )


def test_full_method_transfer_rebuilds_ambiguous_components_without_target_order() -> (
    None
):
    l1_area = ParameterReference(L1, ParameterFamily.AREA)
    l2_area = ParameterReference(L2, ParameterFamily.AREA)
    source = replace_parameter_intent(
        _manual_model(),
        l1_area,
        ManualParameterIntent(0.23, 0.2, 0.3, free=False),
    )
    source = replace_parameter_intent(
        source,
        l2_area,
        ManualParameterIntent(0.71, 0.5, 0.9, free=True),
    )
    method = capture_fitting_method(
        source,
        transfer_defaults=MethodTransferOptions(
            user_bounds=True,
            free_fixed=True,
            parameter_relationships=True,
        ),
    )
    target_l1 = ComponentIdentity(ComponentFamily.LORENTZIAN, "target-l1")
    target_l2 = ComponentIdentity(ComponentFamily.LORENTZIAN, "target-l2")
    target_source = _manual_model()
    target = replace(
        target_source,
        lorentzians=(
            replace(target_source.lorentzians[1], identity=target_l2),
            replace(target_source.lorentzians[0], identity=target_l1),
        ),
    )

    applied = apply_fitting_method(method, target)

    assert tuple(item.identity for item in applied.model.lorentzians) == (L1, L2)
    adopted_l1 = applied.model.parameter_intent(l1_area)
    adopted_l2 = applied.model.parameter_intent(l2_area)
    assert adopted_l1 == ManualParameterIntent(0.23, 0.2, 0.3, free=False)
    assert adopted_l2.user_lower_limit == pytest.approx(0.5)
    assert adopted_l2.user_upper_limit == pytest.approx(0.9)
    assert adopted_l2.free


def test_disabled_inverted_method_bounds_round_trip_and_block_when_reenabled(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    reference = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA)
    dormant = ManualParameterIntent(
        current_value=0.8,
        user_lower_limit=1.5,
        user_upper_limit=0.5,
        user_bounds_enabled=False,
    )
    source = replace_parameter_intent(_manual_model(), reference, dormant)
    method = capture_fitting_method(
        source,
        transfer_defaults=MethodTransferOptions(user_bounds=True),
    )

    applied = apply_fitting_method(method, None).model
    applied_intent = applied.parameter_intent(reference)
    assert applied_intent.user_lower_limit == dormant.user_lower_limit
    assert applied_intent.user_upper_limit == dormant.user_upper_limit
    assert not applied_intent.user_bounds_enabled
    materialize_manual_model(applied, prepared, selection, 0)
    enabled = replace_parameter_intent(
        applied,
        reference,
        replace(applied_intent, user_bounds_enabled=True),
    )
    with pytest.raises(ValueError, match="lower.*upper|upper.*lower"):
        materialize_manual_model(enabled, prepared, selection, 0)


def test_infeasible_target_constraints_block_that_q_without_mutating_intent(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)
    reference = ParameterReference(L1, ParameterFamily.AREA)
    invalid_intent = ManualParameterIntent(
        current_value=0.35,
        user_lower_limit=0.8,
        user_upper_limit=0.2,
        user_bounds_enabled=True,
    )
    target = draft.setup(1).model
    assert target is not None
    draft = draft.with_model(
        1,
        replace_parameter_intent(target, reference, invalid_intent),
    )
    anchor = run_manual_fit(project, draft, 2, max_nfev=800)

    result = fit_all_q_from_current(
        project,
        draft,
        anchor_group_index=2,
        current_fit=anchor,
        max_nfev=800,
    )

    assert result.outcome(1).status is MultiQFitStatus.BLOCKED
    assert result.outcome(1).fit_result is None
    assert result.outcome(0).status is MultiQFitStatus.SUCCESS
    assert result.outcome(0).seed_group_index == 2
    unchanged = draft.setup(1).model
    assert unchanged is not None
    assert unchanged.parameter_intent(reference) == invalid_intent


@pytest.mark.parametrize("variant", ["support", "auto", "review"])
def test_current_result_requires_the_active_prepared_resolution_context(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    variant: str,
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)
    reviewed_keep = ResolutionAcceptance(
        decision=ResolutionAcceptanceDecision.KEEP,
        confirmed=True,
        source=ResolutionAcceptanceSource.USER_REVIEW,
        warnings=(
            (ResolutionAcceptanceWarning.SUSPICIOUS_STRUCTURE_RETAINED,)
            if variant == "review"
            else ()
        ),
    )
    if variant == "support":
        stale_prepared = prepare_measured_resolution(
            prepared.sample_dataset,
            prepared.resolution_dataset,
            acceptance_decisions={
                2: ResolutionAcceptance(
                    decision=(
                        ResolutionAcceptanceDecision.EXCLUDE_BY_CONTIGUOUS_SUPPORT
                    ),
                    confirmed=True,
                    source=ResolutionAcceptanceSource.USER_REVIEW,
                )
            },
            support_overrides={2: ResolutionSupport(-0.2, 0.2)},
        )
    else:
        stale_prepared = prepare_measured_resolution(
            prepared.sample_dataset,
            prepared.resolution_dataset,
            acceptance_decisions={2: reviewed_keep},
            apply_auto_padding={2: False} if variant == "auto" else None,
        )
    stale = fit_single_q(stale_prepared, selection, 2, _model(free=True))

    with pytest.raises(ValueError, match="active context"):
        fit_all_q_from_current(
            project,
            draft,
            anchor_group_index=2,
            current_fit=stale,
            max_nfev=800,
        )


def test_fit_exclusions_remain_excluded_when_cancellation_skips_them(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, configurations, anchor = multi_q_problem
    calls: list[tuple[int, SpectralModelDefinition]] = []
    monkeypatch.setattr(batch_core, "fit_single_q", _recording_fit(anchor, calls))

    result = execute_multi_q_branch(
        prepared,
        selection,
        configurations,
        anchor_group_index=2,
        anchor_fit=anchor,
        fit_excluded_groups={1, 4},
        cancel_requested=lambda: True,
    )

    assert result.status is MultiQExecutionStatus.CANCELLED
    assert result.outcome(1).status is MultiQFitStatus.EXCLUDED
    assert result.outcome(4).status is MultiQFitStatus.EXCLUDED
    assert result.outcome(0).status is MultiQFitStatus.NOT_RUN
    assert result.outcome(3).status is MultiQFitStatus.NOT_RUN
    assert calls == []
