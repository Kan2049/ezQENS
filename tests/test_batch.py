"""Focused tests for anchor-outward independent Multi-Q execution."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pytest

import ezqens.batch.core as batch_core
from ezqens.batch import (
    MultiQExecutionStatus,
    MultiQFitStatus,
    execute_multi_q_branch,
)
from ezqens.convolution import build_convolution_plan
from ezqens.domain import QBins, ReducedDataset, Spectrum, SpectrumRole
from ezqens.fitting import (
    BACKGROUND_COMPONENT,
    ELASTIC_COMPONENT,
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
    FittingWorkspaceState,
    ManualFitContext,
    ManualFitDraft,
    ManualLorentzianState,
    ManualModelState,
    ManualParameterTieState,
    MethodTransferOptions,
    MethodTransferOverrides,
    ProjectDataset,
    WorkflowProject,
    add_project_dataset,
    apply_fitting_method,
    apply_manual_setup_to_all_groups,
    apply_resolution,
    auto_fit_all_q,
    capture_fitting_method,
    commit_fitting_selection,
    create_project,
    fit_all_q_from_current,
    materialize_manual_model,
    open_manual_fit_draft,
    run_manual_fit,
    save_current_result,
    save_working_method,
    set_current_result,
    set_working_method,
)
from ezqens.workflow.manual_state import replace_parameter_intent

FloatArray = npt.NDArray[np.float64]


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
    assert result.outcome(2).fit_result is anchor
    assert result.status is MultiQExecutionStatus.COMPLETED


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
    state = save_current_result(set_current_result(state, anchor), "saved result")
    replacement_result = replace(anchor, fitted_model=anchor.configuration)
    state = set_current_result(state, replacement_result)

    assert state.saved_methods[0].method is first_method
    assert state.working_method is edited_method
    assert state.saved_results[0].result is anchor
    assert state.current_result is replacement_result


def test_fit_all_q_runs_missing_manual_anchor_then_continues(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, _sample, draft = _workflow_problem(prepared, selection)

    result = fit_all_q_from_current(
        project,
        draft,
        anchor_group_index=2,
        max_nfev=800,
    )

    assert result.outcome(2).status is MultiQFitStatus.SUCCESS
    assert all(item.status is MultiQFitStatus.SUCCESS for item in result.outcomes)


def test_fresh_auto_all_q_uses_most_recommended_direct_b0_anchor_only(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection, _configurations, _anchor = multi_q_problem
    project, sample, draft = _workflow_problem(
        prepared,
        selection,
        background=BackgroundModel.CONSTANT,
    )
    anchor = run_manual_fit(project, draft, 2, max_nfev=800)
    assert anchor.context_binding is not None
    candidate = CandidateFitResult(
        StandardModelCandidate(2, BackgroundModel.CONSTANT),
        anchor,
    )
    context = _manual_context(
        anchor.context_binding.prepared_resolution,
        anchor.context_binding.selection,
    )
    context = replace(context, project_id=project.project_id, sample=sample)
    calls = 0

    def fake_auto(*_args: object, **_kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            recommendation=SimpleNamespace(most_recommended=candidate),
            scientific_context=context,
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
    assert retained_anchor.parameters == anchor.parameters
    assert retained_anchor.diagnostics == anchor.diagnostics
    assert anchor.configuration.background is BackgroundModel.CONSTANT


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


def test_applied_auto_b0_result_is_refit_as_its_editable_b1_working_state(
    multi_q_problem: tuple[
        PreparedResolution,
        FittingSelection,
        tuple[SpectralModelDefinition, ...],
        FitResult,
    ],
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

    result = fit_all_q_from_current(
        project,
        editable_draft,
        anchor_group_index=2,
        current_fit=auto_candidate_result,
        max_nfev=800,
    )

    editable_anchor = result.outcome(2).fit_result
    assert editable_anchor is not None
    assert editable_anchor is not auto_candidate_result
    assert editable_anchor.configuration.background is BackgroundModel.LINEAR
    assert editable_anchor.configuration.b1 is not None
    assert editable_anchor.configuration.b1.initial_value == 0.0
    assert not editable_anchor.configuration.b1.free


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
        setups=tuple(replace(setup, model=None) for setup in configured.setups),
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
