"""Independent anchor-outward Multi-Q fitting through the single-Q core."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from ezqens.domain import DiagnosticSeverity
from ezqens.fitting.core import fit_single_q, manual_fit_readiness
from ezqens.fitting.models import (
    BACKGROUND_COMPONENT,
    ELASTIC_COMPONENT,
    CenterGroup,
    FitResult,
    LorentzianComponent,
    ManualFitDiagnosticCode,
    ManualFitReadinessDiagnostic,
    ParameterConfiguration,
    ParameterFamily,
    ParameterReference,
    ParameterTieGroup,
    SpectralModelDefinition,
)
from ezqens.preprocessing import FittingSelection
from ezqens.resolution import PreparedResolution


class MultiQFitStatus(StrEnum):
    """Execution state for one Q group in a single branch."""

    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    NOT_RUN = "not_run"


class MultiQExecutionStatus(StrEnum):
    """Terminal state of one anchor-defined execution."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class MultiQFitOutcome:
    """One independent per-Q outcome, including retained failed evidence."""

    group_index: int
    status: MultiQFitStatus
    fit_result: FitResult | None = None
    seed_group_index: int | None = None
    diagnostics: tuple[ManualFitReadinessDiagnostic, ...] = ()
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        diagnostics = tuple(self.diagnostics)
        if isinstance(self.group_index, bool) or not isinstance(self.group_index, int):
            raise ValueError("group_index must be an integer")
        if self.group_index < 0:
            raise ValueError("group_index must be nonnegative")
        if not isinstance(self.status, MultiQFitStatus):
            raise ValueError("status must be a MultiQFitStatus")
        if self.status is MultiQFitStatus.SUCCESS:
            if (
                self.fit_result is None
                or not self.fit_result.diagnostics.optimizer_success
            ):
                raise ValueError("SUCCESS requires a usable successful FitResult")
        elif self.status is MultiQFitStatus.BLOCKED:
            if self.fit_result is not None:
                raise ValueError("BLOCKED must not contain a FitResult")
            if not any(
                diagnostic.severity is DiagnosticSeverity.ERROR
                for diagnostic in diagnostics
            ):
                raise ValueError("BLOCKED requires an error diagnostic")
        elif self.status is MultiQFitStatus.NOT_RUN:
            if (
                any(
                    value is not None
                    for value in (
                        self.fit_result,
                        self.seed_group_index,
                        self.error_type,
                        self.error_message,
                    )
                )
                or diagnostics
            ):
                raise ValueError("NOT_RUN must not contain execution evidence")
        object.__setattr__(self, "diagnostics", diagnostics)


@dataclass(frozen=True, slots=True)
class MultiQBranchResult:
    """Ordered outcomes for one fixed-topology anchor-defined branch."""

    anchor_group_index: int
    status: MultiQExecutionStatus
    outcomes: tuple[MultiQFitOutcome, ...]

    def __post_init__(self) -> None:
        outcomes = tuple(self.outcomes)
        if not outcomes:
            raise ValueError("a Multi-Q branch requires at least one group")
        if tuple(item.group_index for item in outcomes) != tuple(range(len(outcomes))):
            raise ValueError("Multi-Q outcomes must follow dataset group order")
        if not 0 <= self.anchor_group_index < len(outcomes):
            raise ValueError("anchor_group_index is outside the branch")
        if outcomes[self.anchor_group_index].status is not MultiQFitStatus.SUCCESS:
            raise ValueError("the anchor outcome must retain a successful FitResult")
        if not isinstance(self.status, MultiQExecutionStatus):
            raise ValueError("status must be a MultiQExecutionStatus")
        if self.status is MultiQExecutionStatus.COMPLETED and any(
            item.status is MultiQFitStatus.NOT_RUN for item in outcomes
        ):
            raise ValueError("a completed branch cannot contain NOT_RUN targets")
        object.__setattr__(self, "outcomes", outcomes)

    def outcome(self, group_index: int) -> MultiQFitOutcome:
        """Return one outcome by ordered group index."""

        return self.outcomes[group_index]


def _blocked_diagnostic(
    group_index: int,
    message: str,
) -> ManualFitReadinessDiagnostic:
    return ManualFitReadinessDiagnostic(
        code=ManualFitDiagnosticCode.INVALID_PARAMETER_CONFIGURATION,
        severity=DiagnosticSeverity.ERROR,
        message=message,
        group_index=group_index,
    )


def _center_mode(component: LorentzianComponent) -> tuple[str, str | None]:
    if component.center_group is not None:
        return ("group", component.center_group)
    if component.center is not None:
        return ("independent", None)
    return ("legacy", None)


def _topology(model: SpectralModelDefinition) -> tuple[object, ...]:
    return (
        model.elastic_area is not None,
        model.elastic_center_group,
        model.energy_shift is not None,
        model.background,
        tuple(
            (component.identity, _center_mode(component))
            for component in model.lorentzians
        ),
        tuple(group.group_id for group in model.center_groups),
        tuple((group.group_id, group.members) for group in model.parameter_ties),
        model.parameter_references(),
    )


def _legacy_center_reference(model: SpectralModelDefinition) -> ParameterReference:
    if model.elastic_area is not None and model.elastic_center_group is None:
        return ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER)
    for component in model.lorentzians:
        if component.center is None and component.center_group is None:
            return ParameterReference(component.identity, ParameterFamily.CENTER)
    raise ValueError("model has no legacy shared-center reference")


def _center_group_reference(
    model: SpectralModelDefinition,
    group_id: str,
) -> ParameterReference:
    if model.elastic_area is not None and model.elastic_center_group == group_id:
        return ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER)
    for component in model.lorentzians:
        if component.center_group == group_id:
            return ParameterReference(component.identity, ParameterFamily.CENTER)
    raise ValueError(f"center group is not referenced: {group_id}")


def _seeded_parameter(
    configuration: ParameterConfiguration,
    value: float,
) -> ParameterConfiguration:
    if not configuration.free:
        return configuration
    if not np.isfinite(value):
        raise ValueError("a predecessor fitted parameter value is nonfinite")
    # Target-local bounds and fixed/free state remain authoritative.  A source
    # estimate outside them is reduced only to a legal execution-time start;
    # the single-Q core performs its usual strict-interior adjustment.
    initial = min(max(value, configuration.lower_bound), configuration.upper_bound)
    return ParameterConfiguration(
        initial_value=initial,
        lower_bound=configuration.lower_bound,
        upper_bound=configuration.upper_bound,
        free=configuration.free,
    )


def _seed_model_from_result(
    target: SpectralModelDefinition,
    predecessor: FitResult,
) -> SpectralModelDefinition:
    def seeded(reference: ParameterReference) -> ParameterConfiguration:
        configuration = target.parameter_configuration(reference)
        estimate = predecessor.parameter_by_reference(reference)
        return _seeded_parameter(configuration, estimate.value)

    energy_shift = None
    if target.energy_shift is not None:
        energy_shift = seeded(_legacy_center_reference(target))
    elastic_area = None
    if target.elastic_area is not None:
        elastic_area = seeded(
            ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA)
        )
    lorentzians = tuple(
        LorentzianComponent(
            area=seeded(ParameterReference(component.identity, ParameterFamily.AREA)),
            fwhm=seeded(ParameterReference(component.identity, ParameterFamily.FWHM)),
            center=(
                seeded(ParameterReference(component.identity, ParameterFamily.CENTER))
                if component.center is not None
                else None
            ),
            center_group=component.center_group,
            identity=component.identity,
        )
        for component in target.lorentzians
    )
    b0 = (
        seeded(ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.OFFSET))
        if target.b0 is not None
        else None
    )
    b1 = (
        seeded(ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.SLOPE))
        if target.b1 is not None
        else None
    )
    center_groups = tuple(
        CenterGroup(
            group_id=group.group_id,
            parameter=seeded(_center_group_reference(target, group.group_id)),
        )
        for group in target.center_groups
    )
    parameter_ties = tuple(
        ParameterTieGroup(
            group_id=group.group_id,
            members=group.members,
            parameter=seeded(group.members[0]),
        )
        for group in target.parameter_ties
    )
    return SpectralModelDefinition(
        energy_shift=energy_shift,
        elastic_area=elastic_area,
        lorentzians=lorentzians,
        background=target.background,
        b0=b0,
        b1=b1,
        center_groups=center_groups,
        elastic_center_group=target.elastic_center_group,
        parameter_ties=parameter_ties,
    )


def _usable_result(
    result: FitResult,
    *,
    prepared_resolution: PreparedResolution,
    selection: FittingSelection,
    group_index: int,
    branch_topology: tuple[object, ...],
) -> bool:
    expected_references = tuple(result.configuration.parameter_references())
    result_references = tuple(
        reference
        for parameter in result.parameters
        for reference in parameter.references
    )
    binding = result.context_binding
    return (
        binding is not None
        and binding.prepared_resolution is prepared_resolution
        and binding.selection is selection
        and binding.group_index == group_index
        and result.diagnostics.optimizer_success
        and result.provenance.group_index == group_index
        and _topology(result.configuration) == branch_topology
        and all(np.isfinite(parameter.value) for parameter in result.parameters)
        and len(result_references) == len(set(result_references))
        and set(result_references) == set(expected_references)
    )


def _physical_side_indices(
    prepared_resolution: PreparedResolution,
    anchor_group_index: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    q_bins = prepared_resolution.sample_dataset.q_bins
    if q_bins is None:
        raise ValueError("Multi-Q execution requires assigned representative Q values")
    q_values = q_bins.q_values
    if q_values.size != len(prepared_resolution.spectra):
        raise ValueError("representative Q-value count must match prepared groups")
    if np.unique(q_values).size != q_values.size:
        raise ValueError(
            "Multi-Q execution requires unique representative Q values to define "
            "physical lower-Q and higher-Q traversal"
        )
    anchor_q = float(q_values[anchor_group_index])
    lower = tuple(
        sorted(
            (
                index
                for index, q_value in enumerate(q_values)
                if float(q_value) < anchor_q
            ),
            key=lambda index: float(q_values[index]),
            reverse=True,
        )
    )
    higher = tuple(
        sorted(
            (
                index
                for index, q_value in enumerate(q_values)
                if float(q_value) > anchor_q
            ),
            key=lambda index: float(q_values[index]),
        )
    )
    if len(lower) + len(higher) != q_values.size - 1:
        raise ValueError("representative Q values do not define two physical sides")
    return lower, higher


def execute_multi_q_branch(
    prepared_resolution: PreparedResolution,
    selection: FittingSelection,
    configurations_by_group: Sequence[SpectralModelDefinition | None],
    *,
    anchor_group_index: int,
    anchor_fit: FitResult,
    cancel_requested: Callable[[], bool] | None = None,
    max_nfev: int = 2500,
) -> MultiQBranchResult:
    """Fit one fixed topology independently from a retained anchor outward.

    The lower and higher sides maintain independent most-recent-success seed
    chains.  Per-Q configurations already contain target-local bounds, free
    state, fitting selection, and scientific constraints; only initial values
    are replaced from the predecessor result.
    """

    if not isinstance(prepared_resolution, PreparedResolution):
        raise ValueError("prepared_resolution must be a PreparedResolution")
    if not isinstance(selection, FittingSelection):
        raise ValueError("selection must be a FittingSelection")
    if selection.dataset is not prepared_resolution.sample_dataset:
        raise ValueError("selection must reference the prepared sample dataset")
    if isinstance(anchor_group_index, bool) or not isinstance(anchor_group_index, int):
        raise ValueError("anchor_group_index must be an integer")
    group_count = len(prepared_resolution.spectra)
    if not 0 <= anchor_group_index < group_count:
        raise ValueError("anchor_group_index is outside the prepared resolution")
    configurations = tuple(configurations_by_group)
    if len(configurations) != group_count:
        raise ValueError("configuration count must match prepared Q-group count")
    if not isinstance(anchor_fit, FitResult):
        raise ValueError("anchor_fit must be a FitResult")
    branch_topology = _topology(anchor_fit.configuration)
    if not _usable_result(
        anchor_fit,
        prepared_resolution=prepared_resolution,
        selection=selection,
        group_index=anchor_group_index,
        branch_topology=branch_topology,
    ):
        raise ValueError(
            "anchor_fit must be a usable fit from the active anchor scientific context"
        )
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    lower_indices, higher_indices = _physical_side_indices(
        prepared_resolution,
        anchor_group_index,
    )

    outcomes = [
        MultiQFitOutcome(index, MultiQFitStatus.NOT_RUN) for index in range(group_count)
    ]
    outcomes[anchor_group_index] = MultiQFitOutcome(
        anchor_group_index,
        MultiQFitStatus.SUCCESS,
        fit_result=anchor_fit,
    )
    cancelled = False

    def run_side(indices: Sequence[int]) -> None:
        nonlocal cancelled
        predecessor = anchor_fit
        predecessor_index = anchor_group_index
        for group_index in indices:
            if cancel_requested is not None and cancel_requested():
                cancelled = True
                return
            target = configurations[group_index]
            if target is None:
                diagnostic = _blocked_diagnostic(
                    group_index,
                    "target Q has no resolved spectral-model configuration",
                )
                outcomes[group_index] = MultiQFitOutcome(
                    group_index,
                    MultiQFitStatus.BLOCKED,
                    diagnostics=(diagnostic,),
                )
                continue
            if _topology(target) != branch_topology:
                diagnostic = _blocked_diagnostic(
                    group_index,
                    "target Q model topology does not match the anchor branch",
                )
                outcomes[group_index] = MultiQFitOutcome(
                    group_index,
                    MultiQFitStatus.BLOCKED,
                    diagnostics=(diagnostic,),
                )
                continue
            try:
                seeded = _seed_model_from_result(target, predecessor)
            except (KeyError, ValueError) as error:
                diagnostic = _blocked_diagnostic(group_index, str(error))
                outcomes[group_index] = MultiQFitOutcome(
                    group_index,
                    MultiQFitStatus.BLOCKED,
                    diagnostics=(diagnostic,),
                )
                continue
            readiness = manual_fit_readiness(
                prepared_resolution,
                selection,
                group_index,
                seeded,
            )
            if not readiness.runnable:
                outcomes[group_index] = MultiQFitOutcome(
                    group_index,
                    MultiQFitStatus.BLOCKED,
                    seed_group_index=predecessor_index,
                    diagnostics=readiness.diagnostics,
                )
                continue
            try:
                result = fit_single_q(
                    prepared_resolution,
                    selection,
                    group_index,
                    seeded,
                    max_nfev=max_nfev,
                )
            except (RuntimeError, ValueError) as error:
                outcomes[group_index] = MultiQFitOutcome(
                    group_index,
                    MultiQFitStatus.FAILED,
                    seed_group_index=predecessor_index,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
                continue
            if not _usable_result(
                result,
                prepared_resolution=prepared_resolution,
                selection=selection,
                group_index=group_index,
                branch_topology=branch_topology,
            ):
                outcomes[group_index] = MultiQFitOutcome(
                    group_index,
                    MultiQFitStatus.FAILED,
                    fit_result=result,
                    seed_group_index=predecessor_index,
                    error_type="UnusableFitResult",
                    error_message="single-Q fitting did not produce a usable result",
                )
                continue
            outcomes[group_index] = MultiQFitOutcome(
                group_index,
                MultiQFitStatus.SUCCESS,
                fit_result=result,
                seed_group_index=predecessor_index,
            )
            predecessor = result
            predecessor_index = group_index

    run_side(lower_indices)
    if not cancelled:
        run_side(higher_indices)
    return MultiQBranchResult(
        anchor_group_index=anchor_group_index,
        status=(
            MultiQExecutionStatus.CANCELLED
            if cancelled
            else MultiQExecutionStatus.COMPLETED
        ),
        outcomes=tuple(outcomes),
    )
