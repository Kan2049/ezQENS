"""Issued, immutable per-Q Results and their fit-time working states."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, replace
from enum import StrEnum

import numpy as np

from ezqens.batch import MultiQBranchResult, MultiQFitOutcome, MultiQFitStatus
from ezqens.fitting import (
    ComponentIdentity,
    FitResult,
    ParameterReference,
    StandardModelCandidate,
)

from .manual_state import ManualModelState
from .project import ManualFitContext, measured_spectrum_points_exactly_unchanged


def fit_evidence_is_current(
    fit: FitResult, context: ManualFitContext, group_index: int
) -> bool:
    """Compare only this Q's immutable fit-time scientific inputs by value.

    FitContextBinding retains the source spectra, selection and prepared kernel;
    no mutable application state or whole-context identity participates.
    """

    binding = fit.context_binding
    if (
        binding is None
        or binding.group_index != group_index
        or fit.provenance.group_index != group_index
        or not 0 <= group_index < len(context.sample.dataset.spectra)
        or group_index >= len(binding.prepared_resolution.spectra)
        or group_index >= len(context.prepared_resolution.spectra)
        or group_index >= len(context.selection.ranges)
    ):
        return False
    before = binding.prepared_resolution
    after = context.prepared_resolution
    measured_before = before.sample_dataset.spectra[group_index]
    measured_after = context.sample.dataset.spectra[group_index]
    old_q = before.sample_dataset.q_bins
    new_q = context.sample.dataset.q_bins
    if (
        old_q is None
        or new_q is None
        or old_q.q_values[group_index] != new_q.q_values[group_index]
        or fit.provenance.q_value != after.q_value(group_index)
        or measured_before.energy_unit != measured_after.energy_unit
        or not measured_spectrum_points_exactly_unchanged(
            measured_before, measured_after
        )
        or binding.selection.ranges[group_index]
        != context.selection.ranges[group_index]
        or not np.array_equal(
            binding.selection.excluded_mask(group_index),
            context.selection.excluded_mask(group_index),
        )
    ):
        return False
    old_kernel = before.spectra[group_index]
    new_kernel = after.spectra[group_index]
    return (
        measured_spectrum_points_exactly_unchanged(
            old_kernel.source_spectrum, new_kernel.source_spectrum
        )
        and old_kernel.source_spectrum.energy_unit
        == new_kernel.source_spectrum.energy_unit
        and before.acceptance_provenance(group_index)
        == after.acceptance_provenance(group_index)
        and old_kernel.diagnostics == new_kernel.diagnostics
        and np.array_equal(old_kernel.accepted_mask, new_kernel.accepted_mask)
        and np.array_equal(old_kernel.padding.auto_mask, new_kernel.padding.auto_mask)
        and np.array_equal(
            old_kernel.padding.review_mask, new_kernel.padding.review_mask
        )
        and np.array_equal(old_kernel.energy, new_kernel.energy)
        and np.array_equal(
            old_kernel.normalized_intensity, new_kernel.normalized_intensity
        )
        and np.array_equal(
            old_kernel.normalized_uncertainty,
            new_kernel.normalized_uncertainty,
            equal_nan=True,
        )
    )


class ResultOrigin(StrEnum):
    MANUAL_FIT = "manual_fit"
    MANUAL_FIT_ALL_Q = "manual_fit_all_q"
    AUTO_FIT_SINGLE_Q = "auto_fit_single_q"
    AUTO_FIT_BRANCH = "auto_fit_branch"


class IdentityMark(StrEnum):
    PAIRED_BY_WIDTH = "paired_by_width"
    USER_SWAP = "user_swap"


@dataclass(frozen=True, slots=True)
class QResult:
    """One Q's authoritative outcome and editable state at fit time."""

    outcome: MultiQFitOutcome
    working_state: ManualModelState | None = None
    origin: ResultOrigin | None = None
    candidate: StandardModelCandidate | None = None
    identity_marks: tuple[IdentityMark, ...] = ()
    optimizer_status: int | None = None
    optimizer_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, MultiQFitOutcome):
            raise ValueError("Q Result requires a MultiQFitOutcome")
        fitted = self.outcome.status is MultiQFitStatus.SUCCESS
        failed_fit = None if fitted else self.outcome.fit_result
        if failed_fit is not None:
            diagnostics = failed_fit.diagnostics
            if (
                self.optimizer_status is not None
                and self.optimizer_status != diagnostics.optimizer_status
            ) or (
                self.optimizer_message is not None
                and self.optimizer_message != diagnostics.optimizer_message
            ):
                raise ValueError("optimizer summary must match the failed execution")
            object.__setattr__(self, "optimizer_status", diagnostics.optimizer_status)
            object.__setattr__(self, "optimizer_message", diagnostics.optimizer_message)
            object.__setattr__(self, "outcome", replace(self.outcome, fit_result=None))
        if (self.optimizer_status is None) != (self.optimizer_message is None):
            raise ValueError("optimizer status and message must be supplied together")
        if self.optimizer_status is not None and (
            self.outcome.status is not MultiQFitStatus.FAILED
            or isinstance(self.optimizer_status, bool)
            or not isinstance(self.optimizer_status, int)
            or not isinstance(self.optimizer_message, str)
        ):
            raise ValueError("optimizer failure summary requires a FAILED Q")
        if fitted != (self.working_state is not None):
            raise ValueError("working state must exist exactly at fitted Qs")
        if fitted and self.origin is None:
            raise ValueError("fitted Q requires its execution origin")
        if self.working_state is not None and not isinstance(
            self.working_state, ManualModelState
        ):
            raise ValueError("working state must be a ManualModelState")
        if self.origin is not None and not isinstance(self.origin, ResultOrigin):
            raise ValueError("origin must be a ResultOrigin")
        auto = self.origin in (
            ResultOrigin.AUTO_FIT_SINGLE_Q,
            ResultOrigin.AUTO_FIT_BRANCH,
        )
        if self.candidate is not None and not auto:
            raise ValueError("candidate is only valid for AutoFit provenance")
        if auto and not isinstance(self.candidate, StandardModelCandidate):
            raise ValueError("AutoFit Q requires its typed candidate")
        marks = tuple(self.identity_marks)
        if not fitted and marks:
            raise ValueError("unfitted Q cannot carry identity marks")
        if any(not isinstance(mark, IdentityMark) for mark in marks):
            raise ValueError("identity marks must be typed")
        object.__setattr__(self, "identity_marks", marks)

    @property
    def status(self) -> MultiQFitStatus:
        return self.outcome.status

    @property
    def fit_result(self) -> FitResult | None:
        """Return fitted evidence; non-fitted Qs contain no FitResult."""

        return self.outcome.fit_result

    def _matches_execution_outcome(self, outcome: MultiQFitOutcome) -> bool:
        """Validate the snapshot projection without retaining failed fit objects."""

        fit = outcome.fit_result
        if outcome.status is MultiQFitStatus.SUCCESS or fit is None:
            return (
                self.outcome is outcome
                and self.optimizer_status is None
                and self.optimizer_message is None
            )
        return (
            self.outcome == replace(outcome, fit_result=None)
            and self.optimizer_status == fit.diagnostics.optimizer_status
            and self.optimizer_message == fit.diagnostics.optimizer_message
        )


@dataclass(frozen=True, slots=True)
class UnifiedResult:
    """One Sample's ordered per-Q Result, independent of how it was produced."""

    points: tuple[QResult, ...]
    sample_identity: tuple[str, str]

    def __post_init__(self) -> None:
        points = tuple(self.points)
        if (
            not isinstance(self.sample_identity, tuple)
            or len(self.sample_identity) != 2
            or any(
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
                for value in self.sample_identity
            )
        ):
            raise ValueError("Result requires its stable Project/Sample identity")
        if not points or tuple(point.outcome.group_index for point in points) != tuple(
            range(len(points))
        ):
            raise ValueError("Result requires one ordered record for every Q")
        fitted = [point.fit_result for point in points if point.fit_result is not None]
        if not fitted:
            raise ValueError("Result requires at least one fitted Q")
        for fit in fitted:
            binding = fit.context_binding
            if binding is None or len(binding.prepared_resolution.spectra) != len(
                points
            ):
                raise ValueError("Result requires a bound Sample Q topology")
        object.__setattr__(self, "points", points)

    def point(self, group_index: int) -> QResult:
        return self.points[group_index]

    def outcome(self, group_index: int) -> MultiQFitOutcome:
        return self.points[group_index].outcome

    @property
    def fitted_count(self) -> int:
        return sum(point.status is MultiQFitStatus.SUCCESS for point in self.points)

    @property
    def total_count(self) -> int:
        return len(self.points)

    @property
    def status_counts(self) -> tuple[tuple[MultiQFitStatus, int], ...]:
        return tuple(
            (status, sum(point.status is status for point in self.points))
            for status in MultiQFitStatus
        )

    @property
    def working_states(self) -> tuple[ManualModelState | None, ...]:
        return tuple(point.working_state for point in self.points)


_SNAPSHOT_ISSUER = object()


@dataclass(frozen=True, slots=True)
class ResultSnapshot:
    """Workflow-issued immutable unified Result; external construction is invalid."""

    result: UnifiedResult
    _issuer: InitVar[object | None] = None
    _issued: object = field(init=False, repr=False, compare=False)

    def __post_init__(self, _issuer: object | None) -> None:
        if _issuer is not _SNAPSHOT_ISSUER:
            raise ValueError("Result snapshots must be issued by execution workflow")
        if not isinstance(self.result, UnifiedResult):
            raise ValueError("snapshot requires a unified per-Q Result")
        object.__setattr__(self, "_issued", _SNAPSHOT_ISSUER)

    @property
    def working_states(self) -> tuple[ManualModelState | None, ...]:
        return self.result.working_states

    @property
    def origin(self) -> ResultOrigin | None:
        origins = {
            point.origin for point in self.result.points if point.origin is not None
        }
        return next(iter(origins)) if len(origins) == 1 else None

    @property
    def candidate(self) -> StandardModelCandidate | None:
        candidates = {
            point.candidate
            for point in self.result.points
            if point.candidate is not None
        }
        return next(iter(candidates)) if len(candidates) == 1 else None


def _issue_unified_result(result: UnifiedResult) -> ResultSnapshot:
    return ResultSnapshot(result, _issuer=_SNAPSHOT_ISSUER)


def _issue_result_snapshot(
    result: FitResult | MultiQBranchResult,
    working_states: tuple[ManualModelState | None, ...],
    origin: ResultOrigin,
    candidate: StandardModelCandidate | None = None,
    *,
    sample_identity: tuple[str, str],
) -> ResultSnapshot:
    """Convert one execution outcome to the single per-Q Result representation."""

    states = tuple(working_states)
    if isinstance(result, FitResult):
        binding = result.context_binding
        if binding is None or len(states) != len(binding.prepared_resolution.spectra):
            raise ValueError("snapshot states must match the fitted dataset")
        if not result.diagnostics.optimizer_success or any(
            (state is not None) is not (index == binding.group_index)
            for index, state in enumerate(states)
        ):
            raise ValueError(
                "single-Q snapshot must contain only its successful Q state"
            )
        outcomes = tuple(
            MultiQFitOutcome(index, MultiQFitStatus.SUCCESS, result)
            if index == binding.group_index
            else MultiQFitOutcome(index, MultiQFitStatus.NOT_RUN)
            for index in range(len(states))
        )
    elif isinstance(result, MultiQBranchResult):
        outcomes = result.outcomes
        if len(states) != len(outcomes):
            raise ValueError("snapshot states must match branch Q topology")
        if any(
            (state is not None) is not (outcome.status is MultiQFitStatus.SUCCESS)
            for state, outcome in zip(states, outcomes, strict=True)
        ):
            raise ValueError("snapshot states must exist exactly at successful Qs")
    else:
        raise ValueError("snapshot requires a fit or branch execution output")
    points = tuple(
        QResult(
            outcome,
            state,
            origin
            if isinstance(result, MultiQBranchResult)
            or outcome.status is MultiQFitStatus.SUCCESS
            else None,
            candidate
            if isinstance(result, MultiQBranchResult)
            or outcome.status is MultiQFitStatus.SUCCESS
            else None,
        )
        for outcome, state in zip(outcomes, states, strict=True)
    )
    return _issue_unified_result(UnifiedResult(points, sample_identity))


def _replace_result_point(snapshot: ResultSnapshot, point: QResult) -> ResultSnapshot:
    if not _is_issued_result_snapshot(snapshot):
        raise ValueError("Current must be an issued Result snapshot")
    index = point.outcome.group_index
    points = list(snapshot.result.points)
    points[index] = point
    return _issue_unified_result(replace(snapshot.result, points=tuple(points)))


def _relabel_manual_state(
    state: ManualModelState,
    identities: dict[ComponentIdentity, ComponentIdentity],
) -> ManualModelState:
    """Keep each component's editable state and ties with its exact identity."""

    def reference(item: ParameterReference) -> ParameterReference:
        replacement = identities.get(item.component)
        return item if replacement is None else replace(item, component=replacement)

    return replace(
        state,
        lorentzians=tuple(
            replace(item, identity=identities.get(item.identity, item.identity))
            for item in state.lorentzians
        ),
        parameter_ties=tuple(
            replace(tie, members=tuple(reference(item) for item in tie.members))
            for tie in state.parameter_ties
        ),
    )


def _mark_result_point(
    snapshot: ResultSnapshot, group_index: int, mark: IdentityMark
) -> ResultSnapshot:
    point = snapshot.result.point(group_index)
    return _replace_result_point(
        snapshot,
        replace(point, identity_marks=(*point.identity_marks, mark)),
    )


def _is_issued_result_snapshot(value: object) -> bool:
    return (
        isinstance(value, ResultSnapshot)
        and getattr(value, "_issued", None) is _SNAPSHOT_ISSUER
    )
