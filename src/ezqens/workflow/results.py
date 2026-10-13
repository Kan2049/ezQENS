"""Immutable in-memory Result and execution-time Manual working-state snapshot."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import StrEnum

from ezqens.batch import MultiQBranchResult, MultiQFitStatus
from ezqens.fitting import FitResult, StandardModelCandidate

from .manual_state import ManualModelState


class ResultOrigin(StrEnum):
    MANUAL_FIT = "manual_fit"
    MANUAL_FIT_ALL_Q = "manual_fit_all_q"
    AUTO_FIT_SINGLE_Q = "auto_fit_single_q"
    AUTO_FIT_BRANCH = "auto_fit_branch"


_SNAPSHOT_ISSUER = object()


@dataclass(frozen=True, slots=True)
class ResultSnapshot:
    """A concrete Result and the exact immutable per-Q state at its creation."""

    result: FitResult | MultiQBranchResult
    working_states: tuple[ManualModelState | None, ...]
    origin: ResultOrigin
    candidate: StandardModelCandidate | None = None
    _issuer: InitVar[object | None] = None
    _issued: object = field(init=False, repr=False, compare=False)

    def __post_init__(self, _issuer: object | None) -> None:
        if _issuer is not _SNAPSHOT_ISSUER:
            raise ValueError("Result snapshots must be issued by execution workflow")
        if not isinstance(self.result, (FitResult, MultiQBranchResult)):
            raise ValueError(
                "snapshot result must be a FitResult or MultiQBranchResult"
            )
        states = tuple(self.working_states)
        if not states or any(
            state is not None and not isinstance(state, ManualModelState)
            for state in states
        ):
            raise ValueError("snapshot requires one optional Manual state per Q")
        if not isinstance(self.origin, ResultOrigin):
            raise ValueError("snapshot origin must be a ResultOrigin")
        branch_origin = self.origin in (
            ResultOrigin.MANUAL_FIT_ALL_Q,
            ResultOrigin.AUTO_FIT_BRANCH,
        )
        if branch_origin != isinstance(self.result, MultiQBranchResult):
            raise ValueError("snapshot origin must match Single-Q or Multi-Q Result")
        if isinstance(self.result, MultiQBranchResult):
            if len(states) != len(self.result.outcomes):
                raise ValueError("snapshot states must match branch Q topology")
            if any(
                (state is not None) is not (outcome.status is MultiQFitStatus.SUCCESS)
                for state, outcome in zip(states, self.result.outcomes, strict=True)
            ):
                raise ValueError("snapshot states must exist exactly at successful Qs")
        else:
            binding = self.result.context_binding
            if binding is None or len(states) != len(
                binding.prepared_resolution.spectra
            ):
                raise ValueError("snapshot states must match the fitted dataset")
            if not self.result.diagnostics.optimizer_success or any(
                (state is not None) is not (index == binding.group_index)
                for index, state in enumerate(states)
            ):
                raise ValueError(
                    "single-Q snapshot must contain only its successful Q state"
                )
        auto = self.origin in (
            ResultOrigin.AUTO_FIT_SINGLE_Q,
            ResultOrigin.AUTO_FIT_BRANCH,
        )
        if auto != (self.candidate is not None):
            raise ValueError(
                "AutoFit snapshots require a candidate; Manual snapshots do not"
            )
        if self.candidate is not None and not isinstance(
            self.candidate, StandardModelCandidate
        ):
            raise ValueError("snapshot candidate must be a StandardModelCandidate")
        object.__setattr__(self, "working_states", states)
        object.__setattr__(self, "_issued", _SNAPSHOT_ISSUER)


def _issue_result_snapshot(
    result: FitResult | MultiQBranchResult,
    working_states: tuple[ManualModelState | None, ...],
    origin: ResultOrigin,
    candidate: StandardModelCandidate | None = None,
) -> ResultSnapshot:
    """Issue the exact state captured by a Result-producing workflow path."""

    return ResultSnapshot(
        result,
        working_states,
        origin,
        candidate,
        _issuer=_SNAPSHOT_ISSUER,
    )


def _is_issued_result_snapshot(value: object) -> bool:
    return (
        isinstance(value, ResultSnapshot)
        and getattr(value, "_issued", None) is _SNAPSHOT_ISSUER
    )
