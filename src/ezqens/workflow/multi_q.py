"""GUI-independent M6 Method application and Multi-Q workflow lifecycle."""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Sequence
from dataclasses import InitVar, dataclass, field, replace
from enum import StrEnum
from uuid import uuid4

import numpy as np

from ezqens.batch import (
    MultiQBranchResult,
    MultiQExecutionStatus,
    MultiQFitOutcome,
    MultiQFitStatus,
    execute_multi_q_branch,
)
from ezqens.derived import DerivedUnifiedQENSResult, derive_unified_qens
from ezqens.domain import QBins
from ezqens.fitting import (
    ELASTIC_COMPONENT,
    BackgroundModel,
    CandidateFitResult,
    ComponentFamily,
    ComponentIdentity,
    FitContextBinding,
    FitResult,
    ManualParameterIntent,
    ParameterFamily,
    ParameterReference,
    SpectralModelDefinition,
    StandardModelCandidate,
    convert_standard_auto_fit_result_to_manual,
    relabel_fit_components,
)
from ezqens.preprocessing import EdgePaddingDetectionResult, FittingSelection
from ezqens.resolution import PreparedResolution

from .manual import (
    ManualFitDraft,
    SingleQAutoFitOutcome,
    _adopt_manual_fit_values,
    _manual_state_from_fitted_model,
    _manual_state_from_standard_auto_state,
    _single_q_auto_fit_context_unchanged,
    run_single_q_auto_fit,
)
from .manual_state import (
    ManualCenterGroupState,
    ManualLorentzianState,
    ManualModelMaterialization,
    ManualModelState,
    ManualParameterTieState,
    materialize_manual_model,
    replace_parameter_intent,
)
from .project import (
    ManualFitContext,
    WorkflowProject,
    resolve_manual_fit_context,
)
from .results import (
    IdentityMark,
    ResultOrigin,
    ResultSnapshot,
    UnifiedResult,
    _is_issued_result_snapshot,
    _issue_result_snapshot,
    _relabel_manual_state,
    _replace_result_point,
    fit_evidence_is_current,
)


class MethodTransferCategory(StrEnum):
    """Optional reusable Method categories independent of branch composition."""

    USER_BOUNDS = "user_bounds"
    FREE_FIXED = "free_fixed"
    PARAMETER_RELATIONSHIPS = "parameter_relationships"
    RESOLUTION = "resolution"
    Q_BINS = "q_bins"
    FITTING_SELECTION = "fitting_selection"


class CenterTieKind(StrEnum):
    """Runtime identities for transferable center-sharing topology."""

    LEGACY_SHARED_CENTER = "legacy_shared_center"
    CENTER_GROUP = "center_group"
    PARAMETER_TIE = "parameter_tie"


@dataclass(frozen=True, slots=True)
class CenterTieIdentity:
    """Atomic identity of one active center-sharing relationship."""

    kind: CenterTieKind
    group_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CenterTieKind):
            raise ValueError("center tie kind must be a CenterTieKind")
        if self.kind is CenterTieKind.LEGACY_SHARED_CENTER:
            if self.group_id is not None:
                raise ValueError("legacy shared center must not have a group_id")
            return
        if (
            not isinstance(self.group_id, str)
            or not self.group_id.strip()
            or self.group_id != self.group_id.strip()
        ):
            raise ValueError("center tie group_id must be a canonical nonempty string")


@dataclass(frozen=True, slots=True)
class ActiveBoundsConstraint:
    """One currently enabled Manual user-bounds constraint."""

    reference: ParameterReference
    lower: float | None
    upper: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.reference, ParameterReference):
            raise ValueError("bounds reference must be a ParameterReference")
        if self.lower is None and self.upper is None:
            raise ValueError("active bounds require at least one supplied limit")
        if any(
            value is not None and not np.isfinite(float(value))
            for value in (self.lower, self.upper)
        ):
            raise ValueError("active bounds limits must be finite when supplied")


@dataclass(frozen=True, slots=True)
class ActiveFixedConstraint:
    """One currently fixed Manual parameter and its authoritative value."""

    reference: ParameterReference
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.reference, ParameterReference):
            raise ValueError("fixed reference must be a ParameterReference")
        if not np.isfinite(float(self.value)):
            raise ValueError("fixed value must be finite")


@dataclass(frozen=True, slots=True)
class ActiveCenterTieConstraint:
    """One complete active center-sharing relationship."""

    identity: CenterTieIdentity
    members: tuple[ParameterReference, ...]

    def __post_init__(self) -> None:
        members = tuple(self.members)
        if not isinstance(self.identity, CenterTieIdentity):
            raise ValueError("center tie identity must be a CenterTieIdentity")
        if not members:
            raise ValueError("active center tie requires at least one member")
        if any(
            not isinstance(member, ParameterReference)
            or member.family is not ParameterFamily.CENTER
            or member.component.family
            not in (ComponentFamily.ELASTIC, ComponentFamily.LORENTZIAN)
            for member in members
        ):
            raise ValueError(
                "center tie members must be Elastic/Lorentzian Center references"
            )
        if len(set(members)) != len(members):
            raise ValueError("center tie members must be unique")
        object.__setattr__(self, "members", members)


@dataclass(frozen=True, slots=True)
class ActiveTransferableConstraints:
    """Active Manual constraints that S5 may offer for selective transfer."""

    bounds: tuple[ActiveBoundsConstraint, ...] = ()
    fixed: tuple[ActiveFixedConstraint, ...] = ()
    center_ties: tuple[ActiveCenterTieConstraint, ...] = ()

    def __post_init__(self) -> None:
        bounds = tuple(self.bounds)
        fixed = tuple(self.fixed)
        center_ties = tuple(self.center_ties)
        if any(not isinstance(item, ActiveBoundsConstraint) for item in bounds):
            raise ValueError("bounds must contain ActiveBoundsConstraint values")
        if any(not isinstance(item, ActiveFixedConstraint) for item in fixed):
            raise ValueError("fixed must contain ActiveFixedConstraint values")
        if any(not isinstance(item, ActiveCenterTieConstraint) for item in center_ties):
            raise ValueError(
                "center_ties must contain ActiveCenterTieConstraint values"
            )
        if len({item.reference for item in bounds}) != len(bounds):
            raise ValueError("active bounds references must be unique")
        if len({item.reference for item in fixed}) != len(fixed):
            raise ValueError("active fixed references must be unique")
        if len({item.identity for item in center_ties}) != len(center_ties):
            raise ValueError("active center tie identities must be unique")
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "fixed", fixed)
        object.__setattr__(self, "center_ties", center_ties)


def active_transferable_constraints(
    model: ManualModelState,
) -> ActiveTransferableConstraints:
    """Describe only constraints currently active in one Manual model."""

    references = model.parameter_references()
    bounds: list[ActiveBoundsConstraint] = []
    fixed: list[ActiveFixedConstraint] = []
    for reference in references:
        intent = model.parameter_intent(reference)
        if intent.user_bounds_enabled and (
            intent.user_lower_limit is not None or intent.user_upper_limit is not None
        ):
            bounds.append(
                ActiveBoundsConstraint(
                    reference,
                    intent.user_lower_limit,
                    intent.user_upper_limit,
                )
            )
        if not intent.free:
            fixed.append(ActiveFixedConstraint(reference, intent.current_value))

    center_ties: list[ActiveCenterTieConstraint] = []
    legacy_candidates: list[ParameterReference] = []
    if model.elastic_area is not None and model.elastic_center_group is None:
        legacy_candidates.append(
            ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER)
        )
    legacy_candidates.extend(
        ParameterReference(component.identity, ParameterFamily.CENTER)
        for component in model.lorentzians
        if component.center is None and component.center_group is None
    )
    legacy_members = tuple(
        reference for reference in legacy_candidates if model.tie_for(reference) is None
    )
    if len(legacy_members) >= 2:
        center_ties.append(
            ActiveCenterTieConstraint(
                CenterTieIdentity(CenterTieKind.LEGACY_SHARED_CENTER),
                legacy_members,
            )
        )
    for center_group in model.center_groups:
        center_ties.append(
            ActiveCenterTieConstraint(
                CenterTieIdentity(CenterTieKind.CENTER_GROUP, center_group.group_id),
                model.center_group_members(center_group.group_id),
            )
        )
    for tie_group in model.parameter_ties:
        if tie_group.family is ParameterFamily.CENTER:
            center_ties.append(
                ActiveCenterTieConstraint(
                    CenterTieIdentity(
                        CenterTieKind.PARAMETER_TIE,
                        tie_group.group_id,
                    ),
                    tie_group.members,
                )
            )
    return ActiveTransferableConstraints(
        bounds=tuple(bounds),
        fixed=tuple(fixed),
        center_ties=tuple(center_ties),
    )


@dataclass(frozen=True, slots=True)
class MultiQConstraintTransferPolicy:
    """Effective S5 transfer decisions validated against one active model."""

    model: InitVar[ManualModelState]
    bounds: frozenset[ParameterReference] = field(default_factory=frozenset)
    fixed: frozenset[ParameterReference] = field(default_factory=frozenset)
    center_ties: frozenset[CenterTieIdentity] = field(default_factory=frozenset)

    def __post_init__(self, model: ManualModelState) -> None:
        if not isinstance(model, ManualModelState):
            raise ValueError("model must be a ManualModelState")
        active = active_transferable_constraints(model)
        bounds = frozenset(self.bounds)
        fixed = frozenset(self.fixed)
        center_ties = frozenset(self.center_ties)
        if any(not isinstance(item, ParameterReference) for item in bounds | fixed):
            raise ValueError(
                "bounds/fixed selections must be ParameterReference values"
            )
        if any(not isinstance(item, CenterTieIdentity) for item in center_ties):
            raise ValueError("center tie selections must be CenterTieIdentity values")
        active_bounds = {item.reference for item in active.bounds}
        active_fixed = {item.reference for item in active.fixed}
        active_ties = {item.identity for item in active.center_ties}
        if not bounds <= active_bounds:
            raise ValueError(
                "bounds selection contains an inactive or unknown reference"
            )
        if not fixed <= active_fixed:
            raise ValueError(
                "fixed selection contains an inactive or unknown reference"
            )
        if not center_ties <= active_ties:
            raise ValueError("center tie selection contains an inactive or unknown tie")
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "fixed", fixed)
        object.__setattr__(self, "center_ties", center_ties)


_BRANCH_ISSUER = object()


@dataclass(frozen=True, slots=True)
class AutoFitCandidateBranchResult:
    """One selected anchor candidate together with its independent Q branch."""

    anchor_evidence: CandidateFitResult
    branch_result: MultiQBranchResult
    evaluation: SingleQAutoFitOutcome | None = field(default=None, repr=False)
    working_states: tuple[ManualModelState | None, ...] = ()
    _issuer: InitVar[object | None] = None
    _issued: object = field(init=False, repr=False, compare=False)

    def __post_init__(self, _issuer: object | None) -> None:
        if _issuer is not _BRANCH_ISSUER:
            raise ValueError("AutoFit branch records must be issued by execution")
        if not isinstance(self.anchor_evidence, CandidateFitResult):
            raise ValueError("anchor_evidence must be a CandidateFitResult")
        if not self.anchor_evidence.success or self.anchor_evidence.fit is None:
            raise ValueError("anchor_evidence must contain a successful FitResult")
        if not isinstance(self.branch_result, MultiQBranchResult):
            raise ValueError("branch_result must be a MultiQBranchResult")
        retained_anchor = self.branch_result.outcome(
            self.branch_result.anchor_group_index
        ).fit_result
        original_fit = self.anchor_evidence.fit
        if retained_anchor is not original_fit and not (
            original_fit is not None
            and retained_anchor is not None
            and retained_anchor.configuration is original_fit.configuration
            and retained_anchor.evaluation is original_fit.evaluation
            and retained_anchor.parameters is original_fit.parameters
        ):
            raise ValueError(
                "the branch must retain the selected candidate's anchor FitResult"
            )
        states = tuple(self.working_states)
        if len(states) != len(self.branch_result.outcomes):
            raise ValueError("AutoFit branch working states must match Q topology")
        if any(
            item.status is MultiQFitStatus.SUCCESS and states[item.group_index] is None
            for item in self.branch_result.outcomes
        ):
            raise ValueError("successful AutoFit branch Q lost its working state")
        object.__setattr__(self, "working_states", states)
        object.__setattr__(self, "_issued", _BRANCH_ISSUER)

    @property
    def candidate(self) -> StandardModelCandidate:
        """Return the typed selected candidate identity."""

        return self.anchor_evidence.candidate


def _issue_auto_fit_candidate_branch_result(
    anchor_evidence: CandidateFitResult,
    branch_result: MultiQBranchResult,
    evaluation: SingleQAutoFitOutcome | None,
    working_states: tuple[ManualModelState | None, ...],
) -> AutoFitCandidateBranchResult:
    """Capture the states submitted for one selected execution branch."""

    return AutoFitCandidateBranchResult(
        anchor_evidence,
        branch_result,
        evaluation,
        working_states,
        _issuer=_BRANCH_ISSUER,
    )


def _is_issued_auto_fit_candidate_branch_result(value: object) -> bool:
    return (
        isinstance(value, AutoFitCandidateBranchResult)
        and getattr(value, "_issued", None) is _BRANCH_ISSUER
    )


@dataclass(frozen=True, slots=True)
class SelectedAutoFitProgressEvent:
    """One authoritative terminal Q outcome for a selected AutoFit branch."""

    candidate: StandardModelCandidate
    outcome: MultiQFitOutcome

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, StandardModelCandidate):
            raise ValueError("candidate must be a StandardModelCandidate")
        if not isinstance(self.outcome, MultiQFitOutcome):
            raise ValueError("outcome must be a MultiQFitOutcome")
        if self.outcome.status is MultiQFitStatus.NOT_RUN:
            raise ValueError("selected AutoFit progress cannot publish NOT_RUN")


@dataclass(frozen=True, slots=True)
class SelectedAutoFitMultiQResult:
    """Ordered independent branches for one selected-candidate operation."""

    status: MultiQExecutionStatus
    selected_candidates: tuple[StandardModelCandidate, ...]
    branches: tuple[AutoFitCandidateBranchResult, ...]

    def __post_init__(self) -> None:
        selected = tuple(self.selected_candidates)
        branches = tuple(self.branches)
        if not isinstance(self.status, MultiQExecutionStatus):
            raise ValueError("status must be a MultiQExecutionStatus")
        if not selected:
            raise ValueError("at least one AutoFit candidate must be selected")
        if any(not isinstance(item, StandardModelCandidate) for item in selected):
            raise ValueError(
                "selected candidates must be StandardModelCandidate values"
            )
        if len(set(selected)) != len(selected):
            raise ValueError("selected AutoFit candidates must be unique")
        if any(
            not _is_issued_auto_fit_candidate_branch_result(item) for item in branches
        ):
            raise ValueError(
                "branches must contain AutoFitCandidateBranchResult values"
            )
        if tuple(item.candidate for item in branches) != selected[: len(branches)]:
            raise ValueError("branch results must preserve caller selection order")
        if self.status is MultiQExecutionStatus.COMPLETED:
            if len(branches) != len(selected) or any(
                item.branch_result.status is not MultiQExecutionStatus.COMPLETED
                for item in branches
            ):
                raise ValueError(
                    "a completed operation requires every completed branch"
                )
        else:
            cancelled_positions = tuple(
                index
                for index, item in enumerate(branches)
                if item.branch_result.status is MultiQExecutionStatus.CANCELLED
            )
            if cancelled_positions and cancelled_positions != (len(branches) - 1,):
                raise ValueError("a cancelled branch must be the final retained branch")
            if not cancelled_positions and len(branches) == len(selected):
                raise ValueError(
                    "a cancelled operation must stop before a later selection or "
                    "retain a final cancelled branch"
                )
        object.__setattr__(self, "selected_candidates", selected)
        object.__setattr__(self, "branches", branches)


@dataclass(frozen=True, slots=True)
class MethodTransferOptions:
    """Complete effective or default ON/OFF state for Method application."""

    user_bounds: bool = False
    free_fixed: bool = False
    parameter_relationships: bool = False
    resolution: bool = False
    q_bins: bool = False
    fitting_selection: bool = False

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, bool)
            for value in (
                self.user_bounds,
                self.free_fixed,
                self.parameter_relationships,
                self.resolution,
                self.q_bins,
                self.fitting_selection,
            )
        ):
            raise ValueError("Method transfer options must be boolean")

    def enabled(self, category: MethodTransferCategory) -> bool:
        """Return one typed category's state."""

        if not isinstance(category, MethodTransferCategory):
            raise ValueError("category must be a MethodTransferCategory")
        return {
            MethodTransferCategory.USER_BOUNDS: self.user_bounds,
            MethodTransferCategory.FREE_FIXED: self.free_fixed,
            MethodTransferCategory.PARAMETER_RELATIONSHIPS: (
                self.parameter_relationships
            ),
            MethodTransferCategory.RESOLUTION: self.resolution,
            MethodTransferCategory.Q_BINS: self.q_bins,
            MethodTransferCategory.FITTING_SELECTION: self.fitting_selection,
        }[category]


@dataclass(frozen=True, slots=True)
class MethodTransferOverrides:
    """Per-application overrides; None retains the Method default."""

    user_bounds: bool | None = None
    free_fixed: bool | None = None
    parameter_relationships: bool | None = None
    resolution: bool | None = None
    q_bins: bool | None = None
    fitting_selection: bool | None = None

    def __post_init__(self) -> None:
        if any(
            value is not None and not isinstance(value, bool)
            for value in (
                self.user_bounds,
                self.free_fixed,
                self.parameter_relationships,
                self.resolution,
                self.q_bins,
                self.fitting_selection,
            )
        ):
            raise ValueError("Method transfer overrides must be boolean or None")

    def resolve(self, defaults: MethodTransferOptions) -> MethodTransferOptions:
        """Return complete effective transfer state."""

        return MethodTransferOptions(
            **{
                name: getattr(defaults, name)
                if getattr(self, name) is None
                else getattr(self, name)
                for name in (
                    "user_bounds",
                    "free_fixed",
                    "parameter_relationships",
                    "resolution",
                    "q_bins",
                    "fitting_selection",
                )
            }
        )


@dataclass(frozen=True, slots=True)
class MethodComposition:
    """Mandatory fixed branch composition with stable component identities."""

    elastic_present: bool
    lorentzian_identities: tuple[ComponentIdentity, ...]
    background: BackgroundModel

    def __post_init__(self) -> None:
        identities = tuple(self.lorentzian_identities)
        if not isinstance(self.elastic_present, bool):
            raise ValueError("elastic_present must be boolean")
        if any(
            not isinstance(item, ComponentIdentity)
            or item.family is not ComponentFamily.LORENTZIAN
            for item in identities
        ):
            raise ValueError("Method Lorentzian identities must be typed")
        if len(set(identities)) != len(identities):
            raise ValueError("Method Lorentzian identities must be unique")
        if not isinstance(self.background, BackgroundModel):
            raise ValueError("Method background must be a BackgroundModel")
        if (
            not self.elastic_present
            and not identities
            and self.background is BackgroundModel.NONE
        ):
            raise ValueError("Method composition must contain a scientific component")
        object.__setattr__(self, "lorentzian_identities", identities)

    @property
    def lorentzian_count(self) -> int:
        return len(self.lorentzian_identities)


@dataclass(frozen=True, slots=True)
class MethodParameterState:
    """Reusable constraints with no ordinary free-parameter Current value."""

    reference: ParameterReference
    user_lower_limit: float | None
    user_upper_limit: float | None
    user_bounds_enabled: bool
    free: bool
    fixed_value: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reference, ParameterReference):
            raise ValueError("Method parameter reference must be typed")
        for value in (self.user_lower_limit, self.user_upper_limit):
            if value is not None and not math.isfinite(value):
                raise ValueError("Method user bounds must be finite when present")
        if (
            self.user_bounds_enabled
            and self.user_lower_limit is not None
            and self.user_upper_limit is not None
            and self.user_lower_limit > self.user_upper_limit
        ):
            raise ValueError("Method lower bound must not exceed upper bound")
        if not isinstance(self.user_bounds_enabled, bool) or not isinstance(
            self.free, bool
        ):
            raise ValueError("Method parameter states must be boolean")
        if self.free and self.fixed_value is not None:
            raise ValueError("a free Method parameter must not store a Current value")
        if not self.free and (
            self.fixed_value is None or not math.isfinite(self.fixed_value)
        ):
            raise ValueError("a fixed Method parameter requires a finite fixed value")


class MethodCenterMode(StrEnum):
    LEGACY_SHARED = "legacy_shared"
    INDEPENDENT = "independent"
    CENTER_GROUP = "center_group"


@dataclass(frozen=True, slots=True)
class MethodCenterRelation:
    reference: ParameterReference
    mode: MethodCenterMode
    group_id: str | None = None

    def __post_init__(self) -> None:
        if self.reference.family is not ParameterFamily.CENTER:
            raise ValueError("center relation requires a Center reference")
        if not isinstance(self.mode, MethodCenterMode):
            raise ValueError("center relation mode must be typed")
        if (self.mode is MethodCenterMode.CENTER_GROUP) != (self.group_id is not None):
            raise ValueError("only CENTER_GROUP relations carry group_id")


@dataclass(frozen=True, slots=True)
class MethodRelationshipGroup:
    group_id: str
    members: tuple[ParameterReference, ...]

    def __post_init__(self) -> None:
        members = tuple(self.members)
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError("relationship group_id must be nonempty")
        if self.group_id != self.group_id.strip():
            raise ValueError("relationship group_id must be canonical")
        if not members or len(set(members)) != len(members):
            raise ValueError("relationship members must be nonempty and unique")
        object.__setattr__(self, "members", members)


@dataclass(frozen=True, slots=True)
class MethodRelationships:
    center_relations: tuple[MethodCenterRelation, ...] = ()
    center_groups: tuple[MethodRelationshipGroup, ...] = ()
    parameter_ties: tuple[MethodRelationshipGroup, ...] = ()


@dataclass(frozen=True, slots=True)
class FittingMethod:
    """Reusable fitting recipe, deliberately excluding Results and free values."""

    composition: MethodComposition
    parameters: tuple[MethodParameterState, ...]
    relationships: MethodRelationships
    transfer_defaults: MethodTransferOptions = MethodTransferOptions()
    resolution_dataset_id: str | None = None
    q_bins: QBins | None = None
    fitting_selection: FittingSelection | None = None

    def __post_init__(self) -> None:
        parameters = tuple(self.parameters)
        if len({item.reference for item in parameters}) != len(parameters):
            raise ValueError("Method parameter references must be unique")
        object.__setattr__(self, "parameters", parameters)

    def parameter(self, reference: ParameterReference) -> MethodParameterState:
        for item in self.parameters:
            if item.reference == reference:
                return item
        raise KeyError(reference)


@dataclass(frozen=True, slots=True)
class AppliedFittingMethod:
    model: ManualModelState
    transfer_options: MethodTransferOptions
    resolution_dataset_id: str | None
    q_bins: QBins | None
    fitting_selection: FittingSelection | None


def _composition(model: ManualModelState) -> MethodComposition:
    return MethodComposition(
        elastic_present=model.elastic_area is not None,
        lorentzian_identities=tuple(item.identity for item in model.lorentzians),
        background=model.background,
    )


def _composition_compatible(
    model: ManualModelState,
    composition: MethodComposition,
) -> bool:
    return (
        (model.elastic_area is not None) == composition.elastic_present
        and len(model.lorentzians) == composition.lorentzian_count
        and model.background is composition.background
    )


def _reidentified_reference(
    reference: ParameterReference,
    identities: dict[ComponentIdentity, ComponentIdentity],
) -> ParameterReference:
    return ParameterReference(
        identities.get(reference.component, reference.component),
        reference.family,
    )


class _ComponentCorrespondenceError(ValueError):
    """Raised when target-local component state cannot be mapped safely."""


def _with_branch_component_identities(
    model: ManualModelState,
    composition: MethodComposition,
    *,
    preserve_target_component_state: bool,
) -> ManualModelState | None:
    """Preserve target state while assigning the anchor's branch identities."""

    if not _composition_compatible(model, composition):
        raise ValueError("target model does not have compatible branch composition")
    anchor_identities = composition.lorentzian_identities
    target_identities = tuple(item.identity for item in model.lorentzians)
    anchor_set = set(anchor_identities)
    if set(target_identities) == anchor_set:
        return model
    identities = {
        identity: identity for identity in target_identities if identity in anchor_set
    }
    remaining_target = tuple(
        identity for identity in target_identities if identity not in identities
    )
    remaining_anchor = tuple(
        identity
        for identity in anchor_identities
        if identity not in identities.values()
    )
    if len(remaining_target) > 1:
        if preserve_target_component_state:
            raise _ComponentCorrespondenceError(
                "multiple unmatched Lorentzians have no trustworthy component "
                "correspondence for preserving target-local state"
            )
        return None
    if remaining_target:
        identities[remaining_target[0]] = remaining_anchor[0]
    return replace(
        model,
        lorentzians=tuple(
            replace(component, identity=identities[component.identity])
            for component in model.lorentzians
        ),
        parameter_ties=tuple(
            replace(
                group,
                members=tuple(
                    _reidentified_reference(member, identities)
                    for member in group.members
                ),
            )
            for group in model.parameter_ties
        ),
    )


def _center_relations(model: ManualModelState) -> tuple[MethodCenterRelation, ...]:
    relations: list[MethodCenterRelation] = []
    if model.elastic_area is not None:
        reference = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER)
        relations.append(
            MethodCenterRelation(
                reference,
                MethodCenterMode.CENTER_GROUP
                if model.elastic_center_group is not None
                else MethodCenterMode.LEGACY_SHARED,
                model.elastic_center_group,
            )
        )
    for component in model.lorentzians:
        reference = ParameterReference(component.identity, ParameterFamily.CENTER)
        if component.center is not None:
            mode = MethodCenterMode.INDEPENDENT
            group_id = None
        elif component.center_group is not None:
            mode = MethodCenterMode.CENTER_GROUP
            group_id = component.center_group
        else:
            mode = MethodCenterMode.LEGACY_SHARED
            group_id = None
        relations.append(MethodCenterRelation(reference, mode, group_id))
    return tuple(relations)


def capture_fitting_method(
    model: ManualModelState,
    *,
    context: ManualFitContext | None = None,
    transfer_defaults: MethodTransferOptions | None = None,
) -> FittingMethod:
    """Capture reusable Method state while omitting ordinary free Current values."""

    transfer_defaults = transfer_defaults or MethodTransferOptions()
    parameters = tuple(
        MethodParameterState(
            reference=reference,
            user_lower_limit=model.parameter_intent(reference).user_lower_limit,
            user_upper_limit=model.parameter_intent(reference).user_upper_limit,
            user_bounds_enabled=(model.parameter_intent(reference).user_bounds_enabled),
            free=model.parameter_intent(reference).free,
            fixed_value=(
                None
                if model.parameter_intent(reference).free
                else model.parameter_intent(reference).current_value
            ),
        )
        for reference in model.parameter_references()
    )
    relationships = MethodRelationships(
        center_relations=_center_relations(model),
        center_groups=tuple(
            MethodRelationshipGroup(
                item.group_id,
                model.center_group_members(item.group_id),
            )
            for item in model.center_groups
        ),
        parameter_ties=tuple(
            MethodRelationshipGroup(item.group_id, item.members)
            for item in model.parameter_ties
        ),
    )
    return FittingMethod(
        composition=_composition(model),
        parameters=parameters,
        relationships=relationships,
        transfer_defaults=transfer_defaults,
        resolution_dataset_id=(context.resolution.dataset_id if context else None),
        q_bins=(context.sample.dataset.q_bins if context else None),
        fitting_selection=(context.selection if context else None),
    )


def _default_model(
    composition: MethodComposition,
    *,
    independent_lorentzian_centers: bool = False,
) -> ManualModelState:
    has_center = composition.elastic_present or (
        bool(composition.lorentzian_identities) and not independent_lorentzian_centers
    )
    return ManualModelState(
        energy_shift=ManualParameterIntent(0.0) if has_center else None,
        elastic_area=(
            ManualParameterIntent(1.0) if composition.elastic_present else None
        ),
        lorentzians=tuple(
            ManualLorentzianState(
                area=ManualParameterIntent(1.0),
                fwhm=ManualParameterIntent(0.1),
                center=(
                    ManualParameterIntent(0.0)
                    if independent_lorentzian_centers
                    else None
                ),
                identity=identity,
            )
            for identity in composition.lorentzian_identities
        ),
        background=composition.background,
        b0=(
            ManualParameterIntent(0.0)
            if composition.background is not BackgroundModel.NONE
            else None
        ),
        b1=(
            ManualParameterIntent(0.0)
            if composition.background is BackgroundModel.LINEAR
            else None
        ),
    )


def _relation_model(
    base: ManualModelState,
    relationships: MethodRelationships,
) -> ManualModelState:
    intents = {
        reference: base.parameter_intent(reference)
        for reference in base.parameter_references()
    }
    relations = {item.reference: item for item in relationships.center_relations}
    legacy = tuple(
        reference
        for reference, relation in relations.items()
        if relation.mode is MethodCenterMode.LEGACY_SHARED
    )
    center_groups = tuple(
        ManualCenterGroupState(group.group_id, intents[group.members[0]])
        for group in relationships.center_groups
    )
    lorentzians = tuple(
        replace(
            component,
            center=(
                intents[ParameterReference(component.identity, ParameterFamily.CENTER)]
                if relations[
                    ParameterReference(component.identity, ParameterFamily.CENTER)
                ].mode
                is MethodCenterMode.INDEPENDENT
                else None
            ),
            center_group=relations[
                ParameterReference(component.identity, ParameterFamily.CENTER)
            ].group_id,
        )
        for component in base.lorentzians
    )
    elastic_reference = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER)
    return ManualModelState(
        energy_shift=intents[legacy[0]] if legacy else None,
        elastic_area=base.elastic_area,
        lorentzians=lorentzians,
        background=base.background,
        b0=base.b0,
        b1=base.b1,
        center_groups=center_groups,
        elastic_center_group=(
            relations[elastic_reference].group_id
            if base.elastic_area is not None
            and relations[elastic_reference].mode is MethodCenterMode.CENTER_GROUP
            else None
        ),
        parameter_ties=tuple(
            ManualParameterTieState(
                group.group_id,
                group.members,
                intents[group.members[0]],
            )
            for group in relationships.parameter_ties
        ),
    )


def _slot_key(model: ManualModelState, reference: ParameterReference) -> object:
    tie = model.tie_for(reference)
    if tie is not None:
        return ("tie", tie.group_id)
    center_group = model.center_group_for(reference)
    if center_group is not None:
        return ("center", center_group.group_id)
    if reference.family is ParameterFamily.CENTER:
        if reference.component == ELASTIC_COMPONENT:
            return ("legacy",) if model.elastic_center_group is None else reference
        component = next(
            item for item in model.lorentzians if item.identity == reference.component
        )
        if component.center is None and component.center_group is None:
            return ("legacy",)
    return reference


def _apply_parameter_categories(
    model: ManualModelState,
    method: FittingMethod,
    options: MethodTransferOptions,
) -> ManualModelState:
    settings = {item.reference: item for item in method.parameters}
    groups: dict[object, list[ParameterReference]] = {}
    for reference in model.parameter_references():
        groups.setdefault(_slot_key(model, reference), []).append(reference)
    updated = model
    for members in groups.values():
        source = [settings[item] for item in members]
        if options.user_bounds:
            bounds = {
                (
                    item.user_lower_limit,
                    item.user_upper_limit,
                    item.user_bounds_enabled,
                )
                for item in source
            }
            if len(bounds) != 1:
                raise ValueError(
                    "transferred user bounds conflict within a target shared parameter"
                )
        if options.free_fixed:
            states = {(item.free, item.fixed_value) for item in source}
            if len(states) != 1:
                raise ValueError(
                    "transferred Free/Fixed state conflicts within a target shared "
                    "parameter"
                )
        reference = members[0]
        intent = updated.parameter_intent(reference)
        first = source[0]
        current_value = intent.current_value
        if options.free_fixed and not first.free:
            if first.fixed_value is None:
                raise RuntimeError("fixed Method state has no fixed value")
            current_value = first.fixed_value
        updated = replace_parameter_intent(
            updated,
            reference,
            ManualParameterIntent(
                current_value=current_value,
                user_lower_limit=(
                    first.user_lower_limit
                    if options.user_bounds
                    else intent.user_lower_limit
                ),
                user_upper_limit=(
                    first.user_upper_limit
                    if options.user_bounds
                    else intent.user_upper_limit
                ),
                free=first.free if options.free_fixed else intent.free,
                user_bounds_enabled=(
                    first.user_bounds_enabled
                    if options.user_bounds
                    else intent.user_bounds_enabled
                ),
            ),
        )
    return updated


def apply_fitting_method(
    method: FittingMethod,
    target_model: ManualModelState | None,
    *,
    target_resolution_dataset_id: str | None = None,
    target_q_bins: QBins | None = None,
    target_fitting_selection: FittingSelection | None = None,
    overrides: MethodTransferOverrides | None = None,
) -> AppliedFittingMethod:
    """Apply mandatory composition plus independently selectable Method categories."""

    options = (overrides or MethodTransferOverrides()).resolve(method.transfer_defaults)
    base = None
    if target_model is not None and _composition_compatible(
        target_model,
        method.composition,
    ):
        base = _with_branch_component_identities(
            target_model,
            method.composition,
            preserve_target_component_state=not (
                options.user_bounds
                and options.free_fixed
                and options.parameter_relationships
            ),
        )
    if base is None:
        base = _default_model(method.composition)
    if options.parameter_relationships:
        base = _relation_model(base, method.relationships)
    model = _apply_parameter_categories(base, method, options)
    return AppliedFittingMethod(
        model=model,
        transfer_options=options,
        resolution_dataset_id=(
            method.resolution_dataset_id
            if options.resolution and method.resolution_dataset_id is not None
            else target_resolution_dataset_id
        ),
        q_bins=(
            method.q_bins
            if options.q_bins and method.q_bins is not None
            else target_q_bins
        ),
        fitting_selection=(
            method.fitting_selection
            if options.fitting_selection and method.fitting_selection is not None
            else target_fitting_selection
        ),
    )


@dataclass(frozen=True, slots=True)
class SavedFittingMethod:
    name: str
    method: FittingMethod


_SAVED_ENTRY_ISSUER = object()
_WORKING_COPY_ISSUER = object()


@dataclass(frozen=True, slots=True)
class ResultWorkingCopy:
    """One item's Result and separate editable, possibly unfitted parameters."""

    snapshot: ResultSnapshot | None
    draft_states: tuple[ManualModelState | None, ...]
    sample_identity: tuple[str, str]
    _issuer: InitVar[object | None] = None
    _issued: object = field(init=False, repr=False, compare=False)

    def __post_init__(self, _issuer: object | None) -> None:
        if _issuer is not _WORKING_COPY_ISSUER:
            raise ValueError("working copies must be issued by the workflow")
        states = tuple(self.draft_states)
        if any(
            item is not None and not isinstance(item, ManualModelState)
            for item in states
        ):
            raise ValueError("working copy requires Manual parameter states")
        if self.snapshot is not None:
            if not _is_issued_result_snapshot(self.snapshot):
                raise ValueError("working copy requires an issued ResultSnapshot")
            if (
                len(states) != self.snapshot.result.total_count
                or self.sample_identity != self.snapshot.result.sample_identity
            ):
                raise ValueError(
                    "working copy must match its Result Sample and Q groups"
                )
        if len(self.sample_identity) != 2 or any(
            not isinstance(value, str) or not value.strip()
            for value in self.sample_identity
        ):
            raise ValueError("working copy requires its Project/Sample identity")
        object.__setattr__(self, "draft_states", states)
        object.__setattr__(self, "_issued", _WORKING_COPY_ISSUER)

    @property
    def has_result(self) -> bool:
        return self.snapshot is not None

    @property
    def modified(self) -> bool:
        return self.has_result

    @property
    def has_draft_edits(self) -> bool:
        baseline = (
            (None,) * len(self.draft_states)
            if self.snapshot is None
            else self.snapshot.working_states
        )
        return any(
            current is not fitted
            for current, fitted in zip(self.draft_states, baseline, strict=True)
        )


def _issue_working_copy(
    snapshot: ResultSnapshot | None,
    draft_states: tuple[ManualModelState | None, ...],
    sample_identity: tuple[str, str],
) -> ResultWorkingCopy:
    return ResultWorkingCopy(
        snapshot, draft_states, sample_identity, _issuer=_WORKING_COPY_ISSUER
    )


def _is_issued_working_copy(value: object) -> bool:
    return (
        isinstance(value, ResultWorkingCopy)
        and getattr(value, "_issued", None) is _WORKING_COPY_ISSUER
    )


@dataclass(frozen=True, slots=True)
class SavedFitResult:
    name: str
    snapshot: ResultSnapshot
    best: bool = False
    draft_states: tuple[ManualModelState | None, ...] | None = None
    modified_copy: ResultWorkingCopy | None = None
    _issuer: InitVar[object | None] = None
    _issued: object = field(init=False, repr=False, compare=False)
    _saved_copy: ResultWorkingCopy = field(init=False, repr=False, compare=False)

    def __post_init__(self, _issuer: object | None) -> None:
        if _issuer is not _SAVED_ENTRY_ISSUER:
            raise ValueError("saved entries must be issued by the workflow")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("saved Result name must be nonempty")
        if not _is_issued_result_snapshot(self.snapshot):
            raise ValueError("saved Result requires an execution snapshot")
        if not isinstance(self.best, bool):
            raise ValueError("best mark must be boolean")
        states = (
            self.snapshot.working_states
            if self.draft_states is None
            else tuple(self.draft_states)
        )
        saved_copy = _issue_working_copy(
            self.snapshot, states, self.snapshot.result.sample_identity
        )
        modified = self.modified_copy
        if modified is not None and (
            not _is_issued_working_copy(modified)
            or modified.snapshot is None
            or modified.sample_identity != saved_copy.sample_identity
            or len(modified.draft_states) != len(states)
        ):
            raise ValueError("modified copy must belong to the saved Result's Sample")
        object.__setattr__(self, "draft_states", states)
        object.__setattr__(self, "_saved_copy", saved_copy)
        object.__setattr__(self, "_issued", _SAVED_ENTRY_ISSUER)

    @property
    def result(self) -> UnifiedResult:
        return self.snapshot.result

    @property
    def working_copy(self) -> ResultWorkingCopy:
        return self._saved_copy if self.modified_copy is None else self.modified_copy

    @property
    def has_result(self) -> bool:
        return True

    @property
    def modified(self) -> bool:
        return self.working_copy.snapshot is not self.snapshot

    @property
    def has_draft_edits(self) -> bool:
        return self.working_copy.has_draft_edits


def _issue_saved_result(
    name: str,
    snapshot: ResultSnapshot,
    *,
    best: bool = False,
    draft_states: tuple[ManualModelState | None, ...] | None = None,
    modified_copy: ResultWorkingCopy | None = None,
) -> SavedFitResult:
    return SavedFitResult(
        name,
        snapshot,
        best,
        draft_states,
        modified_copy,
        _issuer=_SAVED_ENTRY_ISSUER,
    )


def _copy_saved_result(
    entry: SavedFitResult,
    *,
    name: str | None = None,
    best: bool | None = None,
) -> SavedFitResult:
    return _issue_saved_result(
        entry.name if name is None else name,
        entry.snapshot,
        best=entry.best if best is None else best,
        draft_states=entry.draft_states,
        modified_copy=entry.modified_copy,
    )


def _is_issued_saved_result(entry: object) -> bool:
    return (
        isinstance(entry, SavedFitResult)
        and getattr(entry, "_issued", None) is _SAVED_ENTRY_ISSUER
    )


@dataclass(frozen=True, slots=True)
class FittingWorkspaceState:
    """Minimal in-memory working/saved Method and Result lifecycle."""

    working_method: FittingMethod | None = None
    saved_methods: tuple[SavedFittingMethod, ...] = ()
    saved_results: tuple[SavedFitResult, ...] = ()
    untitled: ResultWorkingCopy | None = None
    active_item: SavedFitResult | ResultWorkingCopy | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "saved_methods", tuple(self.saved_methods))
        saved_results = tuple(self.saved_results)
        if any(not _is_issued_saved_result(item) for item in saved_results):
            raise ValueError("saved Results must be named snapshots")
        if sum(item.best for item in saved_results) > 1:
            raise ValueError("a Dataset may have at most one Best Result")
        object.__setattr__(self, "saved_results", saved_results)
        if self.untitled is not None and not _is_issued_working_copy(self.untitled):
            raise ValueError("untitled item must be issued by the workflow")
        active = self.active_item
        if active is not None and not (
            any(active is entry for entry in saved_results)
            or active is self.untitled
            and _is_issued_working_copy(active)
        ):
            raise ValueError("active item must belong to this workspace")
        if self.untitled is not None and active is not self.untitled:
            raise ValueError("an untitled item must be active")
        copies = [entry.working_copy for entry in saved_results]
        if self.untitled is not None:
            copies.append(self.untitled)
        if copies and any(
            item.sample_identity != copies[0].sample_identity for item in copies
        ):
            raise ValueError("workspace items must share one Project/Sample")

    @property
    def associated_saved_result(self) -> SavedFitResult | None:
        """Return the saved snapshot associated with Current, when one exists."""

        return (
            self.active_item if isinstance(self.active_item, SavedFitResult) else None
        )

    @property
    def current_saved_result_index(self) -> int | None:
        return next(
            (
                index
                for index, entry in enumerate(self.saved_results)
                if entry is self.active_item
            ),
            None,
        )

    @property
    def active_working_copy(self) -> ResultWorkingCopy | None:
        active = self.active_item
        return active.working_copy if isinstance(active, SavedFitResult) else active

    @property
    def current(self) -> ResultSnapshot | None:
        copy = self.active_working_copy
        return None if copy is None else copy.snapshot

    @property
    def current_result(self) -> UnifiedResult | None:
        """Read-only projection of the Current snapshot's scientific Result."""

        return None if self.current is None else self.current.result

    @property
    def best_saved_result(self) -> SavedFitResult | None:
        return next((item for item in self.saved_results if item.best), None)


class ResultDisplayStatus(StrEnum):
    FITTED_CURRENT = "fitted_current"
    NEEDS_FIT = "needs_fit"
    FAILED = "failed"
    BLOCKED = "blocked"
    EXCLUDED = "excluded"
    NOT_FITTED = "not_fitted"


def result_q_display_status(
    state: FittingWorkspaceState,
    draft: ManualFitDraft,
    group_index: int,
    context: ManualFitContext | None,
) -> ResultDisplayStatus:
    """Derive status from Current, exact draft identity and active Q context."""

    model = draft.setup(group_index).model
    current = state.current
    if current is None:
        return (
            ResultDisplayStatus.NEEDS_FIT
            if model is not None
            else ResultDisplayStatus.NOT_FITTED
        )
    point = current.result.point(group_index)
    if point.status is MultiQFitStatus.SUCCESS:
        if point.working_state is not model or context is None:
            return ResultDisplayStatus.NEEDS_FIT
        fit = point.fit_result
        assert fit is not None
        if current.result.sample_identity != (
            context.project_id,
            context.sample.dataset_id,
        ) or not fit_evidence_is_current(fit, context, group_index):
            return ResultDisplayStatus.NEEDS_FIT
        return ResultDisplayStatus.FITTED_CURRENT
    return {
        MultiQFitStatus.FAILED: ResultDisplayStatus.FAILED,
        MultiQFitStatus.BLOCKED: ResultDisplayStatus.BLOCKED,
        MultiQFitStatus.EXCLUDED: ResultDisplayStatus.EXCLUDED,
        MultiQFitStatus.NOT_RUN: (
            ResultDisplayStatus.NEEDS_FIT
            if model is not None
            else ResultDisplayStatus.NOT_FITTED
        ),
    }[point.status]


def derive_result_in_context(
    snapshot: ResultSnapshot,
    q_bins: QBins,
    contexts: Sequence[ManualFitContext | None],
) -> DerivedUnifiedQENSResult:
    """Derive each Q only when its own stored fit matches active science."""

    if not _is_issued_result_snapshot(snapshot) or len(contexts) != len(
        snapshot.result.points
    ):
        raise ValueError("derivation requires an issued Result and one context per Q")
    valid: list[bool] = []
    for point, context in zip(snapshot.result.points, contexts, strict=True):
        fit = point.fit_result
        if fit is None or context is None:
            valid.append(False)
            continue
        valid.append(
            snapshot.result.sample_identity
            == (context.project_id, context.sample.dataset_id)
            and fit_evidence_is_current(fit, context, point.outcome.group_index)
        )
    return derive_unified_qens(snapshot.result, q_bins, context_valid=valid)


def set_working_method(
    state: FittingWorkspaceState,
    method: FittingMethod,
) -> FittingWorkspaceState:
    return replace(state, working_method=method)


def _replace_saved_entry(
    state: FittingWorkspaceState, entry: SavedFitResult, updated: SavedFitResult
) -> FittingWorkspaceState:
    if not any(item is entry for item in state.saved_results):
        raise ValueError("saved Result entry is not in this workspace")
    return replace(
        state,
        saved_results=tuple(
            updated if item is entry else item for item in state.saved_results
        ),
        active_item=updated if state.active_item is entry else state.active_item,
    )


def _store_working_copy(
    state: FittingWorkspaceState, copy: ResultWorkingCopy
) -> FittingWorkspaceState:
    active = state.active_item
    if isinstance(active, SavedFitResult):
        return _replace_saved_entry(
            state,
            active,
            _issue_saved_result(
                active.name,
                active.snapshot,
                best=active.best,
                draft_states=active.draft_states,
                modified_copy=copy,
            ),
        )
    return replace(state, untitled=copy, active_item=copy)


def update_working_draft(
    state: FittingWorkspaceState, draft: ManualFitDraft
) -> FittingWorkspaceState:
    """Cache parameter edits on the active item without changing fitted evidence."""

    states = tuple(setup.model for setup in draft.setups)
    current = state.active_working_copy
    identity = (draft.project_id, draft.sample_id)
    if current is not None:
        if current.sample_identity != identity or len(current.draft_states) != len(
            states
        ):
            raise ValueError("draft belongs to another Sample or Q topology")
        if all(
            before is after
            for before, after in zip(current.draft_states, states, strict=True)
        ):
            return state
    return _store_working_copy(
        state, _issue_working_copy(state.current, states, identity)
    )


def active_manual_draft(
    state: FittingWorkspaceState, template: ManualFitDraft
) -> ManualFitDraft:
    """Install the active item's cached parameters, or raw/empty analysis state."""

    copy = state.active_working_copy
    states = (None,) * len(template.setups) if copy is None else copy.draft_states
    if copy is not None and (
        copy.sample_identity != (template.project_id, template.sample_id)
        or len(states) != len(template.setups)
    ):
        raise ValueError("active item belongs to another Sample or Q topology")
    draft = template
    for index, model in enumerate(states):
        if draft.setup(index).model is not model:
            draft = draft.with_model(index, model)
    return draft


def set_current_result(
    state: FittingWorkspaceState,
    snapshot: ResultSnapshot,
    *,
    draft: ManualFitDraft | None = None,
) -> FittingWorkspaceState:
    """Replace the Result only on the active item; create untitled when absent."""

    if not _is_issued_result_snapshot(snapshot):
        raise ValueError("Current must be a ResultSnapshot created at execution")
    states = (
        snapshot.working_states
        if draft is None
        else tuple(setup.model for setup in draft.setups)
    )
    if draft is not None and snapshot.result.sample_identity != (
        draft.project_id,
        draft.sample_id,
    ):
        raise ValueError("Result belongs to another Sample")
    return _store_working_copy(
        state, _issue_working_copy(snapshot, states, snapshot.result.sample_identity)
    )


def replace_current_q_result(
    state: FittingWorkspaceState,
    snapshot: ResultSnapshot,
    *,
    draft: ManualFitDraft | None = None,
) -> FittingWorkspaceState:
    """Adopt one Single-Q fit without downgrading other fitted Qs in Current."""

    if not _is_issued_result_snapshot(snapshot):
        raise ValueError("Single-Q adoption requires an issued snapshot")
    fitted = [
        point
        for point in snapshot.result.points
        if point.status is MultiQFitStatus.SUCCESS
    ]
    if len(fitted) != 1:
        raise ValueError("Single-Q adoption requires exactly one fitted Q")
    current = state.current
    if current is not None:
        if current.result.total_count != snapshot.result.total_count:
            raise ValueError("Single-Q Result Q topology differs from Current")
        if current.result.sample_identity != snapshot.result.sample_identity:
            raise ValueError("Single-Q Result belongs to another Sample")
        snapshot = _replace_result_point(current, fitted[0])
    if draft is not None:
        return set_current_result(state, snapshot, draft=draft)
    copy = state.active_working_copy
    states = list(snapshot.working_states if copy is None else copy.draft_states)
    states[fitted[0].outcome.group_index] = fitted[0].working_state
    return _store_working_copy(
        state,
        _issue_working_copy(snapshot, tuple(states), snapshot.result.sample_identity),
    )


def save_working_method(
    state: FittingWorkspaceState,
    name: str,
) -> FittingWorkspaceState:
    if state.working_method is None:
        raise ValueError("there is no working Method to save")
    saved = SavedFittingMethod(name, state.working_method)
    return replace(state, saved_methods=(*state.saved_methods, saved))


def save_current_result(
    state: FittingWorkspaceState,
    name: str | None = None,
) -> FittingWorkspaceState:
    """Save the active Result; an untitled Result requires a name."""

    if name is not None:
        return save_current_result_as(state, name)
    current = state.current
    if current is None:
        raise ValueError("there is no current Result to save")
    active = state.associated_saved_result
    if active is None:
        raise ValueError("untitled Result requires a name; use Save As")
    return _replace_saved_entry(
        state,
        active,
        _issue_saved_result(
            active.name,
            current,
            best=active.best,
            draft_states=active.working_copy.draft_states,
        ),
    )


def save_current_result_as(
    state: FittingWorkspaceState,
    name: str,
) -> FittingWorkspaceState:
    """Save active fitted evidence as a new clean entry and clear its source copy."""

    if state.current is None:
        raise ValueError("there is no current Result to save")
    copy = state.active_working_copy
    assert copy is not None
    saved = _issue_saved_result(name, state.current, draft_states=copy.draft_states)
    active = state.associated_saved_result
    if active is not None and active.modified_copy is not None:
        state = _replace_saved_entry(
            state,
            active,
            _issue_saved_result(
                active.name,
                active.snapshot,
                best=active.best,
                draft_states=active.draft_states,
            ),
        )
    return replace(
        state,
        saved_results=(*state.saved_results, saved),
        untitled=None,
        active_item=saved,
    )


def append_saved_result(
    state: FittingWorkspaceState,
    name: str,
    snapshot: ResultSnapshot,
) -> FittingWorkspaceState:
    """Append an executed snapshot without modifying Current or its association."""

    if not _is_issued_result_snapshot(snapshot):
        raise ValueError("saved entry requires a ResultSnapshot")
    return replace(
        state,
        saved_results=(*state.saved_results, _issue_saved_result(name, snapshot)),
    )


def remove_saved_result(
    state: FittingWorkspaceState,
    entry: SavedFitResult,
) -> FittingWorkspaceState:
    """Remove one entry and its copy; removing the active entry leaves raw data."""

    index = next(
        (
            position
            for position, item in enumerate(state.saved_results)
            if item is entry
        ),
        None,
    )
    if index is None:
        raise ValueError("saved Result entry is not in this workspace")
    return replace(
        state,
        saved_results=state.saved_results[:index] + state.saved_results[index + 1 :],
        active_item=None if state.active_item is entry else state.active_item,
    )


def rename_saved_result(
    state: FittingWorkspaceState,
    entry: SavedFitResult,
    name: str,
) -> FittingWorkspaceState:
    """Rename one entry by identity without changing Current or association."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("saved Result name must be nonempty")
    index = next(
        (index for index, item in enumerate(state.saved_results) if item is entry),
        None,
    )
    if index is None:
        raise ValueError("saved Result entry is not in this workspace")
    return _replace_saved_entry(state, entry, _copy_saved_result(entry, name=name))


def set_best_saved_result(
    state: FittingWorkspaceState,
    entry: SavedFitResult | None,
) -> FittingWorkspaceState:
    """Mark at most one existing entry Best; None clears the mark."""

    if entry is not None and not any(item is entry for item in state.saved_results):
        raise ValueError("Best Result entry is not in this workspace")
    for item in state.saved_results:
        if item.best != (item is entry):
            state = _replace_saved_entry(
                state, item, _copy_saved_result(item, best=item is entry)
            )
    return state


def has_unsaved_result_changes(
    state: FittingWorkspaceState,
    draft: ManualFitDraft | None = None,
) -> bool:
    """Combined convenience query; item queries distinguish Result and draft work."""

    if unsaved_result_items(state) or draft_edit_items(state):
        return True
    if draft is None:
        return False
    copy = state.active_working_copy
    states = (None,) * len(draft.setups) if copy is None else copy.draft_states
    return len(states) != len(draft.setups) or any(
        setup.model is not stored
        for setup, stored in zip(draft.setups, states, strict=True)
    )


def unsaved_result_items(
    state: FittingWorkspaceState,
) -> tuple[SavedFitResult | ResultWorkingCopy, ...]:
    """Close-prompt items: unsaved untitled Result and entries with modified copies."""

    return (
        *(
            (state.untitled,)
            if state.untitled is not None and state.untitled.has_result
            else ()
        ),
        *(entry for entry in state.saved_results if entry.modified_copy is not None),
    )


def draft_edit_items(
    state: FittingWorkspaceState,
) -> tuple[SavedFitResult | ResultWorkingCopy, ...]:
    """Report cached parameters that have not been fitted, separately from Results."""

    return (
        *(
            (state.untitled,)
            if state.untitled is not None and state.untitled.has_draft_edits
            else ()
        ),
        *(entry for entry in state.saved_results if entry.has_draft_edits),
    )


def _require_discard_consent(
    state: FittingWorkspaceState, discard_untitled: bool
) -> None:
    if not isinstance(discard_untitled, bool):
        raise ValueError("discard_untitled must be boolean")
    if (
        state.untitled is not None
        and state.untitled.has_result
        and not discard_untitled
    ):
        raise ValueError("discard consent is required for the untitled Result")


def _retain_outgoing_draft(
    state: FittingWorkspaceState, draft: ManualFitDraft
) -> FittingWorkspaceState:
    copy = state.active_working_copy
    if copy is not None and len(copy.draft_states) != len(draft.setups):
        if copy.sample_identity != (draft.project_id, draft.sample_id):
            raise ValueError("draft belongs to another Sample")
        # A regrouped draft cannot describe edits to the historical item.
        # Retain that item's existing copy rather than rewriting its Q topology.
        return state
    return update_working_draft(state, draft)


def load_saved_result(
    state: FittingWorkspaceState,
    draft: ManualFitDraft,
    entry: SavedFitResult,
    context: ManualFitContext | None = None,
    *,
    discard_untitled: bool = False,
) -> tuple[FittingWorkspaceState, ManualFitDraft]:
    """Open an entry's working copy, retaining edits on the outgoing saved entry."""

    index = next(
        (index for index, item in enumerate(state.saved_results) if item is entry),
        None,
    )
    if index is None:
        raise ValueError("saved Result entry is not in this workspace")
    incoming = entry.snapshot.result
    count = incoming.total_count
    sample_identity = incoming.sample_identity
    if count != len(draft.setups):
        raise ValueError("saved Result Q topology differs from active Sample")
    if (
        sample_identity != (draft.project_id, draft.sample_id)
        or context is not None
        and sample_identity != (context.project_id, context.sample.dataset_id)
    ):
        raise ValueError("saved Result belongs to another Sample")
    _require_discard_consent(state, discard_untitled)
    state = _retain_outgoing_draft(state, draft)
    entry = state.saved_results[index]
    state = replace(state, untitled=None, active_item=entry)
    return state, active_manual_draft(state, draft)


def fresh_analysis(
    state: FittingWorkspaceState,
    draft: ManualFitDraft,
    *,
    discard_untitled: bool = False,
) -> tuple[FittingWorkspaceState, ManualFitDraft]:
    """Start an empty untitled item; retain saved working copies on their entries."""

    _require_discard_consent(state, discard_untitled)
    state = _retain_outgoing_draft(state, draft)
    copy = _issue_working_copy(
        None, (None,) * len(draft.setups), (draft.project_id, draft.sample_id)
    )
    state = replace(state, untitled=copy, active_item=copy)
    return state, active_manual_draft(state, draft)


def revert_saved_result(
    state: FittingWorkspaceState, draft: ManualFitDraft, entry: SavedFitResult
) -> tuple[FittingWorkspaceState, ManualFitDraft]:
    """Drop modifications and restore the saved Result and whole draft baseline."""

    active = state.active_item is entry
    updated = _issue_saved_result(
        entry.name,
        entry.snapshot,
        best=entry.best,
        draft_states=entry.draft_states,
    )
    state = _replace_saved_entry(state, entry, updated)
    return state, active_manual_draft(state, draft) if active else draft


def swap_current_lorentzian_identities(
    state: FittingWorkspaceState,
    draft: ManualFitDraft,
    group_index: int,
    first: ComponentIdentity,
    second: ComponentIdentity,
) -> tuple[FittingWorkspaceState, ManualFitDraft]:
    """Relabel exactly two fitted components at one Q without refitting."""

    if state.current is None or first == second:
        raise ValueError("swap requires two distinct identities in Current")
    point = state.current.result.point(group_index)
    fit = point.fit_result
    working = point.working_state
    draft_model = draft.setup(group_index).model
    if fit is None or working is None or draft_model is None:
        raise ValueError("swap requires fitted Q and editable Manual state")
    fitted = fit.fitted_model
    if fitted is None or {first, second} - {
        item.identity for item in fitted.lorentzians
    }:
        raise ValueError("swap identities must belong to the fitted Q")
    mapping = {item.identity: item.identity for item in fitted.lorentzians}
    mapping.update({first: second, second: first})
    if {first, second} - {item.identity for item in draft_model.lorentzians}:
        raise ValueError("draft topology no longer matches the fitted identities")
    relabeled_fit = relabel_fit_components(fit, mapping)
    relabeled_working = _relabel_manual_state(working, mapping)
    relabeled_draft = (
        relabeled_working
        if draft_model is working
        else _relabel_manual_state(draft_model, mapping)
    )
    updated = replace(
        point,
        outcome=replace(point.outcome, fit_result=relabeled_fit),
        working_state=relabeled_working,
        identity_marks=(*point.identity_marks, IdentityMark.USER_SWAP),
    )
    current = _replace_result_point(state.current, updated)
    return (
        _store_working_copy(
            state,
            _issue_working_copy(
                current,
                tuple(
                    setup.model
                    for setup in draft.with_model(group_index, relabeled_draft).setups
                ),
                current.result.sample_identity,
            ),
        ),
        draft.with_model(group_index, relabeled_draft),
    )


def _resolved_context(
    project: WorkflowProject,
    draft: ManualFitDraft,
    group: int,
) -> ManualFitContext:
    sample = next(
        item for item in project.datasets if item.dataset_id == draft.sample_id
    )
    resolved = resolve_manual_fit_context(project, sample, group)
    if resolved.context is None:
        from .project import WorkflowError

        raise WorkflowError(resolved.diagnostics)
    return resolved.context


def _same_core_topology(
    first: SpectralModelDefinition,
    second: SpectralModelDefinition,
) -> bool:
    return (
        first.background is second.background
        and (first.elastic_area is None) == (second.elastic_area is None)
        and (first.energy_shift is None) == (second.energy_shift is None)
        and set(first.parameter_references()) == set(second.parameter_references())
        and first.elastic_center_group == second.elastic_center_group
        and {
            (item.identity, item.center is not None, item.center_group)
            for item in first.lorentzians
        }
        == {
            (item.identity, item.center is not None, item.center_group)
            for item in second.lorentzians
        }
        and {item.group_id for item in first.center_groups}
        == {item.group_id for item in second.center_groups}
        and {(item.group_id, frozenset(item.members)) for item in first.parameter_ties}
        == {(item.group_id, frozenset(item.members)) for item in second.parameter_ties}
    )


def _padding_context_equal(
    first: EdgePaddingDetectionResult,
    second: EdgePaddingDetectionResult,
) -> bool:
    return (
        first.algorithm_version == second.algorithm_version
        and len(first.spectra) == len(second.spectra)
        and all(
            left.group_index == right.group_index
            and left.group_identity == right.group_identity
            and left.left == right.left
            and left.right == right.right
            and np.array_equal(left.auto_mask, right.auto_mask)
            and np.array_equal(left.review_mask, right.review_mask)
            for left, right in zip(first.spectra, second.spectra, strict=True)
        )
    )


def _prepared_resolution_context_equal(
    first: PreparedResolution,
    second: PreparedResolution,
) -> bool:
    """Compare the complete active measured-kernel context without aliasing it."""

    return (
        first.sample_dataset is second.sample_dataset
        and first.resolution_dataset is second.resolution_dataset
        and first.diagnostics == second.diagnostics
        and first.padding_comparisons == second.padding_comparisons
        and _padding_context_equal(first.sample_padding, second.sample_padding)
        and _padding_context_equal(first.resolution_padding, second.resolution_padding)
        and len(first.spectra) == len(second.spectra)
        and all(
            first.acceptance_provenance(index) == second.acceptance_provenance(index)
            and np.array_equal(
                first.spectra[index].accepted_mask,
                second.spectra[index].accepted_mask,
            )
            and np.array_equal(
                first.spectra[index].normalized_intensity,
                second.spectra[index].normalized_intensity,
            )
            and np.array_equal(
                first.spectra[index].normalized_uncertainty,
                second.spectra[index].normalized_uncertainty,
                equal_nan=True,
            )
            for index in range(len(first.spectra))
        )
    )


def _result_in_active_context(
    result: FitResult,
    context: ManualFitContext,
    group_index: int,
) -> FitResult:
    binding = result.context_binding
    if (
        binding is None
        or binding.selection is not context.selection
        or binding.group_index != group_index
        or not _prepared_resolution_context_equal(
            binding.prepared_resolution,
            context.prepared_resolution,
        )
    ):
        raise ValueError("Current Result does not belong to the active context")
    if binding.prepared_resolution is context.prepared_resolution:
        return result
    return replace(
        result,
        context_binding=FitContextBinding(
            context.prepared_resolution,
            context.selection,
            group_index,
        ),
    )


def _branch_configurations(
    draft: ManualFitDraft,
    context: ManualFitContext,
    method: FittingMethod | None,
    overrides: MethodTransferOverrides,
) -> tuple[SpectralModelDefinition | None, ...]:
    configurations: list[SpectralModelDefinition | None] = []
    for group_index, setup in enumerate(draft.setups):
        model = setup.model
        if method is not None:
            try:
                application = apply_fitting_method(
                    method,
                    model,
                    target_resolution_dataset_id=context.resolution.dataset_id,
                    target_q_bins=context.sample.dataset.q_bins,
                    target_fitting_selection=context.selection,
                    overrides=overrides,
                )
            except _ComponentCorrespondenceError:
                configurations.append(None)
                continue
            if (
                application.resolution_dataset_id != context.resolution.dataset_id
                or application.q_bins is not context.sample.dataset.q_bins
                or application.fitting_selection is not context.selection
            ):
                raise ValueError(
                    "selected Method context must match the active target context"
                )
            model = application.model
        if model is None:
            configurations.append(None)
            continue
        try:
            materialized = materialize_manual_model(
                model,
                context.prepared_resolution,
                context.selection,
                group_index,
            )
        except ValueError:
            configurations.append(None)
        else:
            configurations.append(materialized.fit_model)
    return tuple(configurations)


def _center_modes(
    model: ManualModelState,
) -> dict[ParameterReference, tuple[MethodCenterMode, str | None]]:
    """Return the declared center mode for every component-owned center."""

    modes: dict[ParameterReference, tuple[MethodCenterMode, str | None]] = {}
    if model.elastic_area is not None:
        reference = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER)
        modes[reference] = (
            (
                MethodCenterMode.CENTER_GROUP
                if model.elastic_center_group is not None
                else MethodCenterMode.LEGACY_SHARED
            ),
            model.elastic_center_group,
        )
    for component in model.lorentzians:
        reference = ParameterReference(component.identity, ParameterFamily.CENTER)
        if component.center is not None:
            modes[reference] = (MethodCenterMode.INDEPENDENT, None)
        elif component.center_group is not None:
            modes[reference] = (
                MethodCenterMode.CENTER_GROUP,
                component.center_group,
            )
        else:
            modes[reference] = (MethodCenterMode.LEGACY_SHARED, None)
    return modes


def _compatible_shared_center_intent(
    members: Sequence[ParameterReference],
    resolved_intents: dict[ParameterReference, ManualParameterIntent],
) -> ManualParameterIntent:
    """Return one order-independent intent for a resulting shared center slot."""

    intents = tuple(resolved_intents[member] for member in members)
    states = {
        (
            intent.free,
            intent.user_bounds_enabled,
            intent.user_lower_limit,
            intent.user_upper_limit,
        )
        for intent in intents
    }
    if len(states) != 1:
        raise ValueError(
            "selected center tie members have incompatible resulting "
            "Manual parameter state"
        )
    free, bounds_enabled, lower, upper = states.pop()
    current_values = {intent.current_value for intent in intents}
    if not free:
        if len(current_values) != 1:
            raise ValueError(
                "selected center tie members have incompatible resulting "
                "Manual parameter state"
            )
        current_value = current_values.pop()
    else:
        minimum = min(current_values)
        maximum = max(current_values)
        current_value = minimum + 0.5 * (maximum - minimum)
        if not np.isfinite(current_value):
            current_value = 0.5 * minimum + 0.5 * maximum
        if not np.isfinite(current_value):
            raise ValueError(
                "selected center tie members have no finite temporary current value"
            )
    return ManualParameterIntent(
        current_value,
        lower,
        upper,
        free=free,
        user_bounds_enabled=bounds_enabled,
    )


def _apply_s5_center_ties(
    source: ManualModelState,
    target: ManualModelState,
    policy: MultiQConstraintTransferPolicy,
    resolved_intents: dict[ParameterReference, ManualParameterIntent],
) -> ManualModelState:
    """Overlay only selected complete source center relationships."""

    if not policy.center_ties:
        return target
    active_by_identity = {
        item.identity: item
        for item in active_transferable_constraints(source).center_ties
    }
    selected = tuple(
        relationship
        for relationship in active_by_identity.values()
        if relationship.identity in policy.center_ties
    )

    selected_intents = {
        relationship.identity: _compatible_shared_center_intent(
            relationship.members,
            resolved_intents,
        )
        for relationship in selected
    }
    target_references = set(target.parameter_references())
    if any(
        member not in target_references
        for relationship in selected
        for member in relationship.members
    ):
        raise ValueError(
            "selected center relationship is incompatible with target composition"
        )

    modes = _center_modes(target)
    legacy_slot_members = tuple(
        reference
        for reference in target.parameter_references()
        if _slot_key(target, reference) == ("legacy",)
    )
    selected_members = {
        member for relationship in selected for member in relationship.members
    }
    selected_center_groups = {
        relationship.identity.group_id: relationship
        for relationship in selected
        if relationship.identity.kind is CenterTieKind.CENTER_GROUP
    }

    for group_id, relationship in selected_center_groups.items():
        if group_id is None:
            raise RuntimeError("validated center group lost its group_id")
        existing = next(
            (item for item in target.center_groups if item.group_id == group_id),
            None,
        )
        if existing is not None and set(target.center_group_members(group_id)) != set(
            relationship.members
        ):
            raise ValueError(
                "target center-group identity has incompatible complete membership"
            )

    for relationship in selected:
        if relationship.identity.kind is CenterTieKind.LEGACY_SHARED_CENTER:
            members = set(relationship.members)
            for reference, (mode, _group_id) in tuple(modes.items()):
                if mode is not MethodCenterMode.LEGACY_SHARED or reference in members:
                    continue
                if reference.component == ELASTIC_COMPONENT:
                    raise ValueError(
                        "selected legacy center tie cannot detach target Elastic "
                        "from an incompatible legacy relationship"
                    )
                modes[reference] = (MethodCenterMode.INDEPENDENT, None)
            for member in relationship.members:
                modes[member] = (MethodCenterMode.LEGACY_SHARED, None)
        elif relationship.identity.kind is CenterTieKind.CENTER_GROUP:
            group_id = relationship.identity.group_id
            if group_id is None:
                raise RuntimeError("validated center group lost its group_id")
            for member in relationship.members:
                modes[member] = (MethodCenterMode.CENTER_GROUP, group_id)
        else:
            for member in relationship.members:
                if modes[member][0] is not MethodCenterMode.LEGACY_SHARED:
                    continue
                if member.component == ELASTIC_COMPONENT:
                    if len(legacy_slot_members) > 1:
                        raise ValueError(
                            "selected general center tie cannot detach target Elastic "
                            "from legacy shared-center state"
                        )
                    continue
                modes[member] = (MethodCenterMode.INDEPENDENT, None)

    center_groups: list[ManualCenterGroupState] = []
    for center_group in target.center_groups:
        if center_group.group_id in selected_center_groups:
            continue
        if any(
            mode is MethodCenterMode.CENTER_GROUP and group_id == center_group.group_id
            for mode, group_id in modes.values()
        ):
            center_groups.append(center_group)
    for group_id, relationship in selected_center_groups.items():
        if group_id is None:
            raise RuntimeError("validated center group lost its group_id")
        center_groups.append(
            ManualCenterGroupState(
                group_id,
                selected_intents[relationship.identity],
            )
        )

    legacy_members = tuple(
        reference
        for reference, (mode, _group_id) in modes.items()
        if mode is MethodCenterMode.LEGACY_SHARED
    )
    selected_legacy = next(
        (
            relationship
            for relationship in selected
            if relationship.identity.kind is CenterTieKind.LEGACY_SHARED_CENTER
        ),
        None,
    )
    energy_shift = (
        selected_intents[selected_legacy.identity]
        if selected_legacy is not None
        else resolved_intents[legacy_members[0]]
        if legacy_members
        else None
    )
    elastic_reference = ParameterReference(
        ELASTIC_COMPONENT,
        ParameterFamily.CENTER,
    )
    elastic_center_group = (
        modes[elastic_reference][1]
        if elastic_reference in modes
        and modes[elastic_reference][0] is MethodCenterMode.CENTER_GROUP
        else None
    )
    lorentzians = tuple(
        replace(
            component,
            center=(
                resolved_intents[
                    ParameterReference(component.identity, ParameterFamily.CENTER)
                ]
                if modes[
                    ParameterReference(component.identity, ParameterFamily.CENTER)
                ][0]
                is MethodCenterMode.INDEPENDENT
                else None
            ),
            center_group=(
                modes[ParameterReference(component.identity, ParameterFamily.CENTER)][1]
                if modes[
                    ParameterReference(component.identity, ParameterFamily.CENTER)
                ][0]
                is MethodCenterMode.CENTER_GROUP
                else None
            ),
        )
        for component in target.lorentzians
    )

    selected_parameter_ties = {
        relationship.identity.group_id: relationship
        for relationship in selected
        if relationship.identity.kind is CenterTieKind.PARAMETER_TIE
    }
    parameter_ties: list[ManualParameterTieState] = []
    for tie_group in target.parameter_ties:
        if tie_group.group_id in selected_parameter_ties:
            selected_group = selected_parameter_ties[tie_group.group_id]
            if selected_group is None:
                raise RuntimeError("validated parameter tie lost its identity")
            if tie_group.family is not ParameterFamily.CENTER or set(
                tie_group.members
            ) != set(selected_group.members):
                raise ValueError(
                    "target parameter-tie identity has incompatible membership"
                )
            continue
        if tie_group.family is not ParameterFamily.CENTER:
            parameter_ties.append(tie_group)
            continue
        remaining = tuple(
            member for member in tie_group.members if member not in selected_members
        )
        if remaining:
            parameter_ties.append(replace(tie_group, members=remaining))
    for group_id, relationship in selected_parameter_ties.items():
        if group_id is None:
            raise RuntimeError("validated parameter tie lost its group_id")
        if any(tie_group.group_id == group_id for tie_group in parameter_ties):
            raise ValueError("selected center tie conflicts with a target tie identity")
        parameter_ties.append(
            ManualParameterTieState(
                group_id,
                relationship.members,
                selected_intents[relationship.identity],
            )
        )

    return ManualModelState(
        energy_shift=energy_shift,
        elastic_area=target.elastic_area,
        lorentzians=lorentzians,
        background=target.background,
        b0=target.b0,
        b1=target.b1,
        center_groups=tuple(center_groups),
        elastic_center_group=elastic_center_group,
        parameter_ties=tuple(parameter_ties),
    )


def _resolved_s5_parameter_intents(
    source: ManualModelState,
    target: ManualModelState,
    policy: MultiQConstraintTransferPolicy,
) -> dict[ParameterReference, ManualParameterIntent]:
    """Resolve selected categories per reference before topology can share them."""

    target_references = set(target.parameter_references())
    if not (policy.bounds | policy.fixed) <= target_references:
        raise ValueError("selected parameter constraint is absent from target model")
    resolved = {
        reference: target.parameter_intent(reference)
        for reference in target.parameter_references()
    }
    for reference in target.parameter_references():
        intent = resolved[reference]
        source_intent = source.parameter_intent(reference)
        if reference in policy.bounds:
            intent = replace(
                intent,
                user_lower_limit=source_intent.user_lower_limit,
                user_upper_limit=source_intent.user_upper_limit,
                user_bounds_enabled=source_intent.user_bounds_enabled,
            )
        if reference in policy.fixed:
            if source_intent.free:
                raise RuntimeError("validated Fixed transfer became free")
            intent = replace(
                intent,
                current_value=source_intent.current_value,
                free=False,
            )
        resolved[reference] = intent
    return resolved


def _apply_resolved_s5_parameter_intents(
    model: ManualModelState,
    resolved_intents: dict[ParameterReference, ManualParameterIntent],
) -> ManualModelState:
    """Materialize one compatible intent per final optimizer slot."""

    slots: dict[object, list[ParameterReference]] = {}
    for reference in model.parameter_references():
        slots.setdefault(_slot_key(model, reference), []).append(reference)
    updated = model
    for members in slots.values():
        if members[0].family is ParameterFamily.CENTER:
            intent = _compatible_shared_center_intent(members, resolved_intents)
        else:
            intents = {resolved_intents[member] for member in members}
            if len(intents) != 1:
                raise ValueError(
                    "target shared parameter members have incompatible resulting "
                    "Manual parameter state"
                )
            intent = intents.pop()
        updated = replace_parameter_intent(updated, members[0], intent)
    return updated


def _s5_target_model(
    source: ManualModelState,
    target: ManualModelState | None,
    policy: MultiQConstraintTransferPolicy,
) -> ManualModelState:
    """Build one fixed-composition target with only selected S5 constraints."""

    composition = _composition(source)
    base: ManualModelState | None = None
    if target is not None and _composition_compatible(target, composition):
        base = _with_branch_component_identities(
            target,
            composition,
            preserve_target_component_state=True,
        )
    if base is None:
        base = _default_model(
            composition,
            independent_lorentzian_centers=True,
        )
    resolved_intents = _resolved_s5_parameter_intents(source, base, policy)
    base = _apply_s5_center_ties(source, base, policy, resolved_intents)
    return _apply_resolved_s5_parameter_intents(base, resolved_intents)


@dataclass(frozen=True, slots=True)
class _S5PreparedTarget:
    model: ManualModelState
    materialization: ManualModelMaterialization


@dataclass(frozen=True, slots=True)
class ManualMultiQFitExecutionOutcome:
    """Authoritative branch evidence and its adopted per-Q Manual working state."""

    branch_result: MultiQBranchResult
    adopted_draft: ManualFitDraft
    snapshot: ResultSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.branch_result, MultiQBranchResult):
            raise ValueError("branch_result must be a MultiQBranchResult")
        if not isinstance(self.adopted_draft, ManualFitDraft):
            raise ValueError("adopted_draft must be a ManualFitDraft")
        if len(self.adopted_draft.setups) != len(self.branch_result.outcomes):
            raise ValueError("adopted Manual setups must match the branch Q groups")
        if not _is_issued_result_snapshot(self.snapshot) or any(
            not self.snapshot.result.point(item.group_index)._matches_execution_outcome(
                item
            )
            for item in self.branch_result.outcomes
        ):
            raise ValueError("branch execution snapshot must retain its branch Result")


def _s5_branch_configurations(
    draft: ManualFitDraft,
    context: ManualFitContext,
    source: ManualModelState,
    policy: MultiQConstraintTransferPolicy,
) -> tuple[_S5PreparedTarget | None, ...]:
    prepared: list[_S5PreparedTarget | None] = []
    for group_index, setup in enumerate(draft.setups):
        try:
            target = _s5_target_model(source, setup.model, policy)
            materialized = materialize_manual_model(
                target,
                context.prepared_resolution,
                context.selection,
                group_index,
            )
        except (ValueError, _ComponentCorrespondenceError):
            prepared.append(None)
        else:
            prepared.append(_S5PreparedTarget(target, materialized))
    return tuple(prepared)


def _execute_fit_all_q_from_current(
    project: WorkflowProject,
    draft: ManualFitDraft,
    *,
    anchor_group_index: int,
    current_fit: FitResult | None = None,
    transfer_policy: MultiQConstraintTransferPolicy | None = None,
    method: FittingMethod | None = None,
    transfer_overrides: MethodTransferOverrides | None = None,
    fit_excluded_groups: Collection[int] = (),
    derived_result_excluded_groups: Collection[int] = (),
    cancel_requested: Callable[[], bool] | None = None,
    group_started_callback: Callable[[int], object] | None = None,
    progress_callback: Callable[[MultiQFitOutcome], object] | None = None,
    max_nfev: int = 2500,
) -> tuple[MultiQBranchResult, tuple[_S5PreparedTarget | None, ...]]:
    """Execute S5 while retaining the exact target Manual state used for fitting."""

    if method is not None or transfer_overrides is not None:
        raise ValueError(
            "fit_all_q_from_current uses MultiQConstraintTransferPolicy; apply a "
            "generic Method before creating the current Result"
        )
    if current_fit is None:
        raise ValueError("Fit to all Q requires a current successful Single-Q Result")
    if not isinstance(current_fit, FitResult):
        raise ValueError("current_fit must be a FitResult")
    if not current_fit.diagnostics.optimizer_success:
        raise ValueError("Fit to all Q requires a successful current Result")
    context = _resolved_context(project, draft, anchor_group_index)
    if not fit_evidence_is_current(current_fit, context, anchor_group_index):
        raise ValueError("Current Result does not belong to the active context")
    current_fit = replace(
        current_fit,
        context_binding=FitContextBinding(
            context.prepared_resolution, context.selection, anchor_group_index
        ),
    )
    source = draft.setup(anchor_group_index).model
    if source is None:
        raise ValueError("Fit to all Q requires an active anchor Manual model")
    try:
        anchor_materialization = materialize_manual_model(
            source,
            context.prepared_resolution,
            context.selection,
            anchor_group_index,
        )
    except ValueError as error:
        raise ValueError(
            "the active anchor Manual model is not runnable in its context"
        ) from error
    if not _same_core_topology(
        current_fit.configuration,
        anchor_materialization.fit_model,
    ):
        raise ValueError(
            "current Result topology is incompatible with the active Manual model"
        )
    if transfer_policy is None:
        transfer_policy = MultiQConstraintTransferPolicy(source)
    elif not isinstance(transfer_policy, MultiQConstraintTransferPolicy):
        raise ValueError("transfer_policy must be a MultiQConstraintTransferPolicy")
    else:
        transfer_policy = MultiQConstraintTransferPolicy(
            source,
            bounds=transfer_policy.bounds,
            fixed=transfer_policy.fixed,
            center_ties=transfer_policy.center_ties,
        )
    prepared_targets = _s5_branch_configurations(
        draft,
        context,
        source,
        transfer_policy,
    )
    branch = execute_multi_q_branch(
        context.prepared_resolution,
        context.selection,
        tuple(
            target.materialization.fit_model if target is not None else None
            for target in prepared_targets
        ),
        anchor_group_index=anchor_group_index,
        anchor_fit=current_fit,
        fit_excluded_groups=fit_excluded_groups,
        derived_result_excluded_groups=derived_result_excluded_groups,
        cancel_requested=cancel_requested,
        group_started_callback=group_started_callback,
        progress_callback=progress_callback,
        max_nfev=max_nfev,
    )
    return branch, prepared_targets


def fit_all_q_from_current(
    project: WorkflowProject,
    draft: ManualFitDraft,
    *,
    anchor_group_index: int,
    current_fit: FitResult | None = None,
    transfer_policy: MultiQConstraintTransferPolicy | None = None,
    method: FittingMethod | None = None,
    transfer_overrides: MethodTransferOverrides | None = None,
    fit_excluded_groups: Collection[int] = (),
    derived_result_excluded_groups: Collection[int] = (),
    cancel_requested: Callable[[], bool] | None = None,
    group_started_callback: Callable[[int], object] | None = None,
    progress_callback: Callable[[MultiQFitOutcome], object] | None = None,
    max_nfev: int = 2500,
) -> MultiQBranchResult:
    """Retain a successful Manual Current Result and continue its branch."""

    branch, _prepared_targets = _execute_fit_all_q_from_current(
        project,
        draft,
        anchor_group_index=anchor_group_index,
        current_fit=current_fit,
        transfer_policy=transfer_policy,
        method=method,
        transfer_overrides=transfer_overrides,
        fit_excluded_groups=fit_excluded_groups,
        derived_result_excluded_groups=derived_result_excluded_groups,
        cancel_requested=cancel_requested,
        group_started_callback=group_started_callback,
        progress_callback=progress_callback,
        max_nfev=max_nfev,
    )
    return branch


def fit_all_q_from_current_with_adoption(
    project: WorkflowProject,
    draft: ManualFitDraft,
    *,
    anchor_group_index: int,
    current_fit: FitResult | None = None,
    transfer_policy: MultiQConstraintTransferPolicy | None = None,
    method: FittingMethod | None = None,
    transfer_overrides: MethodTransferOverrides | None = None,
    fit_excluded_groups: Collection[int] = (),
    derived_result_excluded_groups: Collection[int] = (),
    cancel_requested: Callable[[], bool] | None = None,
    group_started_callback: Callable[[int], object] | None = None,
    progress_callback: Callable[[MultiQFitOutcome], object] | None = None,
    max_nfev: int = 2500,
) -> ManualMultiQFitExecutionOutcome:
    """Continue a branch and adopt only successful target values into Manual state."""

    branch, prepared_targets = _execute_fit_all_q_from_current(
        project,
        draft,
        anchor_group_index=anchor_group_index,
        current_fit=current_fit,
        transfer_policy=transfer_policy,
        method=method,
        transfer_overrides=transfer_overrides,
        fit_excluded_groups=fit_excluded_groups,
        derived_result_excluded_groups=derived_result_excluded_groups,
        cancel_requested=cancel_requested,
        group_started_callback=group_started_callback,
        progress_callback=progress_callback,
        max_nfev=max_nfev,
    )
    adopted_draft = draft
    for outcome in branch.outcomes:
        if (
            outcome.group_index == anchor_group_index
            or outcome.status is not MultiQFitStatus.SUCCESS
        ):
            continue
        prepared = prepared_targets[outcome.group_index]
        if prepared is None or outcome.fit_result is None:
            raise RuntimeError("successful target lost its Manual preparation")
        adopted_draft = adopted_draft.with_model(
            outcome.group_index,
            _adopt_manual_fit_values(
                prepared.model,
                outcome.fit_result,
                prepared.materialization,
            ),
        )
    return ManualMultiQFitExecutionOutcome(
        branch,
        adopted_draft,
        _issue_result_snapshot(
            branch,
            tuple(
                adopted_draft.setup(item.group_index).model
                if item.status is MultiQFitStatus.SUCCESS
                else None
                for item in branch.outcomes
            ),
            ResultOrigin.MANUAL_FIT_ALL_Q,
            sample_identity=(draft.project_id, draft.sample_id),
        ),
    )


def _selected_auto_fit_evidence(
    outcome: SingleQAutoFitOutcome,
    selected_candidates: Sequence[StandardModelCandidate],
    context: ManualFitContext,
) -> tuple[CandidateFitResult, ...]:
    """Validate and bind all selected anchor evidence before any branch starts."""

    selected = tuple(selected_candidates)
    if not selected:
        raise ValueError("at least one AutoFit candidate must be selected")
    if any(not isinstance(item, StandardModelCandidate) for item in selected):
        raise ValueError("selected candidates must be StandardModelCandidate values")
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate AutoFit candidate selections are not allowed")
    if not _single_q_auto_fit_context_unchanged(
        outcome.scientific_context,
        context,
    ):
        raise ValueError("AutoFit evaluation does not belong to the active context")

    evidence_by_candidate: dict[StandardModelCandidate, CandidateFitResult] = {}
    for evidence in outcome.recommendation.candidate_results:
        candidate = evidence.candidate
        if candidate in evidence_by_candidate:
            raise ValueError("AutoFit evaluation contains duplicate candidate evidence")
        evidence_by_candidate[candidate] = evidence

    validated: list[CandidateFitResult] = []
    for candidate in selected:
        selected_evidence = evidence_by_candidate.get(candidate)
        if selected_evidence is None:
            raise ValueError(
                f"selected AutoFit candidate {candidate.name} is absent from the "
                "anchor evaluation"
            )
        if not selected_evidence.success or selected_evidence.fit is None:
            raise ValueError(
                f"selected AutoFit candidate {candidate.name} has no successful "
                "anchor FitResult"
            )
        active_fit = _result_in_active_context(
            selected_evidence.fit,
            context,
            context.group_index,
        )
        if (
            active_fit.provenance.group_index != context.group_index
            or active_fit.provenance.q_value
            != context.prepared_resolution.q_value(context.group_index)
        ):
            raise ValueError(
                f"selected AutoFit candidate {candidate.name} has stale anchor "
                "group or Q provenance"
            )
        validated.append(selected_evidence)
    return tuple(validated)


def continue_selected_auto_fit_candidates(
    project: WorkflowProject,
    draft: ManualFitDraft,
    outcome: SingleQAutoFitOutcome,
    selected_candidates: Sequence[StandardModelCandidate],
    *,
    fit_excluded_groups: Collection[int] = (),
    derived_result_excluded_groups: Collection[int] = (),
    cancel_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[SelectedAutoFitProgressEvent], object] | None = None,
    max_nfev: int = 2500,
) -> SelectedAutoFitMultiQResult:
    """Continue selected anchors and optionally publish candidate-tagged progress."""

    anchor_group_index = outcome.scientific_context.group_index
    context = _resolved_context(project, draft, anchor_group_index)
    evidence = _selected_auto_fit_evidence(
        outcome,
        selected_candidates,
        context,
    )
    selected = tuple(item.candidate for item in evidence)
    branches: list[AutoFitCandidateBranchResult] = []
    overall_status = MultiQExecutionStatus.COMPLETED

    for branch_index, anchor_evidence in enumerate(evidence):
        if branch_index > 0 and cancel_requested is not None and cancel_requested():
            overall_status = MultiQExecutionStatus.CANCELLED
            break
        anchor_fit = anchor_evidence.fit
        if anchor_fit is None:  # guarded by _selected_auto_fit_evidence
            raise RuntimeError("validated AutoFit evidence lost its anchor FitResult")
        anchor_fitted_model = anchor_fit.fitted_model
        if anchor_fitted_model is None:
            raise RuntimeError("validated AutoFit evidence lost its fitted model")
        active_anchor = _result_in_active_context(
            anchor_fit, context, anchor_group_index
        )
        composition = MethodComposition(
            elastic_present=True,
            lorentzian_identities=tuple(
                item.identity for item in anchor_fit.configuration.lorentzians
            ),
            background=anchor_fit.configuration.background,
        )
        working_states = tuple(
            _manual_state_from_fitted_model(anchor_fitted_model)
            if group_index == anchor_group_index
            else _default_model(composition)
            for group_index in range(len(draft.setups))
        )
        configurations: list[SpectralModelDefinition | None] = []
        for group_index, state in enumerate(working_states):
            try:
                materialized = materialize_manual_model(
                    state,
                    context.prepared_resolution,
                    context.selection,
                    group_index,
                )
            except ValueError:
                configurations.append(None)
            else:
                configurations.append(materialized.fit_model)

        current_candidate = anchor_evidence.candidate

        def publish(
            outcome: MultiQFitOutcome,
            candidate: StandardModelCandidate = current_candidate,
        ) -> None:
            if progress_callback is not None:
                progress_callback(
                    SelectedAutoFitProgressEvent(
                        candidate=candidate,
                        outcome=outcome,
                    )
                )

        branch = execute_multi_q_branch(
            context.prepared_resolution,
            context.selection,
            configurations,
            anchor_group_index=anchor_group_index,
            anchor_fit=active_anchor,
            fit_excluded_groups=fit_excluded_groups,
            derived_result_excluded_groups=derived_result_excluded_groups,
            cancel_requested=cancel_requested,
            progress_callback=publish if progress_callback is not None else None,
            max_nfev=max_nfev,
        )
        branches.append(
            _issue_auto_fit_candidate_branch_result(
                anchor_evidence,
                branch,
                outcome,
                working_states,
            )
        )
        if branch.status is MultiQExecutionStatus.CANCELLED:
            overall_status = MultiQExecutionStatus.CANCELLED
            break

    return SelectedAutoFitMultiQResult(overall_status, selected, tuple(branches))


def adopt_selected_auto_fit_branch(
    project: WorkflowProject,
    draft: ManualFitDraft,
    evaluation: SingleQAutoFitOutcome,
    selected_branch: AutoFitCandidateBranchResult,
    *,
    center_group_id: str,
) -> ManualMultiQFitExecutionOutcome:
    """Atomically adopt one executed branch owned by the exact AutoFit evaluation."""

    if not _is_issued_auto_fit_candidate_branch_result(selected_branch):
        raise ValueError("selected branch was not issued by AutoFit execution")
    if selected_branch.evaluation is not evaluation or not any(
        item is selected_branch.anchor_evidence
        for item in evaluation.recommendation.candidate_results
    ):
        raise ValueError("selected branch is not owned by this AutoFit evaluation")
    anchor = selected_branch.branch_result.anchor_group_index
    context = _resolved_context(project, draft, anchor)
    if not _single_q_auto_fit_context_unchanged(evaluation.scientific_context, context):
        raise ValueError("AutoFit evaluation no longer matches the active context")
    if len(selected_branch.working_states) != len(draft.setups):
        raise ValueError("selected branch has no complete execution working state")
    adopted = draft
    converted_outcomes: list[MultiQFitOutcome] = []
    snapshot_states: list[ManualModelState | None] = [None] * len(draft.setups)
    anchor_fit = selected_branch.anchor_evidence.fit
    assert anchor_fit is not None
    identities = {
        item.identity: ComponentIdentity(ComponentFamily.LORENTZIAN, uuid4().hex)
        for item in anchor_fit.configuration.lorentzians
    }
    for branch_outcome in selected_branch.branch_result.outcomes:
        if branch_outcome.status is not MultiQFitStatus.SUCCESS:
            converted_outcomes.append(branch_outcome)
            continue
        original_fit = branch_outcome.fit_result
        if original_fit is None:
            raise ValueError("successful branch Q has no FitResult")
        active_fit = _result_in_active_context(
            original_fit, context, branch_outcome.group_index
        )
        converted_fit = convert_standard_auto_fit_result_to_manual(
            active_fit,
            center_group_id=center_group_id,
        )
        if identities:
            converted_fit = relabel_fit_components(converted_fit, identities)
        converted_outcomes.append(replace(branch_outcome, fit_result=converted_fit))
        working = selected_branch.working_states[branch_outcome.group_index]
        if working is None:
            raise ValueError("successful branch Q lost its execution working state")
        model = _manual_state_from_standard_auto_state(
            working, center_group_id=center_group_id
        )
        if identities:
            model = _relabel_manual_state(model, identities)
        materialized = materialize_manual_model(
            model,
            context.prepared_resolution,
            context.selection,
            branch_outcome.group_index,
        )
        model = _adopt_manual_fit_values(model, converted_fit, materialized)
        adopted = adopted.with_model(branch_outcome.group_index, model)
        snapshot_states[branch_outcome.group_index] = model

    converted_branch = replace(
        selected_branch.branch_result,
        outcomes=tuple(converted_outcomes),
    )
    snapshot = _issue_result_snapshot(
        converted_branch,
        tuple(snapshot_states),
        ResultOrigin.AUTO_FIT_BRANCH,
        selected_branch.candidate,
        sample_identity=(draft.project_id, draft.sample_id),
    )
    return ManualMultiQFitExecutionOutcome(converted_branch, adopted, snapshot)


def auto_fit_all_q(
    project: WorkflowProject,
    draft: ManualFitDraft,
    *,
    anchor_group_index: int,
    fit_excluded_groups: Collection[int] = (),
    derived_result_excluded_groups: Collection[int] = (),
    cancel_requested: Callable[[], bool] | None = None,
    max_nfev: int = 2500,
) -> MultiQBranchResult:
    """Run frozen AutoFit only at the anchor and continue its Most Recommended fit."""

    outcome = run_single_q_auto_fit(
        project,
        draft,
        anchor_group_index,
        max_nfev=max_nfev,
    )
    selected = outcome.recommendation.most_recommended
    if selected is None or selected.fit is None:
        raise ValueError("AutoFit All Q requires a Most Recommended anchor Result")
    result = continue_selected_auto_fit_candidates(
        project,
        draft,
        outcome,
        (selected.candidate,),
        fit_excluded_groups=fit_excluded_groups,
        derived_result_excluded_groups=derived_result_excluded_groups,
        cancel_requested=cancel_requested,
        max_nfev=max_nfev,
    )
    return result.branches[0].branch_result
