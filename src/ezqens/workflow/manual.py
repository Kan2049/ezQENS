"""Immutable Single-Q Manual Fit application/workflow operations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import StrEnum
from uuid import uuid4

import numpy as np
import numpy.typing as npt

from ezqens.domain import DiagnosticSeverity, SpectrumRole
from ezqens.fitting import (
    BACKGROUND_COMPONENT,
    ELASTIC_COMPONENT,
    AutoFitRecommendation,
    BackgroundModel,
    CandidateFitResult,
    ComponentFamily,
    ComponentIdentity,
    FitResult,
    FittingError,
    ManualFitReadiness,
    ManualInitializationError,
    ManualMaterializationError,
    ManualModelPreview,
    ManualParameterIntent,
    ModelEvaluation,
    ParameterConfiguration,
    ParameterEstimate,
    ParameterFamily,
    ParameterReference,
    SpectralModelDefinition,
    auto_fit_single_q,
    fit_single_q,
    initialize_background_from_interaction,
    initialize_elastic_from_interaction,
    initialize_lorentzian_from_interaction,
    manual_fit_readiness,
    preview_manual_model,
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
    ProjectDataset,
    WorkflowDiagnostic,
    WorkflowDiagnosticCode,
    WorkflowError,
    WorkflowProject,
    project_dataset,
    resolve_manual_fit_context,
)


class ManualComponentKind(StrEnum):
    """Component interaction modes owned by application state."""

    ELASTIC = "elastic"
    LORENTZIAN = "lorentzian"
    BACKGROUND = "background"


@dataclass(frozen=True, slots=True)
class ManualGroupSetup:
    """One group-local optional scientific model."""

    group_index: int
    model: ManualModelState | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.group_index, bool)
            or not isinstance(self.group_index, int)
            or self.group_index < 0
        ):
            raise ValueError("group_index must be a nonnegative integer")
        if self.model is not None and not isinstance(self.model, ManualModelState):
            raise ValueError("model must be a ManualModelState or None")


@dataclass(frozen=True, slots=True)
class ManualFitDraft:
    """Application draft whose empty state is represented outside the core model."""

    project_id: str
    sample_id: str
    setups: tuple[ManualGroupSetup, ...]

    def __post_init__(self) -> None:
        setups = tuple(self.setups)
        if tuple(item.group_index for item in setups) != tuple(range(len(setups))):
            raise ValueError("Manual group setups must be ordered and contiguous")
        object.__setattr__(self, "setups", setups)

    def setup(self, group_index: int) -> ManualGroupSetup:
        """Return one group setup by stable group index."""

        if (
            isinstance(group_index, bool)
            or not isinstance(group_index, int)
            or not 0 <= group_index < len(self.setups)
        ):
            raise _workflow_error(
                WorkflowDiagnosticCode.INVALID_GROUP,
                "Fitting Parameters group index is outside the draft",
                group_index=group_index if isinstance(group_index, int) else None,
            )
        return self.setups[group_index]

    def with_model(
        self,
        group_index: int,
        model: ManualModelState | None,
    ) -> ManualFitDraft:
        """Return a draft with exactly one group-local model replaced."""

        self.setup(group_index)
        setups = list(self.setups)
        setups[group_index] = ManualGroupSetup(group_index, model)
        return replace(self, setups=tuple(setups))


@dataclass(frozen=True, slots=True)
class PendingManualInteraction:
    """An Add command awaiting visual geometry; it owns no scientific model."""

    project_id: str
    sample_id: str
    group_index: int
    component_kind: ManualComponentKind
    component_identity: ComponentIdentity


@dataclass(frozen=True, slots=True)
class PendingManualComponentPreview:
    """Directly plottable evaluation of one immutable interaction proposal."""

    component_identity: ComponentIdentity
    evaluation: ModelEvaluation

    def __post_init__(self) -> None:
        if not isinstance(self.component_identity, ComponentIdentity):
            raise ValueError("component_identity must be a ComponentIdentity")
        if not isinstance(self.evaluation, ModelEvaluation):
            raise ValueError("evaluation must be a ModelEvaluation")
        try:
            self.evaluation.component_curve(self.component_identity)
        except KeyError as error:
            raise ValueError(
                "pending evaluation must contain its provisional component"
            ) from error


@dataclass(frozen=True, slots=True)
class ManualParameterEdit:
    """Complete immutable parameter-row state supplied by the application."""

    current_value: float
    user_lower_limit: float | None
    user_upper_limit: float | None
    free: bool
    user_bounds_enabled: bool = True

    def intent(self) -> ManualParameterIntent:
        """Return the persistent editable state without materializing fit bounds."""

        return ManualParameterIntent(
            current_value=self.current_value,
            user_lower_limit=self.user_lower_limit,
            user_upper_limit=self.user_upper_limit,
            free=self.free,
            user_bounds_enabled=self.user_bounds_enabled,
        )


@dataclass(frozen=True, slots=True)
class ManualWorkflowReadiness:
    """Combined workflow context and frozen scientific-core readiness."""

    runnable: bool
    context: ManualFitContext | None
    workflow_diagnostics: tuple[WorkflowDiagnostic, ...] = ()
    scientific_readiness: ManualFitReadiness | None = None


@dataclass(frozen=True, slots=True)
class ManualFitExecutionOutcome:
    """Atomic result of running and optionally adopting one Manual fit."""

    fit_result: FitResult | None
    adopted_draft: ManualFitDraft | None
    diagnostics: tuple[WorkflowDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(item, WorkflowDiagnostic) for item in diagnostics):
            raise ValueError("diagnostics must contain WorkflowDiagnostic values")
        if self.adopted_draft is not None:
            if self.fit_result is None:
                raise ValueError("an adopted draft requires a FitResult")
            if not self.fit_result.diagnostics.optimizer_success:
                raise ValueError(
                    "a nonconverged FitResult cannot have an adopted draft"
                )
            if any(item.severity is DiagnosticSeverity.ERROR for item in diagnostics):
                raise ValueError(
                    "a successful execution cannot contain error diagnostics"
                )
        elif not any(item.severity is DiagnosticSeverity.ERROR for item in diagnostics):
            raise ValueError("a failed execution requires an error diagnostic")
        object.__setattr__(self, "diagnostics", diagnostics)

    @property
    def success(self) -> bool:
        """Return whether a complete adopted draft was produced."""

        return self.fit_result is not None and self.adopted_draft is not None


@dataclass(frozen=True, slots=True)
class SingleQAutoFitOutcome:
    """Project-bound authoritative AutoFit evidence for one Sample Group."""

    scientific_context: ManualFitContext = field(repr=False)
    recommendation: AutoFitRecommendation
    measured_energy: npt.NDArray[np.float64] = field(repr=False)
    measured_intensity: npt.NDArray[np.float64] = field(repr=False)
    measured_uncertainty: npt.NDArray[np.float64] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.scientific_context, ManualFitContext):
            raise ValueError("scientific_context must be a ManualFitContext")
        arrays = (
            ("measured_energy", self.measured_energy),
            ("measured_intensity", self.measured_intensity),
            ("measured_uncertainty", self.measured_uncertainty),
        )
        copied: list[npt.NDArray[np.float64]] = []
        for name, value in arrays:
            array = np.array(value, dtype=np.float64, copy=True)
            if array.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            array.setflags(write=False)
            copied.append(array)
        if len({array.size for array in copied}) != 1:
            raise ValueError("AutoFit measured arrays must have equal lengths")
        object.__setattr__(self, "measured_energy", copied[0])
        object.__setattr__(self, "measured_intensity", copied[1])
        object.__setattr__(self, "measured_uncertainty", copied[2])

    @property
    def project_id(self) -> str:
        """Return the owning Project identity for application routing."""

        return self.scientific_context.project_id

    @property
    def sample_id(self) -> str:
        """Return the bound Sample identity for application routing."""

        return self.scientific_context.sample.dataset_id

    @property
    def group_index(self) -> int:
        """Return the exact Group evaluated by AutoFit."""

        return self.scientific_context.group_index


def _workflow_error(
    code: WorkflowDiagnosticCode,
    message: str,
    *,
    group_index: int | None = None,
) -> WorkflowError:
    return WorkflowError(
        (
            WorkflowDiagnostic(
                code=code,
                severity=DiagnosticSeverity.ERROR,
                message=message,
                group_index=group_index,
            ),
        )
    )


def _require_draft_project(
    project: WorkflowProject,
    draft: ManualFitDraft,
) -> ProjectDataset:
    if draft.project_id != project.project_id:
        raise _workflow_error(
            WorkflowDiagnosticCode.DATASET_PROJECT_MISMATCH,
            "Manual draft belongs to a different Project",
        )
    reference = next(
        (item for item in project.datasets if item.dataset_id == draft.sample_id),
        None,
    )
    if reference is None:
        raise _workflow_error(
            WorkflowDiagnosticCode.DATASET_NOT_FOUND,
            "Manual draft Sample is not present in the Project",
        )
    return project_dataset(project, reference)


def open_manual_fit_draft(
    project: WorkflowProject,
    sample: ProjectDataset,
) -> ManualFitDraft:
    """Open an empty draft without requiring or constructing a scientific model."""

    current = project_dataset(project, sample)
    if current.dataset.role is not SpectrumRole.SAMPLE:
        raise _workflow_error(
            WorkflowDiagnosticCode.SAMPLE_ROLE_REQUIRED,
            "Fitting Parameters require a Sample dataset",
        )
    return ManualFitDraft(
        project_id=project.project_id,
        sample_id=current.dataset_id,
        setups=tuple(
            ManualGroupSetup(group_index)
            for group_index in range(len(current.dataset.spectra))
        ),
    )


def begin_component_interaction(
    draft: ManualFitDraft,
    group_index: int,
    component_kind: ManualComponentKind,
) -> PendingManualInteraction:
    """Enter interaction mode without changing any scientific configuration."""

    draft.setup(group_index)
    if not isinstance(component_kind, ManualComponentKind):
        raise ValueError("component_kind must be a ManualComponentKind")
    if component_kind is ManualComponentKind.ELASTIC:
        component_identity = ELASTIC_COMPONENT
    elif component_kind is ManualComponentKind.BACKGROUND:
        component_identity = BACKGROUND_COMPONENT
    else:
        component_identity = ComponentIdentity(
            ComponentFamily.LORENTZIAN,
            uuid4().hex,
        )
    return PendingManualInteraction(
        project_id=draft.project_id,
        sample_id=draft.sample_id,
        group_index=group_index,
        component_kind=component_kind,
        component_identity=component_identity,
    )


def _validate_pending(
    draft: ManualFitDraft,
    pending: PendingManualInteraction,
    expected: ManualComponentKind,
) -> None:
    if (
        pending.project_id != draft.project_id
        or pending.sample_id != draft.sample_id
        or pending.group_index >= len(draft.setups)
    ):
        raise _workflow_error(
            WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
            "pending component interaction does not belong to this Manual draft",
        )
    if pending.component_kind is not expected:
        raise _workflow_error(
            WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
            f"pending interaction is not for {expected.value}",
            group_index=pending.group_index,
        )


def _require_context(
    project: WorkflowProject,
    draft: ManualFitDraft,
    group_index: int,
    *,
    interaction: bool = False,
) -> ManualFitContext:
    """Resolve context while retaining the original structured blockers."""

    sample = _require_draft_project(project, draft)
    resolution = resolve_manual_fit_context(project, sample, group_index)
    if resolution.context is None:
        if not interaction:
            raise WorkflowError(resolution.diagnostics)
        outer = WorkflowDiagnostic(
            code=WorkflowDiagnosticCode.INTERACTION_CONTEXT_UNAVAILABLE,
            severity=DiagnosticSeverity.ERROR,
            message="completed component interaction requires usable Manual context",
            group_index=group_index,
        )
        raise WorkflowError((outer, *resolution.diagnostics))
    return resolution.context


def _interaction_initialization_failed(
    error: ManualInitializationError,
    pending: PendingManualInteraction,
) -> WorkflowError:
    return _workflow_error(
        WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
        str(error),
        group_index=pending.group_index,
    )


def _elastic_interaction_proposal(
    project: WorkflowProject,
    draft: ManualFitDraft,
    pending: PendingManualInteraction,
    *,
    component_peak_center: float,
    component_peak_height: float,
) -> tuple[ManualFitContext, ManualModelState]:
    """Build the one elastic proposal shared by preview and completion."""

    _validate_pending(draft, pending, ManualComponentKind.ELASTIC)
    context = _require_context(
        project,
        draft,
        pending.group_index,
        interaction=True,
    )
    try:
        seed = initialize_elastic_from_interaction(
            context.prepared_resolution,
            context.group_index,
            component_peak_center=component_peak_center,
            component_peak_height=component_peak_height,
        )
    except ManualInitializationError as error:
        raise _interaction_initialization_failed(error, pending) from error
    area = ManualParameterIntent(seed.integrated_area)
    center = ManualParameterIntent(seed.center_parameter_value)
    current = draft.setup(pending.group_index).model
    if current is None:
        model = ManualModelState(
            energy_shift=center,
            elastic_area=area,
        )
    else:
        if current.elastic_area is not None:
            raise _workflow_error(
                WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
                "Manual setup already contains an elastic component",
                group_index=pending.group_index,
            )
        if current.energy_shift is not None:
            raise _workflow_error(
                WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
                "cannot add an independent elastic center while legacy energy_shift "
                "is already used by another component",
                group_index=pending.group_index,
            )
        model = replace(current, energy_shift=center, elastic_area=area)
    return context, model


def _observed_fwhm_from_endpoint(
    component_peak_center: float,
    width_endpoint_energy: float,
    pending: PendingManualInteraction,
) -> float:
    half_width_extent = abs(width_endpoint_energy - component_peak_center)
    observed_fwhm = 2.0 * half_width_extent
    if not math.isfinite(observed_fwhm) or observed_fwhm <= 0.0:
        raise _workflow_error(
            WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
            "Lorentzian width interaction must define a finite positive extent",
            group_index=pending.group_index,
        )
    return observed_fwhm


def _lorentzian_interaction_proposal(
    project: WorkflowProject,
    draft: ManualFitDraft,
    pending: PendingManualInteraction,
    *,
    component_peak_center: float,
    component_peak_height: float,
    width_endpoint_energy: float,
) -> tuple[ManualFitContext, ManualModelState]:
    """Build the one Lorentzian proposal shared by preview and completion."""

    _validate_pending(draft, pending, ManualComponentKind.LORENTZIAN)
    observed_fwhm = _observed_fwhm_from_endpoint(
        component_peak_center,
        width_endpoint_energy,
        pending,
    )
    context = _require_context(
        project,
        draft,
        pending.group_index,
        interaction=True,
    )
    try:
        seed = initialize_lorentzian_from_interaction(
            context.prepared_resolution,
            context.group_index,
            component_peak_center=component_peak_center,
            component_peak_height=component_peak_height,
            observed_fwhm=observed_fwhm,
        )
    except ManualInitializationError as error:
        raise _interaction_initialization_failed(error, pending) from error
    component = ManualLorentzianState(
        area=ManualParameterIntent(seed.integrated_area),
        fwhm=ManualParameterIntent(seed.intrinsic_fwhm),
        center=ManualParameterIntent(seed.center_parameter_value),
        identity=pending.component_identity,
    )
    current = draft.setup(pending.group_index).model
    model = (
        ManualModelState(lorentzians=(component,))
        if current is None
        else replace(current, lorentzians=(*current.lorentzians, component))
    )
    return context, model


def _background_interaction_proposal(
    draft: ManualFitDraft,
    pending: PendingManualInteraction,
    *,
    first_energy: float,
    first_height: float,
    second_energy: float,
    second_height: float,
) -> ManualModelState:
    """Build the one B1 proposal shared by preview and completion."""

    _validate_pending(draft, pending, ManualComponentKind.BACKGROUND)
    try:
        seed = initialize_background_from_interaction(
            first_energy=first_energy,
            first_height=first_height,
            second_energy=second_energy,
            second_height=second_height,
        )
    except ManualInitializationError as error:
        raise _interaction_initialization_failed(error, pending) from error
    current = draft.setup(pending.group_index).model
    if current is not None and current.background is not BackgroundModel.NONE:
        raise _workflow_error(
            WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
            "Manual setup already contains a background component",
            group_index=pending.group_index,
        )
    b0 = ManualParameterIntent(seed.b0_initial_value)
    b1 = ManualParameterIntent(seed.b1_initial_value)
    model = (
        ManualModelState(
            background=BackgroundModel.LINEAR,
            b0=b0,
            b1=b1,
        )
        if current is None
        else replace(
            current,
            background=BackgroundModel.LINEAR,
            b0=b0,
            b1=b1,
        )
    )
    return model


def _preview_model_or_error(
    context: ManualFitContext,
    model: ManualModelState,
    *,
    display_energy: npt.ArrayLike | None,
) -> ManualModelPreview:
    """Contain deterministic scientific-preview failures at the workflow boundary."""

    materialized = _materialize_or_error(context, model)
    try:
        return preview_manual_model(
            context.prepared_resolution,
            context.selection,
            context.group_index,
            materialized.preview_model,
            display_energy=display_energy,
        )
    except (FittingError, ValueError) as error:
        raise _workflow_error(
            WorkflowDiagnosticCode.MANUAL_PREVIEW_FAILED,
            str(error),
            group_index=context.group_index,
        ) from error


def _preview_interaction_proposal(
    context: ManualFitContext,
    model: ManualModelState,
    pending: PendingManualInteraction,
    *,
    display_energy: npt.ArrayLike | None,
) -> PendingManualComponentPreview:
    preview = _preview_model_or_error(
        context,
        model,
        display_energy=display_energy,
    )
    return PendingManualComponentPreview(
        component_identity=pending.component_identity,
        evaluation=preview.display_evaluation,
    )


def preview_pending_elastic_interaction(
    project: WorkflowProject,
    draft: ManualFitDraft,
    pending: PendingManualInteraction,
    *,
    component_peak_center: float,
    component_peak_height: float,
    display_energy: npt.ArrayLike | None = None,
) -> PendingManualComponentPreview:
    """Evaluate an elastic proposal without mutating the Manual draft."""

    context, model = _elastic_interaction_proposal(
        project,
        draft,
        pending,
        component_peak_center=component_peak_center,
        component_peak_height=component_peak_height,
    )
    return _preview_interaction_proposal(
        context,
        model,
        pending,
        display_energy=display_energy,
    )


def preview_pending_lorentzian_interaction(
    project: WorkflowProject,
    draft: ManualFitDraft,
    pending: PendingManualInteraction,
    *,
    component_peak_center: float,
    component_peak_height: float,
    width_endpoint_energy: float,
    display_energy: npt.ArrayLike | None = None,
) -> PendingManualComponentPreview:
    """Evaluate a one-sided-width Lorentzian proposal without draft mutation."""

    context, model = _lorentzian_interaction_proposal(
        project,
        draft,
        pending,
        component_peak_center=component_peak_center,
        component_peak_height=component_peak_height,
        width_endpoint_energy=width_endpoint_energy,
    )
    return _preview_interaction_proposal(
        context,
        model,
        pending,
        display_energy=display_energy,
    )


def preview_pending_background_interaction(
    project: WorkflowProject,
    draft: ManualFitDraft,
    pending: PendingManualInteraction,
    *,
    first_energy: float,
    first_height: float,
    second_energy: float,
    second_height: float,
    display_energy: npt.ArrayLike | None = None,
) -> PendingManualComponentPreview:
    """Evaluate a full-domain B1 proposal without mutating the Manual draft."""

    context = _require_context(
        project,
        draft,
        pending.group_index,
        interaction=True,
    )
    model = _background_interaction_proposal(
        draft,
        pending,
        first_energy=first_energy,
        first_height=first_height,
        second_energy=second_energy,
        second_height=second_height,
    )
    return _preview_interaction_proposal(
        context,
        model,
        pending,
        display_energy=display_energy,
    )


def complete_elastic_interaction(
    project: WorkflowProject,
    draft: ManualFitDraft,
    pending: PendingManualInteraction,
    *,
    component_peak_center: float,
    component_peak_height: float,
) -> ManualFitDraft:
    """Commit the same measured-resolution-aware proposal used for preview."""

    _context, model = _elastic_interaction_proposal(
        project,
        draft,
        pending,
        component_peak_center=component_peak_center,
        component_peak_height=component_peak_height,
    )
    return draft.with_model(pending.group_index, model)


def complete_lorentzian_interaction(
    project: WorkflowProject,
    draft: ManualFitDraft,
    pending: PendingManualInteraction,
    *,
    component_peak_center: float,
    component_peak_height: float,
    width_endpoint_energy: float,
) -> ManualFitDraft:
    """Commit the same one-sided-width proposal used for preview."""

    _context, model = _lorentzian_interaction_proposal(
        project,
        draft,
        pending,
        component_peak_center=component_peak_center,
        component_peak_height=component_peak_height,
        width_endpoint_energy=width_endpoint_energy,
    )
    return draft.with_model(pending.group_index, model)


def complete_background_interaction(
    draft: ManualFitDraft,
    pending: PendingManualInteraction,
    *,
    first_energy: float,
    first_height: float,
    second_energy: float,
    second_height: float,
) -> ManualFitDraft:
    """Commit the same full-domain B1 proposal used for preview."""

    model = _background_interaction_proposal(
        draft,
        pending,
        first_energy=first_energy,
        first_height=first_height,
        second_energy=second_energy,
        second_height=second_height,
    )
    return draft.with_model(pending.group_index, model)


def _copy_intent(intent: ManualParameterIntent) -> ManualParameterIntent:
    return ManualParameterIntent(
        intent.current_value,
        intent.user_lower_limit,
        intent.user_upper_limit,
        intent.free,
        intent.user_bounds_enabled,
    )


def _lorentzian_index(
    model: ManualModelState,
    identity: ComponentIdentity,
) -> int:
    for index, component in enumerate(model.lorentzians):
        if component.identity == identity:
            return index
    raise _workflow_error(
        WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
        "parameter reference points to an unknown Lorentzian component",
    )


def update_manual_parameter(
    draft: ManualFitDraft,
    group_index: int,
    reference: ParameterReference,
    edit: ManualParameterEdit,
) -> ManualFitDraft:
    """Immutably replace one independent or explicitly shared parameter slot."""

    model = _require_model(draft, group_index)
    try:
        updated = replace_parameter_intent(
            model,
            reference,
            edit.intent(),
        )
    except ValueError as error:
        if isinstance(error, WorkflowError):
            raise
        raise _workflow_error(
            WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
            str(error),
            group_index=group_index,
        ) from error
    return draft.with_model(group_index, updated)


def _require_model(
    draft: ManualFitDraft,
    group_index: int,
) -> ManualModelState:
    model = draft.setup(group_index).model
    if model is None:
        raise _workflow_error(
            WorkflowDiagnosticCode.EMPTY_MANUAL_DRAFT,
            "Manual setup contains no scientific components",
            group_index=group_index,
        )
    return model


def _new_group_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def create_parameter_tie(
    draft: ManualFitDraft,
    group_index: int,
    members: tuple[ParameterReference, ...],
    *,
    source_member: ParameterReference,
    tie_group_id: str | None = None,
) -> ManualFitDraft:
    """Create one explicit same-family equality tie with no persistent master."""

    model = _require_model(draft, group_index)
    members = tuple(members)
    if source_member not in members:
        raise _workflow_error(
            WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
            "tie source member must be included in the selected members",
            group_index=group_index,
        )
    if any(model.tie_for(member) is not None for member in members):
        raise _workflow_error(
            WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
            "selected member already belongs to an explicit equality tie",
            group_index=group_index,
        )
    try:
        tie = ManualParameterTieState(
            tie_group_id or _new_group_id("tie"),
            members,
            _copy_intent(model.parameter_intent(source_member)),
        )
        updated = replace(model, parameter_ties=(*model.parameter_ties, tie))
    except (KeyError, ValueError) as error:
        raise _workflow_error(
            WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
            str(error),
            group_index=group_index,
        ) from error
    return draft.with_model(group_index, updated)


def join_parameter_tie(
    draft: ManualFitDraft,
    group_index: int,
    tie_group_id: str,
    member: ParameterReference,
) -> ManualFitDraft:
    """Join an existing compatible tie and adopt its current shared state."""

    model = _require_model(draft, group_index)
    if model.tie_for(member) is not None:
        raise _workflow_error(
            WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
            "parameter already belongs to an explicit equality tie",
            group_index=group_index,
        )
    group = next(
        (item for item in model.parameter_ties if item.group_id == tie_group_id),
        None,
    )
    if group is None:
        raise _workflow_error(
            WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
            "unknown parameter tie group",
            group_index=group_index,
        )
    try:
        updated_group = ManualParameterTieState(
            group.group_id,
            (*group.members, member),
            group.intent,
        )
        updated = replace(
            model,
            parameter_ties=tuple(
                updated_group if item.group_id == group.group_id else item
                for item in model.parameter_ties
            ),
        )
    except ValueError as error:
        raise _workflow_error(
            WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
            str(error),
            group_index=group_index,
        ) from error
    return draft.with_model(group_index, updated)


def untie_parameter(
    draft: ManualFitDraft,
    group_index: int,
    member: ParameterReference,
) -> ManualFitDraft:
    """Remove one membership while preserving every surviving tie member."""

    model = _require_model(draft, group_index)
    target = model.tie_for(member)
    if target is None:
        raise _workflow_error(
            WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
            "parameter is not a member of an explicit equality tie",
            group_index=group_index,
        )
    remaining = tuple(item for item in target.members if item != member)
    ties: list[ManualParameterTieState] = []
    for group in model.parameter_ties:
        if group.group_id != target.group_id:
            ties.append(group)
        elif remaining:
            ties.append(
                ManualParameterTieState(group.group_id, remaining, group.intent)
            )
    base = replace(model, parameter_ties=tuple(ties))
    updated = replace_parameter_intent(
        base,
        member,
        _copy_intent(target.intent),
    )
    return draft.with_model(group_index, updated)


def untie_center_group_parameter(
    draft: ManualFitDraft,
    group_index: int,
    member: ParameterReference,
) -> ManualFitDraft:
    """Leave one ordinary center group while preserving its shared intent."""

    model = _require_model(draft, group_index)
    target = model.center_group_for(member)
    if target is None:
        raise _workflow_error(
            WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
            "parameter is not a member of a center group",
            group_index=group_index,
        )
    remaining = tuple(
        reference
        for reference in model.center_group_members(target.group_id)
        if reference != member
    )
    center_groups = tuple(
        group
        for group in model.center_groups
        if group.group_id != target.group_id or remaining
    )
    elastic_center_group = model.elastic_center_group
    energy_shift = model.energy_shift
    lorentzians = model.lorentzians
    if member.component == ELASTIC_COMPONENT:
        existing_legacy = any(
            component.center is None and component.center_group is None
            for component in model.lorentzians
        )
        if existing_legacy:
            raise _workflow_error(
                WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
                "elastic Center cannot leave this group while another legacy "
                "shared Center is present",
                group_index=group_index,
            )
        elastic_center_group = None
        energy_shift = _copy_intent(target.intent)
    else:
        component_index = next(
            (
                index
                for index, component in enumerate(model.lorentzians)
                if component.identity == member.component
            ),
            None,
        )
        if component_index is None or member.family is not ParameterFamily.CENTER:
            raise _workflow_error(
                WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
                "center group member is not a Lorentzian Center",
                group_index=group_index,
            )
        items = list(model.lorentzians)
        items[component_index] = replace(
            items[component_index],
            center=_copy_intent(target.intent),
            center_group=None,
        )
        lorentzians = tuple(items)
    updated = ManualModelState(
        energy_shift=energy_shift,
        elastic_area=model.elastic_area,
        lorentzians=lorentzians,
        background=model.background,
        b0=model.b0,
        b1=model.b1,
        center_groups=center_groups,
        elastic_center_group=elastic_center_group,
        parameter_ties=model.parameter_ties,
    )
    return draft.with_model(group_index, updated)


def join_center_group(
    draft: ManualFitDraft,
    group_index: int,
    center_group_id: str,
    member: ParameterReference,
) -> ManualFitDraft:
    """Join an unshared Center to an existing ordinary center group."""

    model = _require_model(draft, group_index)
    if member.family is not ParameterFamily.CENTER:
        raise _workflow_error(
            WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
            "only Center parameters can join a center group",
            group_index=group_index,
        )
    if model.tie_for(member) is not None or model.center_group_for(member) is not None:
        raise _workflow_error(
            WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
            "Center parameter already belongs to a shared group",
            group_index=group_index,
        )
    if not any(group.group_id == center_group_id for group in model.center_groups):
        raise _workflow_error(
            WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
            "unknown center group",
            group_index=group_index,
        )
    elastic_center_group = model.elastic_center_group
    lorentzians = model.lorentzians
    if member.component == ELASTIC_COMPONENT:
        if model.elastic_area is None:
            raise _workflow_error(
                WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
                "elastic Center is not present",
                group_index=group_index,
            )
        elastic_center_group = center_group_id
    else:
        component_index = next(
            (
                index
                for index, component in enumerate(model.lorentzians)
                if component.identity == member.component
            ),
            None,
        )
        if component_index is None:
            raise _workflow_error(
                WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
                "Lorentzian Center is not present",
                group_index=group_index,
            )
        items = list(model.lorentzians)
        items[component_index] = replace(
            items[component_index],
            center=None,
            center_group=center_group_id,
        )
        lorentzians = tuple(items)
    legacy_center_used = (
        model.elastic_area is not None and elastic_center_group is None
    ) or any(
        component.center is None and component.center_group is None
        for component in lorentzians
    )
    updated = ManualModelState(
        energy_shift=model.energy_shift if legacy_center_used else None,
        elastic_area=model.elastic_area,
        lorentzians=lorentzians,
        background=model.background,
        b0=model.b0,
        b1=model.b1,
        center_groups=model.center_groups,
        elastic_center_group=elastic_center_group,
        parameter_ties=model.parameter_ties,
    )
    return draft.with_model(group_index, updated)


def _component_references(
    model: ManualModelState,
    identity: ComponentIdentity,
) -> set[ParameterReference]:
    return {
        reference
        for reference in model.parameter_references()
        if reference.component == identity
    }


def remove_manual_component(
    draft: ManualFitDraft,
    group_index: int,
    identity: ComponentIdentity,
) -> ManualFitDraft:
    """Remove one component and leave only valid surviving tie references."""

    model = _require_model(draft, group_index)
    removed = _component_references(model, identity)
    if not removed:
        raise _workflow_error(
            WorkflowDiagnosticCode.INVALID_MANUAL_OPERATION,
            "component identity is not present in the Manual model",
            group_index=group_index,
        )
    base = replace(model, parameter_ties=())
    surviving_ties: list[ManualParameterTieState] = []
    for group in model.parameter_ties:
        surviving_members = tuple(
            member for member in group.members if member not in removed
        )
        if surviving_members:
            surviving_ties.append(
                ManualParameterTieState(
                    group.group_id,
                    surviving_members,
                    group.intent,
                )
            )
    elastic_area = base.elastic_area
    elastic_center_group = base.elastic_center_group
    lorentzians = base.lorentzians
    background = base.background
    b0 = base.b0
    b1 = base.b1
    if identity == ELASTIC_COMPONENT:
        elastic_area = None
        elastic_center_group = None
    elif identity == BACKGROUND_COMPONENT:
        background = BackgroundModel.NONE
        b0 = None
        b1 = None
    else:
        lorentzians = tuple(
            item for item in base.lorentzians if item.identity != identity
        )
    used_center_groups = {
        item.center_group for item in lorentzians if item.center_group is not None
    }
    if elastic_center_group is not None:
        used_center_groups.add(elastic_center_group)
    center_groups = tuple(
        group for group in base.center_groups if group.group_id in used_center_groups
    )
    legacy_center_used = (
        elastic_area is not None and elastic_center_group is None
    ) or any(item.center is None and item.center_group is None for item in lorentzians)
    energy_shift = base.energy_shift if legacy_center_used else None
    if elastic_area is None and not lorentzians and background is BackgroundModel.NONE:
        updated: ManualModelState | None = None
    else:
        updated = ManualModelState(
            energy_shift=energy_shift,
            elastic_area=elastic_area,
            lorentzians=lorentzians,
            background=background,
            b0=b0,
            b1=b1,
            center_groups=center_groups,
            elastic_center_group=elastic_center_group,
            parameter_ties=tuple(surviving_ties),
        )
    return draft.with_model(group_index, updated)


def _remap_reference(
    reference: ParameterReference,
    identities: dict[ComponentIdentity, ComponentIdentity],
) -> ParameterReference:
    return ParameterReference(
        identities.get(reference.component, reference.component),
        reference.family,
    )


def _clone_model(model: ManualModelState) -> ManualModelState:
    identities = {
        component.identity: ComponentIdentity(
            ComponentFamily.LORENTZIAN,
            uuid4().hex,
        )
        for component in model.lorentzians
    }
    center_group_ids = {
        group.group_id: _new_group_id("center") for group in model.center_groups
    }
    lorentzians = tuple(
        ManualLorentzianState(
            area=_copy_intent(component.area),
            fwhm=_copy_intent(component.fwhm),
            center=(
                _copy_intent(component.center) if component.center is not None else None
            ),
            center_group=(
                center_group_ids[component.center_group]
                if component.center_group is not None
                else None
            ),
            identity=identities[component.identity],
        )
        for component in model.lorentzians
    )
    ties = tuple(
        ManualParameterTieState(
            _new_group_id("tie"),
            tuple(_remap_reference(member, identities) for member in group.members),
            _copy_intent(group.intent),
        )
        for group in model.parameter_ties
    )
    return ManualModelState(
        energy_shift=(
            _copy_intent(model.energy_shift) if model.energy_shift is not None else None
        ),
        elastic_area=(
            _copy_intent(model.elastic_area) if model.elastic_area is not None else None
        ),
        lorentzians=lorentzians,
        background=model.background,
        b0=_copy_intent(model.b0) if model.b0 is not None else None,
        b1=_copy_intent(model.b1) if model.b1 is not None else None,
        center_groups=tuple(
            ManualCenterGroupState(
                center_group_ids[group.group_id],
                _copy_intent(group.intent),
            )
            for group in model.center_groups
        ),
        elastic_center_group=(
            center_group_ids[model.elastic_center_group]
            if model.elastic_center_group is not None
            else None
        ),
        parameter_ties=ties,
    )


def _target_setup_diagnostics(
    project: WorkflowProject,
    draft: ManualFitDraft,
    group_index: int,
    model: ManualModelState,
) -> tuple[WorkflowDiagnostic, ...]:
    sample = _require_draft_project(project, draft)
    resolved = resolve_manual_fit_context(project, sample, group_index)
    if resolved.context is None:
        return tuple(
            replace(
                item,
                code=WorkflowDiagnosticCode.TARGET_SETUP_INVALID,
                group_index=group_index,
            )
            for item in resolved.diagnostics
        )
    try:
        materialized = materialize_manual_model(
            model,
            resolved.context.prepared_resolution,
            resolved.context.selection,
            group_index,
        )
    except (ManualMaterializationError, ValueError) as error:
        return (
            WorkflowDiagnostic(
                code=WorkflowDiagnosticCode.TARGET_SETUP_INVALID,
                severity=DiagnosticSeverity.ERROR,
                message=str(error),
                group_index=group_index,
            ),
        )
    readiness = manual_fit_readiness(
        resolved.context.prepared_resolution,
        resolved.context.selection,
        group_index,
        materialized.fit_model,
    )
    if readiness.runnable:
        return ()
    return tuple(
        WorkflowDiagnostic(
            code=WorkflowDiagnosticCode.TARGET_SETUP_INVALID,
            severity=item.severity,
            message=item.message,
            group_index=group_index,
        )
        for item in readiness.diagnostics
    )


def _materialize_or_error(
    context: ManualFitContext,
    model: ManualModelState,
) -> ManualModelMaterialization:
    try:
        return materialize_manual_model(
            model,
            context.prepared_resolution,
            context.selection,
            context.group_index,
        )
    except (ManualMaterializationError, ValueError) as error:
        raise _workflow_error(
            WorkflowDiagnosticCode.MANUAL_MATERIALIZATION_FAILED,
            str(error),
            group_index=context.group_index,
        ) from error


def materialize_manual_setup(
    project: WorkflowProject,
    draft: ManualFitDraft,
    group_index: int,
) -> ManualModelMaterialization:
    """Materialize one group's persistent intent for preview and fitting."""

    context = _require_context(project, draft, group_index)
    return _materialize_or_error(context, _require_model(draft, group_index))


def clone_manual_setup_to_group(
    project: WorkflowProject,
    draft: ManualFitDraft,
    source_group: int,
    target_group: int,
) -> ManualFitDraft:
    """Clone one setup with fresh group-local scientific identities."""

    source = _require_model(draft, source_group)
    draft.setup(target_group)
    if source_group == target_group:
        diagnostics = _target_setup_diagnostics(
            project,
            draft,
            target_group,
            source,
        )
        if diagnostics:
            raise WorkflowError(diagnostics)
        return draft
    clone = _clone_model(source)
    diagnostics = _target_setup_diagnostics(project, draft, target_group, clone)
    if diagnostics:
        raise WorkflowError(diagnostics)
    return draft.with_model(target_group, clone)


def apply_manual_setup_to_all_groups(
    project: WorkflowProject,
    draft: ManualFitDraft,
    source_group: int,
) -> ManualFitDraft:
    """Propose all independent group setups atomically without running a fit."""

    source = _require_model(draft, source_group)
    proposals: list[ManualModelState] = []
    diagnostics: list[WorkflowDiagnostic] = []
    for target_group in range(len(draft.setups)):
        model = source if target_group == source_group else _clone_model(source)
        proposals.append(model)
        diagnostics.extend(
            _target_setup_diagnostics(project, draft, target_group, model)
        )
    if diagnostics:
        raise WorkflowError(tuple(diagnostics))
    updated = draft
    for group_index, model in enumerate(proposals):
        updated = updated.with_model(group_index, model)
    return updated


def manual_workflow_readiness(
    project: WorkflowProject,
    draft: ManualFitDraft,
    group_index: int,
) -> ManualWorkflowReadiness:
    """Resolve context then delegate readiness to the frozen scientific core."""

    sample = _require_draft_project(project, draft)
    resolved = resolve_manual_fit_context(project, sample, group_index)
    if resolved.context is None:
        return ManualWorkflowReadiness(
            False,
            None,
            resolved.diagnostics,
            None,
        )
    model = draft.setup(group_index).model
    if model is None:
        diagnostic = WorkflowDiagnostic(
            code=WorkflowDiagnosticCode.EMPTY_MANUAL_DRAFT,
            severity=DiagnosticSeverity.ERROR,
            message="Manual setup contains no scientific components",
            group_index=group_index,
        )
        return ManualWorkflowReadiness(False, resolved.context, (diagnostic,), None)
    try:
        materialized = materialize_manual_model(
            model,
            resolved.context.prepared_resolution,
            resolved.context.selection,
            group_index,
        )
    except (ManualMaterializationError, ValueError) as error:
        diagnostic = WorkflowDiagnostic(
            code=WorkflowDiagnosticCode.MANUAL_MATERIALIZATION_FAILED,
            severity=DiagnosticSeverity.ERROR,
            message=str(error),
            group_index=group_index,
        )
        return ManualWorkflowReadiness(
            False,
            resolved.context,
            (diagnostic,),
            None,
        )
    readiness = manual_fit_readiness(
        resolved.context.prepared_resolution,
        resolved.context.selection,
        group_index,
        materialized.fit_model,
    )
    return ManualWorkflowReadiness(
        readiness.runnable,
        resolved.context,
        (),
        readiness,
    )


def preview_manual_fit(
    project: WorkflowProject,
    draft: ManualFitDraft,
    group_index: int,
    *,
    display_energy: npt.ArrayLike | None = None,
) -> ManualModelPreview:
    """Resolve application context and delegate frozen Manual preview."""

    context = _require_context(project, draft, group_index)
    model = _require_model(draft, group_index)
    return _preview_model_or_error(
        context,
        model,
        display_energy=display_energy,
    )


def run_manual_fit(
    project: WorkflowProject,
    draft: ManualFitDraft,
    group_index: int,
    *,
    max_nfev: int = 2500,
) -> FitResult:
    """Resolve application context and delegate frozen single-Q fitting."""

    context = _require_context(project, draft, group_index)
    model = _require_model(draft, group_index)
    materialized = _materialize_or_error(context, model)
    return fit_single_q(
        context.prepared_resolution,
        context.selection,
        group_index,
        materialized.fit_model,
        max_nfev=max_nfev,
    )


def run_single_q_auto_fit(
    project: WorkflowProject,
    draft: ManualFitDraft,
    group_index: int,
    *,
    max_nfev: int = 2500,
) -> SingleQAutoFitOutcome:
    """Run the public production AutoFit path for one bound Sample Group."""

    context = _require_context(project, draft, group_index)
    recommendation = auto_fit_single_q(
        context.prepared_resolution,
        context.selection,
        group_index,
        max_nfev=max_nfev,
    )
    spectrum = context.selection.dataset.spectra[group_index]
    retained = context.selection.retained_mask(group_index)
    return SingleQAutoFitOutcome(
        scientific_context=context,
        recommendation=recommendation,
        measured_energy=spectrum.energy[retained],
        measured_intensity=spectrum.intensity[retained],
        measured_uncertainty=spectrum.uncertainty[retained],
    )


def _manual_intent_from_configuration(
    configuration: ParameterConfiguration,
) -> ManualParameterIntent:
    """Preserve a core configuration as editable working-state intent."""

    return ManualParameterIntent(
        current_value=configuration.initial_value,
        user_lower_limit=(
            configuration.lower_bound
            if math.isfinite(configuration.lower_bound)
            else None
        ),
        user_upper_limit=(
            configuration.upper_bound
            if math.isfinite(configuration.upper_bound)
            else None
        ),
        free=configuration.free,
        user_bounds_enabled=True,
    )


def _manual_state_from_fitted_model(
    model: SpectralModelDefinition,
) -> ManualModelState:
    """Translate typed fitted topology without labels or scientific inference."""

    intent = _manual_intent_from_configuration
    background = model.background
    b0 = intent(model.b0) if model.b0 else None
    b1 = intent(model.b1) if model.b1 else None
    if background is BackgroundModel.CONSTANT:
        background = BackgroundModel.LINEAR
        b1 = ManualParameterIntent(current_value=0.0, free=False)
    adopted_center_groups = [
        ManualCenterGroupState(group.group_id, intent(group.parameter))
        for group in model.center_groups
    ]
    legacy_center_group: str | None = None
    if model.energy_shift is not None:
        legacy_center_group = _new_group_id("center")
        adopted_center_groups.append(
            ManualCenterGroupState(
                legacy_center_group,
                intent(model.energy_shift),
            )
        )
    return ManualModelState(
        energy_shift=None,
        elastic_area=intent(model.elastic_area) if model.elastic_area else None,
        lorentzians=tuple(
            ManualLorentzianState(
                area=intent(component.area),
                fwhm=intent(component.fwhm),
                center=intent(component.center) if component.center else None,
                center_group=(
                    component.center_group
                    if component.center is not None
                    or component.center_group is not None
                    else legacy_center_group
                ),
                identity=component.identity,
            )
            for component in model.lorentzians
        ),
        background=background,
        b0=b0,
        b1=b1,
        center_groups=tuple(adopted_center_groups),
        elastic_center_group=(
            model.elastic_center_group
            if model.elastic_center_group is not None or model.elastic_area is None
            else legacy_center_group
        ),
        parameter_ties=tuple(
            ManualParameterTieState(
                group.group_id,
                group.members,
                intent(group.parameter),
            )
            for group in model.parameter_ties
        ),
    )


def _single_q_auto_fit_context_unchanged(
    captured: ManualFitContext,
    current: ManualFitContext,
) -> bool:
    """Require the exact immutable inputs and prepared Group provenance."""

    if (
        captured.project_id != current.project_id
        or captured.sample is not current.sample
        or captured.group_index != current.group_index
        or captured.group_identity != current.group_identity
        or captured.resolution is not current.resolution
        or captured.prepared_resolution.sample_dataset
        is not current.prepared_resolution.sample_dataset
        or captured.prepared_resolution.resolution_dataset
        is not current.prepared_resolution.resolution_dataset
    ):
        return False
    group_index = captured.group_index
    return (
        captured.selection.ranges[group_index] == current.selection.ranges[group_index]
        and np.array_equal(
            captured.selection.excluded_mask(group_index),
            current.selection.excluded_mask(group_index),
        )
        and captured.prepared_resolution.acceptance_provenance(group_index)
        == current.prepared_resolution.acceptance_provenance(group_index)
    )


def adopt_single_q_auto_fit_candidate(
    project: WorkflowProject,
    draft: ManualFitDraft,
    outcome: SingleQAutoFitOutcome,
    candidate: CandidateFitResult,
    *,
    group_index: int | None = None,
) -> ManualFitExecutionOutcome:
    """Atomically adopt one exact successful AutoFit candidate into the draft."""

    outcome_group = outcome.group_index
    if (
        outcome.project_id != project.project_id
        or outcome.project_id != draft.project_id
        or outcome.sample_id != draft.sample_id
    ):
        return _failed_execution(
            WorkflowDiagnosticCode.MANUAL_FIT_ADOPTION_FAILED,
            "AutoFit evidence belongs to a different Project or Sample",
            group_index=outcome_group,
        )
    if group_index is not None and group_index != outcome_group:
        return _failed_execution(
            WorkflowDiagnosticCode.AUTO_FIT_CONTEXT_CHANGED,
            "the active Group changed after AutoFit; run AutoFit again",
            group_index=group_index,
        )
    try:
        draft.setup(outcome_group)
        current_context = _require_context(project, draft, outcome_group)
    except WorkflowError as error:
        return ManualFitExecutionOutcome(None, None, error.diagnostics)
    if not _single_q_auto_fit_context_unchanged(
        outcome.scientific_context,
        current_context,
    ):
        return _failed_execution(
            WorkflowDiagnosticCode.AUTO_FIT_CONTEXT_CHANGED,
            "the scientific context changed after AutoFit; run AutoFit again",
            group_index=outcome_group,
        )
    if not any(item is candidate for item in outcome.recommendation.candidate_results):
        return _failed_execution(
            WorkflowDiagnosticCode.MANUAL_FIT_ADOPTION_FAILED,
            "candidate is not part of this AutoFit evaluation",
            group_index=outcome_group,
        )
    fit_result = candidate.fit
    if fit_result is None or not fit_result.diagnostics.optimizer_success:
        return _failed_execution(
            WorkflowDiagnosticCode.MANUAL_FIT_ADOPTION_FAILED,
            "only a successful AutoFit candidate can be adopted",
            group_index=outcome_group,
            fit_result=fit_result,
        )
    if fit_result.provenance.group_index != outcome_group:
        return _failed_execution(
            WorkflowDiagnosticCode.MANUAL_FIT_ADOPTION_FAILED,
            "AutoFit candidate belongs to a different Group",
            group_index=outcome_group,
            fit_result=fit_result,
        )
    fitted_model = fit_result.fitted_model
    if fitted_model is None:
        return _failed_execution(
            WorkflowDiagnosticCode.MANUAL_FIT_ADOPTION_FAILED,
            "AutoFit candidate has no fitted model",
            group_index=outcome_group,
            fit_result=fit_result,
        )
    try:
        adopted_model = _manual_state_from_fitted_model(fitted_model)
        materialized = materialize_manual_setup(
            project,
            draft.with_model(outcome_group, adopted_model),
            outcome_group,
        )
        readiness = manual_fit_readiness(
            current_context.prepared_resolution,
            current_context.selection,
            outcome_group,
            materialized.fit_model,
        )
    except (ManualMaterializationError, ValueError) as error:
        return _failed_execution(
            WorkflowDiagnosticCode.MANUAL_FIT_ADOPTION_FAILED,
            str(error),
            group_index=outcome_group,
            fit_result=fit_result,
        )
    if not readiness.runnable:
        return _failed_execution(
            WorkflowDiagnosticCode.MANUAL_FIT_ADOPTION_FAILED,
            "; ".join(item.message for item in readiness.diagnostics),
            group_index=outcome_group,
            fit_result=fit_result,
        )
    return ManualFitExecutionOutcome(
        fit_result,
        draft.with_model(outcome_group, adopted_model),
    )


class _ManualFitAdoptionError(ValueError):
    """Expected validation failure while constructing an adopted Manual state."""


def _execution_diagnostic(
    code: WorkflowDiagnosticCode,
    message: str,
    *,
    group_index: int,
) -> WorkflowDiagnostic:
    return WorkflowDiagnostic(
        code=code,
        severity=DiagnosticSeverity.ERROR,
        message=message,
        group_index=group_index,
    )


def _failed_execution(
    code: WorkflowDiagnosticCode,
    message: str,
    *,
    group_index: int,
    fit_result: FitResult | None = None,
) -> ManualFitExecutionOutcome:
    return ManualFitExecutionOutcome(
        fit_result=fit_result,
        adopted_draft=None,
        diagnostics=(_execution_diagnostic(code, message, group_index=group_index),),
    )


def _readiness_execution_diagnostics(
    readiness: ManualWorkflowReadiness,
    *,
    group_index: int,
) -> tuple[WorkflowDiagnostic, ...]:
    if readiness.workflow_diagnostics:
        return readiness.workflow_diagnostics
    scientific = readiness.scientific_readiness
    if scientific is not None and scientific.diagnostics:
        return tuple(
            WorkflowDiagnostic(
                code=WorkflowDiagnosticCode.TARGET_SETUP_INVALID,
                severity=item.severity,
                message=item.message,
                group_index=group_index,
            )
            for item in scientific.diagnostics
        )
    return (
        _execution_diagnostic(
            WorkflowDiagnosticCode.MANUAL_FIT_EXECUTION_FAILED,
            "Fitting Parameters is not ready to run",
            group_index=group_index,
        ),
    )


def _materialized_reference_slots(
    materialized: ManualModelMaterialization,
) -> tuple[tuple[ParameterReference, ...], ...]:
    slots: list[list[ParameterReference]] = []
    owners: list[object] = []
    for state in materialized.parameters:
        for index, owner in enumerate(owners):
            if state.materialization is owner:
                slots[index].append(state.reference)
                break
        else:
            owners.append(state.materialization)
            slots.append([state.reference])
    return tuple(tuple(slot) for slot in slots)


def _validated_estimates_by_slot(
    fit_result: FitResult,
    materialized: ManualModelMaterialization,
) -> tuple[tuple[tuple[ParameterReference, ...], ParameterEstimate], ...]:
    expected_slots = _materialized_reference_slots(materialized)
    expected_sets = tuple(frozenset(slot) for slot in expected_slots)
    if any(not isinstance(item, ParameterEstimate) for item in fit_result.parameters):
        raise _ManualFitAdoptionError(
            "Fitting result contains an invalid parameter estimate"
        )
    actual_sets = tuple(
        frozenset(estimate.references) for estimate in fit_result.parameters
    )
    if (
        len(actual_sets) != len(expected_sets)
        or len(set(actual_sets)) != len(actual_sets)
        or set(actual_sets) != set(expected_sets)
    ):
        raise _ManualFitAdoptionError(
            "Fitting result does not provide exact parameter-reference coverage"
        )
    estimates = {
        frozenset(estimate.references): estimate for estimate in fit_result.parameters
    }
    validated: list[tuple[tuple[ParameterReference, ...], ParameterEstimate]] = []
    for slot, slot_set in zip(expected_slots, expected_sets, strict=True):
        estimate = estimates[slot_set]
        configuration = materialized.parameter(slot[0]).fit_configuration
        if not math.isfinite(estimate.value):
            raise _ManualFitAdoptionError(
                "Fitting result contains a nonfinite fitted parameter value"
            )
        if not configuration.lower_bound <= estimate.value <= configuration.upper_bound:
            raise _ManualFitAdoptionError(
                "Fitting result contains a value outside submitted fit bounds"
            )
        if estimate.free is not configuration.free:
            raise _ManualFitAdoptionError(
                "Fitting result free/fixed state differs from submitted state"
            )
        validated.append((slot, estimate))
    return tuple(validated)


def _adopt_manual_fit_values(
    model: ManualModelState,
    fit_result: FitResult,
    materialized: ManualModelMaterialization,
) -> ManualModelState:
    adopted = model
    for references, estimate in _validated_estimates_by_slot(
        fit_result,
        materialized,
    ):
        reference = references[0]
        submitted = adopted.parameter_intent(reference)
        adopted = replace_parameter_intent(
            adopted,
            reference,
            ManualParameterIntent(
                current_value=estimate.value,
                user_lower_limit=submitted.user_lower_limit,
                user_upper_limit=submitted.user_upper_limit,
                free=submitted.free,
                user_bounds_enabled=submitted.user_bounds_enabled,
            ),
        )
    return adopted


def run_and_adopt_manual_fit(
    project: WorkflowProject,
    draft: ManualFitDraft,
    group_index: int,
    *,
    max_nfev: int = 2500,
) -> ManualFitExecutionOutcome:
    """Run one Manual fit and atomically adopt its identity-mapped estimates."""

    try:
        readiness = manual_workflow_readiness(project, draft, group_index)
    except WorkflowError as error:
        return ManualFitExecutionOutcome(None, None, error.diagnostics)
    if not readiness.runnable:
        return ManualFitExecutionOutcome(
            None,
            None,
            _readiness_execution_diagnostics(readiness, group_index=group_index),
        )
    if readiness.context is None:
        raise RuntimeError("runnable Manual readiness must contain context")
    model = _require_model(draft, group_index)
    try:
        materialized = materialize_manual_model(
            model,
            readiness.context.prepared_resolution,
            readiness.context.selection,
            group_index,
        )
    except (ManualMaterializationError, ValueError) as error:
        return _failed_execution(
            WorkflowDiagnosticCode.MANUAL_FIT_EXECUTION_FAILED,
            str(error),
            group_index=group_index,
        )
    try:
        fit_result = fit_single_q(
            readiness.context.prepared_resolution,
            readiness.context.selection,
            group_index,
            materialized.fit_model,
            max_nfev=max_nfev,
        )
    except FittingError as error:
        return _failed_execution(
            WorkflowDiagnosticCode.MANUAL_FIT_EXECUTION_FAILED,
            str(error),
            group_index=group_index,
        )
    if not isinstance(fit_result, FitResult):
        return _failed_execution(
            WorkflowDiagnosticCode.MANUAL_FIT_ADOPTION_FAILED,
            "Fitting did not return a FitResult",
            group_index=group_index,
        )
    if not fit_result.diagnostics.optimizer_success:
        return _failed_execution(
            WorkflowDiagnosticCode.MANUAL_FIT_DID_NOT_CONVERGE,
            "Fitting optimizer did not converge",
            group_index=group_index,
            fit_result=fit_result,
        )
    try:
        adopted_model = _adopt_manual_fit_values(model, fit_result, materialized)
        adopted_materialized = materialize_manual_model(
            adopted_model,
            readiness.context.prepared_resolution,
            readiness.context.selection,
            group_index,
        )
    except (KeyError, ManualMaterializationError, ValueError) as error:
        return _failed_execution(
            WorkflowDiagnosticCode.MANUAL_FIT_ADOPTION_FAILED,
            str(error),
            group_index=group_index,
            fit_result=fit_result,
        )
    adopted_readiness = manual_fit_readiness(
        readiness.context.prepared_resolution,
        readiness.context.selection,
        group_index,
        adopted_materialized.fit_model,
    )
    if not adopted_readiness.runnable:
        message = "; ".join(item.message for item in adopted_readiness.diagnostics)
        return _failed_execution(
            WorkflowDiagnosticCode.MANUAL_FIT_ADOPTION_FAILED,
            message or "adopted Manual state is not runnable",
            group_index=group_index,
            fit_result=fit_result,
        )
    adopted_draft = draft.with_model(group_index, adopted_model)
    return ManualFitExecutionOutcome(fit_result, adopted_draft)
