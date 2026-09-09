"""Immutable GUI-independent project and measured-resolution workflow state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Final
from uuid import uuid4

import numpy as np

from ezqens.domain import (
    DiagnosticSeverity,
    QRebinSpecification,
    ReducedDataset,
    SpectrumRole,
)
from ezqens.preprocessing import (
    FittingSelection,
    QRebinDiagnostic,
    QRebinError,
    rebin_fractional_q,
)
from ezqens.resolution import (
    PreparedResolution,
    ResolutionDiagnostic,
    ResolutionPreparationError,
    prepare_measured_resolution,
    validate_exact_q_association,
)


class WorkflowDiagnosticCode(StrEnum):
    """Stable application/workflow blocker categories."""

    DATASET_PROJECT_MISMATCH = "dataset_project_mismatch"
    DATASET_NOT_FOUND = "dataset_not_found"
    SAMPLE_ROLE_REQUIRED = "sample_role_required"
    RESOLUTION_ROLE_REQUIRED = "resolution_role_required"
    Q_BINS_REQUIRED = "q_bins_required"
    Q_GROUP_COUNT_MISMATCH = "q_group_count_mismatch"
    Q_VALUE_MISMATCH = "q_value_mismatch"
    Q_EDGE_MISMATCH = "q_edge_mismatch"
    RESOLUTION_REPLACEMENT_CONFIRMATION_REQUIRED = (
        "resolution_replacement_confirmation_required"
    )
    NO_APPLIED_RESOLUTION = "no_applied_resolution"
    ASSOCIATED_RESOLUTION_INVALID = "associated_resolution_invalid"
    FITTING_SELECTION_UNAVAILABLE = "fitting_selection_unavailable"
    INVALID_GROUP = "invalid_group"
    RESOLUTION_PREPARATION_FAILED = "resolution_preparation_failed"
    EMPTY_MANUAL_DRAFT = "empty_manual_draft"
    MANUAL_MATERIALIZATION_FAILED = "manual_materialization_failed"
    MANUAL_PREVIEW_FAILED = "manual_preview_failed"
    MANUAL_FIT_EXECUTION_FAILED = "manual_fit_execution_failed"
    MANUAL_FIT_DID_NOT_CONVERGE = "manual_fit_did_not_converge"
    MANUAL_FIT_ADOPTION_FAILED = "manual_fit_adoption_failed"
    AUTO_FIT_CONTEXT_CHANGED = "auto_fit_context_changed"
    INTERACTION_CONTEXT_UNAVAILABLE = "interaction_context_unavailable"
    INVALID_MANUAL_OPERATION = "invalid_manual_operation"
    TARGET_SETUP_INVALID = "target_setup_invalid"
    Q_REBIN_FAILED = "q_rebin_failed"
    RESOLUTION_Q_REBIN_REPLAY_FAILED = "resolution_q_rebin_replay_failed"


@dataclass(frozen=True, slots=True)
class WorkflowDiagnostic:
    """One privacy-safe workflow blocker with optional core resolution details."""

    code: WorkflowDiagnosticCode
    severity: DiagnosticSeverity
    message: str
    group_index: int | None = None
    resolution_diagnostics: tuple[ResolutionDiagnostic, ...] = ()
    q_rebin_diagnostics: tuple[QRebinDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.code, WorkflowDiagnosticCode):
            raise ValueError("workflow diagnostic code must be typed")
        if not isinstance(self.severity, DiagnosticSeverity):
            raise ValueError("workflow diagnostic severity must be typed")
        if not self.message:
            raise ValueError("workflow diagnostic message must not be empty")
        details = tuple(self.resolution_diagnostics)
        if any(not isinstance(item, ResolutionDiagnostic) for item in details):
            raise ValueError(
                "resolution_diagnostics must contain ResolutionDiagnostic values"
            )
        object.__setattr__(self, "resolution_diagnostics", details)
        q_rebin_details = tuple(self.q_rebin_diagnostics)
        if any(not isinstance(item, QRebinDiagnostic) for item in q_rebin_details):
            raise ValueError("q_rebin_diagnostics must contain QRebinDiagnostic values")
        object.__setattr__(self, "q_rebin_diagnostics", q_rebin_details)


class WorkflowError(ValueError):
    """Raised when an immutable workflow operation is blocked."""

    diagnostics: Final[tuple[WorkflowDiagnostic, ...]]

    def __init__(self, diagnostics: tuple[WorkflowDiagnostic, ...]) -> None:
        if not diagnostics:
            raise ValueError("WorkflowError requires at least one diagnostic")
        self.diagnostics = tuple(diagnostics)
        super().__init__(
            "; ".join(f"{item.code.value}: {item.message}" for item in diagnostics)
        )


@dataclass(frozen=True, slots=True, eq=False)
class ProjectDataset:
    """One stable project-local identity resolving to an immutable dataset."""

    project_id: str
    dataset_id: str
    dataset: ReducedDataset

    def __post_init__(self) -> None:
        _validate_identifier(self.project_id, name="project_id")
        _validate_identifier(self.dataset_id, name="dataset_id")
        if not isinstance(self.dataset, ReducedDataset):
            raise ValueError("dataset must be a ReducedDataset")


@dataclass(frozen=True, slots=True)
class ResolutionAssociation:
    """One explicit Sample-level application association."""

    sample_id: str
    resolution_id: str

    def __post_init__(self) -> None:
        _validate_identifier(self.sample_id, name="sample_id")
        _validate_identifier(self.resolution_id, name="resolution_id")
        if self.sample_id == self.resolution_id:
            raise ValueError("sample and resolution identities must differ")


@dataclass(frozen=True, slots=True)
class CommittedFittingSelection:
    """One effective committed selection keyed by stable Sample identity."""

    sample_id: str
    selection: FittingSelection

    def __post_init__(self) -> None:
        _validate_identifier(self.sample_id, name="sample_id")
        if not isinstance(self.selection, FittingSelection):
            raise ValueError("selection must be a FittingSelection")


@dataclass(frozen=True, slots=True, eq=False)
class WorkflowProject:
    """Minimal immutable application project state before persistence."""

    project_id: str
    name: str
    datasets: tuple[ProjectDataset, ...] = ()
    resolution_associations: tuple[ResolutionAssociation, ...] = ()
    fitting_selections: tuple[CommittedFittingSelection, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.project_id, name="project_id")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("project name must be a nonempty string")
        datasets = tuple(self.datasets)
        associations = tuple(self.resolution_associations)
        selections = tuple(self.fitting_selections)
        if any(
            not isinstance(item, ProjectDataset) or item.project_id != self.project_id
            for item in datasets
        ):
            raise ValueError("all project datasets must belong to this project")
        dataset_ids = tuple(item.dataset_id for item in datasets)
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError("project dataset identities must be unique")
        if any(not isinstance(item, ResolutionAssociation) for item in associations):
            raise ValueError("resolution associations must be typed")
        if len({item.sample_id for item in associations}) != len(associations):
            raise ValueError("a Sample may have at most one applied Resolution")
        if any(not isinstance(item, CommittedFittingSelection) for item in selections):
            raise ValueError("committed fitting selections must be typed")
        if len({item.sample_id for item in selections}) != len(selections):
            raise ValueError(
                "a Sample may have at most one committed fitting selection"
            )
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "resolution_associations", associations)
        object.__setattr__(self, "fitting_selections", selections)


@dataclass(frozen=True, slots=True)
class ResolutionApplyPreflight:
    """State needed to request replacement confirmation without mutating mapping."""

    sample: ProjectDataset
    proposed_resolution: ProjectDataset
    existing_resolution: ProjectDataset | None
    replacement_confirmation_required: bool
    already_applied: bool


@dataclass(frozen=True, slots=True)
class ManualFitContext:
    """Resolved scientific inputs for one Sample Q group."""

    project_id: str
    sample: ProjectDataset
    group_index: int
    group_identity: str
    selection: FittingSelection
    resolution: ProjectDataset
    prepared_resolution: PreparedResolution


@dataclass(frozen=True, slots=True)
class ManualFitContextResolution:
    """Structured context resolution that never invents a resolution fallback."""

    context: ManualFitContext | None
    diagnostics: tuple[WorkflowDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        diagnostics = tuple(self.diagnostics)
        if self.context is None and not diagnostics:
            raise ValueError("unresolved Manual context requires diagnostics")
        if self.context is not None and any(
            item.severity is DiagnosticSeverity.ERROR for item in diagnostics
        ):
            raise ValueError("resolved Manual context cannot contain error diagnostics")
        object.__setattr__(self, "diagnostics", diagnostics)

    @property
    def runnable(self) -> bool:
        """Return whether scientific Manual operations have all context inputs."""

        return self.context is not None


def _validate_identifier(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a canonical nonempty string")


def _new_identifier(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def create_project(
    name: str,
    *,
    project_id: str | None = None,
) -> WorkflowProject:
    """Create an empty immutable application project."""

    return WorkflowProject(
        project_id=project_id or _new_identifier("project"),
        name=name,
    )


def _error(
    code: WorkflowDiagnosticCode,
    message: str,
    *,
    group_index: int | None = None,
    resolution_diagnostics: tuple[ResolutionDiagnostic, ...] = (),
    q_rebin_diagnostics: tuple[QRebinDiagnostic, ...] = (),
) -> WorkflowError:
    return WorkflowError(
        (
            WorkflowDiagnostic(
                code=code,
                severity=DiagnosticSeverity.ERROR,
                message=message,
                group_index=group_index,
                resolution_diagnostics=resolution_diagnostics,
                q_rebin_diagnostics=q_rebin_diagnostics,
            ),
        )
    )


def _current_dataset(
    project: WorkflowProject,
    reference: ProjectDataset,
) -> ProjectDataset:
    if not isinstance(project, WorkflowProject):
        raise ValueError("project must be a WorkflowProject")
    if not isinstance(reference, ProjectDataset):
        raise _error(
            WorkflowDiagnosticCode.DATASET_NOT_FOUND,
            "dataset reference must be a ProjectDataset",
        )
    if reference.project_id != project.project_id:
        raise _error(
            WorkflowDiagnosticCode.DATASET_PROJECT_MISMATCH,
            "dataset belongs to a different Project",
        )
    for item in project.datasets:
        if item.dataset_id == reference.dataset_id:
            return item
    raise _error(
        WorkflowDiagnosticCode.DATASET_NOT_FOUND,
        "dataset is not present in the Project",
    )


def add_project_dataset(
    project: WorkflowProject,
    dataset: ReducedDataset,
    *,
    dataset_id: str | None = None,
) -> tuple[WorkflowProject, ProjectDataset]:
    """Add one immutable dataset under a stable project-local identity."""

    item = ProjectDataset(
        project_id=project.project_id,
        dataset_id=dataset_id or _new_identifier("dataset"),
        dataset=dataset,
    )
    if any(existing.dataset_id == item.dataset_id for existing in project.datasets):
        raise ValueError("dataset_id is already present in the Project")
    return replace(project, datasets=(*project.datasets, item)), item


def project_dataset(
    project: WorkflowProject,
    reference: ProjectDataset,
) -> ProjectDataset:
    """Resolve the current immutable dataset replacement by stable identity."""

    return _current_dataset(project, reference)


def _rebind_selection(
    committed: CommittedFittingSelection,
    replacement: ReducedDataset,
) -> CommittedFittingSelection | None:
    if not _measured_spectra_exactly_unchanged(
        committed.selection.dataset,
        replacement,
    ):
        return None
    try:
        rebound = replace(committed.selection, dataset=replacement)
    except ValueError:
        return None
    return CommittedFittingSelection(committed.sample_id, rebound)


def _same_array(first: np.ndarray, second: np.ndarray) -> bool:
    return bool(np.array_equal(first, second, equal_nan=True))


def _measured_spectra_exactly_unchanged(
    original: ReducedDataset,
    replacement: ReducedDataset,
) -> bool:
    """Return whether a point-indexed selection can safely bind to replacement."""

    if original.role is not replacement.role:
        return False
    if not measured_point_correspondence_exactly_unchanged(original, replacement):
        return False
    for before, after in zip(original.spectra, replacement.spectra, strict=True):
        if (
            before.role is not after.role
            or before.energy_unit != after.energy_unit
            or before.intensity_unit != after.intensity_unit
            or before.uncertainty_unit != after.uncertainty_unit
        ):
            return False
    return True


def measured_point_correspondence_exactly_unchanged(
    original: ReducedDataset,
    replacement: ReducedDataset,
) -> bool:
    """Return whether point-indexed masks retain exact measured correspondence."""

    if len(original.spectra) != len(replacement.spectra):
        return False
    for before, after in zip(original.spectra, replacement.spectra, strict=True):
        if (
            before.group_index != after.group_index
            or before.group_label != after.group_label
            or not _same_array(before.energy, after.energy)
            or not _same_array(before.intensity, after.intensity)
            or not _same_array(before.uncertainty, after.uncertainty)
        ):
            return False
    return True


def _rebind_fractional_coverage(
    original: ReducedDataset,
    replacement: ReducedDataset,
) -> ReducedDataset:
    """Preserve or invalidate coverage according to exact X/Y/E correspondence."""

    correspondence_unchanged = measured_point_correspondence_exactly_unchanged(
        original,
        replacement,
    )
    if correspondence_unchanged:
        if (
            replacement.fractional_coverage is None
            and original.fractional_coverage is not None
        ):
            return replace(
                replacement,
                fractional_coverage=original.fractional_coverage,
            )
        return replacement
    if (
        original.fractional_coverage is not None
        and replacement.fractional_coverage is original.fractional_coverage
    ):
        return replace(replacement, fractional_coverage=None)
    return replacement


def replace_project_dataset(
    project: WorkflowProject,
    reference: ProjectDataset,
    replacement: ReducedDataset,
) -> tuple[WorkflowProject, ProjectDataset]:
    """Replace immutable dataset content while preserving its project identity."""

    current = _current_dataset(project, reference)
    replacement = _rebind_fractional_coverage(current.dataset, replacement)
    updated = ProjectDataset(project.project_id, current.dataset_id, replacement)
    datasets = tuple(
        updated if item.dataset_id == current.dataset_id else item
        for item in project.datasets
    )
    role_changed = current.dataset.role is not replacement.role
    associations = (
        tuple(
            item
            for item in project.resolution_associations
            if item.sample_id != current.dataset_id
            and item.resolution_id != current.dataset_id
        )
        if role_changed
        else project.resolution_associations
    )
    selections: list[CommittedFittingSelection] = []
    for committed in project.fitting_selections:
        if committed.sample_id != current.dataset_id:
            selections.append(committed)
            continue
        if replacement.role is not SpectrumRole.SAMPLE:
            continue
        rebound = _rebind_selection(committed, replacement)
        if rebound is not None:
            selections.append(rebound)
    return (
        replace(
            project,
            datasets=datasets,
            resolution_associations=associations,
            fitting_selections=tuple(selections),
        ),
        updated,
    )


def remove_project_dataset(
    project: WorkflowProject,
    reference: ProjectDataset,
) -> WorkflowProject:
    """Remove a dataset and every incoming/outgoing workflow association."""

    current = _current_dataset(project, reference)
    return replace(
        project,
        datasets=tuple(
            item for item in project.datasets if item.dataset_id != current.dataset_id
        ),
        resolution_associations=tuple(
            item
            for item in project.resolution_associations
            if item.sample_id != current.dataset_id
            and item.resolution_id != current.dataset_id
        ),
        fitting_selections=tuple(
            item
            for item in project.fitting_selections
            if item.sample_id != current.dataset_id
        ),
    )


def rerole_project_dataset(
    project: WorkflowProject,
    reference: ProjectDataset,
    role: SpectrumRole,
) -> tuple[WorkflowProject, ProjectDataset]:
    """Return a validated role replacement and invalidate affected mappings."""

    current = _current_dataset(project, reference)
    if not isinstance(role, SpectrumRole):
        raise ValueError("role must be a SpectrumRole")
    spectra = tuple(replace(item, role=role) for item in current.dataset.spectra)
    replacement = replace(current.dataset, role=role, spectra=spectra)
    return replace_project_dataset(project, current, replacement)


def commit_fitting_selection(
    project: WorkflowProject,
    sample: ProjectDataset,
    selection: FittingSelection,
) -> WorkflowProject:
    """Commit one effective selection under stable Sample identity."""

    current = _current_dataset(project, sample)
    if current.dataset.role is not SpectrumRole.SAMPLE:
        raise _error(
            WorkflowDiagnosticCode.SAMPLE_ROLE_REQUIRED,
            "only a Sample may own a fitting selection",
        )
    if selection.dataset is not current.dataset:
        raise _error(
            WorkflowDiagnosticCode.FITTING_SELECTION_UNAVAILABLE,
            "fitting selection must reference the current immutable Sample dataset",
        )
    committed = CommittedFittingSelection(current.dataset_id, selection)
    return replace(
        project,
        fitting_selections=(
            *(
                item
                for item in project.fitting_selections
                if item.sample_id != current.dataset_id
            ),
            committed,
        ),
    )


def applied_resolution(
    project: WorkflowProject,
    sample: ProjectDataset,
) -> ProjectDataset | None:
    """Return the explicitly applied Resolution, never an inferred candidate."""

    current = _current_dataset(project, sample)
    association = next(
        (
            item
            for item in project.resolution_associations
            if item.sample_id == current.dataset_id
        ),
        None,
    )
    if association is None:
        return None
    return next(
        (
            item
            for item in project.datasets
            if item.dataset_id == association.resolution_id
        ),
        None,
    )


def preflight_apply_resolution(
    project: WorkflowProject,
    sample: ProjectDataset,
    resolution: ProjectDataset,
) -> ResolutionApplyPreflight:
    """Expose overwrite state before proposed Resolution validation."""

    current_sample = _current_dataset(project, sample)
    proposed = _current_dataset(project, resolution)
    existing = applied_resolution(project, current_sample)
    return ResolutionApplyPreflight(
        sample=current_sample,
        proposed_resolution=proposed,
        existing_resolution=existing,
        replacement_confirmation_required=(
            existing is not None and existing.dataset_id != proposed.dataset_id
        ),
        already_applied=(
            existing is not None and existing.dataset_id == proposed.dataset_id
        ),
    )


_Q_CODE_MAP = {
    "sample_role_required": WorkflowDiagnosticCode.SAMPLE_ROLE_REQUIRED,
    "resolution_role_required": WorkflowDiagnosticCode.RESOLUTION_ROLE_REQUIRED,
    "q_bins_required": WorkflowDiagnosticCode.Q_BINS_REQUIRED,
    "q_group_count_mismatch": WorkflowDiagnosticCode.Q_GROUP_COUNT_MISMATCH,
    "q_value_mismatch": WorkflowDiagnosticCode.Q_VALUE_MISMATCH,
    "q_edge_mismatch": WorkflowDiagnosticCode.Q_EDGE_MISMATCH,
}


def _association_error(error: ResolutionPreparationError) -> WorkflowError:
    first = error.diagnostics[0]
    return _error(
        _Q_CODE_MAP.get(
            first.code,
            WorkflowDiagnosticCode.RESOLUTION_PREPARATION_FAILED,
        ),
        first.message,
        group_index=first.group_index,
        resolution_diagnostics=error.diagnostics,
    )


def _q_rebin_error(
    error: QRebinError,
    *,
    replay: bool = False,
) -> WorkflowError:
    first = error.diagnostics[0]
    return _error(
        (
            WorkflowDiagnosticCode.RESOLUTION_Q_REBIN_REPLAY_FAILED
            if replay
            else WorkflowDiagnosticCode.Q_REBIN_FAILED
        ),
        first.message,
        group_index=first.target_group_index,
        q_rebin_diagnostics=error.diagnostics,
    )


def _replay_q_rebin_history(
    resolution: ReducedDataset,
    history: tuple[QRebinSpecification, ...],
) -> ReducedDataset:
    coverage = resolution.fractional_coverage
    completed = () if coverage is None else coverage.q_rebin_history
    if len(completed) > len(history) or completed != history[: len(completed)]:
        raise _error(
            WorkflowDiagnosticCode.RESOLUTION_Q_REBIN_REPLAY_FAILED,
            "Resolution Q-rebin history is not compatible with the active Sample",
        )
    replayed = resolution
    for specification in history[len(completed) :]:
        try:
            replayed = rebin_fractional_q(replayed, specification)
        except QRebinError as error:
            raise _q_rebin_error(error, replay=True) from error
    return replayed


def _resolution_is_shared_with_another_sample(
    project: WorkflowProject,
    resolution_id: str,
    sample_id: str,
) -> bool:
    return any(
        association.resolution_id == resolution_id
        and association.sample_id != sample_id
        for association in project.resolution_associations
    )


def _replace_or_branch_rebinned_resolution(
    project: WorkflowProject,
    sample_id: str,
    resolution: ProjectDataset,
    rebinned: ReducedDataset,
) -> tuple[WorkflowProject, ProjectDataset]:
    if not _resolution_is_shared_with_another_sample(
        project,
        resolution.dataset_id,
        sample_id,
    ):
        return replace_project_dataset(project, resolution, rebinned)
    updated_project, branched = add_project_dataset(project, rebinned)
    association = ResolutionAssociation(sample_id, branched.dataset_id)
    return (
        replace(
            updated_project,
            resolution_associations=(
                *(
                    item
                    for item in updated_project.resolution_associations
                    if item.sample_id != sample_id
                ),
                association,
            ),
        ),
        branched,
    )


def rebin_project_sample_q(
    project: WorkflowProject,
    sample: ProjectDataset,
    specification: QRebinSpecification,
) -> tuple[WorkflowProject, ProjectDataset, ProjectDataset | None]:
    """Rebin a Sample and its applied Resolution as one immutable transaction."""

    current_sample = _current_dataset(project, sample)
    if current_sample.dataset.role is not SpectrumRole.SAMPLE:
        raise _error(
            WorkflowDiagnosticCode.SAMPLE_ROLE_REQUIRED,
            "only a Sample dataset may start paired Q rebinning",
        )
    current_resolution = applied_resolution(project, current_sample)
    try:
        rebinned_sample_data = rebin_fractional_q(
            current_sample.dataset,
            specification,
        )
        rebinned_resolution_data = (
            None
            if current_resolution is None
            else rebin_fractional_q(current_resolution.dataset, specification)
        )
    except QRebinError as error:
        raise _q_rebin_error(error) from error

    if rebinned_resolution_data is not None:
        try:
            validate_exact_q_association(
                rebinned_sample_data,
                rebinned_resolution_data,
            )
        except ResolutionPreparationError as error:
            raise _association_error(error) from error

    updated_project, updated_sample = replace_project_dataset(
        project,
        current_sample,
        rebinned_sample_data,
    )
    updated_resolution: ProjectDataset | None = None
    if current_resolution is not None and rebinned_resolution_data is not None:
        updated_project, updated_resolution = _replace_or_branch_rebinned_resolution(
            updated_project,
            updated_sample.dataset_id,
            current_resolution,
            rebinned_resolution_data,
        )
    return updated_project, updated_sample, updated_resolution


def apply_resolution(
    project: WorkflowProject,
    sample: ProjectDataset,
    resolution: ProjectDataset,
    *,
    replace_confirmed: bool = False,
) -> WorkflowProject:
    """Explicitly apply one exact-Q Resolution with transactional replacement."""

    preflight = preflight_apply_resolution(project, sample, resolution)
    if preflight.already_applied:
        return project
    if preflight.replacement_confirmation_required and not replace_confirmed:
        raise _error(
            WorkflowDiagnosticCode.RESOLUTION_REPLACEMENT_CONFIRMATION_REQUIRED,
            "an applied Resolution already exists; explicit replacement "
            "confirmation is required before validating a new Resolution",
        )
    proposed_resolution = preflight.proposed_resolution
    sample_coverage = preflight.sample.dataset.fractional_coverage
    history = () if sample_coverage is None else sample_coverage.q_rebin_history
    if history:
        replayed = _replay_q_rebin_history(proposed_resolution.dataset, history)
        if replayed is not proposed_resolution.dataset:
            project, proposed_resolution = _replace_or_branch_rebinned_resolution(
                project,
                preflight.sample.dataset_id,
                proposed_resolution,
                replayed,
            )
    try:
        validate_exact_q_association(
            preflight.sample.dataset,
            proposed_resolution.dataset,
        )
    except ResolutionPreparationError as error:
        raise _association_error(error) from error
    association = ResolutionAssociation(
        preflight.sample.dataset_id,
        proposed_resolution.dataset_id,
    )
    return replace(
        project,
        resolution_associations=(
            *(
                item
                for item in project.resolution_associations
                if item.sample_id != preflight.sample.dataset_id
            ),
            association,
        ),
    )


def _selection_for(
    project: WorkflowProject,
    sample_id: str,
) -> FittingSelection | None:
    return next(
        (
            item.selection
            for item in project.fitting_selections
            if item.sample_id == sample_id
        ),
        None,
    )


def _context_failure(
    code: WorkflowDiagnosticCode,
    message: str,
    *,
    group_index: int | None = None,
    resolution_diagnostics: tuple[ResolutionDiagnostic, ...] = (),
) -> ManualFitContextResolution:
    return ManualFitContextResolution(
        None,
        (
            WorkflowDiagnostic(
                code=code,
                severity=DiagnosticSeverity.ERROR,
                message=message,
                group_index=group_index,
                resolution_diagnostics=resolution_diagnostics,
            ),
        ),
    )


def resolve_manual_fit_context(
    project: WorkflowProject,
    sample: ProjectDataset,
    group_index: int,
) -> ManualFitContextResolution:
    """Resolve one Sample group to existing frozen scientific fitting inputs."""

    try:
        current_sample = _current_dataset(project, sample)
    except WorkflowError as error:
        return ManualFitContextResolution(None, error.diagnostics)
    if current_sample.dataset.role is not SpectrumRole.SAMPLE:
        return _context_failure(
            WorkflowDiagnosticCode.SAMPLE_ROLE_REQUIRED,
            "Fitting Parameters require a Sample dataset",
            group_index=group_index if isinstance(group_index, int) else None,
        )
    if (
        isinstance(group_index, bool)
        or not isinstance(group_index, int)
        or not 0 <= group_index < len(current_sample.dataset.spectra)
    ):
        return _context_failure(
            WorkflowDiagnosticCode.INVALID_GROUP,
            "Fitting Parameters group index is outside the Sample dataset",
            group_index=group_index if isinstance(group_index, int) else None,
        )
    selection = _selection_for(project, current_sample.dataset_id)
    if selection is None or selection.dataset is not current_sample.dataset:
        return _context_failure(
            WorkflowDiagnosticCode.FITTING_SELECTION_UNAVAILABLE,
            "a committed fitting selection for the current Sample is unavailable",
            group_index=group_index,
        )
    association = next(
        (
            item
            for item in project.resolution_associations
            if item.sample_id == current_sample.dataset_id
        ),
        None,
    )
    if association is None:
        return _context_failure(
            WorkflowDiagnosticCode.NO_APPLIED_RESOLUTION,
            "no Resolution has been explicitly applied to this Sample",
            group_index=group_index,
        )
    resolution = next(
        (
            item
            for item in project.datasets
            if item.dataset_id == association.resolution_id
        ),
        None,
    )
    if resolution is None or resolution.dataset.role is not SpectrumRole.RESOLUTION:
        return _context_failure(
            WorkflowDiagnosticCode.ASSOCIATED_RESOLUTION_INVALID,
            "the applied Resolution is missing or no longer has Resolution role",
            group_index=group_index,
        )
    try:
        prepared = prepare_measured_resolution(
            current_sample.dataset,
            resolution.dataset,
        )
    except ResolutionPreparationError as error:
        return _context_failure(
            WorkflowDiagnosticCode.RESOLUTION_PREPARATION_FAILED,
            "the applied Resolution could not be prepared",
            group_index=group_index,
            resolution_diagnostics=error.diagnostics,
        )
    spectrum = current_sample.dataset.spectra[group_index]
    return ManualFitContextResolution(
        ManualFitContext(
            project_id=project.project_id,
            sample=current_sample,
            group_index=group_index,
            group_identity=spectrum.group_label,
            selection=selection,
            resolution=resolution,
            prepared_resolution=prepared,
        )
    )
