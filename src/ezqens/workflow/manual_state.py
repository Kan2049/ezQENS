"""Immutable Manual Fit intent state and context-local materialization."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from uuid import uuid4

import numpy as np

from ezqens.fitting import (
    BACKGROUND_COMPONENT,
    ELASTIC_COMPONENT,
    BackgroundModel,
    CenterGroup,
    ComponentFamily,
    ComponentIdentity,
    LorentzianComponent,
    ManualParameterIntent,
    ManualParameterKind,
    ManualParameterMaterialization,
    ParameterConfiguration,
    ParameterFamily,
    ParameterReference,
    ParameterTieGroup,
    SpectralModelDefinition,
    materialize_manual_parameter,
)
from ezqens.preprocessing import FittingSelection
from ezqens.resolution import PreparedResolution


def _validate_group_id(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a canonical nonempty string")


def _kind(reference: ParameterReference) -> ManualParameterKind:
    if reference.family is ParameterFamily.AREA:
        return ManualParameterKind.NONNEGATIVE_AREA
    if reference.family is ParameterFamily.FWHM:
        return ManualParameterKind.POSITIVE_FWHM
    if reference.family is ParameterFamily.CENTER:
        return ManualParameterKind.CENTER
    return ManualParameterKind.UNCONSTRAINED


def _structural_configuration(
    intent: ManualParameterIntent,
    kind: ManualParameterKind,
) -> ParameterConfiguration:
    """Validate context-free intent fields and build an ephemeral topology value."""

    if not isinstance(intent, ManualParameterIntent):
        raise ValueError("parameter state must be a ManualParameterIntent")
    values = (
        intent.current_value,
        intent.user_lower_limit,
        intent.user_upper_limit,
    )
    for value in values:
        if value is not None and (
            isinstance(value, bool) or not np.isfinite(float(value))
        ):
            raise ValueError(
                "Manual parameter values and supplied limits must be finite"
            )
    if (
        intent.user_lower_limit is not None
        and intent.user_upper_limit is not None
        and intent.user_lower_limit > intent.user_upper_limit
    ):
        raise ValueError("user lower limit must not exceed user upper limit")
    if not isinstance(intent.free, bool):
        raise ValueError("Manual parameter free state must be boolean")
    current = float(intent.current_value)
    lower = -np.inf
    if kind is ManualParameterKind.NONNEGATIVE_AREA:
        if current < 0.0:
            raise ValueError("Manual area current value must be nonnegative")
        lower = 0.0
    elif kind is ManualParameterKind.POSITIVE_FWHM:
        if current <= 0.0:
            raise ValueError("Manual FWHM current value must be strictly positive")
        lower = float(np.nextafter(0.0, 1.0))
    return ParameterConfiguration(current, lower, np.inf, intent.free)


@dataclass(frozen=True, slots=True)
class ManualCenterGroupState:
    """One center-group identity owning one editable Manual intent."""

    group_id: str
    intent: ManualParameterIntent

    def __post_init__(self) -> None:
        _validate_group_id(self.group_id, name="center group_id")
        _structural_configuration(self.intent, ManualParameterKind.CENTER)


@dataclass(frozen=True, slots=True)
class ManualParameterTieState:
    """One explicit same-family equality tie owning one shared Manual intent."""

    group_id: str
    members: tuple[ParameterReference, ...]
    intent: ManualParameterIntent

    def __post_init__(self) -> None:
        _validate_group_id(self.group_id, name="parameter tie group_id")
        members = tuple(self.members)
        if len(members) < 2:
            raise ValueError("parameter tie group requires at least two members")
        if any(not isinstance(item, ParameterReference) for item in members):
            raise ValueError("parameter tie members must be ParameterReference values")
        if len(set(members)) != len(members):
            raise ValueError("parameter tie members must be unique")
        if len({item.family for item in members}) != 1:
            raise ValueError("parameter tie members must have the same family")
        _structural_configuration(self.intent, _kind(members[0]))
        object.__setattr__(self, "members", members)

    @property
    def family(self) -> ParameterFamily:
        """Return the shared typed parameter family."""

        return self.members[0].family


@dataclass(frozen=True, slots=True)
class ManualLorentzianState:
    """One identity-bearing Lorentzian with editable Manual intents."""

    area: ManualParameterIntent
    fwhm: ManualParameterIntent
    center: ManualParameterIntent | None = None
    center_group: str | None = None
    identity: ComponentIdentity = field(
        default_factory=lambda: ComponentIdentity(
            ComponentFamily.LORENTZIAN,
            uuid4().hex,
        )
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.identity, ComponentIdentity)
            or self.identity.family is not ComponentFamily.LORENTZIAN
        ):
            raise ValueError("Lorentzian identity must have Lorentzian family")
        _structural_configuration(self.area, ManualParameterKind.NONNEGATIVE_AREA)
        _structural_configuration(self.fwhm, ManualParameterKind.POSITIVE_FWHM)
        if self.center is not None:
            _structural_configuration(self.center, ManualParameterKind.CENTER)
        if self.center is not None and self.center_group is not None:
            raise ValueError(
                "Lorentzian center and center_group are mutually exclusive"
            )
        if self.center_group is not None:
            _validate_group_id(self.center_group, name="Lorentzian center_group")


@dataclass(frozen=True, slots=True)
class ManualParameterStateMaterialization:
    """One reference mapped to its shared or independent materialization."""

    reference: ParameterReference
    materialization: ManualParameterMaterialization


@dataclass(frozen=True, slots=True)
class ManualModelMaterialization:
    """Context-local preview and fit models derived from one immutable intent model."""

    preview_model: SpectralModelDefinition
    fit_model: SpectralModelDefinition
    parameters: tuple[ManualParameterStateMaterialization, ...]

    def parameter(
        self,
        reference: ParameterReference,
    ) -> ManualParameterMaterialization:
        """Return one reference's materialization, including shared slots."""

        for item in self.parameters:
            if item.reference == reference:
                return item.materialization
        raise KeyError(reference)


@dataclass(frozen=True, slots=True)
class ManualModelState:
    """Application-owned model topology whose editable values are Manual intents."""

    energy_shift: ManualParameterIntent | None = None
    elastic_area: ManualParameterIntent | None = None
    lorentzians: tuple[ManualLorentzianState, ...] = ()
    background: BackgroundModel = BackgroundModel.NONE
    b0: ManualParameterIntent | None = None
    b1: ManualParameterIntent | None = None
    center_groups: tuple[ManualCenterGroupState, ...] = ()
    elastic_center_group: str | None = None
    parameter_ties: tuple[ManualParameterTieState, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "lorentzians", tuple(self.lorentzians))
        object.__setattr__(self, "center_groups", tuple(self.center_groups))
        object.__setattr__(self, "parameter_ties", tuple(self.parameter_ties))
        _structural_model(self)

    def parameter_references(self) -> tuple[ParameterReference, ...]:
        """Return stable component-owned references in display order."""

        references: list[ParameterReference] = []
        if self.elastic_area is not None:
            references.extend(
                (
                    ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA),
                    ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER),
                )
            )
        for component in self.lorentzians:
            references.extend(
                (
                    ParameterReference(component.identity, ParameterFamily.AREA),
                    ParameterReference(component.identity, ParameterFamily.FWHM),
                    ParameterReference(component.identity, ParameterFamily.CENTER),
                )
            )
        if self.b0 is not None:
            references.append(
                ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.OFFSET)
            )
        if self.b1 is not None:
            references.append(
                ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.SLOPE)
            )
        return tuple(references)

    def tie_for(self, reference: ParameterReference) -> ManualParameterTieState | None:
        """Return the explicit equality tie containing a reference."""

        return next(
            (group for group in self.parameter_ties if reference in group.members),
            None,
        )

    def parameter_intent(self, reference: ParameterReference) -> ManualParameterIntent:
        """Resolve the authoritative independent or shared Manual intent."""

        if reference not in set(self.parameter_references()):
            raise KeyError(reference)
        tie = self.tie_for(reference)
        if tie is not None:
            return tie.intent
        center_group = _center_group_id(self, reference)
        if center_group is not None:
            group = next(
                (item for item in self.center_groups if item.group_id == center_group),
                None,
            )
            if group is None:
                raise ValueError(f"dangling center group reference: {center_group}")
            return group.intent
        if reference.family is ParameterFamily.CENTER and reference in set(
            _legacy_center_references(self)
        ):
            if self.energy_shift is None:
                raise RuntimeError("legacy center has no Manual intent")
            return self.energy_shift
        return _direct_intent(self, reference)


def _lorentzian_index(
    model: ManualModelState,
    identity: ComponentIdentity,
) -> int:
    for index, component in enumerate(model.lorentzians):
        if component.identity == identity:
            return index
    raise KeyError(identity)


def _center_group_id(
    model: ManualModelState,
    reference: ParameterReference,
) -> str | None:
    if reference.family is not ParameterFamily.CENTER:
        return None
    if reference.component == ELASTIC_COMPONENT:
        return model.elastic_center_group
    if reference.component.family is ComponentFamily.LORENTZIAN:
        return model.lorentzians[
            _lorentzian_index(model, reference.component)
        ].center_group
    return None


def _legacy_center_references(
    model: ManualModelState,
) -> tuple[ParameterReference, ...]:
    references: list[ParameterReference] = []
    if model.elastic_area is not None and model.elastic_center_group is None:
        references.append(ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER))
    references.extend(
        ParameterReference(component.identity, ParameterFamily.CENTER)
        for component in model.lorentzians
        if component.center is None and component.center_group is None
    )
    return tuple(references)


def _center_group_references(
    model: ManualModelState,
    group_id: str,
) -> tuple[ParameterReference, ...]:
    references: list[ParameterReference] = []
    if model.elastic_area is not None and model.elastic_center_group == group_id:
        references.append(ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER))
    references.extend(
        ParameterReference(component.identity, ParameterFamily.CENTER)
        for component in model.lorentzians
        if component.center_group == group_id
    )
    return tuple(references)


def _direct_intent(
    model: ManualModelState,
    reference: ParameterReference,
) -> ManualParameterIntent:
    if reference.component == ELASTIC_COMPONENT:
        if reference.family is ParameterFamily.AREA and model.elastic_area is not None:
            return model.elastic_area
        raise KeyError(reference)
    if reference.component == BACKGROUND_COMPONENT:
        if reference.family is ParameterFamily.OFFSET and model.b0 is not None:
            return model.b0
        if reference.family is ParameterFamily.SLOPE and model.b1 is not None:
            return model.b1
        raise KeyError(reference)
    component = model.lorentzians[_lorentzian_index(model, reference.component)]
    if reference.family is ParameterFamily.AREA:
        return component.area
    if reference.family is ParameterFamily.FWHM:
        return component.fwhm
    if component.center is not None:
        return component.center
    raise KeyError(reference)


def replace_parameter_intent(
    model: ManualModelState,
    reference: ParameterReference,
    intent: ManualParameterIntent,
) -> ManualModelState:
    """Replace one independent or explicitly shared editable Manual intent."""

    model.parameter_intent(reference)
    tie = model.tie_for(reference)
    if tie is not None:
        ties = tuple(
            replace(group, intent=intent) if group.group_id == tie.group_id else group
            for group in model.parameter_ties
        )
        return replace(model, parameter_ties=ties)
    center_group = _center_group_id(model, reference)
    if center_group is not None:
        groups = tuple(
            replace(group, intent=intent) if group.group_id == center_group else group
            for group in model.center_groups
        )
        return replace(model, center_groups=groups)
    if reference.family is ParameterFamily.CENTER and reference in set(
        _legacy_center_references(model)
    ):
        return replace(model, energy_shift=intent)
    if reference.component == ELASTIC_COMPONENT:
        return replace(model, elastic_area=intent)
    if reference.component == BACKGROUND_COMPONENT:
        return replace(
            model,
            b0=intent if reference.family is ParameterFamily.OFFSET else model.b0,
            b1=intent if reference.family is ParameterFamily.SLOPE else model.b1,
        )
    index = _lorentzian_index(model, reference.component)
    component = model.lorentzians[index]
    if reference.family is ParameterFamily.AREA:
        component = replace(component, area=intent)
    elif reference.family is ParameterFamily.FWHM:
        component = replace(component, fwhm=intent)
    else:
        component = replace(component, center=intent)
    components = list(model.lorentzians)
    components[index] = component
    return replace(model, lorentzians=tuple(components))


def _structural_model(model: ManualModelState) -> SpectralModelDefinition:
    """Use ephemeral configurations solely to reuse frozen topology validation."""

    direct: dict[ParameterReference, ParameterConfiguration] = {
        reference: _structural_configuration(
            model.parameter_intent(reference),
            _kind(reference),
        )
        for reference in model.parameter_references()
    }
    return _core_model(model, direct)


def _core_model(
    model: ManualModelState,
    configurations: dict[ParameterReference, ParameterConfiguration],
) -> SpectralModelDefinition:
    def configuration(reference: ParameterReference) -> ParameterConfiguration:
        return configurations[reference]

    legacy_references = _legacy_center_references(model)
    energy_shift = configuration(legacy_references[0]) if legacy_references else None
    center_groups_list: list[CenterGroup] = []
    for group in model.center_groups:
        group_references = _center_group_references(model, group.group_id)
        if not group_references:
            raise ValueError(f"center group is not referenced: {group.group_id}")
        center_groups_list.append(
            CenterGroup(group.group_id, configuration(group_references[0]))
        )
    center_groups = tuple(center_groups_list)
    lorentzians = tuple(
        LorentzianComponent(
            area=configuration(
                ParameterReference(component.identity, ParameterFamily.AREA)
            ),
            fwhm=configuration(
                ParameterReference(component.identity, ParameterFamily.FWHM)
            ),
            center=(
                configuration(
                    ParameterReference(component.identity, ParameterFamily.CENTER)
                )
                if component.center is not None
                else None
            ),
            center_group=component.center_group,
            identity=component.identity,
        )
        for component in model.lorentzians
    )
    ties = tuple(
        ParameterTieGroup(
            group.group_id,
            group.members,
            configuration(group.members[0]),
        )
        for group in model.parameter_ties
    )
    return SpectralModelDefinition(
        energy_shift=energy_shift,
        elastic_area=(
            configuration(ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA))
            if model.elastic_area is not None
            else None
        ),
        lorentzians=lorentzians,
        background=model.background,
        b0=(
            configuration(
                ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.OFFSET)
            )
            if model.b0 is not None
            else None
        ),
        b1=(
            configuration(
                ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.SLOPE)
            )
            if model.b1 is not None
            else None
        ),
        center_groups=center_groups,
        elastic_center_group=model.elastic_center_group,
        parameter_ties=ties,
    )


def materialize_manual_model(
    model: ManualModelState,
    prepared_resolution: PreparedResolution,
    selection: FittingSelection,
    group_index: int,
) -> ManualModelMaterialization:
    """Materialize all slots for one active Group without mutating Manual intent."""

    references = model.parameter_references()
    materializations: dict[ParameterReference, ManualParameterMaterialization] = {}

    def materialize_slot(
        members: tuple[ParameterReference, ...],
        intent: ManualParameterIntent,
    ) -> None:
        result = materialize_manual_parameter(
            intent,
            _kind(members[0]),
            prepared_resolution=prepared_resolution,
            selection=selection,
            group_index=group_index,
        )
        for member in members:
            materializations[member] = result

    for tie_state in model.parameter_ties:
        materialize_slot(tie_state.members, tie_state.intent)
    for center_state in model.center_groups:
        materialize_slot(
            _center_group_references(model, center_state.group_id),
            center_state.intent,
        )
    legacy = _legacy_center_references(model)
    if legacy:
        if model.energy_shift is None:
            raise RuntimeError("legacy shared center has no Manual intent")
        materialize_slot(legacy, model.energy_shift)
    for reference in references:
        if reference not in materializations:
            materialize_slot((reference,), _direct_intent(model, reference))

    preview_configurations = {
        reference: materializations[reference].preview_configuration
        for reference in references
    }
    fit_configurations = {
        reference: materializations[reference].fit_configuration
        for reference in references
    }
    return ManualModelMaterialization(
        preview_model=_core_model(model, preview_configurations),
        fit_model=_core_model(model, fit_configurations),
        parameters=tuple(
            ManualParameterStateMaterialization(reference, materializations[reference])
            for reference in references
        ),
    )
