"""Typed single-Q spectral-model and fit-result values."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4

import numpy as np
import numpy.typing as npt

from ezqens.domain import DiagnosticSeverity
from ezqens.resolution import ResolutionAcceptanceProvenance

FloatArray = npt.NDArray[np.float64]


def _readonly_float_array(value: npt.ArrayLike, *, name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    array.setflags(write=False)
    return array


def _readonly_float_matrix(value: npt.ArrayLike, *, name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    array.setflags(write=False)
    return array


class BackgroundModel(StrEnum):
    """Supported additive background forms."""

    NONE = "none"
    CONSTANT = "B0"
    LINEAR = "B1"


class ComponentFamily(StrEnum):
    """Stable scientific function families used by Manual Fit."""

    ELASTIC = "elastic"
    LORENTZIAN = "lorentzian"
    BACKGROUND = "background"


class ParameterFamily(StrEnum):
    """Supported Manual parameter families, independent of display text."""

    AREA = "area"
    CENTER = "center"
    FWHM = "fwhm"
    OFFSET = "offset"
    SLOPE = "slope"


class ParameterDimension(StrEnum):
    """Scientific dimensions needed to render Manual parameter units."""

    INTEGRATED_INTENSITY = "integrated_intensity"
    ENERGY = "energy"
    INTENSITY = "intensity"
    INTENSITY_PER_ENERGY = "intensity_per_energy"


@dataclass(frozen=True, slots=True)
class ComponentIdentity:
    """One stable machine identity for a spectral function/component."""

    family: ComponentFamily
    component_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.family, ComponentFamily):
            raise ValueError("component family must be a ComponentFamily")
        if not isinstance(self.component_id, str) or not self.component_id.strip():
            raise ValueError("component_id must be a nonempty string")
        if self.component_id != self.component_id.strip():
            raise ValueError("component_id must not have surrounding whitespace")


ELASTIC_COMPONENT = ComponentIdentity(ComponentFamily.ELASTIC, "elastic")
BACKGROUND_COMPONENT = ComponentIdentity(ComponentFamily.BACKGROUND, "background")


def _new_lorentzian_identity() -> ComponentIdentity:
    return ComponentIdentity(ComponentFamily.LORENTZIAN, uuid4().hex)


@dataclass(frozen=True, slots=True)
class ParameterReference:
    """Stable reference to one component-owned scientific parameter."""

    component: ComponentIdentity
    family: ParameterFamily

    def __post_init__(self) -> None:
        if not isinstance(self.component, ComponentIdentity):
            raise ValueError("parameter component must be a ComponentIdentity")
        if not isinstance(self.family, ParameterFamily):
            raise ValueError("parameter family must be a ParameterFamily")
        allowed = {
            ComponentFamily.ELASTIC: {ParameterFamily.AREA, ParameterFamily.CENTER},
            ComponentFamily.LORENTZIAN: {
                ParameterFamily.AREA,
                ParameterFamily.CENTER,
                ParameterFamily.FWHM,
            },
            ComponentFamily.BACKGROUND: {
                ParameterFamily.OFFSET,
                ParameterFamily.SLOPE,
            },
        }[self.component.family]
        if self.family not in allowed:
            raise ValueError(
                f"{self.family.value} is incompatible with "
                f"{self.component.family.value}"
            )


@dataclass(frozen=True, slots=True)
class ParameterMetadata:
    """GUI-facing typed metadata for one supported Manual parameter."""

    reference: ParameterReference
    display_label: str
    dimension: ParameterDimension
    unit: str

    @property
    def component(self) -> ComponentIdentity:
        """Return the owning function identity without label parsing."""

        return self.reference.component

    @property
    def family(self) -> ParameterFamily:
        """Return the typed parameter family without label parsing."""

        return self.reference.family


@dataclass(frozen=True, slots=True)
class ParameterConfiguration:
    """One manual/expert parameter initial value, bounds, and free state."""

    initial_value: float
    lower_bound: float = -np.inf
    upper_bound: float = np.inf
    free: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.initial_value):
            raise ValueError("parameter initial_value must be finite")
        if np.isnan(self.lower_bound) or np.isnan(self.upper_bound):
            raise ValueError("parameter bounds must not be NaN")
        if self.lower_bound > self.upper_bound:
            raise ValueError("parameter lower_bound must not exceed upper_bound")
        if not self.lower_bound <= self.initial_value <= self.upper_bound:
            raise ValueError("parameter initial_value must lie within its bounds")
        if self.free and self.lower_bound == self.upper_bound:
            raise ValueError("a free parameter requires a nonzero bound interval")
        if not isinstance(self.free, bool):
            raise ValueError("parameter free state must be boolean")


@dataclass(frozen=True, slots=True)
class CenterGroup:
    """One stable center identity owning one shared parameter configuration."""

    group_id: str
    parameter: ParameterConfiguration

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError("center group_id must be a nonempty string")
        if self.group_id != self.group_id.strip():
            raise ValueError("center group_id must not have surrounding whitespace")
        if not isinstance(self.parameter, ParameterConfiguration):
            raise ValueError("center group parameter must be a ParameterConfiguration")


@dataclass(frozen=True, slots=True)
class ParameterTieGroup:
    """One same-family equality tie owning one optimizer configuration."""

    group_id: str
    members: tuple[ParameterReference, ...]
    parameter: ParameterConfiguration

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError("parameter tie group_id must be a nonempty string")
        if self.group_id != self.group_id.strip():
            raise ValueError(
                "parameter tie group_id must not have surrounding whitespace"
            )
        members = tuple(self.members)
        if len(members) < 2:
            raise ValueError("parameter tie group requires at least two members")
        if any(not isinstance(member, ParameterReference) for member in members):
            raise ValueError("parameter tie members must be ParameterReference values")
        if len(set(members)) != len(members):
            raise ValueError("parameter tie members must be unique")
        families = {member.family for member in members}
        if len(families) != 1:
            raise ValueError("parameter tie members must have the same family")
        if not isinstance(self.parameter, ParameterConfiguration):
            raise ValueError("parameter tie parameter must be a ParameterConfiguration")
        if self.family is ParameterFamily.AREA and (
            self.parameter.lower_bound < 0.0 or self.parameter.initial_value < 0.0
        ):
            raise ValueError("tied integrated area must be nonnegative")
        if self.family is ParameterFamily.FWHM and (
            self.parameter.lower_bound <= 0.0 or self.parameter.initial_value <= 0.0
        ):
            raise ValueError("tied FWHM must be strictly positive")
        object.__setattr__(self, "members", members)

    @property
    def family(self) -> ParameterFamily:
        """Return the one compatible parameter family owned by this tie."""

        return self.members[0].family


@dataclass(frozen=True, slots=True)
class LorentzianComponent:
    """One unit-area Lorentzian with an explicit or legacy center reference."""

    area: ParameterConfiguration
    fwhm: ParameterConfiguration
    center: ParameterConfiguration | None = None
    center_group: str | None = None
    identity: ComponentIdentity = field(default_factory=_new_lorentzian_identity)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.identity, ComponentIdentity)
            or self.identity.family is not ComponentFamily.LORENTZIAN
        ):
            raise ValueError("Lorentzian identity must have lorentzian family")
        if self.area.lower_bound < 0.0 or self.area.initial_value < 0.0:
            raise ValueError("Lorentzian integrated area must be nonnegative")
        if self.fwhm.lower_bound <= 0.0 or self.fwhm.initial_value <= 0.0:
            raise ValueError("Lorentzian FWHM must be strictly positive")
        if self.center is not None and self.center_group is not None:
            raise ValueError(
                "Lorentzian center and center_group are mutually exclusive"
            )
        if self.center_group is not None and (
            not isinstance(self.center_group, str) or not self.center_group.strip()
        ):
            raise ValueError("Lorentzian center_group must be a nonempty string")
        if (
            self.center_group is not None
            and self.center_group != self.center_group.strip()
        ):
            raise ValueError(
                "Lorentzian center_group must not have surrounding whitespace"
            )


@dataclass(frozen=True, slots=True)
class SpectralModelDefinition:
    """Optional elastic, variable Lorentzians, center groups, and background."""

    energy_shift: ParameterConfiguration | None = None
    elastic_area: ParameterConfiguration | None = None
    lorentzians: tuple[LorentzianComponent, ...] = ()
    background: BackgroundModel = BackgroundModel.NONE
    b0: ParameterConfiguration | None = None
    b1: ParameterConfiguration | None = None
    center_groups: tuple[CenterGroup, ...] = ()
    elastic_center_group: str | None = None
    parameter_ties: tuple[ParameterTieGroup, ...] = ()

    def __post_init__(self) -> None:
        lorentzians = tuple(self.lorentzians)
        center_groups = tuple(self.center_groups)
        parameter_ties = tuple(self.parameter_ties)
        if any(not isinstance(item, LorentzianComponent) for item in lorentzians):
            raise ValueError("lorentzians must contain LorentzianComponent values")
        if any(not isinstance(item, CenterGroup) for item in center_groups):
            raise ValueError("center_groups must contain CenterGroup values")
        if any(not isinstance(item, ParameterTieGroup) for item in parameter_ties):
            raise ValueError("parameter_ties must contain ParameterTieGroup values")
        identities = tuple(component.identity for component in lorentzians)
        if len(set(identities)) != len(identities):
            raise ValueError("Lorentzian component identities must be unique")
        if self.energy_shift is not None and not isinstance(
            self.energy_shift, ParameterConfiguration
        ):
            raise ValueError("energy_shift must be a ParameterConfiguration or None")
        if self.elastic_area is not None and not isinstance(
            self.elastic_area, ParameterConfiguration
        ):
            raise ValueError("elastic_area must be a ParameterConfiguration or None")
        if not isinstance(self.background, BackgroundModel):
            raise ValueError("background must be a BackgroundModel")
        if self.elastic_center_group is not None and (
            not isinstance(self.elastic_center_group, str)
            or not self.elastic_center_group.strip()
        ):
            raise ValueError("elastic_center_group must be a nonempty string")
        if (
            self.elastic_center_group is not None
            and self.elastic_center_group != self.elastic_center_group.strip()
        ):
            raise ValueError(
                "elastic_center_group must not have surrounding whitespace"
            )

        group_ids = tuple(group.group_id for group in center_groups)
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("center group identities must be unique")
        groups_by_id = {group.group_id: group for group in center_groups}
        referenced_groups: set[str] = set()
        legacy_center_used = False

        if self.elastic_area is not None:
            if (
                self.elastic_area.lower_bound < 0.0
                or self.elastic_area.initial_value < 0.0
            ):
                raise ValueError("elastic integrated area must be nonnegative")
            if self.elastic_center_group is None:
                if self.energy_shift is None:
                    raise ValueError(
                        "elastic component requires energy_shift or "
                        "elastic_center_group"
                    )
                legacy_center_used = True
            else:
                referenced_groups.add(self.elastic_center_group)
        elif self.elastic_center_group is not None:
            raise ValueError("elastic_center_group requires an elastic component")

        for component in lorentzians:
            if component.center_group is not None:
                referenced_groups.add(component.center_group)
            elif component.center is None:
                if self.energy_shift is None:
                    raise ValueError(
                        "Lorentzian requires energy_shift, center, or center_group"
                    )
                legacy_center_used = True

        dangling = referenced_groups - groups_by_id.keys()
        if dangling:
            raise ValueError(
                "center group reference is dangling: " + ", ".join(sorted(dangling))
            )
        unused = groups_by_id.keys() - referenced_groups
        if unused:
            raise ValueError(
                "center group is not referenced: " + ", ".join(sorted(unused))
            )
        if self.energy_shift is not None and not legacy_center_used:
            raise ValueError("energy_shift is not referenced by any component")

        if self.background is BackgroundModel.NONE:
            if self.b0 is not None or self.b1 is not None:
                raise ValueError("NONE background must not define b0 or b1")
        elif self.background is BackgroundModel.CONSTANT:
            if self.b0 is None or self.b1 is not None:
                raise ValueError("B0 background requires b0 and no b1")
        elif self.b0 is None or self.b1 is None:
            raise ValueError("B1 background requires both b0 and b1")

        tie_ids = tuple(group.group_id for group in parameter_ties)
        if len(set(tie_ids)) != len(tie_ids):
            raise ValueError("parameter tie group identities must be unique")
        available_references = set(self.parameter_references())
        tied_references: set[ParameterReference] = set()
        for group in parameter_ties:
            dangling_tie_members = set(group.members) - available_references
            if dangling_tie_members:
                raise ValueError(
                    "parameter tie reference is dangling: "
                    + ", ".join(
                        sorted(
                            f"{member.component.component_id}.{member.family.value}"
                            for member in dangling_tie_members
                        )
                    )
                )
            if tied_references.intersection(group.members):
                raise ValueError("a parameter reference may belong to only one tie")
            tied_references.update(group.members)
        if tied_references.intersection(self._legacy_shared_center_references()):
            raise ValueError(
                "general parameter ties cannot overlap legacy shared-center state"
            )
        if (
            self.elastic_area is None
            and not lorentzians
            and self.background is BackgroundModel.NONE
        ):
            raise ValueError("spectral model must contain at least one component")

        tied_configuration = {
            member: group.parameter
            for group in parameter_ties
            for member in group.members
        }

        def tied_or(
            reference: ParameterReference,
            configuration: ParameterConfiguration,
        ) -> ParameterConfiguration:
            return tied_configuration.get(reference, configuration)

        normalized_lorentzians = tuple(
            LorentzianComponent(
                area=tied_or(
                    ParameterReference(component.identity, ParameterFamily.AREA),
                    component.area,
                ),
                fwhm=tied_or(
                    ParameterReference(component.identity, ParameterFamily.FWHM),
                    component.fwhm,
                ),
                center=(
                    tied_or(
                        ParameterReference(
                            component.identity,
                            ParameterFamily.CENTER,
                        ),
                        component.center,
                    )
                    if component.center is not None
                    else None
                ),
                center_group=component.center_group,
                identity=component.identity,
            )
            for component in lorentzians
        )
        elastic_area = self.elastic_area
        energy_shift = self.energy_shift
        b0 = self.b0
        b1 = self.b1
        if elastic_area is not None:
            elastic_area = tied_or(
                ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA),
                elastic_area,
            )
            if self.elastic_center_group is None and energy_shift is not None:
                energy_shift = tied_or(
                    ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER),
                    energy_shift,
                )
        if b0 is not None:
            b0 = tied_or(
                ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.OFFSET),
                b0,
            )
        if b1 is not None:
            b1 = tied_or(
                ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.SLOPE),
                b1,
            )
        object.__setattr__(self, "energy_shift", energy_shift)
        object.__setattr__(self, "elastic_area", elastic_area)
        object.__setattr__(self, "lorentzians", normalized_lorentzians)
        object.__setattr__(self, "b0", b0)
        object.__setattr__(self, "b1", b1)
        object.__setattr__(self, "center_groups", center_groups)
        object.__setattr__(self, "parameter_ties", parameter_ties)

    @property
    def lorentzian_count(self) -> int:
        """Return the unrestricted component count represented by this model."""

        return len(self.lorentzians)

    def center_group_parameter(self, group_id: str) -> ParameterConfiguration:
        """Return the one parameter owned by a stable center group."""

        for group in self.center_groups:
            if group.group_id == group_id:
                return group.parameter
        raise ValueError(f"unknown center group: {group_id}")

    def elastic_center(self) -> ParameterConfiguration | None:
        """Return the elastic center, or None when no elastic component exists."""

        if self.elastic_area is None:
            return None
        if self.elastic_center_group is not None:
            return self.center_group_parameter(self.elastic_center_group)
        if self.energy_shift is None:  # guarded by validation
            raise RuntimeError("elastic component has no center")
        return self.energy_shift

    def lorentzian_center(
        self,
        component: LorentzianComponent,
    ) -> ParameterConfiguration:
        """Resolve one Lorentzian center without duplicating tied parameters."""

        if component.center_group is not None:
            return self.center_group_parameter(component.center_group)
        if component.center is not None:
            return component.center
        if self.energy_shift is None:  # guarded by validation
            raise RuntimeError("Lorentzian component has no center")
        return self.energy_shift

    def _legacy_shared_center_references(self) -> set[ParameterReference]:
        shared: set[ParameterReference] = set()
        energy_shift_references: list[ParameterReference] = []
        if self.elastic_area is not None:
            reference = ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER)
            if self.elastic_center_group is not None:
                shared.add(reference)
            else:
                energy_shift_references.append(reference)
        for component in self.lorentzians:
            reference = ParameterReference(component.identity, ParameterFamily.CENTER)
            if component.center_group is not None:
                shared.add(reference)
            elif component.center is None:
                energy_shift_references.append(reference)
        if len(energy_shift_references) > 1:
            shared.update(energy_shift_references)
        return shared

    def parameter_references(self) -> tuple[ParameterReference, ...]:
        """Return component-owned parameter references in model display order."""

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

    def base_parameter_configuration(
        self,
        reference: ParameterReference,
    ) -> ParameterConfiguration:
        """Resolve a member's untied/legacy configuration."""

        if reference.component == ELASTIC_COMPONENT:
            if self.elastic_area is None:
                raise KeyError(reference)
            if reference.family is ParameterFamily.AREA:
                return self.elastic_area
            if reference.family is ParameterFamily.CENTER:
                center = self.elastic_center()
                if center is None:
                    raise KeyError(reference)
                return center
        elif reference.component == BACKGROUND_COMPONENT:
            if reference.family is ParameterFamily.OFFSET and self.b0 is not None:
                return self.b0
            if reference.family is ParameterFamily.SLOPE and self.b1 is not None:
                return self.b1
        else:
            component = next(
                (
                    item
                    for item in self.lorentzians
                    if item.identity == reference.component
                ),
                None,
            )
            if component is None:
                raise KeyError(reference)
            if reference.family is ParameterFamily.AREA:
                return component.area
            if reference.family is ParameterFamily.FWHM:
                return component.fwhm
            if reference.family is ParameterFamily.CENTER:
                return self.lorentzian_center(component)
        raise KeyError(reference)

    def parameter_configuration(
        self,
        reference: ParameterReference,
    ) -> ParameterConfiguration:
        """Resolve one parameter, applying a general equality tie when present."""

        for group in self.parameter_ties:
            if reference in group.members:
                return group.parameter
        return self.base_parameter_configuration(reference)

    def tie_for(self, reference: ParameterReference) -> ParameterTieGroup | None:
        """Return the equality tie containing a parameter reference, if any."""

        return next(
            (group for group in self.parameter_ties if reference in group.members),
            None,
        )


def manual_parameter_metadata(
    model: SpectralModelDefinition,
    *,
    energy_unit: str = "meV",
    intensity_unit: str = "intensity",
) -> tuple[ParameterMetadata, ...]:
    """Return typed GUI metadata without creating a second model description."""

    if not isinstance(model, SpectralModelDefinition):
        raise ValueError("model must be a SpectralModelDefinition")
    if not energy_unit.strip() or not intensity_unit.strip():
        raise ValueError("parameter metadata units must be nonempty")
    details = {
        ParameterFamily.AREA: (
            "Area",
            ParameterDimension.INTEGRATED_INTENSITY,
            f"{intensity_unit}·{energy_unit}",
        ),
        ParameterFamily.CENTER: (
            "Center",
            ParameterDimension.ENERGY,
            energy_unit,
        ),
        ParameterFamily.FWHM: (
            "FWHM",
            ParameterDimension.ENERGY,
            energy_unit,
        ),
        ParameterFamily.OFFSET: (
            "Offset",
            ParameterDimension.INTENSITY,
            intensity_unit,
        ),
        ParameterFamily.SLOPE: (
            "Slope",
            ParameterDimension.INTENSITY_PER_ENERGY,
            f"{intensity_unit}/{energy_unit}",
        ),
    }
    return tuple(
        ParameterMetadata(reference, *details[reference.family])
        for reference in model.parameter_references()
    )


@dataclass(frozen=True, slots=True)
class StandardModelCandidate:
    """One generated standard search candidate, separate from fit policy."""

    lorentzian_count: int
    background: BackgroundModel

    def __post_init__(self) -> None:
        if isinstance(self.lorentzian_count, bool) or not isinstance(
            self.lorentzian_count, int
        ):
            raise ValueError("lorentzian_count must be an integer")
        if self.lorentzian_count < 0:
            raise ValueError("lorentzian_count must be nonnegative")
        if not isinstance(self.background, BackgroundModel):
            raise ValueError("background must be a BackgroundModel")

    @property
    def name(self) -> str:
        """Return a compact human-readable candidate label."""

        components = "E" + "".join(
            f"+L{index}" for index in range(1, self.lorentzian_count + 1)
        )
        if self.background is not BackgroundModel.NONE:
            components += f"+{self.background.value}"
        return components

    @property
    def nominal_parameter_count(self) -> int:
        """Return the all-free parameter count for complexity reporting."""

        background_count = {
            BackgroundModel.NONE: 0,
            BackgroundModel.CONSTANT: 1,
            BackgroundModel.LINEAR: 2,
        }[self.background]
        return 2 + 2 * self.lorentzian_count + background_count


@dataclass(frozen=True, slots=True)
class ParameterEstimate:
    """One fitted or fixed parameter value and local uncertainty state."""

    name: str
    value: float
    standard_error: float | None
    lower_bound: float
    upper_bound: float
    free: bool
    active_lower_bound: bool = False
    active_upper_bound: bool = False
    references: tuple[ParameterReference, ...] = ()

    def __post_init__(self) -> None:
        references = tuple(self.references)
        if any(not isinstance(item, ParameterReference) for item in references):
            raise ValueError("parameter estimate references must be typed references")
        if len(set(references)) != len(references):
            raise ValueError("parameter estimate references must be unique")
        object.__setattr__(self, "references", references)


@dataclass(frozen=True, slots=True)
class AlternativeStartResult:
    """Machine-readable outcome from one optimizer start."""

    start_index: int
    success: bool
    status: int
    chi_square: float
    evaluations: int
    elapsed_seconds: float
    start_parameter_values: tuple[float, ...]
    fitted_parameter_values: tuple[float, ...]
    canonical_component_order: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ResidualDiagnostics:
    """Threshold-free structure metrics for standardized residuals."""

    mean: float
    rms: float
    maximum_absolute: float
    linear_trend: float
    lag1_correlation: float | None
    longest_same_sign_run: int


@dataclass(frozen=True, slots=True)
class FitStatistics:
    """Weighted absolute-sigma fit statistics under one declared convention."""

    chi_square: float
    reduced_chi_square: float
    observations: int
    free_parameters: int
    nominal_degrees_of_freedom: int
    aic: float
    aicc: float
    bic: float
    information_criterion_convention: str = (
        "Gaussian absolute-sigma log likelihood with data-only constant omitted"
    )


@dataclass(frozen=True, slots=True)
class FitDiagnostics:
    """Raw numerical and identifiability evidence without policy thresholds."""

    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    function_evaluations: int
    jacobian_evaluations: int | None
    jacobian_rank: int
    jacobian_singular_values: FloatArray = field(repr=False)
    condition_number: float
    covariance_available: bool
    maximum_absolute_correlation: float | None
    active_bounds: tuple[str, ...]
    relative_standard_errors: tuple[float | None, ...]
    lorentzian_areas: tuple[float, ...]
    lorentzian_fwhm: tuple[float, ...]
    adjacent_fwhm_ratios: tuple[float, ...]
    component_area_to_standard_error: tuple[float | None, ...]
    lorentzian_full_convolution_areas: tuple[float, ...]
    lorentzian_retained_sampled_trapezoid_areas: tuple[float, ...]
    fwhm_to_resolution_fwhm: tuple[float | None, ...]
    fwhm_to_fitting_window: tuple[float, ...]
    resolution_fwhm: float | None
    median_sample_spacing: float
    resolution_fwhm_to_sample_spacing: float | None
    residual: ResidualDiagnostics
    alternative_starts: tuple[AlternativeStartResult, ...]
    selected_start_index: int
    total_elapsed_seconds: float

    def __post_init__(self) -> None:
        singular_values = _readonly_float_array(
            self.jacobian_singular_values,
            name="jacobian_singular_values",
        )
        object.__setattr__(self, "jacobian_singular_values", singular_values)


@dataclass(frozen=True, slots=True)
class ComponentCurve:
    """One component-resolved curve keyed by stable function identity."""

    component: ComponentIdentity
    values: FloatArray = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.component, ComponentIdentity):
            raise ValueError("component curve requires a ComponentIdentity")
        object.__setattr__(
            self,
            "values",
            _readonly_float_array(self.values, name="component curve"),
        )


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    """Component-resolved model values at caller-supplied physical coordinates."""

    energy: FloatArray = field(repr=False)
    total: FloatArray = field(repr=False)
    elastic: FloatArray = field(repr=False)
    lorentzians: tuple[FloatArray, ...] = field(repr=False)
    background: FloatArray = field(repr=False)
    component_curves: tuple[ComponentCurve, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        energy = _readonly_float_array(self.energy, name="model energy")
        total = _readonly_float_array(self.total, name="total model")
        elastic = _readonly_float_array(self.elastic, name="elastic model")
        background = _readonly_float_array(self.background, name="background model")
        lorentzians = tuple(
            _readonly_float_array(value, name="Lorentzian model")
            for value in self.lorentzians
        )
        size = energy.size
        if any(value.size != size for value in (total, elastic, background)) or any(
            value.size != size for value in lorentzians
        ):
            raise ValueError("all model-evaluation arrays must have equal lengths")
        object.__setattr__(self, "energy", energy)
        object.__setattr__(self, "total", total)
        component_curves = tuple(self.component_curves)
        if any(not isinstance(item, ComponentCurve) for item in component_curves):
            raise ValueError("component_curves must contain ComponentCurve values")
        if len({item.component for item in component_curves}) != len(component_curves):
            raise ValueError("component curve identities must be unique")
        if any(item.values.size != size for item in component_curves):
            raise ValueError("component curves must match model-evaluation length")
        object.__setattr__(self, "elastic", elastic)
        object.__setattr__(self, "background", background)
        object.__setattr__(self, "lorentzians", lorentzians)
        object.__setattr__(self, "component_curves", component_curves)

    def component_curve(self, identity: ComponentIdentity) -> FloatArray:
        """Return one resolved component curve by stable identity."""

        for curve in self.component_curves:
            if curve.component == identity:
                return curve.values
        raise KeyError(identity)


@dataclass(frozen=True, slots=True)
class FitProvenance:
    """Small scientific provenance record for one single-Q fit."""

    group_index: int
    group_label: str
    q_value: float
    energy_unit: str
    optimizer: str
    optimizer_method: str
    residual_definition: str
    sigma_interpretation: str
    convolution_spacing: float
    retained_energy_bounds: tuple[float, float]
    model_energy_bounds: tuple[float, float]
    convolution_energy_bounds: tuple[float, float]
    resolution_acceptance: ResolutionAcceptanceProvenance


@dataclass(frozen=True, slots=True)
class FitResult:
    """One fit with its submitted model configuration and separate estimates."""

    configuration: SpectralModelDefinition
    parameters: tuple[ParameterEstimate, ...]
    covariance: FloatArray | None = field(repr=False)
    correlation: FloatArray | None = field(repr=False)
    evaluation: ModelEvaluation = field(repr=False)
    raw_residuals: FloatArray = field(repr=False)
    standardized_residuals: FloatArray = field(repr=False)
    statistics: FitStatistics
    diagnostics: FitDiagnostics
    provenance: FitProvenance
    fitted_model: SpectralModelDefinition | None = None

    def __post_init__(self) -> None:
        parameter_count = len(self.parameters)
        covariance = None
        correlation = None
        if self.covariance is not None:
            covariance = _readonly_float_matrix(self.covariance, name="covariance")
            if covariance.shape != (parameter_count, parameter_count):
                raise ValueError("covariance shape must match parameter count")
        if self.correlation is not None:
            correlation = _readonly_float_matrix(
                self.correlation,
                name="correlation",
            )
            if correlation.shape != (parameter_count, parameter_count):
                raise ValueError("correlation shape must match parameter count")
        raw = _readonly_float_array(self.raw_residuals, name="raw_residuals")
        standardized = _readonly_float_array(
            self.standardized_residuals,
            name="standardized_residuals",
        )
        if raw.size != self.evaluation.energy.size or standardized.size != raw.size:
            raise ValueError("residual arrays must match evaluated fit points")
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "correlation", correlation)
        object.__setattr__(self, "raw_residuals", raw)
        object.__setattr__(self, "standardized_residuals", standardized)
        if self.fitted_model is None:
            object.__setattr__(self, "fitted_model", self.configuration)

    def parameter(self, name: str) -> ParameterEstimate:
        """Return one estimate by its canonical result name."""

        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(name)

    def parameter_by_reference(
        self,
        reference: ParameterReference,
    ) -> ParameterEstimate:
        """Return an estimate by stable component/parameter identity."""

        for parameter in self.parameters:
            if reference in parameter.references:
                return parameter
        raise KeyError(reference)

    @property
    def model(self) -> SpectralModelDefinition:
        """Return the submitted model configuration."""

        return self.configuration


class ManualFitDiagnosticCode(StrEnum):
    """Stable pre-fit blocker categories for Manual Fit readiness."""

    INVALID_SELECTION = "invalid_selection"
    INVALID_GROUP = "invalid_group"
    UNUSABLE_RETAINED_DATA = "unusable_retained_data"
    INSUFFICIENT_RETAINED_POINTS = "insufficient_retained_points"
    NONPOSITIVE_NOMINAL_DOF = "nonpositive_nominal_dof"
    INVALID_PARAMETER_CONFIGURATION = "invalid_parameter_configuration"
    INVALID_BOUNDS = "invalid_bounds"
    INVALID_TIES = "invalid_ties"
    CENTER_OUTSIDE_COVERAGE = "center_outside_coverage"
    NONFINITE_INITIAL_EVALUATION = "nonfinite_initial_evaluation"
    UNUSABLE_PREPARED_RESOLUTION = "unusable_prepared_resolution"


@dataclass(frozen=True, slots=True)
class ManualFitReadinessDiagnostic:
    """One machine-readable reason a Manual fit cannot currently run."""

    code: ManualFitDiagnosticCode
    severity: DiagnosticSeverity
    message: str
    group_index: int | None = None
    component: ComponentIdentity | None = None
    parameter: ParameterReference | None = None


@dataclass(frozen=True, slots=True)
class ManualFitReadiness:
    """Structured outcome from the same validation used by fit_single_q()."""

    runnable: bool
    diagnostics: tuple[ManualFitReadinessDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        diagnostics = tuple(self.diagnostics)
        if self.runnable and any(
            item.severity is DiagnosticSeverity.ERROR for item in diagnostics
        ):
            raise ValueError("runnable readiness cannot contain error diagnostics")
        if not self.runnable and not any(
            item.severity is DiagnosticSeverity.ERROR for item in diagnostics
        ):
            raise ValueError("non-runnable readiness requires an error diagnostic")
        object.__setattr__(self, "diagnostics", diagnostics)


@dataclass(frozen=True, slots=True)
class CandidateFitResult:
    """One standard candidate outcome without recommendation semantics."""

    candidate: StandardModelCandidate
    fit: FitResult | None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def success(self) -> bool:
        """Return whether this candidate produced a converged fit result."""

        return self.fit is not None and self.fit.diagnostics.optimizer_success
