"""Immutable FWHM, relaxation-time, and experimental-EISF results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from ezqens.batch import MultiQBranchResult, MultiQFitOutcome, MultiQFitStatus
from ezqens.domain import QBins
from ezqens.fitting import (
    ELASTIC_COMPONENT,
    ComponentFamily,
    ComponentIdentity,
    FitResult,
    ParameterEstimate,
    ParameterFamily,
    ParameterReference,
)

_TAU_NUMERATOR_MEV_PS = 1.3164239138
_CANONICAL_ENERGY_UNIT = "meV"
_COVARIANCE_ROUNDOFF_FACTOR = 64.0


class DerivedPointStatus(StrEnum):
    """Availability of derived science for one retained branch point."""

    AVAILABLE = "available"
    BLOCKED = "blocked"
    EXCLUDED = "excluded"
    NOT_APPLICABLE = "not_applicable"


class DerivedValueStatus(StrEnum):
    """Validity of one derived scientific value."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class StatisticalUncertaintyStatus(StrEnum):
    """How a reported statistical uncertainty was obtained or withheld."""

    AVAILABLE = "available"
    AVAILABLE_CONDITIONAL_ON_FIXED = "available_conditional_on_fixed"
    FIXED_NOT_ESTIMATED = "fixed_not_estimated"
    ALL_FIXED_NOT_ESTIMATED = "all_fixed_not_estimated"
    STRUCTURAL_VALUE_NOT_ESTIMATED = "structural_value_not_estimated"
    COVARIANCE_UNAVAILABLE = "covariance_unavailable"
    COVARIANCE_UNUSABLE = "covariance_unusable"
    VALUE_UNAVAILABLE = "value_unavailable"


class DerivedWarningCode(StrEnum):
    """Stable reasons for unavailable or qualified derived values."""

    NON_CANONICAL_ENERGY_UNIT = "non_canonical_energy_unit"
    INVALID_FWHM = "invalid_fwhm"
    FWHM_FIXED = "fwhm_fixed"
    FWHM_UNCERTAINTY_UNAVAILABLE = "fwhm_uncertainty_unavailable"
    EISF_UNDEFINED = "eisf_undefined"
    EISF_COVARIANCE_UNAVAILABLE = "eisf_covariance_unavailable"
    EISF_COVARIANCE_UNUSABLE = "eisf_covariance_unusable"
    EISF_FIXED_CONTRIBUTORS = "eisf_fixed_contributors"
    EISF_ALL_CONTRIBUTORS_FIXED = "eisf_all_contributors_fixed"
    INCOMPATIBLE_COMPONENT_TOPOLOGY = "incompatible_component_topology"


@dataclass(frozen=True, slots=True)
class DerivedWarning:
    """One typed derived-result qualification with no policy threshold."""

    code: DerivedWarningCode
    message: str
    component: ComponentIdentity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, DerivedWarningCode):
            raise ValueError("derived warning code must be a DerivedWarningCode")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("derived warning message must be nonempty")


@dataclass(frozen=True, slots=True)
class ComponentAreaEstimate:
    """One fitted integrated area and its optimizer-sharing provenance."""

    component: ComponentIdentity
    value: float
    free: bool
    parameter_name: str
    shared_references: tuple[ParameterReference, ...]

    def __post_init__(self) -> None:
        references = tuple(self.shared_references)
        if not isinstance(self.component, ComponentIdentity):
            raise ValueError("component area requires a ComponentIdentity")
        if not isinstance(self.free, bool):
            raise ValueError("component area free state must be boolean")
        if not isinstance(self.parameter_name, str) or not self.parameter_name:
            raise ValueError("component area parameter name must be nonempty")
        if not references or any(
            not isinstance(reference, ParameterReference) for reference in references
        ):
            raise ValueError("component area requires typed shared references")
        if ParameterReference(self.component, ParameterFamily.AREA) not in references:
            raise ValueError("component area references must include its owning area")
        object.__setattr__(self, "shared_references", references)

    @property
    def reference(self) -> ParameterReference:
        """Return this component's stable area reference."""

        return ParameterReference(self.component, ParameterFamily.AREA)


@dataclass(frozen=True, slots=True)
class LorentzianDerivedQuantity:
    """One identity-preserving intrinsic FWHM and relaxation-time result."""

    component: ComponentIdentity
    fwhm_mev: float
    fwhm_standard_error_mev: float | None
    tau_ps: float | None
    tau_standard_error_ps: float | None
    validity: DerivedValueStatus
    uncertainty_status: StatisticalUncertaintyStatus
    warnings: tuple[DerivedWarning, ...] = ()

    def __post_init__(self) -> None:
        warnings = tuple(self.warnings)
        if not isinstance(self.component, ComponentIdentity):
            raise ValueError("Lorentzian quantity requires a ComponentIdentity")
        if not isinstance(self.validity, DerivedValueStatus):
            raise ValueError("Lorentzian validity must be a DerivedValueStatus")
        if not isinstance(self.uncertainty_status, StatisticalUncertaintyStatus):
            raise ValueError("invalid Lorentzian uncertainty status")
        if any(not isinstance(item, DerivedWarning) for item in warnings):
            raise ValueError("Lorentzian warnings must be DerivedWarning values")
        if self.validity is DerivedValueStatus.AVAILABLE:
            if not np.isfinite(self.fwhm_mev) or self.fwhm_mev <= 0.0:
                raise ValueError(
                    "available Lorentzian FWHM must be finite and positive"
                )
            expected_tau = _TAU_NUMERATOR_MEV_PS / self.fwhm_mev
            if (
                self.tau_ps is None
                or not np.isfinite(self.tau_ps)
                or self.tau_ps <= 0.0
                or not math.isclose(self.tau_ps, expected_tau, rel_tol=1.0e-12)
            ):
                raise ValueError("available relaxation time must match its FWHM")
        elif self.tau_ps is not None or self.tau_standard_error_ps is not None:
            raise ValueError("unavailable Lorentzian values must not contain tau")

        uncertainty_available = (
            self.uncertainty_status is StatisticalUncertaintyStatus.AVAILABLE
        )
        if uncertainty_available:
            if self.validity is not DerivedValueStatus.AVAILABLE:
                raise ValueError("available uncertainty requires an available value")
            if (
                self.fwhm_standard_error_mev is None
                or self.tau_standard_error_ps is None
                or not np.isfinite(self.fwhm_standard_error_mev)
                or not np.isfinite(self.tau_standard_error_ps)
                or self.fwhm_standard_error_mev < 0.0
                or self.tau_standard_error_ps < 0.0
            ):
                raise ValueError(
                    "available uncertainty requires finite nonnegative errors"
                )
            expected_tau_error = (
                _TAU_NUMERATOR_MEV_PS
                * self.fwhm_standard_error_mev
                / (self.fwhm_mev * self.fwhm_mev)
            )
            if not math.isclose(
                self.tau_standard_error_ps,
                expected_tau_error,
                rel_tol=1.0e-12,
            ):
                raise ValueError("tau uncertainty must be propagated from FWHM")
        elif (
            self.fwhm_standard_error_mev is not None
            or self.tau_standard_error_ps is not None
        ):
            raise ValueError(
                "fixed or unavailable Lorentzian uncertainty must not carry errors"
            )
        allowed_statuses = (
            {StatisticalUncertaintyStatus.VALUE_UNAVAILABLE}
            if self.validity is DerivedValueStatus.UNAVAILABLE
            else {
                StatisticalUncertaintyStatus.AVAILABLE,
                StatisticalUncertaintyStatus.FIXED_NOT_ESTIMATED,
                StatisticalUncertaintyStatus.COVARIANCE_UNAVAILABLE,
                StatisticalUncertaintyStatus.COVARIANCE_UNUSABLE,
            }
        )
        if self.uncertainty_status not in allowed_statuses:
            raise ValueError("Lorentzian value and uncertainty states are inconsistent")
        object.__setattr__(self, "warnings", warnings)


@dataclass(frozen=True, slots=True)
class EISFResult:
    """Experimental EISF with full area and covariance provenance."""

    source_fit: FitResult
    q_value: float
    value: float | None
    standard_error: float | None
    validity: DerivedValueStatus
    uncertainty_status: StatisticalUncertaintyStatus
    elastic_area: ComponentAreaEstimate | None
    quasielastic_areas: tuple[ComponentAreaEstimate, ...]
    warnings: tuple[DerivedWarning, ...] = ()

    def __post_init__(self) -> None:
        areas = tuple(self.quasielastic_areas)
        warnings = tuple(self.warnings)
        if not isinstance(self.source_fit, FitResult):
            raise ValueError("EISF source_fit must be a FitResult")
        if not np.isfinite(self.q_value):
            raise ValueError("EISF Q value must be finite")
        if self.q_value != self.source_fit.provenance.q_value:
            raise ValueError("EISF Q value must match its source FitResult")
        if not isinstance(self.validity, DerivedValueStatus):
            raise ValueError("EISF validity must be a DerivedValueStatus")
        if not isinstance(self.uncertainty_status, StatisticalUncertaintyStatus):
            raise ValueError("invalid EISF uncertainty status")
        if self.elastic_area is not None and not isinstance(
            self.elastic_area, ComponentAreaEstimate
        ):
            raise ValueError("elastic_area must be a ComponentAreaEstimate or None")
        if any(not isinstance(item, ComponentAreaEstimate) for item in areas):
            raise ValueError("quasielastic_areas must contain component areas")
        if any(not isinstance(item, DerivedWarning) for item in warnings):
            raise ValueError("EISF warnings must be DerivedWarning values")
        if self.standard_error is not None and (
            not np.isfinite(self.standard_error) or self.standard_error < 0.0
        ):
            raise ValueError("EISF standard error must be finite and nonnegative")
        uncertainty_available = self.uncertainty_status in {
            StatisticalUncertaintyStatus.AVAILABLE,
            StatisticalUncertaintyStatus.AVAILABLE_CONDITIONAL_ON_FIXED,
        }
        if uncertainty_available and (
            self.validity is not DerivedValueStatus.AVAILABLE
            or self.standard_error is None
        ):
            raise ValueError("available EISF uncertainty requires value and error")
        if not uncertainty_available and self.standard_error is not None:
            raise ValueError("unavailable EISF uncertainty must not carry an error")
        if self.validity is DerivedValueStatus.AVAILABLE:
            if (
                self.value is None
                or not np.isfinite(self.value)
                or not 0.0 <= self.value <= 1.0
            ):
                raise ValueError("available EISF must be finite and within [0, 1]")
        elif self.value is not None or self.standard_error is not None:
            raise ValueError("unavailable EISF must not carry value or uncertainty")
        allowed_statuses = (
            {StatisticalUncertaintyStatus.VALUE_UNAVAILABLE}
            if self.validity is DerivedValueStatus.UNAVAILABLE
            else {
                StatisticalUncertaintyStatus.AVAILABLE,
                StatisticalUncertaintyStatus.AVAILABLE_CONDITIONAL_ON_FIXED,
                StatisticalUncertaintyStatus.ALL_FIXED_NOT_ESTIMATED,
                StatisticalUncertaintyStatus.STRUCTURAL_VALUE_NOT_ESTIMATED,
                StatisticalUncertaintyStatus.COVARIANCE_UNAVAILABLE,
                StatisticalUncertaintyStatus.COVARIANCE_UNUSABLE,
            }
        )
        if self.uncertainty_status not in allowed_statuses:
            raise ValueError("EISF value and uncertainty states are inconsistent")

        model = self.source_fit.fitted_model
        if model is None:
            raise ValueError("EISF source FitResult must contain a fitted model")
        if (self.elastic_area is None) != (model.elastic_area is None):
            raise ValueError("EISF elastic area must match its source FitResult")
        if self.elastic_area is not None and (
            self.elastic_area.component != ELASTIC_COMPONENT
        ):
            raise ValueError("EISF elastic area must use the elastic identity")
        expected_lorentzians = {component.identity for component in model.lorentzians}
        actual_lorentzians = {area.component for area in areas}
        if len(actual_lorentzians) != len(areas) or (
            actual_lorentzians != expected_lorentzians
        ):
            raise ValueError("EISF Lorentzian areas must match its source FitResult")
        if any(
            area.component.family is not ComponentFamily.LORENTZIAN for area in areas
        ):
            raise ValueError("quasielastic EISF areas must be Lorentzian components")
        contributors = (
            () if self.elastic_area is None else (self.elastic_area,)
        ) + areas
        for area in contributors:
            try:
                parameter = self.source_fit.parameter_by_reference(area.reference)
            except KeyError as error:
                raise ValueError(
                    "EISF component area is absent from its source FitResult"
                ) from error
            values_match = area.value == parameter.value or (
                np.isnan(area.value) and np.isnan(parameter.value)
            )
            if (
                not values_match
                or area.free is not parameter.free
                or area.parameter_name != parameter.name
                or area.shared_references != parameter.references
            ):
                raise ValueError(
                    "EISF component-area provenance must match its source FitResult"
                )
        free_present = any(area.free for area in contributors)
        fixed_present = any(not area.free for area in contributors)
        structural_value = self.elastic_area is None or not areas
        if self.validity is DerivedValueStatus.AVAILABLE:
            if (
                self.uncertainty_status
                is StatisticalUncertaintyStatus.AVAILABLE_CONDITIONAL_ON_FIXED
                and not (free_present and fixed_present)
            ):
                raise ValueError(
                    "conditional EISF uncertainty requires fixed and free contributors"
                )
            if (
                self.uncertainty_status is StatisticalUncertaintyStatus.AVAILABLE
                and fixed_present
            ):
                raise ValueError(
                    "unconditional EISF uncertainty requires all contributors free"
                )
            if (
                self.uncertainty_status
                is StatisticalUncertaintyStatus.ALL_FIXED_NOT_ESTIMATED
                and (free_present or not fixed_present)
            ):
                raise ValueError("all-fixed EISF state requires all contributors fixed")
            if (
                self.uncertainty_status
                is StatisticalUncertaintyStatus.STRUCTURAL_VALUE_NOT_ESTIMATED
                and not structural_value
            ):
                raise ValueError(
                    "structural EISF uncertainty state requires one component family"
                )
            if structural_value and (
                self.uncertainty_status
                is not StatisticalUncertaintyStatus.STRUCTURAL_VALUE_NOT_ESTIMATED
            ):
                raise ValueError(
                    "elastic-only or Lorentzian-only EISF is structurally exact"
                )
        if self.validity is DerivedValueStatus.AVAILABLE:
            values = np.asarray([area.value for area in contributors], dtype=np.float64)
            total = float(np.sum(values))
            if (
                not np.all(np.isfinite(values))
                or np.any(values < 0.0)
                or not np.isfinite(total)
                or total <= 0.0
            ):
                raise ValueError("available EISF requires valid contributing areas")
            elastic_value = (
                0.0 if self.elastic_area is None else self.elastic_area.value
            )
            expected = elastic_value / total
            if self.value is None or not math.isclose(
                self.value,
                expected,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ):
                raise ValueError("EISF value must match its contributing areas")
        object.__setattr__(self, "quasielastic_areas", areas)
        object.__setattr__(self, "warnings", warnings)


@dataclass(frozen=True, slots=True)
class DerivedQENSPoint:
    """One Q point retaining its original branch execution state."""

    group_index: int
    q_value: float
    execution_status: MultiQFitStatus
    derivation_status: DerivedPointStatus
    source_outcome: MultiQFitOutcome
    lorentzians: tuple[LorentzianDerivedQuantity, ...] = ()
    eisf: EISFResult | None = None
    warnings: tuple[DerivedWarning, ...] = ()

    def __post_init__(self) -> None:
        lorentzians = tuple(self.lorentzians)
        warnings = tuple(self.warnings)
        if self.group_index != self.source_outcome.group_index:
            raise ValueError("derived point group does not match its source outcome")
        if self.execution_status is not self.source_outcome.status:
            raise ValueError("derived point status does not match its source outcome")
        if not isinstance(self.execution_status, MultiQFitStatus):
            raise ValueError("execution_status must be a MultiQFitStatus")
        if not isinstance(self.derivation_status, DerivedPointStatus):
            raise ValueError("derivation_status must be a DerivedPointStatus")
        if not np.isfinite(self.q_value):
            raise ValueError("derived point Q value must be finite")
        if any(not isinstance(item, LorentzianDerivedQuantity) for item in lorentzians):
            raise ValueError("lorentzians must contain derived Lorentzian quantities")
        identities = tuple(item.component for item in lorentzians)
        if len(set(identities)) != len(identities):
            raise ValueError("derived Lorentzian component identities must be unique")
        if any(not isinstance(item, DerivedWarning) for item in warnings):
            raise ValueError("point warnings must be DerivedWarning values")
        if self.derivation_status is not DerivedPointStatus.AVAILABLE and (
            lorentzians or self.eisf is not None
        ):
            raise ValueError("unavailable derived points must not contain values")
        if (
            self.derivation_status is DerivedPointStatus.AVAILABLE
            and self.execution_status is not MultiQFitStatus.SUCCESS
        ):
            raise ValueError("available derived science requires a successful fit")
        source_fit = self.source_outcome.fit_result
        if source_fit is not None and self.q_value != source_fit.provenance.q_value:
            raise ValueError("derived point Q must match its source FitResult")
        if self.derivation_status is DerivedPointStatus.AVAILABLE:
            if source_fit is None or self.eisf is None:
                raise ValueError("available derived science requires its source fit")
            if self.eisf.source_fit is not source_fit:
                raise ValueError("derived EISF must retain the exact source FitResult")
            if self.source_outcome.derived_result_excluded:
                raise ValueError("derived-excluded outcomes cannot contain values")
            model = source_fit.fitted_model
            if model is None or {item.component for item in lorentzians} != {
                component.identity for component in model.lorentzians
            }:
                raise ValueError(
                    "derived Lorentzian quantities must match source-fit identities"
                )
            for item in lorentzians:
                parameter = source_fit.parameter_by_reference(
                    ParameterReference(item.component, ParameterFamily.FWHM)
                )
                values_match = item.fwhm_mev == parameter.value or (
                    np.isnan(item.fwhm_mev) and np.isnan(parameter.value)
                )
                if not values_match:
                    raise ValueError(
                        "derived Lorentzian FWHM must match its source FitResult"
                    )
        elif self.derivation_status is DerivedPointStatus.BLOCKED:
            if self.execution_status is not MultiQFitStatus.SUCCESS:
                raise ValueError("blocked derivation requires a successful source fit")
        elif self.derivation_status is DerivedPointStatus.EXCLUDED:
            if (
                self.execution_status is not MultiQFitStatus.SUCCESS
                or not self.source_outcome.derived_result_excluded
            ):
                raise ValueError(
                    "derived exclusion requires an excluded successful fit"
                )
        elif self.execution_status is MultiQFitStatus.SUCCESS:
            raise ValueError("successful source fits require a derived disposition")
        object.__setattr__(self, "lorentzians", lorentzians)
        object.__setattr__(self, "warnings", warnings)


@dataclass(frozen=True, slots=True)
class DerivedQENSResult:
    """One immutable derived view of an existing Multi-Q branch."""

    source_branch: MultiQBranchResult
    q_bins: QBins
    points: tuple[DerivedQENSPoint, ...]

    def __post_init__(self) -> None:
        points = tuple(self.points)
        if not isinstance(self.source_branch, MultiQBranchResult):
            raise ValueError("source_branch must be a MultiQBranchResult")
        if not isinstance(self.q_bins, QBins):
            raise ValueError("q_bins must be QBins")
        if len(points) != len(self.source_branch.outcomes):
            raise ValueError("derived points must preserve branch Q topology")
        if len(points) != self.q_bins.group_count:
            raise ValueError("Q-bin count must match branch outcome count")
        if tuple(point.group_index for point in points) != tuple(range(len(points))):
            raise ValueError("derived points must remain in dataset group order")
        if any(
            point.source_outcome is not outcome
            for point, outcome in zip(
                points,
                self.source_branch.outcomes,
                strict=True,
            )
        ):
            raise ValueError("derived points must retain their exact source outcomes")
        if tuple(point.q_value for point in points) != tuple(
            float(value) for value in self.q_bins.q_values
        ):
            raise ValueError("derived point Q values must match supplied QBins")
        object.__setattr__(self, "points", points)

    def point(self, group_index: int) -> DerivedQENSPoint:
        """Return one derived point by original dataset group index."""

        return self.points[group_index]


def _parameter_index(
    fit: FitResult,
    reference: ParameterReference,
) -> tuple[int, ParameterEstimate]:
    matches = tuple(
        (index, parameter)
        for index, parameter in enumerate(fit.parameters)
        if reference in parameter.references
    )
    if len(matches) != 1:
        raise ValueError(
            "fitted parameter references must map to exactly one optimizer parameter"
        )
    return matches[0]


def _component_area(
    fit: FitResult,
    component: ComponentIdentity,
) -> tuple[int, ComponentAreaEstimate]:
    reference = ParameterReference(component, ParameterFamily.AREA)
    index, parameter = _parameter_index(fit, reference)
    return index, ComponentAreaEstimate(
        component=component,
        value=float(parameter.value),
        free=parameter.free,
        parameter_name=parameter.name,
        shared_references=parameter.references,
    )


def _derive_lorentzian(
    fit: FitResult,
    component: ComponentIdentity,
) -> LorentzianDerivedQuantity:
    reference = ParameterReference(component, ParameterFamily.FWHM)
    _, parameter = _parameter_index(fit, reference)
    fwhm = float(parameter.value)
    if not np.isfinite(fwhm) or fwhm <= 0.0:
        warning = DerivedWarning(
            DerivedWarningCode.INVALID_FWHM,
            "relaxation time requires a finite positive intrinsic FWHM",
            component,
        )
        return LorentzianDerivedQuantity(
            component,
            fwhm,
            None,
            None,
            None,
            DerivedValueStatus.UNAVAILABLE,
            StatisticalUncertaintyStatus.VALUE_UNAVAILABLE,
            (warning,),
        )

    tau = _TAU_NUMERATOR_MEV_PS / fwhm
    if not parameter.free:
        warning = DerivedWarning(
            DerivedWarningCode.FWHM_FIXED,
            "fixed FWHM has no estimated statistical uncertainty",
            component,
        )
        return LorentzianDerivedQuantity(
            component,
            fwhm,
            None,
            tau,
            None,
            DerivedValueStatus.AVAILABLE,
            StatisticalUncertaintyStatus.FIXED_NOT_ESTIMATED,
            (warning,),
        )

    standard_error = parameter.standard_error
    if (
        standard_error is None
        or not np.isfinite(standard_error)
        or standard_error < 0.0
    ):
        warning = DerivedWarning(
            DerivedWarningCode.FWHM_UNCERTAINTY_UNAVAILABLE,
            "fitted FWHM statistical uncertainty is unavailable",
            component,
        )
        return LorentzianDerivedQuantity(
            component,
            fwhm,
            None,
            tau,
            None,
            DerivedValueStatus.AVAILABLE,
            StatisticalUncertaintyStatus.COVARIANCE_UNAVAILABLE,
            (warning,),
        )
    fwhm_error = float(standard_error)
    tau_error = _TAU_NUMERATOR_MEV_PS * fwhm_error / (fwhm * fwhm)
    return LorentzianDerivedQuantity(
        component,
        fwhm,
        fwhm_error,
        tau,
        tau_error,
        DerivedValueStatus.AVAILABLE,
        StatisticalUncertaintyStatus.AVAILABLE,
    )


def _unavailable_eisf(
    fit: FitResult,
    elastic: ComponentAreaEstimate | None,
    quasielastic: tuple[ComponentAreaEstimate, ...],
    message: str,
) -> EISFResult:
    return EISFResult(
        fit,
        float(fit.provenance.q_value),
        None,
        None,
        DerivedValueStatus.UNAVAILABLE,
        StatisticalUncertaintyStatus.VALUE_UNAVAILABLE,
        elastic,
        quasielastic,
        (DerivedWarning(DerivedWarningCode.EISF_UNDEFINED, message),),
    )


def _covariance_is_positive_semidefinite(covariance: np.ndarray) -> bool:
    """Validate covariance structure within tightly scaled IEEE-754 roundoff."""

    if (
        covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or not np.all(np.isfinite(covariance))
    ):
        return False
    size = covariance.shape[0]
    scale = float(np.max(np.abs(covariance), initial=0.0))
    roundoff_scale = max(scale, np.finfo(np.float64).tiny)
    tolerance = (
        _COVARIANCE_ROUNDOFF_FACTOR
        * np.finfo(np.float64).eps
        * max(size, 1)
        * roundoff_scale
    )
    if float(np.max(np.abs(covariance - covariance.T), initial=0.0)) > tolerance:
        return False
    symmetric = 0.5 * (covariance + covariance.T)
    try:
        eigenvalues = np.linalg.eigvalsh(symmetric)
    except np.linalg.LinAlgError:
        return False
    if not np.all(np.isfinite(eigenvalues)):
        return False
    eigenvalue_scale = max(
        scale,
        float(np.max(np.abs(eigenvalues), initial=0.0)),
        np.finfo(np.float64).tiny,
    )
    eigenvalue_tolerance = (
        _COVARIANCE_ROUNDOFF_FACTOR
        * np.finfo(np.float64).eps
        * max(size, 1)
        * eigenvalue_scale
    )
    return bool(float(np.min(eigenvalues, initial=0.0)) >= -eigenvalue_tolerance)


def _derive_eisf(
    fit: FitResult,
    component_order: tuple[ComponentIdentity, ...],
) -> EISFResult:
    model = fit.fitted_model
    if model is None:  # FitResult normalizes this, kept defensive for typing.
        raise ValueError("FitResult has no fitted model")

    elastic_index: int | None = None
    elastic: ComponentAreaEstimate | None = None
    if model.elastic_area is not None:
        elastic_index, elastic = _component_area(fit, ELASTIC_COMPONENT)
    quasielastic_pairs = tuple(
        _component_area(fit, identity) for identity in component_order
    )
    quasielastic = tuple(area for _, area in quasielastic_pairs)
    all_areas = (() if elastic is None else (elastic,)) + quasielastic
    if not all_areas:
        return _unavailable_eisf(
            fit,
            elastic,
            quasielastic,
            "EISF is undefined for a background-only model",
        )
    values = np.asarray([area.value for area in all_areas], dtype=np.float64)
    total = float(np.sum(values))
    if (
        not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or not np.isfinite(total)
        or total <= 0.0
    ):
        return _unavailable_eisf(
            fit,
            elastic,
            quasielastic,
            "EISF requires finite nonnegative component areas with positive total area",
        )

    elastic_value = 0.0 if elastic is None else elastic.value
    value = elastic_value / total
    if elastic is None or not quasielastic:
        return EISFResult(
            fit,
            float(fit.provenance.q_value),
            value,
            None,
            DerivedValueStatus.AVAILABLE,
            StatisticalUncertaintyStatus.STRUCTURAL_VALUE_NOT_ESTIMATED,
            elastic,
            quasielastic,
        )

    derivatives_by_index: dict[int, float] = {}
    if elastic_index is None:  # guarded by the structural case above
        raise RuntimeError("mixed EISF has no elastic parameter")
    derivatives_by_index[elastic_index] = derivatives_by_index.get(
        elastic_index, 0.0
    ) + (total - elastic_value) / (total * total)
    for index, _area in quasielastic_pairs:
        derivatives_by_index[index] = derivatives_by_index.get(
            index, 0.0
        ) - elastic_value / (total * total)

    contributor_indices = tuple(derivatives_by_index)
    free_indices = tuple(
        index for index in contributor_indices if fit.parameters[index].free
    )
    fixed_present = len(free_indices) != len(contributor_indices)
    warnings: list[DerivedWarning] = []
    if fixed_present:
        warnings.append(
            DerivedWarning(
                DerivedWarningCode.EISF_FIXED_CONTRIBUTORS,
                "EISF uncertainty is conditional on fixed contributing areas",
            )
        )
    if not free_indices:
        warnings.append(
            DerivedWarning(
                DerivedWarningCode.EISF_ALL_CONTRIBUTORS_FIXED,
                "all EISF area contributors are fixed; uncertainty was not estimated",
            )
        )
        return EISFResult(
            fit,
            float(fit.provenance.q_value),
            value,
            None,
            DerivedValueStatus.AVAILABLE,
            StatisticalUncertaintyStatus.ALL_FIXED_NOT_ESTIMATED,
            elastic,
            quasielastic,
            tuple(warnings),
        )

    if fit.covariance is None:
        warnings.append(
            DerivedWarning(
                DerivedWarningCode.EISF_COVARIANCE_UNAVAILABLE,
                "relevant fitted-area covariance is unavailable",
            )
        )
        return EISFResult(
            fit,
            float(fit.provenance.q_value),
            value,
            None,
            DerivedValueStatus.AVAILABLE,
            StatisticalUncertaintyStatus.COVARIANCE_UNAVAILABLE,
            elastic,
            quasielastic,
            tuple(warnings),
        )

    indices = np.asarray(free_indices, dtype=np.int64)
    relevant_covariance = fit.covariance[np.ix_(indices, indices)]
    gradient = np.asarray(
        [derivatives_by_index[index] for index in free_indices],
        dtype=np.float64,
    )
    covariance_usable = _covariance_is_positive_semidefinite(relevant_covariance)
    variance = (
        float(gradient @ relevant_covariance @ gradient)
        if covariance_usable
        else math.nan
    )
    if not covariance_usable or not np.isfinite(variance) or variance < 0.0:
        warnings.append(
            DerivedWarning(
                DerivedWarningCode.EISF_COVARIANCE_UNUSABLE,
                "relevant fitted-area covariance is unusable",
            )
        )
        return EISFResult(
            fit,
            float(fit.provenance.q_value),
            value,
            None,
            DerivedValueStatus.AVAILABLE,
            StatisticalUncertaintyStatus.COVARIANCE_UNUSABLE,
            elastic,
            quasielastic,
            tuple(warnings),
        )

    uncertainty_status = (
        StatisticalUncertaintyStatus.AVAILABLE_CONDITIONAL_ON_FIXED
        if fixed_present
        else StatisticalUncertaintyStatus.AVAILABLE
    )
    return EISFResult(
        fit,
        float(fit.provenance.q_value),
        value,
        math.sqrt(variance),
        DerivedValueStatus.AVAILABLE,
        uncertainty_status,
        elastic,
        quasielastic,
        tuple(warnings),
    )


def _blocked_point(
    outcome: MultiQFitOutcome,
    q_value: float,
    warning: DerivedWarning,
) -> DerivedQENSPoint:
    return DerivedQENSPoint(
        outcome.group_index,
        q_value,
        outcome.status,
        DerivedPointStatus.BLOCKED,
        outcome,
        warnings=(warning,),
    )


def derive_qens(
    branch_result: MultiQBranchResult,
    q_bins: QBins,
) -> DerivedQENSResult:
    """Derive FWHM/tau/EISF without changing or re-executing a branch."""

    if not isinstance(branch_result, MultiQBranchResult):
        raise ValueError("branch_result must be a MultiQBranchResult")
    if not isinstance(q_bins, QBins):
        raise ValueError("q_bins must be QBins")
    if q_bins.group_count != len(branch_result.outcomes):
        raise ValueError("Q-bin count must match branch outcome count")

    anchor_fit = branch_result.outcome(branch_result.anchor_group_index).fit_result
    if anchor_fit is None:  # guarded by MultiQBranchResult validation
        raise ValueError("branch anchor has no FitResult")
    anchor_model = anchor_fit.fitted_model
    if anchor_model is None:
        raise ValueError("branch anchor has no fitted model")
    component_order = tuple(
        component.identity for component in anchor_fit.configuration.lorentzians
    )
    anchor_has_elastic = anchor_model.elastic_area is not None
    if set(component_order) != {
        component.identity for component in anchor_model.lorentzians
    }:
        raise ValueError("anchor fitted component topology is inconsistent")

    points: list[DerivedQENSPoint] = []
    for outcome, q_raw in zip(
        branch_result.outcomes,
        q_bins.q_values,
        strict=True,
    ):
        q_value = float(q_raw)
        if outcome.status is not MultiQFitStatus.SUCCESS:
            points.append(
                DerivedQENSPoint(
                    outcome.group_index,
                    q_value,
                    outcome.status,
                    DerivedPointStatus.NOT_APPLICABLE,
                    outcome,
                )
            )
            continue
        if outcome.derived_result_excluded:
            points.append(
                DerivedQENSPoint(
                    outcome.group_index,
                    q_value,
                    outcome.status,
                    DerivedPointStatus.EXCLUDED,
                    outcome,
                )
            )
            continue
        fit = outcome.fit_result
        if fit is None:  # guarded by MultiQFitOutcome validation
            raise ValueError("successful branch outcome has no FitResult")
        if fit.provenance.q_value != q_value:
            raise ValueError(
                "FitResult Q value does not match supplied branch Q topology"
            )
        if fit.provenance.energy_unit != _CANONICAL_ENERGY_UNIT:
            points.append(
                _blocked_point(
                    outcome,
                    q_value,
                    DerivedWarning(
                        DerivedWarningCode.NON_CANONICAL_ENERGY_UNIT,
                        "derived linewidth and relaxation time require canonical meV "
                        "FitResult energy quantities",
                    ),
                )
            )
            continue
        fitted_model = fit.fitted_model
        if fitted_model is None:
            raise ValueError("successful FitResult has no fitted model")
        fitted_identities = {
            component.identity for component in fitted_model.lorentzians
        }
        if (
            fitted_identities != set(component_order)
            or (fitted_model.elastic_area is not None) != anchor_has_elastic
        ):
            points.append(
                _blocked_point(
                    outcome,
                    q_value,
                    DerivedWarning(
                        DerivedWarningCode.INCOMPATIBLE_COMPONENT_TOPOLOGY,
                        "FitResult component topology differs from the anchor branch",
                    ),
                )
            )
            continue
        lorentzians = tuple(
            _derive_lorentzian(fit, identity) for identity in component_order
        )
        eisf = _derive_eisf(fit, component_order)
        points.append(
            DerivedQENSPoint(
                outcome.group_index,
                q_value,
                outcome.status,
                DerivedPointStatus.AVAILABLE,
                outcome,
                lorentzians,
                eisf,
                tuple(warning for item in lorentzians for warning in item.warnings)
                + eisf.warnings,
            )
        )
    return DerivedQENSResult(branch_result, q_bins, tuple(points))
