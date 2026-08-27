"""Reusable weighted single-Q fitting using frozen measured-resolution convolution."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from ezqens.convolution import (
    ConvolutionError,
    ConvolutionPlan,
    build_convolution_plan,
    cell_integrated_lorentzian,
)
from ezqens.domain import DiagnosticSeverity
from ezqens.preprocessing import FittingSelection
from ezqens.resolution import PreparedResolution

from .models import (
    BACKGROUND_COMPONENT,
    ELASTIC_COMPONENT,
    AlternativeStartResult,
    BackgroundModel,
    CandidateFitResult,
    CenterGroup,
    ComponentCurve,
    ComponentFamily,
    ComponentIdentity,
    FitDiagnostics,
    FitProvenance,
    FitResult,
    FitStatistics,
    LorentzianComponent,
    ManualFitDiagnosticCode,
    ManualFitReadiness,
    ManualFitReadinessDiagnostic,
    ModelEvaluation,
    ParameterConfiguration,
    ParameterEstimate,
    ParameterFamily,
    ParameterReference,
    ParameterTieGroup,
    ResidualDiagnostics,
    SpectralModelDefinition,
    StandardModelCandidate,
)

FloatArray = npt.NDArray[np.float64]
_BOUND_INTERIOR_FRACTION: Final[float] = 1.0e-10
_OPTIMIZER: Final[str] = "scipy.optimize.least_squares"
_METHOD: Final[str] = "trf"


class FittingError(ValueError):
    """Raised when a requested fit is scientifically or numerically invalid."""


class _ManualFitBlocker(FittingError):
    """Internal typed blocker shared by readiness and actual fitting."""

    def __init__(
        self,
        code: ManualFitDiagnosticCode,
        message: str,
        *,
        group_index: int | None = None,
        parameter: ParameterReference | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.group_index = group_index
        self.parameter = parameter

    def diagnostic(self) -> ManualFitReadinessDiagnostic:
        return ManualFitReadinessDiagnostic(
            code=self.code,
            severity=DiagnosticSeverity.ERROR,
            message=str(self),
            group_index=self.group_index,
            component=(
                self.parameter.component if self.parameter is not None else None
            ),
            parameter=self.parameter,
        )


@dataclass(frozen=True, slots=True)
class _ParameterSlot:
    name: str
    configuration: ParameterConfiguration
    references: tuple[ParameterReference, ...]

    @property
    def key(self) -> frozenset[ParameterReference]:
        return frozenset(self.references)


@dataclass(frozen=True, slots=True)
class _FitInputs:
    plan: ConvolutionPlan
    energy: FloatArray
    intensity: FloatArray
    sigma: FloatArray
    group_label: str
    q_value: float


@dataclass(frozen=True, slots=True)
class _Solution:
    optimized: Any
    chi_square: float
    elapsed_seconds: float
    start_index: int


def _center_group_slot_name(group_id: str) -> str:
    return f"center_group_{group_id}"


def _tie_group_slot_name(group_id: str) -> str:
    return f"tie_{group_id}"


def _reference_name(
    model: SpectralModelDefinition,
    reference: ParameterReference,
) -> str:
    if reference.component == ELASTIC_COMPONENT:
        return (
            "elastic_area"
            if reference.family is ParameterFamily.AREA
            else "energy_shift"
        )
    if reference.component == BACKGROUND_COMPONENT:
        return "b0" if reference.family is ParameterFamily.OFFSET else "b1"
    index = next(
        index
        for index, component in enumerate(model.lorentzians, start=1)
        if component.identity == reference.component
    )
    return f"lorentzian_{index}_{reference.family.value}"


def _energy_shift_references(
    model: SpectralModelDefinition,
) -> tuple[ParameterReference, ...]:
    """Return only centers structurally assigned to legacy energy_shift."""

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
    model: SpectralModelDefinition,
    group_id: str,
) -> tuple[ParameterReference, ...]:
    """Return only centers structurally assigned to one declared center group."""

    references: list[ParameterReference] = []
    if model.elastic_area is not None and model.elastic_center_group == group_id:
        references.append(ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER))
    references.extend(
        ParameterReference(component.identity, ParameterFamily.CENTER)
        for component in model.lorentzians
        if component.center_group == group_id
    )
    return tuple(references)


def _validate_parameter_slot_partition(
    model: SpectralModelDefinition,
    slots: Sequence[_ParameterSlot],
) -> None:
    """Require an exact, scientifically compatible partition of references."""

    expected = tuple(model.parameter_references())
    assigned = tuple(reference for slot in slots for reference in slot.references)
    if len(assigned) != len(set(assigned)):
        raise FittingError(
            "a parameter reference belongs to more than one optimizer slot"
        )
    if set(assigned) != set(expected):
        raise FittingError(
            "optimizer slots do not cover every model parameter reference"
        )
    if any(
        len(slot.references) > 1
        and len({reference.family for reference in slot.references}) != 1
        for slot in slots
    ):
        raise FittingError(
            "a shared optimizer slot must contain scientifically compatible parameters"
        )


def _parameter_slots(model: SpectralModelDefinition) -> tuple[_ParameterSlot, ...]:
    slots: list[_ParameterSlot] = []
    tied = {member for group in model.parameter_ties for member in group.members}
    covered: set[ParameterReference] = set()
    if model.energy_shift is not None:
        references = tuple(
            reference
            for reference in _energy_shift_references(model)
            if reference not in tied
        )
        if references:
            slots.append(_ParameterSlot("energy_shift", model.energy_shift, references))
            covered.update(references)
    for group in model.center_groups:
        references = _center_group_references(model, group.group_id)
        slots.append(
            _ParameterSlot(
                _center_group_slot_name(group.group_id),
                group.parameter,
                references,
            )
        )
        covered.update(references)
    for tie_group in model.parameter_ties:
        slots.append(
            _ParameterSlot(
                _tie_group_slot_name(tie_group.group_id),
                tie_group.parameter,
                tie_group.members,
            )
        )
        covered.update(tie_group.members)
    for reference in model.parameter_references():
        if reference in covered:
            continue
        slots.append(
            _ParameterSlot(
                _reference_name(model, reference),
                model.parameter_configuration(reference),
                (reference,),
            )
        )
        covered.add(reference)
    _validate_parameter_slot_partition(model, slots)
    return tuple(slots)


def _configuration_values(
    template: SpectralModelDefinition,
    values: npt.ArrayLike,
) -> tuple[
    dict[ParameterReference, ParameterConfiguration], dict[str, ParameterConfiguration]
]:
    slots = _parameter_slots(template)
    parameter_values = np.asarray(values, dtype=np.float64)
    if parameter_values.ndim != 1 or parameter_values.size != len(slots):
        raise FittingError("parameter value count does not match the spectral model")
    by_reference: dict[ParameterReference, ParameterConfiguration] = {}
    by_name: dict[str, ParameterConfiguration] = {}
    for slot, value in zip(slots, parameter_values, strict=True):
        configuration = ParameterConfiguration(
            initial_value=float(value),
            lower_bound=slot.configuration.lower_bound,
            upper_bound=slot.configuration.upper_bound,
            free=slot.configuration.free,
        )
        by_name[slot.name] = configuration
        for reference in slot.references:
            by_reference[reference] = configuration
    return by_reference, by_name


def _model_from_values(
    template: SpectralModelDefinition,
    values: npt.ArrayLike,
) -> SpectralModelDefinition:
    configurations, named = _configuration_values(template, values)

    def resolved(reference: ParameterReference) -> ParameterConfiguration:
        return configurations[reference]

    components = tuple(
        LorentzianComponent(
            area=resolved(ParameterReference(component.identity, ParameterFamily.AREA)),
            fwhm=resolved(ParameterReference(component.identity, ParameterFamily.FWHM)),
            center=(
                resolved(ParameterReference(component.identity, ParameterFamily.CENTER))
                if component.center is not None
                else None
            ),
            center_group=component.center_group,
            identity=component.identity,
        )
        for component in template.lorentzians
    )
    energy_shift = None
    if template.energy_shift is not None:
        references = _energy_shift_references(template)
        energy_shift = resolved(references[0])
    return SpectralModelDefinition(
        energy_shift=energy_shift,
        elastic_area=(
            resolved(ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA))
            if template.elastic_area is not None
            else None
        ),
        lorentzians=components,
        background=template.background,
        b0=(
            resolved(ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.OFFSET))
            if template.b0 is not None
            else None
        ),
        b1=(
            resolved(ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.SLOPE))
            if template.b1 is not None
            else None
        ),
        center_groups=tuple(
            CenterGroup(
                group_id=group.group_id,
                parameter=named[_center_group_slot_name(group.group_id)],
            )
            for group in template.center_groups
        ),
        elastic_center_group=template.elastic_center_group,
        parameter_ties=tuple(
            ParameterTieGroup(
                group_id=group.group_id,
                members=group.members,
                parameter=named[_tie_group_slot_name(group.group_id)],
            )
            for group in template.parameter_ties
        ),
    )


def evaluate_spectral_model(
    plan: ConvolutionPlan,
    model: SpectralModelDefinition,
    energy: npt.ArrayLike | None = None,
) -> ModelEvaluation:
    """Evaluate a configured model without changing experimental coordinates.

    The elastic contribution directly scales the unit-area measured-resolution
    representation. Lorentzian amplitudes scale finite-domain convolution
    results without silently renormalizing truncated tails.
    """

    coordinates = np.asarray(
        plan.target_energy if energy is None else energy,
        dtype=np.float64,
    )
    if coordinates.ndim != 1 or coordinates.size == 0:
        raise FittingError("model evaluation energy must be a nonempty vector")
    if not np.all(np.isfinite(coordinates)):
        raise FittingError("model evaluation energy must be finite")
    elastic = np.zeros(coordinates.size, dtype=np.float64)
    if model.elastic_area is not None:
        elastic_area = model.parameter_configuration(
            ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA)
        )
        elastic_center = model.parameter_configuration(
            ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER)
        )
        elastic_shape = np.interp(
            coordinates - elastic_center.initial_value,
            plan.resolution_energy,
            plan.resolution_values,
            left=0.0,
            right=0.0,
        )
        elastic = np.asarray(elastic_area.initial_value * elastic_shape)
    lorentzian_contributions: list[FloatArray] = []
    for component in model.lorentzians:
        center = model.parameter_configuration(
            ParameterReference(component.identity, ParameterFamily.CENTER)
        ).initial_value
        fwhm = model.parameter_configuration(
            ParameterReference(component.identity, ParameterFamily.FWHM)
        ).initial_value
        area = model.parameter_configuration(
            ParameterReference(component.identity, ParameterFamily.AREA)
        ).initial_value
        intrinsic = cell_integrated_lorentzian(
            plan.model_energy,
            fwhm=fwhm,
            spacing=plan.spacing,
        )
        profile = plan.convolve(intrinsic).evaluate(
            coordinates,
            energy_shift=center,
        )
        lorentzian_contributions.append(np.asarray(area * profile, dtype=np.float64))
    background = np.zeros(coordinates.size, dtype=np.float64)
    if model.b0 is not None:
        background += model.parameter_configuration(
            ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.OFFSET)
        ).initial_value
    if model.b1 is not None:
        background += (
            model.parameter_configuration(
                ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.SLOPE)
            ).initial_value
            * coordinates
        )
    total = elastic + background
    for contribution in lorentzian_contributions:
        total = total + contribution
    component_curves: list[ComponentCurve] = []
    if model.elastic_area is not None:
        component_curves.append(ComponentCurve(ELASTIC_COMPONENT, elastic))
    component_curves.extend(
        ComponentCurve(component.identity, contribution)
        for component, contribution in zip(
            model.lorentzians,
            lorentzian_contributions,
            strict=True,
        )
    )
    if model.background is not BackgroundModel.NONE:
        component_curves.append(ComponentCurve(BACKGROUND_COMPONENT, background))
    return ModelEvaluation(
        energy=coordinates,
        total=total,
        elastic=elastic,
        lorentzians=tuple(lorentzian_contributions),
        background=background,
        component_curves=tuple(component_curves),
    )


def _fit_inputs(
    prepared_resolution: PreparedResolution,
    selection: FittingSelection,
    group_index: int,
    free_parameter_count: int,
) -> _FitInputs:
    if not isinstance(prepared_resolution, PreparedResolution):
        raise _ManualFitBlocker(
            ManualFitDiagnosticCode.UNUSABLE_PREPARED_RESOLUTION,
            "prepared_resolution must be a PreparedResolution",
            group_index=group_index if isinstance(group_index, int) else None,
        )
    if not isinstance(selection, FittingSelection):
        raise _ManualFitBlocker(
            ManualFitDiagnosticCode.INVALID_SELECTION,
            "selection must be a FittingSelection",
            group_index=group_index if isinstance(group_index, int) else None,
        )
    if selection.dataset is not prepared_resolution.sample_dataset:
        raise _ManualFitBlocker(
            ManualFitDiagnosticCode.INVALID_SELECTION,
            "fitting selection must reference the prepared sample dataset",
            group_index=group_index if isinstance(group_index, int) else None,
        )
    if isinstance(group_index, bool) or not isinstance(group_index, int):
        raise _ManualFitBlocker(
            ManualFitDiagnosticCode.INVALID_GROUP,
            "group_index must be an integer",
        )
    if not 0 <= group_index < len(prepared_resolution.spectra):
        raise _ManualFitBlocker(
            ManualFitDiagnosticCode.INVALID_GROUP,
            "group_index is outside the prepared resolution",
            group_index=group_index,
        )
    try:
        retained = selection.retained_mask(group_index)
    except (IndexError, ValueError) as error:
        raise _ManualFitBlocker(
            ManualFitDiagnosticCode.INVALID_SELECTION,
            str(error),
            group_index=group_index,
        ) from error
    spectrum = prepared_resolution.sample_dataset.spectra[group_index]
    energy = np.asarray(spectrum.energy[retained], dtype=np.float64)
    intensity = np.asarray(spectrum.intensity[retained], dtype=np.float64)
    sigma = np.asarray(spectrum.uncertainty[retained], dtype=np.float64)
    if not (
        energy.ndim == intensity.ndim == sigma.ndim == 1
        and energy.size == intensity.size == sigma.size
    ):
        raise _ManualFitBlocker(
            ManualFitDiagnosticCode.UNUSABLE_RETAINED_DATA,
            "retained sample arrays must be equal-length vectors",
            group_index=group_index,
        )
    if energy.size < 2:
        raise _ManualFitBlocker(
            ManualFitDiagnosticCode.INSUFFICIENT_RETAINED_POINTS,
            "single-Q fitting and diagnostics require at least two retained "
            "sample energy coordinates",
            group_index=group_index,
        )
    if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(intensity)):
        raise _ManualFitBlocker(
            ManualFitDiagnosticCode.UNUSABLE_RETAINED_DATA,
            "retained sample energy and intensity must be finite",
            group_index=group_index,
        )
    if not np.all(np.isfinite(sigma) & (sigma > 0.0)):
        raise _ManualFitBlocker(
            ManualFitDiagnosticCode.UNUSABLE_RETAINED_DATA,
            "retained absolute uncertainties must be finite and strictly positive",
            group_index=group_index,
        )
    if energy.size - free_parameter_count <= 0:
        raise _ManualFitBlocker(
            ManualFitDiagnosticCode.NONPOSITIVE_NOMINAL_DOF,
            "fit requires positive nominal statistical degrees of freedom",
            group_index=group_index,
        )
    try:
        plan = build_convolution_plan(prepared_resolution, group_index)
        q_value = prepared_resolution.q_value(group_index)
    except (ConvolutionError, IndexError, ValueError) as error:
        raise _ManualFitBlocker(
            ManualFitDiagnosticCode.UNUSABLE_PREPARED_RESOLUTION,
            str(error),
            group_index=group_index,
        ) from error
    return _FitInputs(
        plan=plan,
        energy=energy,
        intensity=intensity,
        sigma=sigma,
        group_label=spectrum.group_label,
        q_value=q_value,
    )


def _interior_initial(
    configuration: ParameterConfiguration,
    reference: ParameterReference | None = None,
) -> float:
    value = configuration.initial_value
    lower = configuration.lower_bound
    upper = configuration.upper_bound
    if np.isfinite(lower) and value <= lower:
        scale = max(1.0, abs(lower))
        value = lower + _BOUND_INTERIOR_FRACTION * scale
    if np.isfinite(upper) and value >= upper:
        scale = max(1.0, abs(upper))
        value = upper - _BOUND_INTERIOR_FRACTION * scale
    if not lower < value < upper:
        raise _ManualFitBlocker(
            ManualFitDiagnosticCode.INVALID_BOUNDS,
            "free parameter cannot be initialized inside its bounds",
            parameter=reference,
        )
    return float(value)


def _free_problem(
    model: SpectralModelDefinition,
) -> tuple[FloatArray, FloatArray, FloatArray, tuple[int, ...]]:
    slots = _parameter_slots(model)
    free_indices = tuple(
        index for index, slot in enumerate(slots) if slot.configuration.free
    )
    initial = np.asarray(
        [
            _interior_initial(
                slots[index].configuration,
                slots[index].references[0] if slots[index].references else None,
            )
            for index in free_indices
        ],
        dtype=np.float64,
    )
    lower = np.asarray(
        [slots[index].configuration.lower_bound for index in free_indices],
        dtype=np.float64,
    )
    upper = np.asarray(
        [slots[index].configuration.upper_bound for index in free_indices],
        dtype=np.float64,
    )
    return initial, lower, upper, free_indices


def _expand_free_values(
    model: SpectralModelDefinition,
    free_indices: tuple[int, ...],
    free_values: npt.ArrayLike,
) -> FloatArray:
    values = np.asarray(
        [slot.configuration.initial_value for slot in _parameter_slots(model)],
        dtype=np.float64,
    )
    values[np.asarray(free_indices, dtype=np.int64)] = np.asarray(
        free_values,
        dtype=np.float64,
    )
    return values


def _slot_index_by_reference(
    model: SpectralModelDefinition,
) -> dict[ParameterReference, int]:
    positions: dict[ParameterReference, int] = {}
    for index, slot in enumerate(_parameter_slots(model)):
        for reference in slot.references:
            positions[reference] = index
    return positions


def _canonical_component_order(
    values: FloatArray,
    model: SpectralModelDefinition,
) -> tuple[int, ...]:
    positions = _slot_index_by_reference(model)
    return tuple(
        sorted(
            range(model.lorentzian_count),
            key=lambda index: float(
                values[
                    positions[
                        ParameterReference(
                            model.lorentzians[index].identity,
                            ParameterFamily.FWHM,
                        )
                    ]
                ]
            ),
        )
    )


def _canonical_permutation(
    values: FloatArray,
    model: SpectralModelDefinition,
) -> tuple[int, ...]:
    order = _canonical_component_order(values, model)
    canonical_model = SpectralModelDefinition(
        energy_shift=model.energy_shift,
        elastic_area=model.elastic_area,
        lorentzians=tuple(model.lorentzians[index] for index in order),
        background=model.background,
        b0=model.b0,
        b1=model.b1,
        center_groups=model.center_groups,
        elastic_center_group=model.elastic_center_group,
        parameter_ties=model.parameter_ties,
    )
    original = {slot.key: index for index, slot in enumerate(_parameter_slots(model))}
    try:
        permutation = tuple(
            original[slot.key] for slot in _parameter_slots(canonical_model)
        )
    except KeyError as error:
        raise FittingError("canonical parameter permutation is incomplete") from error
    if sorted(permutation) != list(range(len(original))):
        raise FittingError("canonical parameter permutation is incomplete")
    return permutation


def _profile_fwhm(energy: FloatArray, values: FloatArray) -> float:
    peak_index = int(np.argmax(values))
    peak_value = float(values[peak_index])
    if not np.isfinite(peak_value) or peak_value <= 0.0:
        raise FittingError("measured resolution does not define a positive peak")
    half = peak_value / 2.0
    left = np.flatnonzero(values[:peak_index] <= half)
    right = np.flatnonzero(values[peak_index + 1 :] <= half)
    if left.size == 0 or right.size == 0:
        raise FittingError("measured-resolution FWHM is not contained in its support")
    left_low = int(left[-1])
    right_high = peak_index + 1 + int(right[0])
    left_crossing = float(
        np.interp(
            half,
            values[left_low : left_low + 2],
            energy[left_low : left_low + 2],
        )
    )
    right_crossing = float(
        np.interp(
            half,
            values[right_high - 1 : right_high + 1][::-1],
            energy[right_high - 1 : right_high + 1][::-1],
        )
    )
    width = right_crossing - left_crossing
    if not np.isfinite(width) or width <= 0.0:
        raise FittingError("measured-resolution FWHM is not usable")
    return width


def _residual_diagnostics(residuals: FloatArray) -> ResidualDiagnostics:
    centered_coordinate = np.linspace(-0.5, 0.5, residuals.size)
    linear_trend = float(np.polyfit(centered_coordinate, residuals, 1)[0])
    lag1: float | None = None
    if (
        residuals.size > 2
        and float(np.std(residuals[:-1])) > 0.0
        and float(np.std(residuals[1:])) > 0.0
    ):
        correlation = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
        if np.isfinite(correlation):
            lag1 = correlation
    signs = residuals >= 0.0
    longest = 1
    current = 1
    for index in range(1, signs.size):
        current = current + 1 if signs[index] == signs[index - 1] else 1
        longest = max(longest, current)
    return ResidualDiagnostics(
        mean=float(np.mean(residuals)),
        rms=float(np.sqrt(np.mean(np.square(residuals)))),
        maximum_absolute=float(np.max(np.abs(residuals))),
        linear_trend=linear_trend,
        lag1_correlation=lag1,
        longest_same_sign_run=longest,
    )


def _information_criteria(
    chi_square: float,
    observations: int,
    free_parameters: int,
) -> tuple[float, float, float]:
    aic = chi_square + 2.0 * free_parameters
    aicc = math.inf
    if observations > free_parameters + 1:
        aicc = aic + (
            2.0
            * free_parameters
            * (free_parameters + 1)
            / (observations - free_parameters - 1)
        )
    bic = chi_square + free_parameters * math.log(observations)
    return aic, aicc, bic


def _covariance_from_jacobian(
    jacobian: FloatArray,
) -> tuple[FloatArray | None, FloatArray, int, float]:
    if jacobian.size == 0:
        return None, np.empty(0, dtype=np.float64), 0, math.nan
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    tolerance = (
        np.finfo(np.float64).eps * max(jacobian.shape) * float(singular_values[0])
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0.0
        else math.inf
    )
    if rank != jacobian.shape[1] or not np.isfinite(condition):
        return None, singular_values, rank, condition
    try:
        covariance = np.linalg.inv(jacobian.T @ jacobian)
    except np.linalg.LinAlgError:
        covariance = None
    return covariance, singular_values, rank, condition


def _full_covariance(
    free_covariance: FloatArray | None,
    parameter_count: int,
    free_indices: tuple[int, ...],
) -> FloatArray | None:
    if free_covariance is None:
        return None
    covariance = np.full((parameter_count, parameter_count), np.nan)
    indices = np.asarray(free_indices, dtype=np.int64)
    covariance[np.ix_(indices, indices)] = free_covariance
    return covariance


def _correlation_from_covariance(
    covariance: FloatArray | None,
) -> tuple[FloatArray | None, float | None]:
    if covariance is None:
        return None, None
    scales = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / np.outer(scales, scales)
    finite = correlation[
        np.isfinite(correlation) & ~np.eye(correlation.shape[0], dtype=np.bool_)
    ]
    maximum = float(np.max(np.abs(finite))) if finite.size else None
    return np.asarray(correlation, dtype=np.float64), maximum


def _validate_center_coverage(
    name: str,
    configuration: ParameterConfiguration,
    inputs: _FitInputs,
    plan: ConvolutionPlan,
    *,
    group_index: int | None = None,
    reference: ParameterReference | None = None,
) -> None:
    allowed_lower = float(inputs.energy[-1] - plan.convolution_energy[-1])
    allowed_upper = float(inputs.energy[0] - plan.convolution_energy[0])
    if configuration.free and (
        not np.isfinite(configuration.lower_bound)
        or not np.isfinite(configuration.upper_bound)
        or configuration.lower_bound < allowed_lower
        or configuration.upper_bound > allowed_upper
    ):
        raise _ManualFitBlocker(
            ManualFitDiagnosticCode.CENTER_OUTSIDE_COVERAGE,
            f"free {name} requires finite bounds inside the fixed "
            "convolution-domain coverage",
            group_index=group_index,
            parameter=reference,
        )
    if not allowed_lower <= configuration.initial_value <= allowed_upper:
        raise _ManualFitBlocker(
            ManualFitDiagnosticCode.CENTER_OUTSIDE_COVERAGE,
            f"{name} initial value falls outside fixed convolution-domain coverage",
            group_index=group_index,
            parameter=reference,
        )


def _validate_parameter_tie_integrity(model: SpectralModelDefinition) -> None:
    ties = tuple(model.parameter_ties)
    if any(not isinstance(group, ParameterTieGroup) for group in ties):
        raise _ManualFitBlocker(
            ManualFitDiagnosticCode.INVALID_TIES,
            "parameter_ties must contain ParameterTieGroup values",
        )
    if len({group.group_id for group in ties}) != len(ties):
        raise _ManualFitBlocker(
            ManualFitDiagnosticCode.INVALID_TIES,
            "parameter tie group identities must be unique",
        )
    available = set(model.parameter_references())
    seen: set[ParameterReference] = set()
    for group in ties:
        if len({member.family for member in group.members}) != 1:
            raise _ManualFitBlocker(
                ManualFitDiagnosticCode.INVALID_TIES,
                "parameter tie members must have the same family",
            )
        if set(group.members) - available:
            raise _ManualFitBlocker(
                ManualFitDiagnosticCode.INVALID_TIES,
                "parameter tie reference is dangling",
            )
        if seen.intersection(group.members):
            raise _ManualFitBlocker(
                ManualFitDiagnosticCode.INVALID_TIES,
                "a parameter reference may belong to only one tie",
            )
        seen.update(group.members)
    if seen.intersection(model._legacy_shared_center_references()):
        raise _ManualFitBlocker(
            ManualFitDiagnosticCode.INVALID_TIES,
            "general parameter ties cannot overlap legacy shared-center state",
        )


def _validate_initial_evaluation(
    model: SpectralModelDefinition,
    inputs: _FitInputs,
    *,
    group_index: int,
) -> None:
    """Validate the deterministic residual state submitted to the optimizer."""

    initial, _, _, free_indices = _free_problem(model)
    try:
        initial_model = _model_from_values(
            model,
            _expand_free_values(model, free_indices, initial),
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            evaluation = evaluate_spectral_model(
                inputs.plan,
                initial_model,
                inputs.energy,
            )
            raw_residuals = evaluation.total - inputs.intensity
            standardized_residuals = raw_residuals / inputs.sigma
    except (ConvolutionError, ValueError, FloatingPointError) as error:
        raise _ManualFitBlocker(
            ManualFitDiagnosticCode.NONFINITE_INITIAL_EVALUATION,
            f"initial model/residual evaluation failed: {error}",
            group_index=group_index,
        ) from error
    if not (
        np.all(np.isfinite(evaluation.total))
        and np.all(np.isfinite(raw_residuals))
        and np.all(np.isfinite(standardized_residuals))
    ):
        raise _ManualFitBlocker(
            ManualFitDiagnosticCode.NONFINITE_INITIAL_EVALUATION,
            "initial model, raw residuals, and standardized residuals "
            "must be finite before optimizer entry",
            group_index=group_index,
        )


def _validate_manual_fit_request(
    prepared_resolution: PreparedResolution,
    selection: FittingSelection,
    group_index: int,
    model: SpectralModelDefinition,
) -> _FitInputs:
    if not isinstance(model, SpectralModelDefinition):
        raise _ManualFitBlocker(
            ManualFitDiagnosticCode.INVALID_PARAMETER_CONFIGURATION,
            "model must be a SpectralModelDefinition",
            group_index=group_index if isinstance(group_index, int) else None,
        )
    _validate_parameter_tie_integrity(model)
    try:
        slots = _parameter_slots(model)
    except ValueError as error:
        raise _ManualFitBlocker(
            ManualFitDiagnosticCode.INVALID_TIES,
            str(error),
            group_index=group_index if isinstance(group_index, int) else None,
        ) from error
    free_count = sum(slot.configuration.free for slot in slots)
    inputs = _fit_inputs(prepared_resolution, selection, group_index, free_count)
    plan = inputs.plan
    center_slots = tuple(
        slot
        for slot in slots
        if slot.references
        and all(
            reference.family is ParameterFamily.CENTER for reference in slot.references
        )
    )
    for slot in center_slots:
        _validate_center_coverage(
            slot.name,
            slot.configuration,
            inputs,
            plan,
            group_index=group_index,
            reference=slot.references[0],
        )
    _validate_initial_evaluation(model, inputs, group_index=group_index)
    return inputs


def manual_fit_readiness(
    prepared_resolution: PreparedResolution,
    selection: FittingSelection,
    group_index: int,
    model: SpectralModelDefinition,
) -> ManualFitReadiness:
    """Return structured readiness using the exact fit-time validation path."""

    try:
        _validate_manual_fit_request(
            prepared_resolution,
            selection,
            group_index,
            model,
        )
    except _ManualFitBlocker as error:
        return ManualFitReadiness(False, (error.diagnostic(),))
    except ValueError as error:
        diagnostic = ManualFitReadinessDiagnostic(
            code=ManualFitDiagnosticCode.INVALID_PARAMETER_CONFIGURATION,
            severity=DiagnosticSeverity.ERROR,
            message=str(error),
            group_index=group_index if isinstance(group_index, int) else None,
        )
        return ManualFitReadiness(False, (diagnostic,))
    return ManualFitReadiness(True)


def _fit_with_starts(
    prepared_resolution: PreparedResolution,
    selection: FittingSelection,
    group_index: int,
    models: Sequence[SpectralModelDefinition],
    *,
    max_nfev: int,
) -> FitResult:
    if not models:
        raise FittingError("at least one model initialization is required")
    template = models[0]
    template_slots = _parameter_slots(template)
    if any(
        tuple(
            (
                slot.name,
                slot.key,
                slot.configuration.lower_bound,
                slot.configuration.upper_bound,
                slot.configuration.free,
            )
            for slot in _parameter_slots(model)
        )
        != tuple(
            (
                slot.name,
                slot.key,
                slot.configuration.lower_bound,
                slot.configuration.upper_bound,
                slot.configuration.free,
            )
            for slot in template_slots
        )
        or model.background is not template.background
        or model.lorentzian_count != template.lorentzian_count
        or model.elastic_center_group != template.elastic_center_group
        or tuple(group.group_id for group in model.center_groups)
        != tuple(group.group_id for group in template.center_groups)
        or tuple(component.center_group for component in model.lorentzians)
        != tuple(component.center_group for component in template.lorentzians)
        or tuple(component.identity for component in model.lorentzians)
        != tuple(component.identity for component in template.lorentzians)
        for model in models[1:]
    ):
        raise FittingError(
            "all starts must share model structure, bounds, and free state"
        )
    free_count = sum(slot.configuration.free for slot in template_slots)
    inputs = _validate_manual_fit_request(
        prepared_resolution,
        selection,
        group_index,
        template,
    )
    plan = inputs.plan
    initial, lower, upper, free_indices = _free_problem(template)
    del initial

    def residual(
        free_values: npt.ArrayLike, start_model: SpectralModelDefinition
    ) -> FloatArray:
        full_values = _expand_free_values(start_model, free_indices, free_values)
        evaluation = evaluate_spectral_model(
            plan,
            _model_from_values(start_model, full_values),
            inputs.energy,
        )
        return np.asarray(
            (evaluation.total - inputs.intensity) / inputs.sigma,
            dtype=np.float64,
        )

    solutions: list[_Solution] = []
    if free_count == 0:
        residuals = residual(np.empty(0), template)
        optimized = type(
            "FixedResult",
            (),
            {
                "x": np.empty(0),
                "fun": residuals,
                "jac": np.empty((inputs.energy.size, 0)),
                "success": True,
                "status": 0,
                "message": "all parameters fixed; optimizer not called",
                "nfev": 1,
                "njev": None,
                "active_mask": np.empty(0, dtype=np.int8),
            },
        )()
        solutions.append(
            _Solution(
                optimized=optimized,
                chi_square=float(np.sum(np.square(residuals))),
                elapsed_seconds=0.0,
                start_index=0,
            )
        )
    else:
        for start_index, start_model in enumerate(models):
            start, _, _, start_free_indices = _free_problem(start_model)
            if start_free_indices != free_indices:
                raise FittingError("optimizer starts have inconsistent free parameters")
            started = time.perf_counter()
            optimized = least_squares(
                lambda values, current=start_model: residual(values, current),
                start,
                bounds=(lower, upper),
                method=_METHOD,
                x_scale="jac",
                max_nfev=max_nfev,
            )
            elapsed = time.perf_counter() - started
            solutions.append(
                _Solution(
                    optimized=optimized,
                    chi_square=float(np.sum(np.square(optimized.fun))),
                    elapsed_seconds=elapsed,
                    start_index=start_index,
                )
            )
    successful_solutions = [
        solution for solution in solutions if bool(solution.optimized.success)
    ]
    best = min(
        successful_solutions if successful_solutions else solutions,
        key=lambda item: item.chi_square,
    )
    best_model = models[best.start_index]
    full_values = _expand_free_values(best_model, free_indices, best.optimized.x)
    free_covariance, singular_values, rank, condition = _covariance_from_jacobian(
        np.asarray(best.optimized.jac, dtype=np.float64)
    )
    if not best.optimized.success:
        free_covariance = None
    active_lower = np.zeros(len(template_slots), dtype=np.bool_)
    active_upper = np.zeros(len(template_slots), dtype=np.bool_)
    if free_indices:
        optimizer_active = np.asarray(best.optimized.active_mask)
        indices = np.asarray(free_indices, dtype=np.int64)
        active_lower[indices] = optimizer_active == -1
        active_upper[indices] = optimizer_active == 1
    if np.any(active_lower | active_upper):
        free_covariance = None
    covariance = _full_covariance(free_covariance, len(template_slots), free_indices)
    component_order = _canonical_component_order(full_values, best_model)
    permutation = _canonical_permutation(full_values, best_model)
    full_values = full_values[np.asarray(permutation, dtype=np.int64)]
    active_lower = active_lower[np.asarray(permutation, dtype=np.int64)]
    active_upper = active_upper[np.asarray(permutation, dtype=np.int64)]
    if covariance is not None:
        covariance = covariance[np.ix_(permutation, permutation)]
    canonical_template = SpectralModelDefinition(
        energy_shift=best_model.energy_shift,
        elastic_area=best_model.elastic_area,
        lorentzians=tuple(best_model.lorentzians[index] for index in component_order),
        background=best_model.background,
        b0=best_model.b0,
        b1=best_model.b1,
        center_groups=best_model.center_groups,
        elastic_center_group=best_model.elastic_center_group,
        parameter_ties=best_model.parameter_ties,
    )
    canonical_configurations = tuple(
        slot.configuration for slot in _parameter_slots(canonical_template)
    )
    # Values and their component-specific bounds/free states move together.
    fitted_model = _model_from_values(canonical_template, full_values)
    evaluation = evaluate_spectral_model(plan, fitted_model, inputs.energy)
    raw_residuals = np.asarray(evaluation.total - inputs.intensity)
    standardized = np.asarray(raw_residuals / inputs.sigma)
    correlation, maximum_correlation = _correlation_from_covariance(covariance)
    standard_errors: list[float | None] = []
    for index, configuration in enumerate(canonical_configurations):
        error: float | None = None
        if (
            configuration.free
            and covariance is not None
            and not active_lower[index]
            and not active_upper[index]
        ):
            variance = float(covariance[index, index])
            if np.isfinite(variance) and variance >= 0.0:
                error = math.sqrt(variance)
        standard_errors.append(error)
    names = tuple(slot.name for slot in _parameter_slots(fitted_model))
    parameters = tuple(
        ParameterEstimate(
            name=name,
            value=float(value),
            standard_error=error,
            lower_bound=configuration.lower_bound,
            upper_bound=configuration.upper_bound,
            free=configuration.free,
            active_lower_bound=bool(lower_hit),
            active_upper_bound=bool(upper_hit),
            references=slot.references,
        )
        for slot, name, value, error, configuration, lower_hit, upper_hit in zip(
            _parameter_slots(fitted_model),
            names,
            full_values,
            standard_errors,
            canonical_configurations,
            active_lower,
            active_upper,
            strict=True,
        )
    )
    active_bounds = tuple(
        f"{parameter.name}:{'lower' if parameter.active_lower_bound else 'upper'}"
        for parameter in parameters
        if parameter.active_lower_bound or parameter.active_upper_bound
    )
    relative_errors = tuple(
        None
        if parameter.standard_error is None
        else (
            parameter.standard_error / abs(parameter.value)
            if parameter.value != 0.0
            else math.inf
        )
        for parameter in parameters
    )
    lorentzian_areas = tuple(
        component.area.initial_value for component in fitted_model.lorentzians
    )
    linewidths = tuple(
        component.fwhm.initial_value for component in fitted_model.lorentzians
    )
    adjacent_ratios = tuple(
        linewidths[index + 1] / linewidths[index]
        for index in range(len(linewidths) - 1)
    )
    component_area_to_error: list[float | None] = []
    for component, area in zip(
        fitted_model.lorentzians,
        lorentzian_areas,
        strict=True,
    ):
        area_reference = ParameterReference(component.identity, ParameterFamily.AREA)
        error = next(
            parameter.standard_error
            for parameter in parameters
            if area_reference in parameter.references
        )
        component_area_to_error.append(
            None if error is None else (area / error if error > 0.0 else math.inf)
        )
    try:
        resolution_fwhm: float | None = _profile_fwhm(
            plan.resolution_energy,
            plan.resolution_values,
        )
    except FittingError:
        resolution_fwhm = None
    fitting_window = float(inputs.energy[-1] - inputs.energy[0])
    median_sample_spacing = float(np.median(np.diff(inputs.energy)))
    full_areas: list[float] = []
    retained_areas: list[float] = []
    for component, retained_contribution in zip(
        fitted_model.lorentzians,
        evaluation.lorentzians,
        strict=True,
    ):
        intrinsic = cell_integrated_lorentzian(
            plan.model_energy,
            fwhm=component.fwhm.initial_value,
            spacing=plan.spacing,
        )
        full_profile = plan.convolve(intrinsic)
        full_areas.append(
            component.area.initial_value
            * float(np.trapezoid(full_profile.values, full_profile.energy))
        )
        retained_areas.append(float(np.trapezoid(retained_contribution, inputs.energy)))
    alternative_starts: list[AlternativeStartResult] = []
    for solution in solutions:
        start_model = models[solution.start_index]
        start_free_values, _, _, _ = _free_problem(start_model)
        submitted_start_values = _expand_free_values(
            start_model,
            free_indices,
            start_free_values,
        )
        fitted_start_values = _expand_free_values(
            start_model,
            free_indices,
            solution.optimized.x,
        )
        canonical_component_order = _canonical_component_order(
            fitted_start_values,
            start_model,
        )
        start_permutation = _canonical_permutation(
            fitted_start_values,
            start_model,
        )
        fitted_start_values = fitted_start_values[
            np.asarray(start_permutation, dtype=np.int64)
        ]
        alternative_starts.append(
            AlternativeStartResult(
                start_index=solution.start_index,
                success=bool(solution.optimized.success),
                status=int(solution.optimized.status),
                chi_square=solution.chi_square,
                evaluations=int(solution.optimized.nfev),
                elapsed_seconds=solution.elapsed_seconds,
                start_parameter_values=tuple(
                    float(value) for value in submitted_start_values
                ),
                fitted_parameter_values=tuple(
                    float(value) for value in fitted_start_values
                ),
                canonical_component_order=canonical_component_order,
            )
        )
    observations = inputs.energy.size
    degrees_of_freedom = observations - free_count
    chi_square = float(np.sum(np.square(standardized)))
    aic, aicc, bic = _information_criteria(chi_square, observations, free_count)
    diagnostics = FitDiagnostics(
        optimizer_success=bool(best.optimized.success),
        optimizer_status=int(best.optimized.status),
        optimizer_message=str(best.optimized.message),
        function_evaluations=int(best.optimized.nfev),
        jacobian_evaluations=(
            int(best.optimized.njev) if best.optimized.njev is not None else None
        ),
        jacobian_rank=rank,
        jacobian_singular_values=singular_values,
        condition_number=condition,
        covariance_available=covariance is not None,
        maximum_absolute_correlation=maximum_correlation,
        active_bounds=active_bounds,
        relative_standard_errors=relative_errors,
        lorentzian_areas=lorentzian_areas,
        lorentzian_fwhm=linewidths,
        adjacent_fwhm_ratios=adjacent_ratios,
        component_area_to_standard_error=tuple(component_area_to_error),
        lorentzian_full_convolution_areas=tuple(full_areas),
        lorentzian_retained_sampled_trapezoid_areas=tuple(retained_areas),
        fwhm_to_resolution_fwhm=tuple(
            None if resolution_fwhm is None else width / resolution_fwhm
            for width in linewidths
        ),
        fwhm_to_fitting_window=tuple(width / fitting_window for width in linewidths),
        resolution_fwhm=resolution_fwhm,
        median_sample_spacing=median_sample_spacing,
        resolution_fwhm_to_sample_spacing=(
            None if resolution_fwhm is None else resolution_fwhm / median_sample_spacing
        ),
        residual=_residual_diagnostics(standardized),
        alternative_starts=tuple(alternative_starts),
        selected_start_index=best.start_index,
        total_elapsed_seconds=sum(item.elapsed_seconds for item in solutions),
    )
    return FitResult(
        configuration=best_model,
        parameters=parameters,
        covariance=covariance,
        correlation=correlation,
        evaluation=evaluation,
        raw_residuals=raw_residuals,
        standardized_residuals=standardized,
        statistics=FitStatistics(
            chi_square=chi_square,
            reduced_chi_square=chi_square / degrees_of_freedom,
            observations=observations,
            free_parameters=free_count,
            nominal_degrees_of_freedom=degrees_of_freedom,
            aic=aic,
            aicc=aicc,
            bic=bic,
        ),
        diagnostics=diagnostics,
        provenance=FitProvenance(
            group_index=group_index,
            group_label=inputs.group_label,
            q_value=inputs.q_value,
            energy_unit="meV",
            optimizer=_OPTIMIZER,
            optimizer_method=_METHOD,
            residual_definition="(model - data) / sigma",
            sigma_interpretation="absolute experimental standard deviation",
            convolution_spacing=plan.spacing,
            retained_energy_bounds=(float(inputs.energy[0]), float(inputs.energy[-1])),
            model_energy_bounds=(
                float(plan.model_energy[0]),
                float(plan.model_energy[-1]),
            ),
            convolution_energy_bounds=(
                float(plan.convolution_energy[0]),
                float(plan.convolution_energy[-1]),
            ),
            resolution_acceptance=prepared_resolution.acceptance_provenance(
                group_index
            ),
        ),
        fitted_model=fitted_model,
    )


def fit_single_q(
    prepared_resolution: PreparedResolution,
    selection: FittingSelection,
    group_index: int,
    model: SpectralModelDefinition,
    *,
    max_nfev: int = 2500,
) -> FitResult:
    """Fit one explicitly configured arbitrary-N model to one selected Q group."""

    if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev < 1:
        raise FittingError("max_nfev must be a positive integer")
    return _fit_with_starts(
        prepared_resolution,
        selection,
        group_index,
        (model,),
        max_nfev=max_nfev,
    )


def _provisional_elastic(
    plan: ConvolutionPlan,
    energy: FloatArray,
    data: FloatArray,
    sigma: FloatArray,
) -> tuple[float, float]:
    window = float(energy[-1] - energy[0])
    resolution_width = _profile_fwhm(plan.resolution_energy, plan.resolution_values)
    span = min(0.15 * window, max(3.0 * resolution_width, 5.0 * plan.spacing))
    weights = 1.0 / np.square(sigma)
    edge_count = max(3, energy.size // 12)
    baseline = float(np.median(np.r_[data[:edge_count], data[-edge_count:]]))
    centered = data - baseline
    best = (math.inf, 0.0, 0.0)
    for shift in np.linspace(-span, span, 41):
        shape = np.interp(
            energy - shift,
            plan.resolution_energy,
            plan.resolution_values,
            left=0.0,
            right=0.0,
        )
        denominator = float(np.sum(weights * shape * shape))
        if denominator <= 0.0:
            continue
        area = max(0.0, float(np.sum(weights * shape * centered)) / denominator)
        chi_square = float(np.sum(np.square((area * shape + baseline - data) / sigma)))
        if chi_square < best[0]:
            best = (chi_square, float(shift), area)
    if not np.isfinite(best[0]):
        raise FittingError("provisional measured-resolution alignment failed")
    return best[1], best[2]


def _standard_initializations(
    inputs: _FitInputs,
    candidate: StandardModelCandidate,
    one_lorentzian_fit: FitResult | None,
) -> tuple[SpectralModelDefinition, ...]:
    plan = inputs.plan
    energy = inputs.energy
    data = inputs.intensity
    sigma = inputs.sigma
    e0, elastic_area = _provisional_elastic(plan, energy, data, sigma)
    elastic_shape = np.interp(
        energy - e0,
        plan.resolution_energy,
        plan.resolution_values,
        left=0.0,
        right=0.0,
    )
    residual = data - elastic_area * elastic_shape
    edge_count = max(3, energy.size // 10)
    left_x = float(np.median(energy[:edge_count]))
    right_x = float(np.median(energy[-edge_count:]))
    left_y = float(np.median(residual[:edge_count]))
    right_y = float(np.median(residual[-edge_count:]))
    b1 = (right_y - left_y) / (right_x - left_x)
    b0 = 0.5 * ((left_y - b1 * left_x) + (right_y - b1 * right_x))
    qe_residual = residual - b0 - b1 * energy
    qe_area = max(
        float(np.trapezoid(np.maximum(qe_residual, 0.0), energy)),
        0.02 * max(float(np.trapezoid(np.abs(data), energy)), 1.0e-10),
    )
    resolution_width = _profile_fwhm(plan.resolution_energy, plan.resolution_values)
    width_seeds = (
        0.25 * resolution_width,
        resolution_width,
        4.0 * resolution_width,
    )
    window = float(energy[-1] - energy[0])
    e0_span = min(0.2 * window, 0.3)
    gamma_floor = max(plan.spacing * 1.0e-5, np.finfo(float).eps * window)

    def parameter(
        initial: float,
        lower: float = -math.inf,
        upper: float = math.inf,
    ) -> ParameterConfiguration:
        clipped = float(np.clip(initial, lower, upper))
        return ParameterConfiguration(clipped, lower, upper)

    component_identities = tuple(
        ComponentIdentity(ComponentFamily.LORENTZIAN, f"standard_lorentzian_{index}")
        for index in range(1, candidate.lorentzian_count + 1)
    )

    def build(components: Sequence[tuple[float, float]]) -> SpectralModelDefinition:
        return SpectralModelDefinition(
            energy_shift=parameter(e0, -e0_span, e0_span),
            elastic_area=parameter(max(elastic_area, 1.0e-10), 0.0),
            lorentzians=tuple(
                LorentzianComponent(
                    area=parameter(max(area, 0.0), 0.0),
                    fwhm=parameter(max(width, gamma_floor), gamma_floor),
                    identity=identity,
                )
                for identity, (area, width) in zip(
                    component_identities,
                    components,
                    strict=True,
                )
            ),
            background=candidate.background,
            b0=(
                parameter(b0)
                if candidate.background is not BackgroundModel.NONE
                else None
            ),
            b1=(
                parameter(b1)
                if candidate.background is BackgroundModel.LINEAR
                else None
            ),
        )

    if candidate.lorentzian_count == 0:
        return (build(()),)
    if candidate.lorentzian_count == 1:
        return tuple(build(((qe_area, width),)) for width in width_seeds)
    if candidate.lorentzian_count == 2:
        if one_lorentzian_fit is None:
            raise FittingError("2L standard initialization requires a fitted 1L seed")
        old_area = one_lorentzian_fit.parameter("lorentzian_1_area").value
        old_width = one_lorentzian_fit.parameter("lorentzian_1_fwhm").value
        e0 = one_lorentzian_fit.parameter("energy_shift").value
        elastic_area = one_lorentzian_fit.parameter("elastic_area").value
        if candidate.background is not BackgroundModel.NONE:
            b0 = one_lorentzian_fit.parameter("b0").value
        if candidate.background is BackgroundModel.LINEAR:
            b1 = one_lorentzian_fit.parameter("b1").value
        return tuple(
            build(
                (
                    (0.7 * old_area, old_width),
                    (0.3 * old_area, new_width),
                )
            )
            for new_width in (0.35 * old_width, 3.0 * old_width)
        )
    raise FittingError(
        "automatic initialization above 2L is not validated; use fit_single_q "
        "with explicit manual initials"
    )


def fit_standard_candidate(
    prepared_resolution: PreparedResolution,
    selection: FittingSelection,
    group_index: int,
    candidate: StandardModelCandidate,
    *,
    one_lorentzian_fit: FitResult | None = None,
    max_nfev: int = 2500,
) -> FitResult:
    """Fit one generated 0L/1L/2L candidate with validated multistart initials."""

    if candidate.lorentzian_count > 2:
        raise FittingError(
            "automatic initialization above 2L is not validated; use fit_single_q"
        )
    nominal_free = candidate.nominal_parameter_count
    inputs = _fit_inputs(prepared_resolution, selection, group_index, nominal_free)
    if candidate.lorentzian_count == 2 and one_lorentzian_fit is None:
        one_lorentzian_fit = fit_standard_candidate(
            prepared_resolution,
            selection,
            group_index,
            StandardModelCandidate(1, candidate.background),
            max_nfev=max_nfev,
        )
    starts = _standard_initializations(inputs, candidate, one_lorentzian_fit)
    return _fit_with_starts(
        prepared_resolution,
        selection,
        group_index,
        starts,
        max_nfev=max_nfev,
    )


def generate_standard_candidates(
    *,
    max_lorentzians: int = 2,
    allow_linear_background: bool = True,
) -> tuple[StandardModelCandidate, ...]:
    """Generate candidate structure only; no recommendation policy is applied."""

    if isinstance(max_lorentzians, bool) or not isinstance(max_lorentzians, int):
        raise ValueError("max_lorentzians must be an integer")
    if max_lorentzians < 0:
        raise ValueError("max_lorentzians must be nonnegative")
    if not isinstance(allow_linear_background, bool):
        raise ValueError("allow_linear_background must be boolean")
    backgrounds = (
        (BackgroundModel.NONE, BackgroundModel.CONSTANT, BackgroundModel.LINEAR)
        if allow_linear_background
        else (BackgroundModel.NONE, BackgroundModel.CONSTANT)
    )
    return tuple(
        StandardModelCandidate(count, background)
        for count in range(max_lorentzians + 1)
        for background in backgrounds
    )


def evaluate_standard_candidates(
    prepared_resolution: PreparedResolution,
    selection: FittingSelection,
    group_index: int,
    *,
    max_lorentzians: int = 2,
    allow_linear_background: bool = True,
    max_nfev: int = 2500,
) -> tuple[CandidateFitResult, ...]:
    """Fit standard candidates and return evidence without choosing a winner."""

    candidates = generate_standard_candidates(
        max_lorentzians=max_lorentzians,
        allow_linear_background=allow_linear_background,
    )
    results: list[CandidateFitResult] = []
    one_lorentzian: dict[BackgroundModel, FitResult] = {}
    for candidate in candidates:
        try:
            fit = fit_standard_candidate(
                prepared_resolution,
                selection,
                group_index,
                candidate,
                one_lorentzian_fit=one_lorentzian.get(candidate.background),
                max_nfev=max_nfev,
            )
        except (FittingError, ValueError) as error:
            results.append(
                CandidateFitResult(
                    candidate=candidate,
                    fit=None,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )
            continue
        results.append(CandidateFitResult(candidate=candidate, fit=fit))
        if candidate.lorentzian_count == 1:
            one_lorentzian[candidate.background] = fit
    return tuple(results)
