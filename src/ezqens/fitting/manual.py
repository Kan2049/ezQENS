"""Thin GUI-independent scientific bridges for single-Q Manual Fit."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Real
from typing import Final

import numpy as np
import numpy.typing as npt

from ezqens.convolution import (
    ConvolutionError,
    ConvolutionPlan,
    build_convolution_plan,
    cell_integrated_lorentzian,
)
from ezqens.preprocessing import FittingSelection
from ezqens.resolution import PreparedResolution

from .core import FittingError, _fit_inputs, _profile_fwhm, evaluate_spectral_model
from .models import (
    BackgroundModel,
    ModelEvaluation,
    ParameterConfiguration,
    SpectralModelDefinition,
)

FloatArray = npt.NDArray[np.float64]
_BOUND_INTERIOR_FRACTION: Final[float] = 1.0e-10
_NARROW_SEED_SPACING_FRACTION: Final[float] = 1.0e-6
_WIDTH_SEARCH_COARSE_POINTS: Final[int] = 97
_WIDTH_SEARCH_REFINEMENT_POINTS: Final[int] = 49
_WIDTH_SEARCH_REFINEMENT_ROUNDS: Final[int] = 6


class ManualInitializationError(FittingError):
    """Raised when interaction geometry cannot define a valid component seed."""


class ManualMaterializationError(FittingError):
    """Raised when a Manual parameter intention cannot become a legal configuration."""


class ManualParameterKind(StrEnum):
    """Current Manual parameter categories with distinct core constraints."""

    NONNEGATIVE_AREA = "nonnegative_area"
    POSITIVE_FWHM = "positive_fwhm"
    CENTER = "center"
    UNCONSTRAINED = "unconstrained"


@dataclass(frozen=True, slots=True)
class ElasticInteractionSeed:
    """Integrated elastic area and center-parameter value from one visual hint."""

    integrated_area: float
    center_parameter_value: float


@dataclass(frozen=True, slots=True)
class BackgroundInteractionSeed:
    """Background model and initial coefficients from two visual points."""

    background: BackgroundModel
    b0_initial_value: float
    b1_initial_value: float


@dataclass(frozen=True, slots=True)
class LorentzianInteractionSeed:
    """Approximate positive initialization derived from visual geometry.

    The forward-width fields are interaction diagnostics only. The fitted
    intrinsic FWHM, rather than this deterministic seed, is the scientific
    result.
    """

    integrated_area: float
    intrinsic_fwhm: float
    center_parameter_value: float
    forward_observed_fwhm: float
    absolute_width_mismatch: float


@dataclass(frozen=True, slots=True)
class ManualParameterIntent:
    """Caller-owned value, dormant-capable user limits, and free state."""

    current_value: float
    user_lower_limit: float | None = None
    user_upper_limit: float | None = None
    free: bool = True
    user_bounds_enabled: bool = True


@dataclass(frozen=True, slots=True)
class ManualParameterMaterialization:
    """Separate preview and fit configurations derived without mutating the intent."""

    intent: ManualParameterIntent
    kind: ManualParameterKind
    preview_configuration: ParameterConfiguration
    fit_configuration: ParameterConfiguration
    fit_start_adjusted: bool


def _readonly_float_array(value: npt.ArrayLike, *, name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class ManualModelPreview:
    """Display evaluation plus exact retained-point residual evaluation."""

    display_evaluation: ModelEvaluation
    retained_evaluation: ModelEvaluation
    retained_energy: FloatArray = field(repr=False)
    retained_intensity: FloatArray = field(repr=False)
    retained_sigma: FloatArray = field(repr=False)
    raw_residuals: FloatArray = field(repr=False)
    standardized_residuals: FloatArray = field(repr=False)

    def __post_init__(self) -> None:
        retained_energy = _readonly_float_array(
            self.retained_energy,
            name="retained_energy",
        )
        retained_intensity = _readonly_float_array(
            self.retained_intensity,
            name="retained_intensity",
        )
        retained_sigma = _readonly_float_array(
            self.retained_sigma,
            name="retained_sigma",
        )
        raw_residuals = _readonly_float_array(
            self.raw_residuals,
            name="raw_residuals",
        )
        standardized_residuals = _readonly_float_array(
            self.standardized_residuals,
            name="standardized_residuals",
        )
        size = retained_energy.size
        if any(
            value.size != size
            for value in (
                retained_intensity,
                retained_sigma,
                raw_residuals,
                standardized_residuals,
            )
        ):
            raise ValueError("Manual preview retained arrays must have equal lengths")
        if self.retained_evaluation.energy.size != size:
            raise ValueError("retained evaluation must use retained sample coordinates")
        if np.any(retained_sigma <= 0.0):
            raise ValueError("retained_sigma must be strictly positive")
        object.__setattr__(self, "retained_energy", retained_energy)
        object.__setattr__(self, "retained_intensity", retained_intensity)
        object.__setattr__(self, "retained_sigma", retained_sigma)
        object.__setattr__(self, "raw_residuals", raw_residuals)
        object.__setattr__(self, "standardized_residuals", standardized_residuals)


def _finite_hint(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ManualInitializationError(f"{name} must be a finite number")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ManualInitializationError(f"{name} must be finite")
    return numeric


def initialize_background_from_interaction(
    *,
    first_energy: float,
    first_height: float,
    second_energy: float,
    second_height: float,
) -> BackgroundInteractionSeed:
    """Convert press-release geometry into an unconstrained B1 initial seed."""

    energy_1 = _finite_hint(first_energy, name="first_energy")
    height_1 = _finite_hint(first_height, name="first_height")
    energy_2 = _finite_hint(second_energy, name="second_energy")
    height_2 = _finite_hint(second_height, name="second_height")
    if energy_1 == energy_2:
        if height_1 != height_2:
            raise ManualInitializationError(
                "different background heights at equal energies do not define a "
                "finite linear background"
            )
        b1 = 0.0
        b0 = height_1
    else:
        b1 = (height_2 - height_1) / (energy_2 - energy_1)
        b0 = height_1 - b1 * energy_1
    if not np.isfinite(b0) or not np.isfinite(b1):
        raise ManualInitializationError(
            "two-point interaction does not define finite background coefficients"
        )
    return BackgroundInteractionSeed(
        background=BackgroundModel.LINEAR,
        b0_initial_value=b0,
        b1_initial_value=b1,
    )


def _center_coverage(
    plan: ConvolutionPlan,
    target_energy: FloatArray | None = None,
) -> tuple[float, float]:
    targets = plan.target_energy if target_energy is None else target_energy
    return (
        float(targets[-1] - plan.convolution_energy[-1]),
        float(targets[0] - plan.convolution_energy[0]),
    )


def _validate_center_seed(plan: ConvolutionPlan, center: float) -> None:
    lower, upper = _center_coverage(plan)
    if not lower <= center <= upper:
        raise ManualInitializationError(
            "interaction center cannot be represented on the fixed convolution domain"
        )


def initialize_elastic_from_interaction(
    prepared_resolution: PreparedResolution,
    group_index: int,
    *,
    component_peak_center: float,
    component_peak_height: float,
) -> ElasticInteractionSeed:
    """Convert an elastic-component peak hint into area and center seeds.

    The height is the desired elastic-component height, not total-model height.
    The returned center value shifts the actual measured-resolution profile so
    its numerical peak lies at ``component_peak_center``.
    """

    peak_center = _finite_hint(component_peak_center, name="component_peak_center")
    peak_height = _finite_hint(component_peak_height, name="component_peak_height")
    if peak_height < 0.0:
        raise ManualInitializationError("component_peak_height must be nonnegative")
    try:
        plan = build_convolution_plan(prepared_resolution, group_index)
    except (ConvolutionError, ValueError) as error:
        raise ManualInitializationError(str(error)) from error
    peak_index = int(np.argmax(plan.resolution_values))
    unit_peak_height = float(plan.resolution_values[peak_index])
    if not np.isfinite(unit_peak_height) or unit_peak_height <= 0.0:
        raise ManualInitializationError(
            "measured resolution does not define a positive elastic peak"
        )
    center = peak_center - float(plan.resolution_energy[peak_index])
    _validate_center_seed(plan, center)
    return ElasticInteractionSeed(
        integrated_area=peak_height / unit_peak_height,
        center_parameter_value=center,
    )


def _unit_lorentzian_profile(
    plan: ConvolutionPlan,
    intrinsic_fwhm: float,
) -> tuple[FloatArray, FloatArray]:
    intrinsic = cell_integrated_lorentzian(
        plan.model_energy,
        fwhm=intrinsic_fwhm,
        spacing=plan.spacing,
    )
    profile = plan.convolve(intrinsic)
    return profile.energy, profile.values


@dataclass(frozen=True, slots=True)
class _WidthCandidate:
    intrinsic_fwhm: float
    forward_observed_fwhm: float
    absolute_mismatch: float


def _evaluate_width_candidate(
    plan: ConvolutionPlan,
    intrinsic_fwhm: float,
    observed_fwhm: float,
) -> _WidthCandidate | None:
    if not np.isfinite(intrinsic_fwhm) or intrinsic_fwhm <= 0.0:
        return None
    try:
        energy, values = _unit_lorentzian_profile(plan, intrinsic_fwhm)
        forward_width = _profile_fwhm(energy, values)
    except (ConvolutionError, FittingError, ValueError):
        return None
    mismatch = abs(forward_width - observed_fwhm)
    if (
        not np.isfinite(forward_width)
        or forward_width <= 0.0
        or not np.isfinite(mismatch)
    ):
        return None
    return _WidthCandidate(
        intrinsic_fwhm=float(intrinsic_fwhm),
        forward_observed_fwhm=float(forward_width),
        absolute_mismatch=float(mismatch),
    )


def _select_width_candidate(
    candidates: tuple[_WidthCandidate, ...],
) -> _WidthCandidate:
    if not candidates:
        raise ManualInitializationError(
            "measured-resolution convolution does not define a usable positive "
            "Lorentzian width seed"
        )
    return min(
        candidates,
        key=lambda candidate: (
            candidate.absolute_mismatch,
            candidate.intrinsic_fwhm,
        ),
    )


def _intrinsic_fwhm_seed(
    plan: ConvolutionPlan,
    observed_fwhm: float,
) -> _WidthCandidate:
    """Find a deterministic approximate seed by forward profile evaluation.

    The measured profile can be asymmetric or structured, so no monotonic or
    unique inverse relation between intrinsic and visible FWHM is assumed.
    """

    lower = max(
        np.finfo(np.float64).tiny,
        plan.spacing * _NARROW_SEED_SPACING_FRACTION,
    )
    convolution_span = float(plan.convolution_energy[-1] - plan.convolution_energy[0])
    upper = max(
        lower * 2.0,
        plan.spacing * 256.0,
        convolution_span * 4.0,
    )
    coarse_widths = np.geomspace(lower, upper, _WIDTH_SEARCH_COARSE_POINTS)
    candidates = tuple(
        candidate
        for intrinsic_fwhm in coarse_widths
        if (
            candidate := _evaluate_width_candidate(
                plan,
                float(intrinsic_fwhm),
                observed_fwhm,
            )
        )
        is not None
    )
    best = _select_width_candidate(candidates)

    coarse_factor = float(
        np.exp(np.log(upper / lower) / (_WIDTH_SEARCH_COARSE_POINTS - 1))
    )
    refinement_factor = coarse_factor
    for _ in range(_WIDTH_SEARCH_REFINEMENT_ROUNDS):
        refinement_widths = np.geomspace(
            max(lower, best.intrinsic_fwhm / refinement_factor),
            min(upper, best.intrinsic_fwhm * refinement_factor),
            _WIDTH_SEARCH_REFINEMENT_POINTS,
        )
        refined = tuple(
            candidate
            for intrinsic_fwhm in refinement_widths
            if (
                candidate := _evaluate_width_candidate(
                    plan,
                    float(intrinsic_fwhm),
                    observed_fwhm,
                )
            )
            is not None
        )
        best = _select_width_candidate((best, *refined))
        refinement_factor = float(np.sqrt(refinement_factor))
    return best


def initialize_lorentzian_from_interaction(
    prepared_resolution: PreparedResolution,
    group_index: int,
    *,
    component_peak_center: float,
    component_peak_height: float,
    observed_fwhm: float,
) -> LorentzianInteractionSeed:
    """Convert convolved Lorentzian geometry into non-physical initial seeds.

    A deterministic coarse-to-fine forward search approximates the observed
    width after measured-resolution convolution. Structured profiles can make
    the inverse non-unique, so this is an initial-guess operation rather than
    physical deconvolution or a mechanism classification.
    """

    peak_center = _finite_hint(component_peak_center, name="component_peak_center")
    peak_height = _finite_hint(component_peak_height, name="component_peak_height")
    width = _finite_hint(observed_fwhm, name="observed_fwhm")
    if peak_height < 0.0:
        raise ManualInitializationError("component_peak_height must be nonnegative")
    if width <= 0.0:
        raise ManualInitializationError("observed_fwhm must be strictly positive")
    try:
        plan = build_convolution_plan(prepared_resolution, group_index)
        width_candidate = _intrinsic_fwhm_seed(plan, width)
        profile_energy, profile_values = _unit_lorentzian_profile(
            plan,
            width_candidate.intrinsic_fwhm,
        )
    except ManualInitializationError:
        raise
    except (ConvolutionError, FittingError, ValueError) as error:
        raise ManualInitializationError(str(error)) from error
    peak_index = int(np.argmax(profile_values))
    unit_peak_height = float(profile_values[peak_index])
    if not np.isfinite(unit_peak_height) or unit_peak_height <= 0.0:
        raise ManualInitializationError(
            "convolved Lorentzian does not define a positive component peak"
        )
    center = peak_center - float(profile_energy[peak_index])
    _validate_center_seed(plan, center)
    return LorentzianInteractionSeed(
        integrated_area=peak_height / unit_peak_height,
        intrinsic_fwhm=width_candidate.intrinsic_fwhm,
        center_parameter_value=center,
        forward_observed_fwhm=width_candidate.forward_observed_fwhm,
        absolute_width_mismatch=width_candidate.absolute_mismatch,
    )


def _intent_number(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ManualMaterializationError(f"{name} must be a finite number or None")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ManualMaterializationError(f"{name} must be finite when supplied")
    return numeric


def _strict_interior_start(value: float, lower: float, upper: float) -> float:
    if lower < value < upper:
        return value
    if value <= lower:
        scale = max(1.0, abs(lower))
        candidate = lower + _BOUND_INTERIOR_FRACTION * scale
        if not candidate < upper:
            candidate = float(np.nextafter(lower, upper))
        if lower < candidate < upper:
            return candidate
    if value >= upper:
        scale = max(1.0, abs(upper))
        candidate = upper - _BOUND_INTERIOR_FRACTION * scale
        if not lower < candidate:
            candidate = float(np.nextafter(upper, lower))
        if lower < candidate < upper:
            return candidate
    raise ManualMaterializationError(
        "free parameter has no usable interior initialization point"
    )


def materialize_manual_parameter(
    intent: ManualParameterIntent,
    kind: ManualParameterKind,
    *,
    prepared_resolution: PreparedResolution | None = None,
    selection: FittingSelection | None = None,
    group_index: int | None = None,
) -> ManualParameterMaterialization:
    """Create separate preview and fit configurations from one Manual intention.

    User limits are deliberately absent from the preview configuration. They are
    intersected with current core constraints only for fit materialization.
    Center parameters require the selected prepared-resolution/selection context.
    """

    if not isinstance(intent, ManualParameterIntent):
        raise ManualMaterializationError("intent must be a ManualParameterIntent")
    if not isinstance(kind, ManualParameterKind):
        raise ManualMaterializationError("kind must be a ManualParameterKind")
    if not isinstance(intent.free, bool):
        raise ManualMaterializationError("free must be a boolean")
    if not isinstance(intent.user_bounds_enabled, bool):
        raise ManualMaterializationError("user_bounds_enabled must be a boolean")
    current = _intent_number(intent.current_value, name="current_value")
    if current is None:  # excluded by the non-optional field
        raise ManualMaterializationError("current_value must be finite")
    user_lower = _intent_number(intent.user_lower_limit, name="user_lower_limit")
    user_upper = _intent_number(intent.user_upper_limit, name="user_upper_limit")
    if (
        intent.user_bounds_enabled
        and user_lower is not None
        and user_upper is not None
        and user_lower > user_upper
    ):
        raise ManualMaterializationError("user lower limit must not exceed upper limit")

    scientific_lower = -np.inf
    scientific_upper = np.inf
    if kind is ManualParameterKind.NONNEGATIVE_AREA:
        scientific_lower = 0.0
        if current < 0.0:
            raise ManualMaterializationError("area current value must be nonnegative")
    elif kind is ManualParameterKind.POSITIVE_FWHM:
        scientific_lower = float(np.nextafter(0.0, 1.0))
        if current <= 0.0:
            raise ManualMaterializationError(
                "FWHM current value must be strictly positive"
            )
    elif kind is ManualParameterKind.CENTER:
        if prepared_resolution is None or selection is None or group_index is None:
            raise ManualMaterializationError(
                "center materialization requires prepared resolution, selection, "
                "and group_index"
            )
        try:
            inputs = _fit_inputs(prepared_resolution, selection, group_index, 0)
        except (ConvolutionError, FittingError, ValueError) as error:
            raise ManualMaterializationError(str(error)) from error
        scientific_lower, scientific_upper = _center_coverage(
            inputs.plan,
            inputs.energy,
        )
        if not scientific_lower <= current <= scientific_upper:
            raise ManualMaterializationError(
                "center current value falls outside fixed convolution-domain coverage"
            )

    preview = ParameterConfiguration(
        initial_value=current,
        lower_bound=scientific_lower,
        upper_bound=scientific_upper,
        free=intent.free,
    )
    active_user_lower = user_lower if intent.user_bounds_enabled else None
    active_user_upper = user_upper if intent.user_bounds_enabled else None
    fit_lower = max(
        scientific_lower,
        active_user_lower if active_user_lower is not None else -np.inf,
    )
    fit_upper = min(
        scientific_upper,
        active_user_upper if active_user_upper is not None else np.inf,
    )
    if fit_lower > fit_upper:
        raise ManualMaterializationError(
            "user limits have no intersection with required core constraints"
        )
    if intent.free:
        if fit_lower >= fit_upper:
            raise ManualMaterializationError(
                "free parameter requires a usable interval after applying user limits"
            )
        fit_start = _strict_interior_start(current, fit_lower, fit_upper)
    else:
        if not fit_lower <= current <= fit_upper:
            raise ManualMaterializationError(
                "fixed current value lies outside the final allowed interval"
            )
        fit_start = current
    fit = ParameterConfiguration(
        initial_value=fit_start,
        lower_bound=fit_lower,
        upper_bound=fit_upper,
        free=intent.free,
    )
    return ManualParameterMaterialization(
        intent=intent,
        kind=kind,
        preview_configuration=preview,
        fit_configuration=fit,
        fit_start_adjusted=fit_start != current,
    )


def preview_manual_model(
    prepared_resolution: PreparedResolution,
    selection: FittingSelection,
    group_index: int,
    model: SpectralModelDefinition,
    *,
    display_energy: npt.ArrayLike | None = None,
) -> ManualModelPreview:
    """Evaluate one Manual model and exact retained-point standardized residuals."""

    if not isinstance(model, SpectralModelDefinition):
        raise FittingError("model must be a SpectralModelDefinition")
    inputs = _fit_inputs(prepared_resolution, selection, group_index, 0)
    display_coordinates = (
        inputs.plan.target_energy if display_energy is None else display_energy
    )
    display_evaluation = evaluate_spectral_model(
        inputs.plan,
        model,
        display_coordinates,
    )
    retained_evaluation = evaluate_spectral_model(
        inputs.plan,
        model,
        inputs.energy,
    )
    raw_residuals = np.asarray(
        retained_evaluation.total - inputs.intensity,
        dtype=np.float64,
    )
    standardized_residuals = np.asarray(
        raw_residuals / inputs.sigma,
        dtype=np.float64,
    )
    return ManualModelPreview(
        display_evaluation=display_evaluation,
        retained_evaluation=retained_evaluation,
        retained_energy=inputs.energy,
        retained_intensity=inputs.intensity,
        retained_sigma=inputs.sigma,
        raw_residuals=raw_residuals,
        standardized_residuals=standardized_residuals,
    )
