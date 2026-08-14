"""Typed single-Q spectral-model and fit-result values."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import numpy.typing as npt

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
class LorentzianComponent:
    """One unit-area intrinsic Lorentzian with integrated-area amplitude."""

    area: ParameterConfiguration
    fwhm: ParameterConfiguration

    def __post_init__(self) -> None:
        if self.area.lower_bound < 0.0 or self.area.initial_value < 0.0:
            raise ValueError("Lorentzian integrated area must be nonnegative")
        if self.fwhm.lower_bound <= 0.0 or self.fwhm.initial_value <= 0.0:
            raise ValueError("Lorentzian FWHM must be strictly positive")


@dataclass(frozen=True, slots=True)
class SpectralModelDefinition:
    """One elastic component, variable Lorentzians, shared E0, and background."""

    energy_shift: ParameterConfiguration
    elastic_area: ParameterConfiguration
    lorentzians: tuple[LorentzianComponent, ...] = ()
    background: BackgroundModel = BackgroundModel.NONE
    b0: ParameterConfiguration | None = None
    b1: ParameterConfiguration | None = None

    def __post_init__(self) -> None:
        lorentzians = tuple(self.lorentzians)
        if any(not isinstance(item, LorentzianComponent) for item in lorentzians):
            raise ValueError("lorentzians must contain LorentzianComponent values")
        if not isinstance(self.background, BackgroundModel):
            raise ValueError("background must be a BackgroundModel")
        if self.elastic_area.lower_bound < 0.0 or self.elastic_area.initial_value < 0.0:
            raise ValueError("elastic integrated area must be nonnegative")
        if self.background is BackgroundModel.NONE:
            if self.b0 is not None or self.b1 is not None:
                raise ValueError("NONE background must not define b0 or b1")
        elif self.background is BackgroundModel.CONSTANT:
            if self.b0 is None or self.b1 is not None:
                raise ValueError("B0 background requires b0 and no b1")
        elif self.b0 is None or self.b1 is None:
            raise ValueError("B1 background requires both b0 and b1")
        object.__setattr__(self, "lorentzians", lorentzians)

    @property
    def lorentzian_count(self) -> int:
        """Return the unrestricted component count represented by this model."""

        return len(self.lorentzians)


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
class ModelEvaluation:
    """Component-resolved model values at caller-supplied physical coordinates."""

    energy: FloatArray = field(repr=False)
    total: FloatArray = field(repr=False)
    elastic: FloatArray = field(repr=False)
    lorentzians: tuple[FloatArray, ...] = field(repr=False)
    background: FloatArray = field(repr=False)

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
        object.__setattr__(self, "elastic", elastic)
        object.__setattr__(self, "background", background)
        object.__setattr__(self, "lorentzians", lorentzians)


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

    def parameter(self, name: str) -> ParameterEstimate:
        """Return one estimate by its canonical result name."""

        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(name)

    @property
    def model(self) -> SpectralModelDefinition:
        """Return the submitted model configuration."""

        return self.configuration


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
