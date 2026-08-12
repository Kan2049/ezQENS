"""Fixed-grid numerical convolution using prepared measured resolution data."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt

from ezqens.resolution import PreparedResolution

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]

CANONICAL_ENERGY_UNIT: Final[str] = "meV"
_LATTICE_ULPS: Final[float] = 16.0


class ConvolutionError(ValueError):
    """Raised when a scientifically valid numerical convolution cannot be built."""


def _readonly_float_array(value: npt.ArrayLike, *, name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ConvolutionError(f"{name} must be one-dimensional")
    array.setflags(write=False)
    return array


def _readonly_complex_array(value: npt.ArrayLike, *, name: str) -> ComplexArray:
    array = np.array(value, dtype=np.complex128, copy=True)
    if array.ndim != 1:
        raise ConvolutionError(f"{name} must be one-dimensional")
    array.setflags(write=False)
    return array


def _validate_coordinates(
    value: npt.ArrayLike,
    *,
    name: str,
    minimum_size: int = 2,
) -> FloatArray:
    coordinates = _readonly_float_array(value, name=name)
    if coordinates.size < minimum_size:
        raise ConvolutionError(
            f"{name} must contain at least {minimum_size} coordinates"
        )
    if not np.all(np.isfinite(coordinates)):
        raise ConvolutionError(f"{name} must contain only finite coordinates")
    if np.any(np.diff(coordinates) <= 0.0):
        raise ConvolutionError(
            f"{name} must be strictly increasing; values are not sorted or repaired"
        )
    return coordinates


def _characteristic_spacing(coordinates: FloatArray) -> float:
    differences = np.diff(coordinates)
    spacing = float(np.median(differences))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ConvolutionError("energy coordinates do not define a usable spacing")
    return spacing


def _validate_uniform_spacing(
    coordinates: FloatArray,
    spacing: float,
    *,
    name: str,
) -> None:
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ConvolutionError("convolution spacing must be finite and positive")
    coordinate_scale = max(
        abs(float(coordinates[0])),
        abs(float(coordinates[-1])),
        spacing,
    )
    absolute_tolerance = _LATTICE_ULPS * np.finfo(np.float64).eps * coordinate_scale
    if not np.allclose(
        np.diff(coordinates),
        spacing,
        rtol=_LATTICE_ULPS * np.finfo(np.float64).eps,
        atol=absolute_tolerance,
    ):
        raise ConvolutionError(f"{name} increments must equal spacing")


def automatic_grid_spacing(
    sample_energy: npt.ArrayLike,
    resolution_energy: npt.ArrayLike,
) -> float:
    """Return the approved S1/4 internal spacing in canonical energy units."""

    sample = _validate_coordinates(sample_energy, name="sample_energy")
    resolution = _validate_coordinates(resolution_energy, name="resolution_energy")
    spacing = (
        min(
            _characteristic_spacing(sample),
            _characteristic_spacing(resolution),
        )
        / 4.0
    )
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ConvolutionError("automatic convolution spacing is not usable")
    return spacing


def _snap_near_integer(value: float) -> float:
    nearest = float(round(value))
    tolerance = _LATTICE_ULPS * np.finfo(np.float64).eps * max(1.0, abs(value))
    return nearest if abs(value - nearest) <= tolerance else value


def _zero_anchored_axis(lower: float, upper: float, spacing: float) -> FloatArray:
    """Cover physical bounds with the minimal zero-anchored cell-center lattice."""

    lower_index = math.floor(_snap_near_integer(lower / spacing))
    upper_index = math.ceil(_snap_near_integer(upper / spacing))
    if upper_index < lower_index:
        raise ConvolutionError("model convolution domain is unusable")
    indices = np.arange(lower_index, upper_index + 1, dtype=np.float64)
    return np.asarray(indices * spacing, dtype=np.float64)


def _outward_axis(lower: float, upper: float, spacing: float) -> FloatArray:
    """Cover both bounds at fixed spacing without moving the lower coordinate."""

    interval_ratio = _snap_near_integer((upper - lower) / spacing)
    interval_count = math.ceil(interval_ratio)
    if interval_count < 1:
        raise ConvolutionError("uniform convolution axis is unusable")
    return np.asarray(
        lower + spacing * np.arange(interval_count + 1, dtype=np.float64),
        dtype=np.float64,
    )


def _next_power_of_two(required_length: int) -> int:
    if required_length < 1:
        raise ConvolutionError("linear convolution length must be positive")
    return 1 << (required_length - 1).bit_length()


def cell_integrated_lorentzian(
    energy: npt.ArrayLike,
    *,
    fwhm: float,
    spacing: float,
    center: float = 0.0,
) -> FloatArray:
    """Return unit-area Lorentzian cell-average densities using FWHM input.

    The interval probability is evaluated with an ``atan2`` form equivalent to
    the difference of arctangents. This avoids subtractive loss in distant,
    narrow-line tails while preserving the approved analytic cell integral.
    """

    coordinates = _readonly_float_array(energy, name="energy")
    if coordinates.size == 0 or not np.all(np.isfinite(coordinates)):
        raise ConvolutionError("energy must contain finite coordinates")
    if not np.isfinite(fwhm) or fwhm <= 0.0:
        raise ConvolutionError("Lorentzian FWHM must be finite and positive")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ConvolutionError("Lorentzian cell spacing must be finite and positive")
    if not np.isfinite(center):
        raise ConvolutionError("Lorentzian center must be finite")

    gamma = fwhm / 2.0
    lower = coordinates - spacing / 2.0 - center
    upper = coordinates + spacing / 2.0 - center
    interval_probability = (
        np.arctan2(
            gamma * spacing,
            gamma * gamma + upper * lower,
        )
        / math.pi
    )
    values = np.asarray(interval_probability / spacing, dtype=np.float64)
    values.setflags(write=False)
    return values


@dataclass(frozen=True, slots=True)
class ConvolvedProfile:
    """One full linear-convolution result with explicit physical coordinates."""

    energy: FloatArray = field(repr=False)
    values: FloatArray = field(repr=False)
    spacing: float

    def __post_init__(self) -> None:
        energy = _validate_coordinates(self.energy, name="convolution energy")
        values = _readonly_float_array(self.values, name="convolved values")
        if values.size != energy.size:
            raise ConvolutionError(
                "convolution energy and values must have identical lengths"
            )
        if not np.all(np.isfinite(values)):
            raise ConvolutionError("convolved values must be finite")
        _validate_uniform_spacing(
            energy,
            self.spacing,
            name="convolution energy",
        )
        object.__setattr__(self, "energy", energy)
        object.__setattr__(self, "values", values)

    def evaluate(
        self,
        target_energy: npt.ArrayLike,
        *,
        energy_shift: float = 0.0,
    ) -> FloatArray:
        """Linearly evaluate this fixed profile at ``target_energy - energy_shift``."""

        targets = _readonly_float_array(target_energy, name="target_energy")
        if targets.size == 0 or not np.all(np.isfinite(targets)):
            raise ConvolutionError("target_energy must contain finite coordinates")
        if not np.isfinite(energy_shift):
            raise ConvolutionError("energy_shift must be finite")
        query = targets - energy_shift
        scale = max(
            1.0,
            abs(float(self.energy[0])),
            abs(float(self.energy[-1])),
            abs(energy_shift),
        )
        tolerance = _LATTICE_ULPS * np.finfo(np.float64).eps * scale
        if (
            float(np.min(query)) < float(self.energy[0]) - tolerance
            or float(np.max(query)) > float(self.energy[-1]) + tolerance
        ):
            raise ConvolutionError(
                "shifted target coordinates fall outside the calculated "
                "convolution domain"
            )
        clipped_query = np.clip(query, self.energy[0], self.energy[-1])
        evaluated = np.asarray(
            np.interp(clipped_query, self.energy, self.values),
            dtype=np.float64,
        )
        evaluated.setflags(write=False)
        return evaluated


@dataclass(frozen=True, slots=True)
class ConvolutionPlan:
    """Reusable fixed-grid state for one prepared sample/resolution Q group."""

    target_energy: FloatArray = field(repr=False)
    spacing: float
    model_energy: FloatArray = field(repr=False)
    resolution_energy: FloatArray = field(repr=False)
    resolution_values: FloatArray = field(repr=False)
    convolution_energy: FloatArray = field(repr=False)
    resolution_grid_integral_before_correction: float
    fft_length: int
    _resolution_fft: ComplexArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        target = _validate_coordinates(self.target_energy, name="target_energy")
        model = _validate_coordinates(self.model_energy, name="model_energy")
        resolution = _validate_coordinates(
            self.resolution_energy,
            name="resolution_energy",
        )
        resolution_values = _readonly_float_array(
            self.resolution_values,
            name="resolution_values",
        )
        convolution = _validate_coordinates(
            self.convolution_energy,
            name="convolution_energy",
        )
        full_length = model.size + resolution.size - 1
        if resolution_values.size != resolution.size:
            raise ConvolutionError("resolution grid coordinates and values differ")
        if convolution.size != full_length:
            raise ConvolutionError("convolution coordinate length is inconsistent")
        if self.fft_length < full_length:
            raise ConvolutionError(
                "FFT length cannot represent full linear convolution"
            )
        if not np.all(np.isfinite(resolution_values)):
            raise ConvolutionError("resolution grid values must be finite")
        if (
            not np.isfinite(self.resolution_grid_integral_before_correction)
            or self.resolution_grid_integral_before_correction <= 0.0
        ):
            raise ConvolutionError(
                "resolution grid integral before correction must be positive"
            )
        _validate_uniform_spacing(model, self.spacing, name="model energy")
        _validate_uniform_spacing(
            resolution,
            self.spacing,
            name="resolution energy",
        )
        _validate_uniform_spacing(
            convolution,
            self.spacing,
            name="convolution energy",
        )
        coordinate_scale = max(
            abs(float(model[0])),
            abs(float(resolution[0])),
            abs(float(convolution[0])),
            self.spacing,
        )
        coordinate_tolerance = (
            _LATTICE_ULPS * np.finfo(np.float64).eps * coordinate_scale
        )
        if not np.isclose(
            convolution[0],
            model[0] + resolution[0],
            rtol=_LATTICE_ULPS * np.finfo(np.float64).eps,
            atol=coordinate_tolerance,
        ):
            raise ConvolutionError(
                "convolution energy origin must equal the sum of input origins"
            )
        resolution_area = float(np.trapezoid(resolution_values, resolution))
        if not np.isclose(resolution_area, 1.0, rtol=1.0e-12, atol=1.0e-12):
            raise ConvolutionError("resolution grid values must have unit area")
        object.__setattr__(self, "target_energy", target)
        object.__setattr__(self, "model_energy", model)
        object.__setattr__(self, "resolution_energy", resolution)
        object.__setattr__(self, "resolution_values", resolution_values)
        object.__setattr__(self, "convolution_energy", convolution)
        object.__setattr__(
            self,
            "_resolution_fft",
            _readonly_complex_array(
                np.fft.rfft(resolution_values, self.fft_length),
                name="resolution FFT",
            ),
        )

    @property
    def full_length(self) -> int:
        """Return the uncropped linear-convolution length."""

        return int(self.convolution_energy.size)

    @property
    def resolution_grid_integral(self) -> float:
        """Return the corrected numerical resolution area."""

        return float(np.trapezoid(self.resolution_values, self.resolution_energy))

    @property
    def energy_unit(self) -> str:
        """Return the canonical physical energy unit."""

        return CANONICAL_ENERGY_UNIT

    def convolve(self, model_density: npt.ArrayLike) -> ConvolvedProfile:
        """Convolve one theoretical density using FFT-based linear convolution."""

        model = _readonly_float_array(model_density, name="model_density")
        if model.size != self.model_energy.size:
            raise ConvolutionError(
                "model_density must contain one value per internal model coordinate"
            )
        if not np.all(np.isfinite(model)):
            raise ConvolutionError("model_density must contain only finite values")
        transformed = np.fft.rfft(model, self.fft_length)
        full = np.asarray(
            np.fft.irfft(
                transformed * self._resolution_fft,
                self.fft_length,
            )[: self.full_length]
            * self.spacing,
            dtype=np.float64,
        )
        return ConvolvedProfile(
            energy=self.convolution_energy,
            values=full,
            spacing=self.spacing,
        )

    def evaluate_on_sample(
        self,
        model_density: npt.ArrayLike,
        *,
        energy_shift: float = 0.0,
    ) -> FloatArray:
        """Convolve and evaluate only the model on original sample coordinates."""

        return self.convolve(model_density).evaluate(
            self.target_energy,
            energy_shift=energy_shift,
        )

    def approximate_temporary_bytes(self) -> int:
        """Return a transparent upper-order estimate for one FFT evaluation."""

        real_values = self.model_energy.size + 2 * self.full_length
        complex_values = 2 * (self.fft_length // 2 + 1)
        return int(real_values * 8 + complex_values * 16)


def build_convolution_plan(
    prepared_resolution: PreparedResolution,
    group_index: int,
) -> ConvolutionPlan:
    """Build fixed S1/4 numerical state for one exactly associated Q group."""

    if isinstance(group_index, bool) or not isinstance(group_index, int):
        raise ConvolutionError("group_index must be an integer")
    if not 0 <= group_index < len(prepared_resolution.spectra):
        raise ConvolutionError("group_index is outside the prepared resolution")

    sample = prepared_resolution.sample_dataset.spectra[group_index]
    resolution = prepared_resolution.spectra[group_index]
    if sample.energy_unit != CANONICAL_ENERGY_UNIT:
        raise ConvolutionError(
            "sample energy unit must be explicitly canonicalized to meV"
        )
    if resolution.source_spectrum.energy_unit != CANONICAL_ENERGY_UNIT:
        raise ConvolutionError(
            "resolution energy unit must be explicitly canonicalized to meV"
        )

    targets = _validate_coordinates(sample.energy, name="sample energy")
    measured_resolution_energy = _validate_coordinates(
        resolution.energy,
        name="prepared resolution energy",
    )
    measured_resolution_values = _readonly_float_array(
        resolution.normalized_intensity,
        name="prepared resolution intensity",
    )
    if measured_resolution_values.size != measured_resolution_energy.size:
        raise ConvolutionError("prepared resolution arrays have unequal lengths")
    if not np.all(np.isfinite(measured_resolution_values)):
        raise ConvolutionError("prepared resolution intensity must be finite")

    spacing = automatic_grid_spacing(targets, measured_resolution_energy)
    resolution_min = float(measured_resolution_energy[0])
    resolution_max = float(measured_resolution_energy[-1])
    model_min = float(targets[0]) - resolution_max
    model_max = float(targets[-1]) - resolution_min
    model_energy = _zero_anchored_axis(model_min, model_max, spacing)

    resolution_energy = _outward_axis(
        resolution_min - spacing,
        resolution_max + spacing,
        spacing,
    )
    resolution_values = np.interp(
        resolution_energy,
        measured_resolution_energy,
        measured_resolution_values,
        left=0.0,
        right=0.0,
    )
    representation_area = float(np.trapezoid(resolution_values, resolution_energy))
    if not np.isfinite(representation_area) or representation_area <= 0.0:
        raise ConvolutionError(
            "interpolated resolution representation has no finite positive area"
        )
    resolution_values = np.asarray(
        resolution_values / representation_area,
        dtype=np.float64,
    )
    if not np.all(np.isfinite(resolution_values)):
        raise ConvolutionError("normalized resolution representation is not finite")

    full_length = model_energy.size + resolution_energy.size - 1
    fft_length = _next_power_of_two(full_length)
    convolution_energy = np.asarray(
        float(model_energy[0])
        + float(resolution_energy[0])
        + spacing * np.arange(full_length, dtype=np.float64),
        dtype=np.float64,
    )
    coordinate_tolerance = (
        _LATTICE_ULPS
        * np.finfo(np.float64).eps
        * max(1.0, abs(float(targets[0])), abs(float(targets[-1])))
    )
    if (
        float(targets[0]) < float(convolution_energy[0]) - coordinate_tolerance
        or float(targets[-1]) > float(convolution_energy[-1]) + coordinate_tolerance
    ):
        raise ConvolutionError(
            "constructed convolution domain does not cover sample coordinates"
        )

    return ConvolutionPlan(
        target_energy=targets,
        spacing=spacing,
        model_energy=model_energy,
        resolution_energy=resolution_energy,
        resolution_values=resolution_values,
        convolution_energy=convolution_energy,
        resolution_grid_integral_before_correction=representation_area,
        fft_length=fft_length,
    )
