"""Prepare measured resolution spectra without changing their source data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

import numpy as np
import numpy.typing as npt

from ezqens.domain import DiagnosticSeverity, ReducedDataset, Spectrum, SpectrumRole
from ezqens.preprocessing import (
    EdgePaddingDetectionResult,
    SpectrumPaddingResult,
    detect_edge_padding,
)

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]

Q_MATCH_RTOL: Final[float] = 1.0e-10
Q_MATCH_ATOL: Final[float] = 1.0e-12
PADDING_BOUNDARY_RTOL: Final[float] = 1.0e-10
PADDING_BOUNDARY_ATOL: Final[float] = 1.0e-12
NORMALIZATION_METHOD: Final[str] = "trapezoid_measured_grid"


class ResolutionSupportSource(StrEnum):
    """Origin of an accepted measured-resolution energy support."""

    DEFAULT_VALID_DATA = "default_valid_data"
    EXPLICIT_OVERRIDE = "explicit_override"


@dataclass(frozen=True, slots=True)
class ResolutionSupport:
    """Inclusive energy support for one measured-resolution spectrum."""

    lower_energy: float
    upper_energy: float
    source: ResolutionSupportSource = ResolutionSupportSource.EXPLICIT_OVERRIDE

    def __post_init__(self) -> None:
        if not np.isfinite(self.lower_energy) or not np.isfinite(self.upper_energy):
            raise ValueError("resolution-support bounds must be finite")
        if self.lower_energy > self.upper_energy:
            raise ValueError("lower_energy must not exceed upper_energy")
        if not isinstance(self.source, ResolutionSupportSource):
            raise ValueError("source must be a ResolutionSupportSource")


@dataclass(frozen=True, slots=True)
class ResolutionDiagnostic:
    """One privacy-safe measured-resolution preparation diagnostic."""

    code: str
    severity: DiagnosticSeverity
    message: str
    group_index: int | None = None
    group_identity: str | None = None


class ResolutionPreparationError(ValueError):
    """Raised when measured resolution cannot be prepared scientifically."""

    diagnostics: Final[tuple[ResolutionDiagnostic, ...]]

    def __init__(self, diagnostics: tuple[ResolutionDiagnostic, ...]) -> None:
        if not diagnostics:
            raise ValueError("ResolutionPreparationError requires diagnostics")
        self.diagnostics = diagnostics
        super().__init__(
            "; ".join(
                f"{diagnostic.code}: {diagnostic.message}" for diagnostic in diagnostics
            )
        )


def _readonly_mask(value: npt.ArrayLike) -> BoolArray:
    mask = np.array(value, dtype=np.bool_, copy=True)
    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    mask.setflags(write=False)
    return mask


def _readonly_float_array(value: npt.ArrayLike) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError("array must be one-dimensional")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class ResolutionPaddingComparison:
    """Diagnostic comparison of independently detected sample/resolution edges."""

    group_index: int
    sample_group_identity: str
    resolution_group_identity: str
    sample_left_auto_count: int
    sample_right_auto_count: int
    resolution_left_auto_count: int
    resolution_right_auto_count: int
    sample_retained_energy_bounds: tuple[float | None, float | None]
    resolution_retained_energy_bounds: tuple[float | None, float | None]
    is_consistent: bool


@dataclass(frozen=True, slots=True)
class PreparedResolutionSpectrum:
    """One unit-area resolution linked to its immutable measured spectrum."""

    source_spectrum: Spectrum = field(repr=False)
    padding: SpectrumPaddingResult = field(repr=False)
    support: ResolutionSupport
    auto_padding_applied: bool
    normalization_integral: float
    normalization_factor: float
    normalization_method: str = NORMALIZATION_METHOD
    diagnostics: tuple[ResolutionDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if self.source_spectrum.role is not SpectrumRole.RESOLUTION:
            raise ValueError("prepared source spectrum must have resolution role")
        if (
            self.padding.group_index != self.source_spectrum.group_index
            or self.padding.group_identity != self.source_spectrum.group_label
            or self.padding.auto_mask.size != self.source_spectrum.energy.size
        ):
            raise ValueError("padding result must match the source resolution spectrum")
        if (
            not np.isfinite(self.normalization_integral)
            or self.normalization_integral <= 0.0
        ):
            raise ValueError("normalization integral must be finite and positive")
        if (
            not np.isfinite(self.normalization_factor)
            or self.normalization_factor <= 0.0
        ):
            raise ValueError("normalization factor must be finite and positive")
        if self.normalization_method != NORMALIZATION_METHOD:
            raise ValueError("unsupported normalization method")
        if not isinstance(self.auto_padding_applied, bool):
            raise ValueError("auto_padding_applied must be a boolean")

    @property
    def invalid_mask(self) -> BoolArray:
        """Return every imported invalid-value state without changing the source."""

        spectrum = self.source_spectrum
        return _readonly_mask(
            spectrum.invalid_energy_mask
            | spectrum.invalid_intensity_mask
            | spectrum.invalid_uncertainty_mask
        )

    @property
    def support_mask(self) -> BoolArray:
        """Return points inside the explicit inclusive support range."""

        energy = self.source_spectrum.energy
        return _readonly_mask(
            np.isfinite(energy)
            & (energy >= self.support.lower_energy)
            & (energy <= self.support.upper_energy)
        )

    @property
    def accepted_mask(self) -> BoolArray:
        """Return usable kernel points; REVIEW remains accepted by default."""

        spectrum = self.source_spectrum
        applied_auto_mask = (
            self.padding.auto_mask
            if self.auto_padding_applied
            else np.zeros(spectrum.energy.size, dtype=np.bool_)
        )
        return _readonly_mask(
            self.support_mask
            & ~applied_auto_mask
            & ~spectrum.invalid_energy_mask
            & ~spectrum.invalid_intensity_mask
        )

    @property
    def energy(self) -> FloatArray:
        """Return accepted original measured energy coordinates."""

        return _readonly_float_array(self.source_spectrum.energy[self.accepted_mask])

    @property
    def normalized_intensity(self) -> FloatArray:
        """Return accepted intensity scaled to unit trapezoidal area."""

        return _readonly_float_array(
            self.source_spectrum.intensity[self.accepted_mask]
            * self.normalization_factor
        )

    @property
    def normalized_uncertainty(self) -> FloatArray:
        """Return source uncertainty scaled by the deterministic area factor."""

        return _readonly_float_array(
            self.source_spectrum.uncertainty[self.accepted_mask]
            * self.normalization_factor
        )

    @property
    def normalized_integral(self) -> float:
        """Return the area of the derived normalized representation."""

        return float(np.trapezoid(self.normalized_intensity, self.energy))

    @property
    def normalized_intensity_on_source_grid(self) -> FloatArray:
        """Return a derived source-grid view with zero outside accepted support."""

        values = np.zeros_like(self.source_spectrum.intensity)
        accepted = self.accepted_mask
        values[accepted] = (
            self.source_spectrum.intensity[accepted] * self.normalization_factor
        )
        return _readonly_float_array(values)


@dataclass(frozen=True, slots=True)
class PreparedResolution:
    """Ordered measured-resolution preparation associated exactly with sample Q."""

    sample_dataset: ReducedDataset = field(repr=False)
    resolution_dataset: ReducedDataset = field(repr=False)
    sample_padding: EdgePaddingDetectionResult = field(repr=False)
    resolution_padding: EdgePaddingDetectionResult = field(repr=False)
    spectra: tuple[PreparedResolutionSpectrum, ...]
    padding_comparisons: tuple[ResolutionPaddingComparison, ...]
    diagnostics: tuple[ResolutionDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        count = len(self.sample_dataset.spectra)
        if self.sample_dataset.role is not SpectrumRole.SAMPLE:
            raise ValueError("sample_dataset must have sample role")
        if self.resolution_dataset.role is not SpectrumRole.RESOLUTION:
            raise ValueError("resolution_dataset must have resolution role")
        if not (
            len(self.resolution_dataset.spectra)
            == len(self.sample_padding.spectra)
            == len(self.resolution_padding.spectra)
            == len(self.spectra)
            == len(self.padding_comparisons)
            == count
        ):
            raise ValueError("prepared resolution state must align by group")
        if any(
            prepared.source_spectrum is not source
            for prepared, source in zip(
                self.spectra, self.resolution_dataset.spectra, strict=True
            )
        ):
            raise ValueError("prepared spectra must reference resolution sources")

    def q_value(self, group_index: int) -> float:
        """Return the exactly associated dataset-level representative Q value."""

        if not 0 <= group_index < len(self.spectra):
            raise IndexError("group_index is outside the prepared resolution")
        q_bins = self.resolution_dataset.q_bins
        if q_bins is None:  # guarded by prepare_measured_resolution
            raise RuntimeError("prepared resolution has no Q assignment")
        return float(q_bins.q_values[group_index])


def _error(
    code: str,
    message: str,
    *,
    group_index: int | None = None,
    group_identity: str | None = None,
) -> ResolutionPreparationError:
    return ResolutionPreparationError(
        (
            ResolutionDiagnostic(
                code=code,
                severity=DiagnosticSeverity.ERROR,
                message=message,
                group_index=group_index,
                group_identity=group_identity,
            ),
        )
    )


def _validate_exact_q_association(
    sample_dataset: ReducedDataset,
    resolution_dataset: ReducedDataset,
) -> None:
    if sample_dataset.role is not SpectrumRole.SAMPLE:
        raise _error("sample_role_required", "sample dataset must have sample role")
    if resolution_dataset.role is not SpectrumRole.RESOLUTION:
        raise _error(
            "resolution_role_required",
            "resolution dataset must have resolution role",
        )
    sample_q = sample_dataset.q_bins
    resolution_q = resolution_dataset.q_bins
    if sample_q is None or resolution_q is None:
        raise _error(
            "q_bins_required",
            "sample and resolution datasets both require explicit Q bins",
        )
    if sample_q.group_count != resolution_q.group_count:
        raise _error(
            "q_group_count_mismatch",
            "sample and resolution Q-group counts differ; no repair is applied",
        )
    value_matches = np.isclose(
        sample_q.q_values,
        resolution_q.q_values,
        rtol=Q_MATCH_RTOL,
        atol=Q_MATCH_ATOL,
    )
    if not np.all(value_matches):
        group_index = int(np.flatnonzero(~value_matches)[0])
        raise _error(
            "q_value_mismatch",
            "ordered sample and resolution representative Q values differ",
            group_index=group_index,
        )
    if sample_q.edges is not None and resolution_q.edges is not None:
        edge_matches = np.isclose(
            sample_q.edges,
            resolution_q.edges,
            rtol=Q_MATCH_RTOL,
            atol=Q_MATCH_ATOL,
        )
        if not np.all(edge_matches):
            edge_index = int(np.flatnonzero(~edge_matches)[0])
            group_index = min(edge_index, sample_q.group_count - 1)
            raise _error(
                "q_edge_mismatch",
                "known sample and resolution Q-bin edges differ",
                group_index=group_index,
            )


def _retained_energy_bounds(
    spectrum: Spectrum,
    padding: SpectrumPaddingResult,
) -> tuple[float | None, float | None]:
    retained = spectrum.energy[np.isfinite(spectrum.energy) & ~padding.auto_mask]
    if retained.size == 0:
        return (None, None)
    return (float(np.min(retained)), float(np.max(retained)))


def _bounds_match(
    first: tuple[float | None, float | None],
    second: tuple[float | None, float | None],
) -> bool:
    if first[0] is None or first[1] is None:
        return second == (None, None)
    if second[0] is None or second[1] is None:
        return False
    return bool(
        np.isclose(
            first[0],
            second[0],
            rtol=PADDING_BOUNDARY_RTOL,
            atol=PADDING_BOUNDARY_ATOL,
        )
        and np.isclose(
            first[1],
            second[1],
            rtol=PADDING_BOUNDARY_RTOL,
            atol=PADDING_BOUNDARY_ATOL,
        )
    )


def _compare_padding(
    sample_dataset: ReducedDataset,
    resolution_dataset: ReducedDataset,
    sample_padding: EdgePaddingDetectionResult,
    resolution_padding: EdgePaddingDetectionResult,
) -> tuple[tuple[ResolutionPaddingComparison, ...], tuple[ResolutionDiagnostic, ...]]:
    comparisons: list[ResolutionPaddingComparison] = []
    diagnostics: list[ResolutionDiagnostic] = []
    aligned_groups = zip(
        sample_dataset.spectra,
        resolution_dataset.spectra,
        sample_padding.spectra,
        resolution_padding.spectra,
        strict=True,
    )
    for group_index, (
        sample,
        resolution,
        sample_result,
        resolution_result,
    ) in enumerate(aligned_groups):
        sample_bounds = _retained_energy_bounds(sample, sample_result)
        resolution_bounds = _retained_energy_bounds(resolution, resolution_result)
        is_consistent = _bounds_match(sample_bounds, resolution_bounds)
        comparisons.append(
            ResolutionPaddingComparison(
                group_index=group_index,
                sample_group_identity=sample.group_label,
                resolution_group_identity=resolution.group_label,
                sample_left_auto_count=sample_result.left_auto_mask_count,
                sample_right_auto_count=sample_result.right_auto_mask_count,
                resolution_left_auto_count=resolution_result.left_auto_mask_count,
                resolution_right_auto_count=resolution_result.right_auto_mask_count,
                sample_retained_energy_bounds=sample_bounds,
                resolution_retained_energy_bounds=resolution_bounds,
                is_consistent=is_consistent,
            )
        )
        if not is_consistent:
            diagnostics.append(
                ResolutionDiagnostic(
                    code="padding_boundary_mismatch",
                    severity=DiagnosticSeverity.WARNING,
                    message=(
                        "independently detected sample and resolution retained "
                        "energy boundaries differ; neither mask was changed"
                    ),
                    group_index=group_index,
                    group_identity=resolution.group_label,
                )
            )
    return tuple(comparisons), tuple(diagnostics)


def _default_support(spectrum: Spectrum) -> ResolutionSupport:
    usable = ~spectrum.invalid_energy_mask & ~spectrum.invalid_intensity_mask
    energies = spectrum.energy[usable]
    if energies.size < 2:
        raise _error(
            "insufficient_resolution_points",
            "fewer than two valid measured resolution points are available",
            group_index=spectrum.group_index,
            group_identity=spectrum.group_label,
        )
    return ResolutionSupport(
        lower_energy=float(np.min(energies)),
        upper_energy=float(np.max(energies)),
        source=ResolutionSupportSource.DEFAULT_VALID_DATA,
    )


def _prepare_spectrum(
    spectrum: Spectrum,
    padding: SpectrumPaddingResult,
    support: ResolutionSupport,
    *,
    auto_padding_applied: bool,
) -> PreparedResolutionSpectrum:
    support_mask = (
        np.isfinite(spectrum.energy)
        & (spectrum.energy >= support.lower_energy)
        & (spectrum.energy <= support.upper_energy)
    )
    support_indices = np.flatnonzero(support_mask)
    if support_indices.size < 2:
        raise _error(
            "insufficient_resolution_points",
            "resolution support must contain at least two measured coordinates",
            group_index=spectrum.group_index,
            group_identity=spectrum.group_label,
        )
    first_index = int(support_indices[0])
    last_index = int(support_indices[-1])
    support_slice = slice(first_index, last_index + 1)
    if np.any(
        spectrum.invalid_energy_mask[support_slice]
        | spectrum.invalid_intensity_mask[support_slice]
    ):
        raise _error(
            "resolution_support_contains_internal_invalid_hole",
            "accepted resolution support contains an internal invalid energy "
            "or intensity point; M3 does not interpolate or bridge it",
            group_index=spectrum.group_index,
            group_identity=spectrum.group_label,
        )
    measured_energy = spectrum.energy[support_slice]
    if not np.all(np.diff(measured_energy) > 0.0):
        raise _error(
            "resolution_energy_not_strictly_increasing",
            "accepted resolution energy coordinates must be strictly "
            "increasing and unique",
            group_index=spectrum.group_index,
            group_identity=spectrum.group_label,
        )
    applied_auto_mask = (
        padding.auto_mask
        if auto_padding_applied
        else np.zeros(spectrum.energy.size, dtype=np.bool_)
    )
    accepted = (
        support_mask
        & ~applied_auto_mask
        & ~spectrum.invalid_energy_mask
        & ~spectrum.invalid_intensity_mask
    )
    energy = spectrum.energy[accepted]
    intensity = spectrum.intensity[accepted]
    if energy.size < 2:
        raise _error(
            "insufficient_resolution_points",
            "accepted resolution support must contain at least two usable points",
            group_index=spectrum.group_index,
            group_identity=spectrum.group_label,
        )
    with np.errstate(over="ignore", invalid="ignore"):
        integral = float(np.trapezoid(intensity, energy))
    if not np.isfinite(integral):
        raise _error(
            "resolution_integral_nonfinite",
            "measured-resolution normalization integral is not finite",
            group_index=spectrum.group_index,
            group_identity=spectrum.group_label,
        )
    if integral <= 0.0:
        raise _error(
            "resolution_integral_nonpositive",
            "measured-resolution normalization integral must be positive",
            group_index=spectrum.group_index,
            group_identity=spectrum.group_label,
        )

    diagnostics: list[ResolutionDiagnostic] = []
    invalid_uncertainty_count = int(
        np.count_nonzero(spectrum.invalid_uncertainty_mask & accepted)
    )
    if invalid_uncertainty_count:
        diagnostics.append(
            ResolutionDiagnostic(
                code="invalid_resolution_uncertainty_retained",
                severity=DiagnosticSeverity.WARNING,
                message=(
                    "accepted resolution intensity is usable but one or more "
                    "uncertainties remain invalid; no zero replacement was made"
                ),
                group_index=spectrum.group_index,
                group_identity=spectrum.group_label,
            )
        )
    return PreparedResolutionSpectrum(
        source_spectrum=spectrum,
        padding=padding,
        support=support,
        auto_padding_applied=auto_padding_applied,
        normalization_integral=integral,
        normalization_factor=1.0 / integral,
        diagnostics=tuple(diagnostics),
    )


def prepare_measured_resolution(
    sample_dataset: ReducedDataset,
    resolution_dataset: ReducedDataset,
    *,
    support_overrides: Mapping[int, ResolutionSupport] | None = None,
    apply_auto_padding: Mapping[int, bool] | None = None,
) -> PreparedResolution:
    """Validate exact Q identity and prepare independent unit-area kernels.

    Edge padding is always detected independently on the sample and measured
    resolution. ``AUTO`` padding is applied by default and may be disabled
    explicitly per group without changing the independent support range.
    """

    _validate_exact_q_association(sample_dataset, resolution_dataset)
    overrides = dict(support_overrides or {})
    auto_application = dict(apply_auto_padding or {})
    invalid_override_indices = sorted(
        index
        for index in overrides
        if not isinstance(index, int)
        or isinstance(index, bool)
        or not 0 <= index < len(resolution_dataset.spectra)
    )
    if invalid_override_indices:
        raise _error(
            "resolution_support_group_invalid",
            "resolution support override references a group outside the dataset",
        )
    if any(not isinstance(value, ResolutionSupport) for value in overrides.values()):
        raise _error(
            "resolution_support_type_invalid",
            "support overrides must contain ResolutionSupport values",
        )
    invalid_auto_indices = sorted(
        index
        for index in auto_application
        if not isinstance(index, int)
        or isinstance(index, bool)
        or not 0 <= index < len(resolution_dataset.spectra)
    )
    if invalid_auto_indices:
        raise _error(
            "resolution_auto_application_group_invalid",
            "AUTO-padding application references a group outside the dataset",
        )
    if any(not isinstance(value, bool) for value in auto_application.values()):
        raise _error(
            "resolution_auto_application_type_invalid",
            "AUTO-padding application values must be boolean",
        )

    sample_padding = detect_edge_padding(sample_dataset)
    resolution_padding = detect_edge_padding(resolution_dataset)
    comparisons, comparison_diagnostics = _compare_padding(
        sample_dataset,
        resolution_dataset,
        sample_padding,
        resolution_padding,
    )

    prepared_spectra: list[PreparedResolutionSpectrum] = []
    for group_index, (spectrum, padding) in enumerate(
        zip(
            resolution_dataset.spectra,
            resolution_padding.spectra,
            strict=True,
        )
    ):
        support = overrides.get(group_index)
        if support is None:
            support = _default_support(spectrum)
        elif support.source is not ResolutionSupportSource.EXPLICIT_OVERRIDE:
            support = ResolutionSupport(
                lower_energy=support.lower_energy,
                upper_energy=support.upper_energy,
                source=ResolutionSupportSource.EXPLICIT_OVERRIDE,
            )
        prepared_spectra.append(
            _prepare_spectrum(
                spectrum,
                padding,
                support,
                auto_padding_applied=auto_application.get(group_index, True),
            )
        )

    return PreparedResolution(
        sample_dataset=sample_dataset,
        resolution_dataset=resolution_dataset,
        sample_padding=sample_padding,
        resolution_padding=resolution_padding,
        spectra=tuple(prepared_spectra),
        padding_comparisons=comparisons,
        diagnostics=comparison_diagnostics,
    )
