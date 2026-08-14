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


class ResolutionAcceptanceDecision(StrEnum):
    """User-reviewed disposition of one measured-resolution Q group."""

    KEEP = "keep"
    EXCLUDE_BY_CONTIGUOUS_SUPPORT = "exclude_by_contiguous_support"


class ResolutionAcceptanceWarning(StrEnum):
    """Scientifically neutral warning retained with an approved kernel."""

    SUSPICIOUS_STRUCTURE_RETAINED = "suspicious_structure_retained_by_user"


@dataclass(frozen=True, slots=True)
class ResolutionAcceptance:
    """Explicit per-Q review decision and confirmation state."""

    decision: ResolutionAcceptanceDecision | None = None
    confirmed: bool = False
    warnings: tuple[ResolutionAcceptanceWarning, ...] = ()

    def __post_init__(self) -> None:
        warnings = tuple(self.warnings)
        if self.decision is not None and not isinstance(
            self.decision, ResolutionAcceptanceDecision
        ):
            raise ValueError("decision must be a ResolutionAcceptanceDecision or None")
        if not isinstance(self.confirmed, bool):
            raise ValueError("confirmed must be a boolean")
        if self.confirmed and self.decision is None:
            raise ValueError("a confirmed resolution acceptance requires a decision")
        if any(
            not isinstance(warning, ResolutionAcceptanceWarning) for warning in warnings
        ):
            raise ValueError("warnings must contain ResolutionAcceptanceWarning values")
        if warnings and self.decision is not ResolutionAcceptanceDecision.KEEP:
            raise ValueError("resolution warnings may only accompany a KEEP decision")
        object.__setattr__(self, "warnings", warnings)


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
class ResolutionPreparationPreviewSpectrum:
    """One proposed accepted raw kernel and its normalization preview."""

    source_spectrum: Spectrum = field(repr=False)
    padding: SpectrumPaddingResult = field(repr=False)
    original_support: ResolutionSupport
    support: ResolutionSupport
    acceptance: ResolutionAcceptance
    auto_padding_applied: bool
    pre_qc_integral: float
    normalization_integral: float
    normalization_factor: float
    signed_area_ratio: float | None
    normalization_method: str = NORMALIZATION_METHOD
    diagnostics: tuple[ResolutionDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if self.source_spectrum.role is not SpectrumRole.RESOLUTION:
            raise ValueError("resolution source spectrum must have resolution role")
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
        if not np.isclose(
            self.normalization_factor,
            1.0 / self.normalization_integral,
            rtol=1.0e-14,
            atol=0.0,
        ):
            raise ValueError("normalization factor must be the reciprocal integral")
        if self.normalization_method != NORMALIZATION_METHOD:
            raise ValueError("unsupported normalization method")
        if not isinstance(self.auto_padding_applied, bool):
            raise ValueError("auto_padding_applied must be a boolean")
        if not isinstance(self.acceptance, ResolutionAcceptance):
            raise ValueError("acceptance must be a ResolutionAcceptance")
        if self.signed_area_ratio is not None and not np.isfinite(
            self.signed_area_ratio
        ):
            raise ValueError("signed_area_ratio must be finite or None")
        if np.isfinite(self.pre_qc_integral) and self.pre_qc_integral > 0.0:
            expected_ratio = self.normalization_integral / self.pre_qc_integral
            if self.signed_area_ratio is None or not np.isclose(
                self.signed_area_ratio,
                expected_ratio,
                rtol=1.0e-14,
                atol=0.0,
            ):
                raise ValueError("signed_area_ratio must match the pre-QC integral")
        elif self.signed_area_ratio is not None:
            raise ValueError(
                "signed_area_ratio requires a finite positive pre-QC integral"
            )

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
        """Return points inside the proposed inclusive support range."""

        energy = self.source_spectrum.energy
        return _readonly_mask(
            np.isfinite(energy)
            & (energy >= self.support.lower_energy)
            & (energy <= self.support.upper_energy)
        )

    @property
    def accepted_mask(self) -> BoolArray:
        """Return proposed usable kernel points; REVIEW remains accepted."""

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
        """Return proposed accepted original measured energy coordinates."""

        return _readonly_float_array(self.source_spectrum.energy[self.accepted_mask])

    @property
    def intensity(self) -> FloatArray:
        """Return proposed accepted measured intensity before normalization."""

        return _readonly_float_array(self.source_spectrum.intensity[self.accepted_mask])

    @property
    def uncertainty(self) -> FloatArray:
        """Return proposed accepted measured uncertainty before normalization."""

        return _readonly_float_array(
            self.source_spectrum.uncertainty[self.accepted_mask]
        )


@dataclass(frozen=True, slots=True)
class PreparedResolutionSpectrum(ResolutionPreparationPreviewSpectrum):
    """One confirmed unit-area resolution linked to its measured spectrum."""

    def __post_init__(self) -> None:
        super(PreparedResolutionSpectrum, self).__post_init__()
        if not self.acceptance.confirmed:
            raise ValueError("prepared resolution acceptance must be confirmed")
        if self.acceptance.decision is ResolutionAcceptanceDecision.KEEP:
            if self.support != self.original_support:
                raise ValueError("confirmed KEEP must preserve the original support")
        elif self.acceptance.decision is (
            ResolutionAcceptanceDecision.EXCLUDE_BY_CONTIGUOUS_SUPPORT
        ):
            if not _support_is_within(self.support, self.original_support):
                raise ValueError(
                    "confirmed EXCLUDE support must remain inside original"
                )
            if not _support_is_narrower(self.support, self.original_support):
                raise ValueError("confirmed EXCLUDE support must be narrower")
        else:
            raise ValueError("prepared resolution requires an acceptance decision")

    @property
    def normalized_intensity(self) -> FloatArray:
        """Return accepted intensity scaled to unit trapezoidal area."""

        return _readonly_float_array(self.intensity * self.normalization_factor)

    @property
    def normalized_uncertainty(self) -> FloatArray:
        """Return accepted uncertainty scaled by the deterministic area factor."""

        return _readonly_float_array(self.uncertainty * self.normalization_factor)

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
class ResolutionAcceptanceProvenance:
    """Accepted measured-kernel provenance copied into downstream results."""

    group_index: int
    group_label: str
    q_value: float
    source_reference: str | None
    original_support: tuple[float, float]
    accepted_support: tuple[float, float]
    decision: ResolutionAcceptanceDecision
    retained_pre_normalization_area: float
    signed_area_ratio: float | None
    normalization_method: str
    normalization_factor: float
    confirmed: bool
    warnings: tuple[ResolutionAcceptanceWarning, ...]
    auto_padding_applied: bool


def _validate_resolution_collection(
    sample_dataset: ReducedDataset,
    resolution_dataset: ReducedDataset,
    sample_padding: EdgePaddingDetectionResult,
    resolution_padding: EdgePaddingDetectionResult,
    spectra: tuple[ResolutionPreparationPreviewSpectrum, ...],
    comparisons: tuple[ResolutionPaddingComparison, ...],
) -> None:
    count = len(sample_dataset.spectra)
    if sample_dataset.role is not SpectrumRole.SAMPLE:
        raise ValueError("sample_dataset must have sample role")
    if resolution_dataset.role is not SpectrumRole.RESOLUTION:
        raise ValueError("resolution_dataset must have resolution role")
    if not (
        len(resolution_dataset.spectra)
        == len(sample_padding.spectra)
        == len(resolution_padding.spectra)
        == len(spectra)
        == len(comparisons)
        == count
    ):
        raise ValueError("resolution preparation state must align by group")
    if any(
        prepared.source_spectrum is not source
        for prepared, source in zip(spectra, resolution_dataset.spectra, strict=True)
    ):
        raise ValueError("prepared spectra must reference resolution sources")


def _q_value(
    resolution_dataset: ReducedDataset, spectrum_count: int, group_index: int
) -> float:
    if not 0 <= group_index < spectrum_count:
        raise IndexError("group_index is outside the resolution preparation")
    q_bins = resolution_dataset.q_bins
    if q_bins is None:  # guarded by preview_measured_resolution
        raise RuntimeError("resolution preparation has no Q assignment")
    return float(q_bins.q_values[group_index])


@dataclass(frozen=True, slots=True)
class ResolutionPreparationPreview:
    """Ordered per-Q review preview before normalization is authorized."""

    sample_dataset: ReducedDataset = field(repr=False)
    resolution_dataset: ReducedDataset = field(repr=False)
    sample_padding: EdgePaddingDetectionResult = field(repr=False)
    resolution_padding: EdgePaddingDetectionResult = field(repr=False)
    spectra: tuple[ResolutionPreparationPreviewSpectrum, ...]
    padding_comparisons: tuple[ResolutionPaddingComparison, ...]
    diagnostics: tuple[ResolutionDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        _validate_resolution_collection(
            self.sample_dataset,
            self.resolution_dataset,
            self.sample_padding,
            self.resolution_padding,
            self.spectra,
            self.padding_comparisons,
        )

    def q_value(self, group_index: int) -> float:
        """Return the exactly associated dataset-level representative Q value."""

        return _q_value(self.resolution_dataset, len(self.spectra), group_index)


@dataclass(frozen=True, slots=True)
class PreparedResolution:
    """Ordered confirmed measured resolution associated exactly with sample Q."""

    sample_dataset: ReducedDataset = field(repr=False)
    resolution_dataset: ReducedDataset = field(repr=False)
    sample_padding: EdgePaddingDetectionResult = field(repr=False)
    resolution_padding: EdgePaddingDetectionResult = field(repr=False)
    spectra: tuple[PreparedResolutionSpectrum, ...]
    padding_comparisons: tuple[ResolutionPaddingComparison, ...]
    diagnostics: tuple[ResolutionDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        _validate_resolution_collection(
            self.sample_dataset,
            self.resolution_dataset,
            self.sample_padding,
            self.resolution_padding,
            self.spectra,
            self.padding_comparisons,
        )

    def q_value(self, group_index: int) -> float:
        """Return the exactly associated dataset-level representative Q value."""

        return _q_value(self.resolution_dataset, len(self.spectra), group_index)

    def acceptance_provenance(self, group_index: int) -> ResolutionAcceptanceProvenance:
        """Return immutable provenance for the accepted kernel used at one Q."""

        q_value = self.q_value(group_index)
        spectrum = self.spectra[group_index]
        decision = spectrum.acceptance.decision
        if decision is None:  # guarded by PreparedResolutionSpectrum
            raise RuntimeError("prepared resolution has no acceptance decision")
        return ResolutionAcceptanceProvenance(
            group_index=group_index,
            group_label=spectrum.source_spectrum.group_label,
            q_value=q_value,
            source_reference=self.resolution_dataset.source_reference,
            original_support=(
                spectrum.original_support.lower_energy,
                spectrum.original_support.upper_energy,
            ),
            accepted_support=(
                spectrum.support.lower_energy,
                spectrum.support.upper_energy,
            ),
            decision=decision,
            retained_pre_normalization_area=spectrum.normalization_integral,
            signed_area_ratio=spectrum.signed_area_ratio,
            normalization_method=spectrum.normalization_method,
            normalization_factor=spectrum.normalization_factor,
            confirmed=spectrum.acceptance.confirmed,
            warnings=spectrum.acceptance.warnings,
            auto_padding_applied=spectrum.auto_padding_applied,
        )


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


def _support_is_within(
    candidate: ResolutionSupport,
    original: ResolutionSupport,
) -> bool:
    lower_ok = candidate.lower_energy >= original.lower_energy or np.isclose(
        candidate.lower_energy,
        original.lower_energy,
        rtol=PADDING_BOUNDARY_RTOL,
        atol=PADDING_BOUNDARY_ATOL,
    )
    upper_ok = candidate.upper_energy <= original.upper_energy or np.isclose(
        candidate.upper_energy,
        original.upper_energy,
        rtol=PADDING_BOUNDARY_RTOL,
        atol=PADDING_BOUNDARY_ATOL,
    )
    return bool(lower_ok and upper_ok)


def _support_is_narrower(
    candidate: ResolutionSupport,
    original: ResolutionSupport,
) -> bool:
    same_lower = np.isclose(
        candidate.lower_energy,
        original.lower_energy,
        rtol=PADDING_BOUNDARY_RTOL,
        atol=PADDING_BOUNDARY_ATOL,
    )
    same_upper = np.isclose(
        candidate.upper_energy,
        original.upper_energy,
        rtol=PADDING_BOUNDARY_RTOL,
        atol=PADDING_BOUNDARY_ATOL,
    )
    return bool(not (same_lower and same_upper))


def _accepted_values(
    spectrum: Spectrum,
    padding: SpectrumPaddingResult,
    support: ResolutionSupport,
    *,
    auto_padding_applied: bool,
) -> tuple[BoolArray, float]:
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

    return _readonly_mask(accepted), integral


def _pre_qc_integral(
    spectrum: Spectrum,
    padding: SpectrumPaddingResult,
    original_support: ResolutionSupport,
    *,
    auto_padding_applied: bool,
) -> float:
    support_mask = (
        np.isfinite(spectrum.energy)
        & (spectrum.energy >= original_support.lower_energy)
        & (spectrum.energy <= original_support.upper_energy)
    )
    support_indices = np.flatnonzero(support_mask)
    if support_indices.size < 2:
        return float("nan")
    support_slice = slice(int(support_indices[0]), int(support_indices[-1]) + 1)
    if np.any(
        spectrum.invalid_energy_mask[support_slice]
        | spectrum.invalid_intensity_mask[support_slice]
    ):
        return float("nan")
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
    if energy.size < 2 or not np.all(np.diff(energy) > 0.0):
        return float("nan")
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.trapezoid(intensity, energy))


def _preview_spectrum(
    spectrum: Spectrum,
    padding: SpectrumPaddingResult,
    original_support: ResolutionSupport,
    support: ResolutionSupport,
    acceptance: ResolutionAcceptance,
    *,
    auto_padding_applied: bool,
) -> ResolutionPreparationPreviewSpectrum:
    accepted, integral = _accepted_values(
        spectrum,
        padding,
        support,
        auto_padding_applied=auto_padding_applied,
    )
    pre_qc_integral = _pre_qc_integral(
        spectrum,
        padding,
        original_support,
        auto_padding_applied=auto_padding_applied,
    )
    signed_area_ratio = (
        integral / pre_qc_integral
        if np.isfinite(pre_qc_integral) and pre_qc_integral > 0.0
        else None
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
    if acceptance.warnings:
        diagnostics.append(
            ResolutionDiagnostic(
                code="suspicious_resolution_structure_retained_by_user",
                severity=DiagnosticSeverity.WARNING,
                message=(
                    "user retained measured resolution structure of unknown "
                    "physical origin; no automatic correction was applied"
                ),
                group_index=spectrum.group_index,
                group_identity=spectrum.group_label,
            )
        )
    return ResolutionPreparationPreviewSpectrum(
        source_spectrum=spectrum,
        padding=padding,
        original_support=original_support,
        support=support,
        acceptance=acceptance,
        auto_padding_applied=auto_padding_applied,
        pre_qc_integral=pre_qc_integral,
        normalization_integral=integral,
        normalization_factor=1.0 / integral,
        signed_area_ratio=signed_area_ratio,
        diagnostics=tuple(diagnostics),
    )


def _validate_mapping_indices(
    values: Mapping[int, object],
    *,
    group_count: int,
    code: str,
    message: str,
) -> None:
    if any(
        not isinstance(index, int)
        or isinstance(index, bool)
        or not 0 <= index < group_count
        for index in values
    ):
        raise _error(code, message)


def preview_measured_resolution(
    sample_dataset: ReducedDataset,
    resolution_dataset: ReducedDataset,
    *,
    acceptance_decisions: Mapping[int, ResolutionAcceptance] | None = None,
    support_overrides: Mapping[int, ResolutionSupport] | None = None,
    apply_auto_padding: Mapping[int, bool] | None = None,
) -> ResolutionPreparationPreview:
    """Preview per-Q accepted raw kernels before confirmation authorizes use.

    Edge padding is always detected independently on the sample and measured
    resolution. ``AUTO`` padding is applied by default and may be disabled
    explicitly per group without changing the independent support range.
    """

    _validate_exact_q_association(sample_dataset, resolution_dataset)
    acceptances = dict(acceptance_decisions or {})
    overrides = dict(support_overrides or {})
    auto_application = dict(apply_auto_padding or {})
    group_count = len(resolution_dataset.spectra)
    _validate_mapping_indices(
        acceptances,
        group_count=group_count,
        code="resolution_acceptance_group_invalid",
        message="resolution acceptance references a group outside the dataset",
    )
    _validate_mapping_indices(
        overrides,
        group_count=group_count,
        code="resolution_support_group_invalid",
        message="resolution support override references a group outside the dataset",
    )
    _validate_mapping_indices(
        auto_application,
        group_count=group_count,
        code="resolution_auto_application_group_invalid",
        message="AUTO-padding application references a group outside the dataset",
    )
    if any(
        not isinstance(value, ResolutionAcceptance) for value in acceptances.values()
    ):
        raise _error(
            "resolution_acceptance_type_invalid",
            "acceptance decisions must contain ResolutionAcceptance values",
        )
    if any(not isinstance(value, ResolutionSupport) for value in overrides.values()):
        raise _error(
            "resolution_support_type_invalid",
            "support overrides must contain ResolutionSupport values",
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

    preview_spectra: list[ResolutionPreparationPreviewSpectrum] = []
    for group_index, (spectrum, padding) in enumerate(
        zip(
            resolution_dataset.spectra,
            resolution_padding.spectra,
            strict=True,
        )
    ):
        original_support = _default_support(spectrum)
        acceptance = acceptances.get(group_index, ResolutionAcceptance())
        support = overrides.get(group_index)
        if acceptance.decision is ResolutionAcceptanceDecision.KEEP:
            if support is not None:
                raise _error(
                    "resolution_keep_support_override_invalid",
                    "KEEP uses the unchanged pre-QC support; select EXCLUDE "
                    "to supply a narrower contiguous support",
                    group_index=group_index,
                    group_identity=spectrum.group_label,
                )
            support = original_support
        elif (
            acceptance.decision
            is ResolutionAcceptanceDecision.EXCLUDE_BY_CONTIGUOUS_SUPPORT
        ):
            if support is None:
                raise _error(
                    "resolution_exclusion_support_required",
                    "EXCLUDE requires a narrower contiguous support",
                    group_index=group_index,
                    group_identity=spectrum.group_label,
                )
            if not _support_is_within(support, original_support):
                raise _error(
                    "resolution_exclusion_outside_original_support",
                    "excluded resolution support must remain inside the pre-QC support",
                    group_index=group_index,
                    group_identity=spectrum.group_label,
                )
            if not _support_is_narrower(support, original_support):
                raise _error(
                    "resolution_exclusion_not_narrower",
                    "EXCLUDE support must remove at least one outer boundary region",
                    group_index=group_index,
                    group_identity=spectrum.group_label,
                )
        elif support is not None:
            raise _error(
                "resolution_exclusion_decision_required",
                "a support override requires an explicit EXCLUDE decision",
                group_index=group_index,
                group_identity=spectrum.group_label,
            )
        else:
            support = original_support
        if support.source is not ResolutionSupportSource.EXPLICIT_OVERRIDE and (
            acceptance.decision
            is ResolutionAcceptanceDecision.EXCLUDE_BY_CONTIGUOUS_SUPPORT
        ):
            support = ResolutionSupport(
                lower_energy=support.lower_energy,
                upper_energy=support.upper_energy,
                source=ResolutionSupportSource.EXPLICIT_OVERRIDE,
            )
        preview_spectra.append(
            _preview_spectrum(
                spectrum,
                padding,
                original_support,
                support,
                acceptance,
                auto_padding_applied=auto_application.get(group_index, True),
            )
        )

    return ResolutionPreparationPreview(
        sample_dataset=sample_dataset,
        resolution_dataset=resolution_dataset,
        sample_padding=sample_padding,
        resolution_padding=resolution_padding,
        spectra=tuple(preview_spectra),
        padding_comparisons=comparisons,
        diagnostics=comparison_diagnostics,
    )


def prepare_measured_resolution(
    sample_dataset: ReducedDataset,
    resolution_dataset: ReducedDataset,
    *,
    acceptance_decisions: Mapping[int, ResolutionAcceptance] | None = None,
    support_overrides: Mapping[int, ResolutionSupport] | None = None,
    apply_auto_padding: Mapping[int, bool] | None = None,
) -> PreparedResolution:
    """Normalize only explicitly confirmed per-Q measured-resolution kernels."""

    preview = preview_measured_resolution(
        sample_dataset,
        resolution_dataset,
        acceptance_decisions=acceptance_decisions,
        support_overrides=support_overrides,
        apply_auto_padding=apply_auto_padding,
    )
    gate_diagnostics: list[ResolutionDiagnostic] = []
    for group_index, spectrum in enumerate(preview.spectra):
        acceptance = spectrum.acceptance
        if acceptance.decision is None:
            gate_diagnostics.append(
                ResolutionDiagnostic(
                    code="resolution_acceptance_required",
                    severity=DiagnosticSeverity.ERROR,
                    message=(
                        "measured resolution must have an explicit per-Q KEEP or "
                        "EXCLUDE decision before normalization and use"
                    ),
                    group_index=group_index,
                    group_identity=spectrum.source_spectrum.group_label,
                )
            )
        elif not acceptance.confirmed:
            gate_diagnostics.append(
                ResolutionDiagnostic(
                    code="resolution_acceptance_unconfirmed",
                    severity=DiagnosticSeverity.ERROR,
                    message=(
                        "measured-resolution review decision must be explicitly "
                        "confirmed before normalization and use"
                    ),
                    group_index=group_index,
                    group_identity=spectrum.source_spectrum.group_label,
                )
            )
    if gate_diagnostics:
        raise ResolutionPreparationError(tuple(gate_diagnostics))

    prepared_spectra = tuple(
        PreparedResolutionSpectrum(
            source_spectrum=item.source_spectrum,
            padding=item.padding,
            original_support=item.original_support,
            support=item.support,
            acceptance=item.acceptance,
            auto_padding_applied=item.auto_padding_applied,
            pre_qc_integral=item.pre_qc_integral,
            normalization_integral=item.normalization_integral,
            normalization_factor=item.normalization_factor,
            signed_area_ratio=item.signed_area_ratio,
            normalization_method=item.normalization_method,
            diagnostics=item.diagnostics,
        )
        for item in preview.spectra
    )
    return PreparedResolution(
        sample_dataset=preview.sample_dataset,
        resolution_dataset=preview.resolution_dataset,
        sample_padding=preview.sample_padding,
        resolution_padding=preview.resolution_padding,
        spectra=prepared_spectra,
        padding_comparisons=preview.padding_comparisons,
        diagnostics=preview.diagnostics,
    )
