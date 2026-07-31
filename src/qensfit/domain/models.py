"""Minimal typed in-memory scientific models for Milestone 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from qensfit.domain.diagnostics import (
    DiagnosticSeverity,
    ImportDiagnostic,
)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


class SpectrumRole(StrEnum):
    """Scientific role of an imported reduced spectrum."""

    SAMPLE = "sample"
    RESOLUTION = "resolution"


class ReducedDataFormat(StrEnum):
    """Reduced-data layouts recognized by Milestone 1."""

    DAVE_GROUP_BLOCKS = "dave_group_blocks"
    WIDE_QENS_TABLE = "wide_qens_table"
    SINGLE_SPECTRUM_TABLE = "single_spectrum_table"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"


class DetectionConfidence(StrEnum):
    """Confidence assigned by content-based format detection."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class SourceColumnMetadata:
    """Source columns mapped to one scientific spectrum."""

    energy: str
    intensity: str
    uncertainty: str
    extra_columns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FormatDetectionResult:
    """Structured result of content-based reduced-data detection."""

    proposed_format: ReducedDataFormat
    confidence: DetectionConfidence
    evidence: tuple[str, ...]
    detected_required_columns: tuple[str, ...]
    detected_extra_columns: tuple[str, ...]
    detected_count: int
    diagnostics: tuple[ImportDiagnostic, ...] = ()
    alternative_formats: tuple[ReducedDataFormat, ...] = ()
    explicit_override: bool = False
    requires_confirmation: bool = True
    extension_hint: str | None = None

    @property
    def has_errors(self) -> bool:
        """Return whether detection produced at least one error."""

        return any(
            diagnostic.severity is DiagnosticSeverity.ERROR
            for diagnostic in self.diagnostics
        )


def _readonly_float_array(value: npt.ArrayLike, *, name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    array.setflags(write=False)
    return array


def _readonly_bool_array(value: npt.ArrayLike, *, name: str) -> BoolArray:
    array = np.array(value, dtype=np.bool_, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional boolean array")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class Spectrum:
    """One imported spectrum with original values and invalid-value masks."""

    role: SpectrumRole
    group_index: int
    group_label: str
    energy: FloatArray = field(repr=False)
    intensity: FloatArray = field(repr=False)
    uncertainty: FloatArray = field(repr=False)
    energy_unit: str
    intensity_unit: str
    uncertainty_unit: str
    source_row_numbers: tuple[int, ...] = field(repr=False)
    invalid_energy_mask: BoolArray = field(repr=False)
    invalid_intensity_mask: BoolArray = field(repr=False)
    invalid_uncertainty_mask: BoolArray = field(repr=False)
    source_columns: SourceColumnMetadata
    source_layout: ReducedDataFormat
    source_layout_metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.role, SpectrumRole):
            raise ValueError("role must be a SpectrumRole")
        if not isinstance(self.source_layout, ReducedDataFormat):
            raise ValueError("source_layout must be a ReducedDataFormat")
        if self.group_index < 0:
            raise ValueError("group_index must be nonnegative")
        if not self.group_label:
            raise ValueError("group_label must not be empty")
        if not self.energy_unit or not self.intensity_unit or not self.uncertainty_unit:
            raise ValueError("spectrum units must not be empty")

        energy = _readonly_float_array(self.energy, name="energy")
        intensity = _readonly_float_array(self.intensity, name="intensity")
        uncertainty = _readonly_float_array(self.uncertainty, name="uncertainty")
        invalid_energy = _readonly_bool_array(
            self.invalid_energy_mask,
            name="invalid_energy_mask",
        )
        invalid_intensity = _readonly_bool_array(
            self.invalid_intensity_mask,
            name="invalid_intensity_mask",
        )
        invalid_uncertainty = _readonly_bool_array(
            self.invalid_uncertainty_mask,
            name="invalid_uncertainty_mask",
        )

        length = energy.size
        if length == 0:
            raise ValueError("spectrum arrays must not be empty")
        arrays = (
            intensity,
            uncertainty,
            invalid_energy,
            invalid_intensity,
            invalid_uncertainty,
        )
        if any(array.size != length for array in arrays):
            raise ValueError("spectrum arrays and invalid masks must have equal length")
        if len(self.source_row_numbers) != length:
            raise ValueError("source_row_numbers must match the spectrum length")

        object.__setattr__(self, "energy", energy)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "invalid_energy_mask", invalid_energy)
        object.__setattr__(self, "invalid_intensity_mask", invalid_intensity)
        object.__setattr__(self, "invalid_uncertainty_mask", invalid_uncertainty)

    @classmethod
    def from_imported_arrays(
        cls,
        *,
        role: SpectrumRole,
        group_index: int,
        group_label: str,
        energy: npt.ArrayLike,
        intensity: npt.ArrayLike,
        uncertainty: npt.ArrayLike,
        energy_unit: str,
        intensity_unit: str,
        uncertainty_unit: str,
        source_row_numbers: tuple[int, ...],
        source_columns: SourceColumnMetadata,
        source_layout: ReducedDataFormat,
        source_layout_metadata: tuple[tuple[str, str], ...] = (),
    ) -> Spectrum:
        """Build a spectrum while classifying invalid values without repair."""

        energy_array = np.asarray(energy, dtype=np.float64)
        intensity_array = np.asarray(intensity, dtype=np.float64)
        uncertainty_array = np.asarray(uncertainty, dtype=np.float64)
        return cls(
            role=role,
            group_index=group_index,
            group_label=group_label,
            energy=energy_array,
            intensity=intensity_array,
            uncertainty=uncertainty_array,
            energy_unit=energy_unit,
            intensity_unit=intensity_unit,
            uncertainty_unit=uncertainty_unit,
            source_row_numbers=source_row_numbers,
            invalid_energy_mask=~np.isfinite(energy_array),
            invalid_intensity_mask=~np.isfinite(intensity_array),
            invalid_uncertainty_mask=(
                ~np.isfinite(uncertainty_array) | (uncertainty_array <= 0.0)
            ),
            source_columns=source_columns,
            source_layout=source_layout,
            source_layout_metadata=source_layout_metadata,
        )


@dataclass(frozen=True, slots=True)
class InvalidValueCounts:
    """Privacy-safe invalid-value counts for one spectrum."""

    energy: int
    intensity: int
    uncertainty: int


@dataclass(frozen=True, slots=True)
class ImportSummary:
    """Privacy-safe structural summary of an imported reduced dataset."""

    detected_format: ReducedDataFormat
    confidence: DetectionConfidence
    spectrum_count: int
    row_counts: tuple[int, ...]
    detected_required_columns: tuple[str, ...]
    detected_extra_columns: tuple[str, ...]
    finite_energy_ranges: tuple[tuple[float | None, float | None], ...]
    invalid_value_counts: tuple[InvalidValueCounts, ...]
    shared_energy_grid: bool


@dataclass(frozen=True, slots=True)
class ImportedDataset:
    """Ordered in-memory collection of imported reduced spectra."""

    role: SpectrumRole
    source_layout: ReducedDataFormat
    spectra: tuple[Spectrum, ...] = field(repr=False)
    source_reference: str
    diagnostics: tuple[ImportDiagnostic, ...]
    detected_extra_columns: tuple[str, ...]
    shared_energy_grid: bool
    shared_energy_axis: FloatArray | None = field(default=None, repr=False)
    format_detection: FormatDetectionResult = field(
        default_factory=lambda: FormatDetectionResult(
            proposed_format=ReducedDataFormat.UNKNOWN,
            confidence=DetectionConfidence.NONE,
            evidence=(),
            detected_required_columns=(),
            detected_extra_columns=(),
            detected_count=0,
        )
    )

    def __post_init__(self) -> None:
        if not isinstance(self.role, SpectrumRole):
            raise ValueError("role must be a SpectrumRole")
        if not isinstance(self.source_layout, ReducedDataFormat):
            raise ValueError("source_layout must be a ReducedDataFormat")
        if not self.spectra:
            raise ValueError("an imported dataset requires at least one spectrum")
        if any(spectrum.role is not self.role for spectrum in self.spectra):
            raise ValueError("all spectrum roles must match the dataset role")
        if any(
            spectrum.source_layout is not self.source_layout
            for spectrum in self.spectra
        ):
            raise ValueError("all spectrum layouts must match the dataset layout")
        if tuple(spectrum.group_index for spectrum in self.spectra) != tuple(
            range(len(self.spectra))
        ):
            raise ValueError("spectrum group indices must be ordered and contiguous")

        shared_axis: FloatArray | None = None
        if self.shared_energy_axis is not None:
            shared_axis = _readonly_float_array(
                self.shared_energy_axis,
                name="shared_energy_axis",
            )
        if self.shared_energy_grid:
            if shared_axis is None:
                raise ValueError(
                    "shared_energy_axis is required when shared_energy_grid is true"
                )
            if any(
                not np.array_equal(
                    spectrum.energy,
                    shared_axis,
                    equal_nan=True,
                )
                for spectrum in self.spectra
            ):
                raise ValueError("shared_energy_axis must match every spectrum")
        elif shared_axis is not None:
            raise ValueError(
                "shared_energy_axis must be absent when shared_energy_grid is false"
            )
        object.__setattr__(self, "shared_energy_axis", shared_axis)

    def structural_summary(self) -> ImportSummary:
        """Return counts and finite ranges without exposing numerical arrays."""

        energy_ranges: list[tuple[float | None, float | None]] = []
        invalid_counts: list[InvalidValueCounts] = []
        for spectrum in self.spectra:
            finite_energy = spectrum.energy[np.isfinite(spectrum.energy)]
            if finite_energy.size:
                energy_ranges.append(
                    (float(np.min(finite_energy)), float(np.max(finite_energy)))
                )
            else:
                energy_ranges.append((None, None))
            invalid_counts.append(
                InvalidValueCounts(
                    energy=int(np.count_nonzero(spectrum.invalid_energy_mask)),
                    intensity=int(np.count_nonzero(spectrum.invalid_intensity_mask)),
                    uncertainty=int(
                        np.count_nonzero(spectrum.invalid_uncertainty_mask)
                    ),
                )
            )
        return ImportSummary(
            detected_format=self.source_layout,
            confidence=self.format_detection.confidence,
            spectrum_count=len(self.spectra),
            row_counts=tuple(spectrum.energy.size for spectrum in self.spectra),
            detected_required_columns=(
                self.format_detection.detected_required_columns
            ),
            detected_extra_columns=self.detected_extra_columns,
            finite_energy_ranges=tuple(energy_ranges),
            invalid_value_counts=tuple(invalid_counts),
            shared_energy_grid=self.shared_energy_grid,
        )

