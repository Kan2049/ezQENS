"""Small typed scientific models for reduced QENS spectra."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from qensfit.domain.diagnostics import DiagnosticSeverity, ImportDiagnostic

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


class SpectrumRole(StrEnum):
    """Scientific role of a reduced spectrum."""

    SAMPLE = "sample"
    RESOLUTION = "resolution"


class ReducedDataFormat(StrEnum):
    """Reduced text layouts recognized by the current importer."""

    DAVE_GROUP_BLOCKS = "dave_group_blocks"
    WIDE_QENS_TABLE = "wide_qens_table"
    SINGLE_SPECTRUM_TABLE = "single_spectrum_table"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class FormatDetectionResult:
    """Content-based layout proposal and structural diagnostics."""

    proposed_format: ReducedDataFormat
    evidence: tuple[str, ...]
    detected_required_columns: tuple[str, ...]
    detected_extra_columns: tuple[str, ...]
    detected_count: int
    diagnostics: tuple[ImportDiagnostic, ...] = ()
    alternative_formats: tuple[ReducedDataFormat, ...] = ()

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


@dataclass(frozen=True, slots=True)
class Spectrum:
    """One reduced spectrum with immutable original values and invalid masks."""

    role: SpectrumRole
    group_index: int
    group_label: str
    energy: FloatArray = field(repr=False)
    intensity: FloatArray = field(repr=False)
    uncertainty: FloatArray = field(repr=False)
    energy_unit: str
    intensity_unit: str
    uncertainty_unit: str
    invalid_energy_mask: BoolArray = field(init=False, repr=False)
    invalid_intensity_mask: BoolArray = field(init=False, repr=False)
    invalid_uncertainty_mask: BoolArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.role, SpectrumRole):
            raise ValueError("role must be a SpectrumRole")
        if self.group_index < 0:
            raise ValueError("group_index must be nonnegative")
        if not self.group_label:
            raise ValueError("group_label must not be empty")
        if not self.energy_unit or not self.intensity_unit or not self.uncertainty_unit:
            raise ValueError("spectrum units must not be empty")

        energy = _readonly_float_array(self.energy, name="energy")
        intensity = _readonly_float_array(self.intensity, name="intensity")
        uncertainty = _readonly_float_array(self.uncertainty, name="uncertainty")
        if energy.size == 0:
            raise ValueError("spectrum arrays must not be empty")
        if intensity.size != energy.size or uncertainty.size != energy.size:
            raise ValueError("spectrum arrays must have equal length")

        invalid_energy = ~np.isfinite(energy)
        invalid_intensity = ~np.isfinite(intensity)
        invalid_uncertainty = ~np.isfinite(uncertainty) | (uncertainty <= 0.0)
        for mask in (invalid_energy, invalid_intensity, invalid_uncertainty):
            mask.setflags(write=False)

        object.__setattr__(self, "energy", energy)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "invalid_energy_mask", invalid_energy)
        object.__setattr__(self, "invalid_intensity_mask", invalid_intensity)
        object.__setattr__(self, "invalid_uncertainty_mask", invalid_uncertainty)


@dataclass(frozen=True, slots=True)
class SourceColumnMetadata:
    """Text-column traceability kept at the dataset import boundary."""

    group_identity: str
    energy: str
    intensity: str
    uncertainty: str
    extra_columns: tuple[str, ...] = ()
    source_row_numbers: tuple[int, ...] = field(default=(), repr=False)


@dataclass(frozen=True, slots=True)
class InvalidValueCounts:
    """Privacy-safe invalid-value counts for one spectrum."""

    energy: int
    intensity: int
    uncertainty: int


@dataclass(frozen=True, slots=True)
class ImportSummary:
    """Privacy-safe structural summary of a reduced dataset."""

    detected_format: ReducedDataFormat | None
    spectrum_count: int
    row_counts: tuple[int, ...]
    detected_required_columns: tuple[str, ...]
    detected_extra_columns: tuple[str, ...]
    finite_energy_ranges: tuple[tuple[float | None, float | None], ...]
    invalid_value_counts: tuple[InvalidValueCounts, ...]
    shared_energy_grid: bool


@dataclass(frozen=True, slots=True)
class ReducedDataset:
    """Ordered reduced spectra independent of their producing data source."""

    role: SpectrumRole
    spectra: tuple[Spectrum, ...] = field(repr=False)
    source_reference: str | None = None
    source_layout: ReducedDataFormat | None = None
    diagnostics: tuple[ImportDiagnostic, ...] = ()
    source_columns: tuple[SourceColumnMetadata, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.role, SpectrumRole):
            raise ValueError("role must be a SpectrumRole")
        if self.source_layout is not None and not isinstance(
            self.source_layout, ReducedDataFormat
        ):
            raise ValueError("source_layout must be a ReducedDataFormat or None")
        if not self.spectra:
            raise ValueError("a reduced dataset requires at least one spectrum")
        if any(spectrum.role is not self.role for spectrum in self.spectra):
            raise ValueError("all spectrum roles must match the dataset role")
        if tuple(spectrum.group_index for spectrum in self.spectra) != tuple(
            range(len(self.spectra))
        ):
            raise ValueError("spectrum group indices must be ordered and contiguous")
        if self.source_columns:
            if len(self.source_columns) != len(self.spectra):
                raise ValueError("source column metadata must match spectrum count")
            if any(
                metadata.group_identity != spectrum.group_label
                for metadata, spectrum in zip(
                    self.source_columns, self.spectra, strict=True
                )
            ):
                raise ValueError("source column metadata must match spectrum order")

    @property
    def shared_energy_grid(self) -> bool:
        """Return whether every spectrum has the same energy array."""

        first = self.spectra[0].energy
        return all(
            np.array_equal(spectrum.energy, first, equal_nan=True)
            for spectrum in self.spectra[1:]
        )

    @property
    def detected_extra_columns(self) -> tuple[str, ...]:
        """Return unique ignored source columns in source order."""

        return tuple(
            dict.fromkeys(
                column
                for metadata in self.source_columns
                for column in metadata.extra_columns
            )
        )

    @property
    def detected_required_columns(self) -> tuple[str, ...]:
        """Return unique mapped source columns in source order."""

        return tuple(
            dict.fromkeys(
                column
                for metadata in self.source_columns
                for column in (
                    metadata.energy,
                    metadata.intensity,
                    metadata.uncertainty,
                )
            )
        )

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
            spectrum_count=len(self.spectra),
            row_counts=tuple(spectrum.energy.size for spectrum in self.spectra),
            detected_required_columns=self.detected_required_columns,
            detected_extra_columns=self.detected_extra_columns,
            finite_energy_ranges=tuple(energy_ranges),
            invalid_value_counts=tuple(invalid_counts),
            shared_energy_grid=self.shared_energy_grid,
        )
