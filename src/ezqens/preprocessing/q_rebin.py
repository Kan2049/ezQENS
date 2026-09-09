"""Fractional-coverage-aware Q rebinning for reduced spectra."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from ezqens.domain import (
    FractionalCoverage,
    FractionalCoverageOrigin,
    QBins,
    QExclusionInterval,
    QRebinSpecification,
    ReducedDataset,
    Spectrum,
)


class QRebinDiagnosticCode(StrEnum):
    """Stable scientific blockers for fractional Q rebinning."""

    FRACTIONAL_COVERAGE_MISSING = "fractional_coverage_missing"
    SOURCE_Q_EDGES_REQUIRED = "source_q_edges_required"
    TARGET_OUTSIDE_SOURCE_COVERAGE = "target_outside_source_coverage"
    TARGET_REFINES_SOURCE = "target_refines_source"
    INCOMPATIBLE_ENERGY_GRIDS = "incompatible_energy_grids"
    INCOMPATIBLE_UNITS = "incompatible_units"
    EMPTY_TARGET_OVERLAP = "empty_target_overlap"
    OUTPUT_COVERAGE_MISSING = "output_coverage_missing"


@dataclass(frozen=True, slots=True)
class QRebinDiagnostic:
    """One structured Q-rebin failure reason."""

    code: QRebinDiagnosticCode
    message: str
    target_group_index: int | None = None
    source_group_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, QRebinDiagnosticCode):
            raise ValueError("Q-rebin diagnostic code must be typed")
        if not self.message:
            raise ValueError("Q-rebin diagnostic message must not be empty")


class QRebinError(ValueError):
    """Raised when a requested fractional Q rebin is scientifically invalid."""

    def __init__(self, diagnostics: tuple[QRebinDiagnostic, ...]) -> None:
        if not diagnostics:
            raise ValueError("QRebinError requires at least one diagnostic")
        self.diagnostics = tuple(diagnostics)
        super().__init__(
            "; ".join(f"{item.code.value}: {item.message}" for item in diagnostics)
        )


def _fail(
    code: QRebinDiagnosticCode,
    message: str,
    *,
    target_group_index: int | None = None,
    source_group_index: int | None = None,
) -> QRebinError:
    return QRebinError(
        (
            QRebinDiagnostic(
                code=code,
                message=message,
                target_group_index=target_group_index,
                source_group_index=source_group_index,
            ),
        )
    )


def _merged_exclusions(
    exclusions: tuple[QExclusionInterval, ...],
) -> tuple[tuple[float, float], ...]:
    ordered = sorted((item.lower_q, item.upper_q) for item in exclusions)
    merged: list[tuple[float, float]] = []
    for lower, upper in ordered:
        if not merged or lower > merged[-1][1]:
            merged.append((lower, upper))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], upper))
    return tuple(merged)


def _unexcluded_overlap(
    lower: float,
    upper: float,
    exclusions: tuple[tuple[float, float], ...],
) -> float:
    length = upper - lower
    if length <= 0.0:
        return 0.0
    removed = 0.0
    for excluded_lower, excluded_upper in exclusions:
        overlap_lower = max(lower, excluded_lower)
        overlap_upper = min(upper, excluded_upper)
        if overlap_upper > overlap_lower:
            removed += overlap_upper - overlap_lower
    return max(0.0, length - removed)


def _refinement_tolerance(source_width: float, target_width: float) -> float:
    epsilon = np.finfo(np.float64).eps
    return 32.0 * epsilon * max(source_width, target_width)


def _require_matching_scientific_axes(
    spectra: tuple[Spectrum, ...],
    source_indices: tuple[int, ...],
    *,
    target_group_index: int,
) -> Spectrum:
    representative = spectra[source_indices[0]]
    for source_index in source_indices[1:]:
        spectrum = spectra[source_index]
        if not np.array_equal(
            spectrum.energy,
            representative.energy,
            equal_nan=True,
        ):
            raise _fail(
                QRebinDiagnosticCode.INCOMPATIBLE_ENERGY_GRIDS,
                "source spectra contributing to one target Q bin must have "
                "exactly matching energy coordinates",
                target_group_index=target_group_index,
                source_group_index=source_index,
            )
        if (
            spectrum.energy_unit != representative.energy_unit
            or spectrum.intensity_unit != representative.intensity_unit
            or spectrum.uncertainty_unit != representative.uncertainty_unit
        ):
            raise _fail(
                QRebinDiagnosticCode.INCOMPATIBLE_UNITS,
                "source spectra contributing to one target Q bin must have "
                "matching scientific units",
                target_group_index=target_group_index,
                source_group_index=source_index,
            )
    return representative


def rebin_fractional_q(
    dataset: ReducedDataset,
    specification: QRebinSpecification,
) -> ReducedDataset:
    """Return a new dataset rebinned by exact Q overlap and stored coverage."""

    if not isinstance(dataset, ReducedDataset):
        raise ValueError("dataset must be a ReducedDataset")
    if not isinstance(specification, QRebinSpecification):
        raise ValueError("specification must be a QRebinSpecification")
    coverage = dataset.fractional_coverage
    if coverage is None:
        raise _fail(
            QRebinDiagnosticCode.FRACTIONAL_COVERAGE_MISSING,
            "fractional Q-E coverage is required before Q rebinning",
        )
    if dataset.q_bins is None or dataset.q_bins.edges is None:
        raise _fail(
            QRebinDiagnosticCode.SOURCE_Q_EDGES_REQUIRED,
            "explicit source Q-bin edges are required before Q rebinning",
        )

    source_edges = dataset.q_bins.edges
    target_q_bins = QBins.from_edges(specification.target_edges)
    target_edges = target_q_bins.edges
    assert target_edges is not None
    if target_edges[0] < source_edges[0] or target_edges[-1] > source_edges[-1]:
        raise _fail(
            QRebinDiagnosticCode.TARGET_OUTSIDE_SOURCE_COVERAGE,
            "target Q-bin range must lie inside the current source Q coverage",
        )

    exclusions = _merged_exclusions(specification.exclusions)
    source_widths = np.diff(source_edges)
    output_spectra: list[Spectrum] = []
    output_coverage: list[np.ndarray] = []

    for target_index, (target_lower, target_upper) in enumerate(
        zip(target_edges[:-1], target_edges[1:], strict=True)
    ):
        target_width = float(target_upper - target_lower)
        overlap_fractions = np.zeros(dataset.q_bins.group_count, dtype=np.float64)
        for source_index, (source_lower, source_upper, source_width) in enumerate(
            zip(
                source_edges[:-1],
                source_edges[1:],
                source_widths,
                strict=True,
            )
        ):
            overlap_lower = max(float(source_lower), float(target_lower))
            overlap_upper = min(float(source_upper), float(target_upper))
            retained_overlap = _unexcluded_overlap(
                overlap_lower,
                overlap_upper,
                exclusions,
            )
            if retained_overlap <= 0.0:
                continue
            numeric_source_width = float(source_width)
            if (
                target_width
                + _refinement_tolerance(
                    numeric_source_width,
                    target_width,
                )
                < numeric_source_width
            ):
                raise _fail(
                    QRebinDiagnosticCode.TARGET_REFINES_SOURCE,
                    "target Q bins may not refine a contributing current source bin",
                    target_group_index=target_index,
                    source_group_index=source_index,
                )
            overlap_fractions[source_index] = retained_overlap / numeric_source_width

        source_indices = tuple(
            int(index) for index in np.flatnonzero(overlap_fractions > 0.0)
        )
        if not source_indices:
            raise _fail(
                QRebinDiagnosticCode.EMPTY_TARGET_OVERLAP,
                "target Q bin has no retained source overlap after Q exclusions",
                target_group_index=target_index,
            )

        representative = _require_matching_scientific_axes(
            dataset.spectra,
            source_indices,
            target_group_index=target_index,
        )
        point_count = representative.energy.size
        fractional_weight = np.zeros(point_count, dtype=np.float64)
        weighted_signal = np.zeros(point_count, dtype=np.float64)
        weighted_variance = np.zeros(point_count, dtype=np.float64)
        invalid_uncertainty = np.zeros(point_count, dtype=np.bool_)
        for source_index in source_indices:
            source_coverage = coverage.values[source_index]
            geometric_fraction = overlap_fractions[source_index]
            point_weight = source_coverage * geometric_fraction
            fractional_weight += point_weight
            active = source_coverage > 0.0
            source_spectrum = dataset.spectra[source_index]
            weighted_signal[active] += (
                source_spectrum.intensity[active] * point_weight[active]
            )
            invalid_uncertainty |= active & source_spectrum.invalid_uncertainty_mask
            valid_uncertainty = active & ~source_spectrum.invalid_uncertainty_mask
            weighted_variance[valid_uncertainty] += (
                np.square(
                    source_spectrum.uncertainty[valid_uncertainty]
                    * source_coverage[valid_uncertainty]
                )
                * geometric_fraction
            )

        missing_weight = ~np.isfinite(fractional_weight) | (fractional_weight <= 0.0)
        if np.any(missing_weight):
            raise _fail(
                QRebinDiagnosticCode.OUTPUT_COVERAGE_MISSING,
                "every output Q-E point must retain positive finite fractional "
                "coverage",
                target_group_index=target_index,
            )

        intensity = weighted_signal / fractional_weight
        uncertainty = np.sqrt(weighted_variance) / fractional_weight
        uncertainty[invalid_uncertainty] = np.nan
        output_spectra.append(
            Spectrum(
                role=dataset.role,
                group_index=target_index,
                group_label=f"Q={target_q_bins.q_values[target_index]:.12g}",
                energy=representative.energy,
                intensity=intensity,
                uncertainty=uncertainty,
                energy_unit=representative.energy_unit,
                intensity_unit=representative.intensity_unit,
                uncertainty_unit=representative.uncertainty_unit,
            )
        )
        output_coverage.append(fractional_weight)

    spectra = tuple(output_spectra)
    propagated = FractionalCoverage.aligned_with(
        spectra,
        values=tuple(output_coverage),
        origin=FractionalCoverageOrigin.PROPAGATED_Q_REBIN,
        q_rebin_history=(*coverage.q_rebin_history, specification),
    )
    return replace(
        dataset,
        spectra=spectra,
        source_columns=(),
        q_bins=target_q_bins,
        fractional_coverage=propagated,
    )
