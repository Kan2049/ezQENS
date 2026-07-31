"""Detect high-confidence boundary-connected padding without changing data."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from qensfit.domain import DiagnosticSeverity, ImportedDataset, Spectrum

BoolArray = npt.NDArray[np.bool_]

ALGORITHM_VERSION = "edge-padding-v1.0.0"


class PaddingConfidence(StrEnum):
    """Confidence assigned to a boundary-padding interpretation."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class BoundarySide(StrEnum):
    """Array boundary inspected for a connected plateau."""

    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True, slots=True)
class EdgePaddingConfig:
    """Explicit numerical tolerances and evidence thresholds."""

    plateau_relative_tolerance: float = 1.0e-7
    plateau_absolute_tolerance: float = 1.0e-12
    signature_relative_tolerance: float = 1.0e-6
    signature_absolute_tolerance: float = 1.0e-10
    transition_relative_threshold: float = 5.0e-2
    transition_absolute_threshold: float = 1.0e-12
    transition_sigma_threshold: float = 5.0
    minimum_run_points: int = 2
    regular_run_points: int = 3
    long_run_points: int = 5
    long_run_fraction: float = 1.0e-1
    short_run_supporting_spectra: int = 2

    def __post_init__(self) -> None:
        nonnegative = (
            self.plateau_relative_tolerance,
            self.plateau_absolute_tolerance,
            self.signature_relative_tolerance,
            self.signature_absolute_tolerance,
            self.transition_relative_threshold,
            self.transition_absolute_threshold,
            self.transition_sigma_threshold,
        )
        if any(value < 0.0 or not np.isfinite(value) for value in nonnegative):
            raise ValueError("padding tolerances and thresholds must be finite")
        if self.minimum_run_points < 2:
            raise ValueError("minimum_run_points must be at least two")
        if self.regular_run_points < self.minimum_run_points:
            raise ValueError("regular_run_points must not be below the minimum")
        if self.long_run_points < self.regular_run_points:
            raise ValueError("long_run_points must not be below regular_run_points")
        if not 0.0 < self.long_run_fraction < 1.0:
            raise ValueError("long_run_fraction must be strictly between zero and one")
        if self.short_run_supporting_spectra < 1:
            raise ValueError("short-run support requires at least one other spectrum")


@dataclass(frozen=True, slots=True)
class PaddingDiagnostic:
    """Structured edge-padding diagnostic without scientific arrays."""

    code: str
    severity: DiagnosticSeverity
    message: str
    group_identity: str
    side: BoundarySide | None = None


@dataclass(frozen=True, slots=True)
class BoundaryPaddingDetection:
    """Evidence and decision for one connected array boundary."""

    side: BoundarySide
    run_length: int
    energy_bounds: tuple[float, float] | None
    plateau_intensity: float | None
    plateau_uncertainty: float | None
    adjacent_interior_index: int | None
    intensity_transition_absolute: float | None
    intensity_transition_sigma: float | None
    uncertainty_transition_absolute: float | None
    run_fraction: float
    matching_spectrum_count: int
    supporting_long_run_spectrum_count: int
    confidence: PaddingConfidence
    evidence_codes: tuple[str, ...]
    recommended_default_on: bool
    diagnostics: tuple[PaddingDiagnostic, ...] = ()


def _readonly_mask(value: npt.ArrayLike, *, name: str) -> BoolArray:
    mask = np.array(value, dtype=np.bool_, copy=True)
    if mask.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    mask.setflags(write=False)
    return mask


@dataclass(frozen=True, slots=True)
class SpectrumPaddingMask:
    """Immutable high-confidence mask plus confirmation-gated suggestions."""

    group_index: int
    group_identity: str
    padding_mask: BoolArray = field(repr=False)
    suggested_padding_mask: BoolArray = field(repr=False)
    left: BoundaryPaddingDetection
    right: BoundaryPaddingDetection
    confidence: PaddingConfidence
    evidence_codes: tuple[str, ...]
    algorithm_version: str
    recommended_default_on: bool
    diagnostics: tuple[PaddingDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        padding_mask = _readonly_mask(self.padding_mask, name="padding_mask")
        suggestion_mask = _readonly_mask(
            self.suggested_padding_mask,
            name="suggested_padding_mask",
        )
        if padding_mask.size != suggestion_mask.size:
            raise ValueError("padding and suggestion masks must have equal length")
        if np.any(padding_mask & ~suggestion_mask):
            raise ValueError("default-on padding must also be a detected suggestion")
        object.__setattr__(self, "padding_mask", padding_mask)
        object.__setattr__(self, "suggested_padding_mask", suggestion_mask)

    @property
    def left_masked_point_count(self) -> int:
        """Return high-confidence default-on points at the left boundary."""

        return self.left.run_length if self.left.recommended_default_on else 0

    @property
    def right_masked_point_count(self) -> int:
        """Return high-confidence default-on points at the right boundary."""

        return self.right.run_length if self.right.recommended_default_on else 0

    @property
    def total_masked_point_count(self) -> int:
        """Return the total number of high-confidence default-on points."""

        return int(np.count_nonzero(self.padding_mask))


@dataclass(frozen=True, slots=True)
class SpectrumPaddingSummary:
    """Privacy-safe structural padding summary for one spectrum."""

    group_index: int
    group_identity: str
    left_auto_masked_point_count: int
    right_auto_masked_point_count: int
    total_auto_padding_mask_count: int
    derived_valid_energy_range: tuple[float | None, float | None]
    confidence: PaddingConfidence
    evidence_codes: tuple[str, ...]
    recommended_default_on: bool


@dataclass(frozen=True, slots=True)
class EdgePaddingSummary:
    """Privacy-safe structural padding summary for a dataset."""

    algorithm_version: str
    spectra: tuple[SpectrumPaddingSummary, ...]
    total_auto_padding_mask_count: int
    diagnostic_counts_by_severity: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class EdgePaddingDetectionResult:
    """Ordered edge-padding results for an imported dataset."""

    spectra: tuple[SpectrumPaddingMask, ...]
    algorithm_version: str
    configuration: EdgePaddingConfig
    diagnostics: tuple[PaddingDiagnostic, ...]

    def structural_summary(self, dataset: ImportedDataset) -> EdgePaddingSummary:
        """Return mask counts and valid energy ranges without array values."""

        if len(dataset.spectra) != len(self.spectra):
            raise ValueError("padding result does not match dataset spectrum count")
        spectrum_summaries: list[SpectrumPaddingSummary] = []
        for spectrum, result in zip(
            dataset.spectra,
            self.spectra,
            strict=True,
        ):
            if spectrum.group_index != result.group_index:
                raise ValueError("padding result does not match dataset ordering")
            retained_finite = np.isfinite(spectrum.energy) & ~result.padding_mask
            retained_energy = spectrum.energy[retained_finite]
            valid_range: tuple[float | None, float | None]
            if retained_energy.size:
                valid_range = (
                    float(np.min(retained_energy)),
                    float(np.max(retained_energy)),
                )
            else:
                valid_range = (None, None)
            spectrum_summaries.append(
                SpectrumPaddingSummary(
                    group_index=result.group_index,
                    group_identity=result.group_identity,
                    left_auto_masked_point_count=(
                        result.left_masked_point_count
                    ),
                    right_auto_masked_point_count=(
                        result.right_masked_point_count
                    ),
                    total_auto_padding_mask_count=(
                        result.total_masked_point_count
                    ),
                    derived_valid_energy_range=valid_range,
                    confidence=result.confidence,
                    evidence_codes=result.evidence_codes,
                    recommended_default_on=result.recommended_default_on,
                )
            )
        severity_counts = {
            severity.value: sum(
                diagnostic.severity is severity
                for diagnostic in self.diagnostics
            )
            for severity in DiagnosticSeverity
        }
        return EdgePaddingSummary(
            algorithm_version=self.algorithm_version,
            spectra=tuple(spectrum_summaries),
            total_auto_padding_mask_count=sum(
                item.total_auto_padding_mask_count
                for item in spectrum_summaries
            ),
            diagnostic_counts_by_severity=tuple(severity_counts.items()),
        )


@dataclass(frozen=True, slots=True)
class _BoundaryCandidate:
    spectrum_index: int
    group_identity: str
    side: BoundarySide
    point_count: int
    run_length: int
    energy_bounds: tuple[float, float] | None
    plateau_intensity: float | None
    plateau_uncertainty: float | None
    adjacent_interior_index: int | None
    intensity_transition_absolute: float | None
    intensity_transition_sigma: float | None
    uncertainty_transition_absolute: float | None
    numerical_intensity_transition: bool
    statistical_intensity_transition: bool
    uncertainty_transition: bool
    evidence_codes: tuple[str, ...]
    diagnostics: tuple[PaddingDiagnostic, ...]

    @property
    def transition_clear(self) -> bool:
        return (
            self.numerical_intensity_transition
            or self.statistical_intensity_transition
            or self.uncertainty_transition
        )


def _pair_matches(
    intensity: float,
    uncertainty: float,
    reference_intensity: float,
    reference_uncertainty: float,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> bool:
    return bool(
        np.isclose(
            intensity,
            reference_intensity,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
        and np.isclose(
            uncertainty,
            reference_uncertainty,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
    )


def _inspect_boundary(
    spectrum: Spectrum,
    *,
    spectrum_index: int,
    side: BoundarySide,
    configuration: EdgePaddingConfig,
) -> _BoundaryCandidate:
    point_count = spectrum.energy.size
    edge_index = 0 if side is BoundarySide.LEFT else point_count - 1
    edge_values = (
        spectrum.energy[edge_index],
        spectrum.intensity[edge_index],
        spectrum.uncertainty[edge_index],
    )
    if (
        not all(np.isfinite(value) for value in edge_values)
        or edge_values[2] <= 0.0
    ):
        diagnostic = PaddingDiagnostic(
            code="padding_boundary_invalid_value",
            severity=DiagnosticSeverity.INFO,
            message=(
                "Boundary values are handled by invalid-data masks and were "
                "not interpreted as padding"
            ),
            group_identity=spectrum.group_label,
            side=side,
        )
        return _BoundaryCandidate(
            spectrum_index=spectrum_index,
            group_identity=spectrum.group_label,
            side=side,
            point_count=point_count,
            run_length=0,
            energy_bounds=None,
            plateau_intensity=None,
            plateau_uncertainty=None,
            adjacent_interior_index=None,
            intensity_transition_absolute=None,
            intensity_transition_sigma=None,
            uncertainty_transition_absolute=None,
            numerical_intensity_transition=False,
            statistical_intensity_transition=False,
            uncertainty_transition=False,
            evidence_codes=("boundary_invalid_value_separate",),
            diagnostics=(diagnostic,),
        )

    reference_intensity = float(edge_values[1])
    reference_uncertainty = float(edge_values[2])
    step = 1 if side is BoundarySide.LEFT else -1
    indices = [edge_index]
    current = edge_index + step
    while 0 <= current < point_count:
        if not np.isfinite(spectrum.energy[current]):
            break
        intensity = float(spectrum.intensity[current])
        uncertainty = float(spectrum.uncertainty[current])
        if (
            not np.isfinite(intensity)
            or not np.isfinite(uncertainty)
            or uncertainty <= 0.0
            or not _pair_matches(
                intensity,
                uncertainty,
                reference_intensity,
                reference_uncertainty,
                relative_tolerance=configuration.plateau_relative_tolerance,
                absolute_tolerance=configuration.plateau_absolute_tolerance,
            )
        ):
            break
        indices.append(current)
        current += step

    ordered_indices = tuple(sorted(indices))
    plateau_intensity = float(np.mean(spectrum.intensity[list(ordered_indices)]))
    plateau_uncertainty = float(
        np.mean(spectrum.uncertainty[list(ordered_indices)])
    )
    boundary_energy = spectrum.energy[list(ordered_indices)]
    energy_bounds = (
        float(np.min(boundary_energy)),
        float(np.max(boundary_energy)),
    )
    adjacent_index = current if 0 <= current < point_count else None
    evidence: list[str] = []
    qualifying_run = len(indices) >= configuration.minimum_run_points
    if qualifying_run:
        evidence.extend(("boundary_connected_plateau", "repeated_pair_run"))
        if plateau_intensity <= 0.0:
            evidence.append("nonpositive_plateau_intensity_support")
        if len(indices) >= configuration.long_run_points:
            evidence.append("long_absolute_run")
        if len(indices) / point_count >= configuration.long_run_fraction:
            evidence.append("long_relative_run")

    intensity_jump: float | None = None
    transition_sigma: float | None = None
    uncertainty_jump: float | None = None
    numerical_clear = False
    statistical_clear = False
    uncertainty_clear = False
    diagnostics: list[PaddingDiagnostic] = []
    if adjacent_index is None:
        evidence.append("adjacent_interior_point_missing")
    else:
        adjacent_values = (
            spectrum.energy[adjacent_index],
            spectrum.intensity[adjacent_index],
            spectrum.uncertainty[adjacent_index],
        )
        if (
            not all(np.isfinite(value) for value in adjacent_values)
            or adjacent_values[2] <= 0.0
        ):
            evidence.append("adjacent_interior_value_invalid")
            diagnostics.append(
                PaddingDiagnostic(
                    code="padding_transition_invalid_interior",
                    severity=DiagnosticSeverity.INFO,
                    message=(
                        "The adjacent interior point is invalid, so a "
                        "high-confidence transition could not be established"
                    ),
                    group_identity=spectrum.group_label,
                    side=side,
                )
            )
        else:
            adjacent_intensity = float(adjacent_values[1])
            adjacent_uncertainty = float(adjacent_values[2])
            intensity_jump = abs(adjacent_intensity - plateau_intensity)
            combined_uncertainty = float(
                np.hypot(plateau_uncertainty, adjacent_uncertainty)
            )
            if combined_uncertainty > 0.0:
                transition_sigma = intensity_jump / combined_uncertainty
            intensity_scale = max(
                abs(adjacent_intensity),
                abs(plateau_intensity),
            )
            numerical_clear = intensity_jump > (
                configuration.transition_absolute_threshold
                + configuration.transition_relative_threshold * intensity_scale
            )
            statistical_clear = bool(
                transition_sigma is not None
                and transition_sigma >= configuration.transition_sigma_threshold
            )
            uncertainty_jump = abs(
                adjacent_uncertainty - plateau_uncertainty
            )
            uncertainty_scale = max(
                abs(adjacent_uncertainty),
                abs(plateau_uncertainty),
            )
            uncertainty_clear = uncertainty_jump > (
                configuration.transition_absolute_threshold
                + configuration.transition_relative_threshold
                * uncertainty_scale
            )
            if numerical_clear and qualifying_run:
                evidence.append("clear_numerical_intensity_transition")
            if statistical_clear and qualifying_run:
                evidence.append("clear_statistical_intensity_transition")
            if uncertainty_clear and qualifying_run:
                evidence.append("clear_uncertainty_transition")
            if qualifying_run and not (
                numerical_clear or statistical_clear or uncertainty_clear
            ):
                evidence.append("transition_not_clear")

    return _BoundaryCandidate(
        spectrum_index=spectrum_index,
        group_identity=spectrum.group_label,
        side=side,
        point_count=point_count,
        run_length=len(indices),
        energy_bounds=energy_bounds,
        plateau_intensity=plateau_intensity,
        plateau_uncertainty=plateau_uncertainty,
        adjacent_interior_index=adjacent_index,
        intensity_transition_absolute=intensity_jump,
        intensity_transition_sigma=transition_sigma,
        uncertainty_transition_absolute=uncertainty_jump,
        numerical_intensity_transition=numerical_clear,
        statistical_intensity_transition=statistical_clear,
        uncertainty_transition=uncertainty_clear,
        evidence_codes=tuple(evidence),
        diagnostics=tuple(diagnostics),
    )


def _signature_matches(
    first: _BoundaryCandidate,
    second: _BoundaryCandidate,
    configuration: EdgePaddingConfig,
) -> bool:
    if (
        first.run_length < configuration.minimum_run_points
        or second.run_length < configuration.minimum_run_points
        or first.plateau_intensity is None
        or first.plateau_uncertainty is None
        or second.plateau_intensity is None
        or second.plateau_uncertainty is None
    ):
        return False
    return _pair_matches(
        first.plateau_intensity,
        first.plateau_uncertainty,
        second.plateau_intensity,
        second.plateau_uncertainty,
        relative_tolerance=configuration.signature_relative_tolerance,
        absolute_tolerance=configuration.signature_absolute_tolerance,
    )


def _classify_candidate(
    candidate: _BoundaryCandidate,
    candidates: tuple[_BoundaryCandidate, ...],
    configuration: EdgePaddingConfig,
) -> BoundaryPaddingDetection:
    matching = tuple(
        other
        for other in candidates
        if _signature_matches(candidate, other, configuration)
    )
    matching_spectra = {other.spectrum_index for other in matching}
    supporting_long_spectra = {
        other.spectrum_index
        for other in matching
        if other.spectrum_index != candidate.spectrum_index
        and other.run_length >= configuration.long_run_points
        and other.transition_clear
    }

    evidence = list(candidate.evidence_codes)
    if len(matching_spectra) >= 2:
        evidence.append("matching_boundary_signature_multiple_spectra")
    if supporting_long_spectra:
        evidence.append("cross_group_long_run_support")

    confidence = PaddingConfidence.NONE
    if candidate.run_length >= configuration.minimum_run_points:
        if candidate.adjacent_interior_index is None:
            confidence = PaddingConfidence.LOW
        elif candidate.transition_clear:
            is_long_absolute = (
                candidate.run_length >= configuration.long_run_points
            )
            is_long_relative = (
                candidate.run_length / candidate.point_count
                >= configuration.long_run_fraction
            )
            nonpositive_support = bool(
                candidate.plateau_intensity is not None
                and candidate.plateau_intensity <= 0.0
            )
            cross_group_support = len(matching_spectra) >= 2
            if is_long_absolute and (
                is_long_relative
                or nonpositive_support
                or cross_group_support
            ):
                confidence = PaddingConfidence.HIGH
            elif (
                candidate.run_length >= configuration.regular_run_points
                and cross_group_support
                and supporting_long_spectra
            ):
                confidence = PaddingConfidence.HIGH
            elif (
                candidate.run_length == configuration.minimum_run_points
                and len(supporting_long_spectra)
                >= configuration.short_run_supporting_spectra
            ):
                confidence = PaddingConfidence.HIGH
                evidence.append("short_run_promoted_by_cross_group_support")
            elif candidate.run_length >= configuration.regular_run_points:
                confidence = PaddingConfidence.MEDIUM
            else:
                confidence = PaddingConfidence.LOW
        elif candidate.run_length >= configuration.regular_run_points:
            confidence = PaddingConfidence.MEDIUM
        else:
            confidence = PaddingConfidence.LOW

    diagnostics = list(candidate.diagnostics)
    if confidence is PaddingConfidence.HIGH:
        diagnostics.append(
            PaddingDiagnostic(
                code="padding_high_confidence_default_on",
                severity=DiagnosticSeverity.INFO,
                message=(
                    "A high-confidence boundary plateau is recommended "
                    "default-on as a reversible padding mask"
                ),
                group_identity=candidate.group_identity,
                side=candidate.side,
            )
        )
    elif confidence is PaddingConfidence.MEDIUM:
        diagnostics.append(
            PaddingDiagnostic(
                code="padding_medium_confidence_confirmation_required",
                severity=DiagnosticSeverity.WARNING,
                message=(
                    "A boundary plateau is suggested but requires explicit "
                    "confirmation"
                ),
                group_identity=candidate.group_identity,
                side=candidate.side,
            )
        )
    elif confidence is PaddingConfidence.LOW:
        diagnostics.append(
            PaddingDiagnostic(
                code="padding_low_confidence_not_masked",
                severity=DiagnosticSeverity.INFO,
                message=(
                    "Boundary evidence was insufficient for an automatic "
                    "padding mask"
                ),
                group_identity=candidate.group_identity,
                side=candidate.side,
            )
        )

    return BoundaryPaddingDetection(
        side=candidate.side,
        run_length=candidate.run_length,
        energy_bounds=candidate.energy_bounds,
        plateau_intensity=candidate.plateau_intensity,
        plateau_uncertainty=candidate.plateau_uncertainty,
        adjacent_interior_index=candidate.adjacent_interior_index,
        intensity_transition_absolute=(
            candidate.intensity_transition_absolute
        ),
        intensity_transition_sigma=candidate.intensity_transition_sigma,
        uncertainty_transition_absolute=(
            candidate.uncertainty_transition_absolute
        ),
        run_fraction=candidate.run_length / candidate.point_count,
        matching_spectrum_count=len(matching_spectra),
        supporting_long_run_spectrum_count=len(supporting_long_spectra),
        confidence=confidence,
        evidence_codes=tuple(dict.fromkeys(evidence)),
        recommended_default_on=confidence is PaddingConfidence.HIGH,
        diagnostics=tuple(diagnostics),
    )


_CONFIDENCE_RANK = {
    PaddingConfidence.NONE: 0,
    PaddingConfidence.LOW: 1,
    PaddingConfidence.MEDIUM: 2,
    PaddingConfidence.HIGH: 3,
}


def _strongest_confidence(
    left: PaddingConfidence,
    right: PaddingConfidence,
) -> PaddingConfidence:
    return max((left, right), key=_CONFIDENCE_RANK.__getitem__)


def _apply_boundary(
    mask: BoolArray,
    boundary: BoundaryPaddingDetection,
) -> None:
    if boundary.run_length == 0:
        return
    if boundary.side is BoundarySide.LEFT:
        mask[: boundary.run_length] = True
    else:
        mask[-boundary.run_length :] = True


def detect_edge_padding(
    dataset: ImportedDataset,
    *,
    configuration: EdgePaddingConfig | None = None,
) -> EdgePaddingDetectionResult:
    """Detect reversible boundary-padding masks for every spectrum.

    Imported numerical arrays and invalid-value masks are read only. The
    returned ``padding_mask`` contains only high-confidence default-on points;
    ``suggested_padding_mask`` additionally contains medium-confidence points.
    """

    selected_configuration = configuration or EdgePaddingConfig()
    candidates = tuple(
        _inspect_boundary(
            spectrum,
            spectrum_index=spectrum_index,
            side=side,
            configuration=selected_configuration,
        )
        for spectrum_index, spectrum in enumerate(dataset.spectra)
        for side in BoundarySide
    )
    classified = {
        (candidate.spectrum_index, candidate.side): _classify_candidate(
            candidate,
            candidates,
            selected_configuration,
        )
        for candidate in candidates
    }

    results: list[SpectrumPaddingMask] = []
    diagnostics: list[PaddingDiagnostic] = []
    for spectrum_index, spectrum in enumerate(dataset.spectra):
        left = classified[(spectrum_index, BoundarySide.LEFT)]
        right = classified[(spectrum_index, BoundarySide.RIGHT)]
        padding_mask = np.zeros(spectrum.energy.size, dtype=np.bool_)
        suggestion_mask = np.zeros(spectrum.energy.size, dtype=np.bool_)
        for boundary in (left, right):
            if boundary.confidence in {
                PaddingConfidence.HIGH,
                PaddingConfidence.MEDIUM,
            }:
                _apply_boundary(suggestion_mask, boundary)
            if boundary.recommended_default_on:
                _apply_boundary(padding_mask, boundary)

        boundary_diagnostics = left.diagnostics + right.diagnostics
        diagnostics.extend(boundary_diagnostics)
        evidence_codes = tuple(
            dict.fromkeys(
                f"{boundary.side.value}:{code}"
                for boundary in (left, right)
                for code in boundary.evidence_codes
            )
        )
        results.append(
            SpectrumPaddingMask(
                group_index=spectrum.group_index,
                group_identity=spectrum.group_label,
                padding_mask=padding_mask,
                suggested_padding_mask=suggestion_mask,
                left=left,
                right=right,
                confidence=_strongest_confidence(
                    left.confidence,
                    right.confidence,
                ),
                evidence_codes=evidence_codes,
                algorithm_version=ALGORITHM_VERSION,
                recommended_default_on=bool(np.any(padding_mask)),
                diagnostics=boundary_diagnostics,
            )
        )

    return EdgePaddingDetectionResult(
        spectra=tuple(results),
        algorithm_version=ALGORITHM_VERSION,
        configuration=selected_configuration,
        diagnostics=tuple(diagnostics),
    )
