"""Detect reversible boundary-connected padding without changing data."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from ezqens.domain import ReducedDataset, Spectrum

BoolArray = npt.NDArray[np.bool_]

ALGORITHM_VERSION = "edge-padding-v2.0.0"

# The v2 rule intentionally has a small, fixed numerical surface. A new
# scientific validation is required before changing these values.
_PAIR_RELATIVE_TOLERANCE = 1.0e-7
_PAIR_ABSOLUTE_TOLERANCE = 1.0e-12
_TRANSITION_RELATIVE_TOLERANCE = 5.0e-2
_TRANSITION_ABSOLUTE_TOLERANCE = 1.0e-12
_LONG_RUN_POINTS = 5
_SHORT_RUN_SUPPORTING_SPECTRA = 2


class PaddingStatus(StrEnum):
    """Behavior assigned to a boundary-padding candidate."""

    AUTO = "auto"
    REVIEW = "review"
    NONE = "none"


class BoundarySide(StrEnum):
    """Array boundary inspected for a connected repeated pair."""

    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True, slots=True)
class BoundaryPaddingResult:
    """Inspection result for one spectrum boundary."""

    side: BoundarySide
    run_length: int
    energy_bounds: tuple[float, float] | None
    status: PaddingStatus
    reason: str


def _readonly_mask(value: npt.ArrayLike, *, name: str) -> BoolArray:
    mask = np.array(value, dtype=np.bool_, copy=True)
    if mask.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    mask.setflags(write=False)
    return mask


@dataclass(frozen=True, slots=True)
class SpectrumPaddingResult:
    """Immutable automatic and review masks for one spectrum."""

    group_index: int
    group_identity: str
    auto_mask: BoolArray = field(repr=False)
    review_mask: BoolArray = field(repr=False)
    left: BoundaryPaddingResult
    right: BoundaryPaddingResult

    def __post_init__(self) -> None:
        auto_mask = _readonly_mask(self.auto_mask, name="auto_mask")
        review_mask = _readonly_mask(self.review_mask, name="review_mask")
        if auto_mask.size != review_mask.size:
            raise ValueError("automatic and review masks must have equal length")
        if np.any(auto_mask & review_mask):
            raise ValueError("automatic and review masks must be mutually exclusive")
        object.__setattr__(self, "auto_mask", auto_mask)
        object.__setattr__(self, "review_mask", review_mask)

    @property
    def status(self) -> PaddingStatus:
        """Return the strongest behavior while preserving ``needs_review``."""

        if np.any(self.auto_mask):
            return PaddingStatus.AUTO
        if np.any(self.review_mask):
            return PaddingStatus.REVIEW
        return PaddingStatus.NONE

    @property
    def needs_review(self) -> bool:
        """Return whether either boundary has a review-only candidate."""

        return bool(np.any(self.review_mask))

    @property
    def left_auto_mask_count(self) -> int:
        return self.left.run_length if self.left.status is PaddingStatus.AUTO else 0

    @property
    def right_auto_mask_count(self) -> int:
        return self.right.run_length if self.right.status is PaddingStatus.AUTO else 0

    @property
    def total_auto_mask_count(self) -> int:
        return int(np.count_nonzero(self.auto_mask))

    @property
    def total_review_mask_count(self) -> int:
        return int(np.count_nonzero(self.review_mask))


@dataclass(frozen=True, slots=True)
class SpectrumPaddingSummary:
    """Privacy-safe structural padding summary for one spectrum."""

    group_index: int
    group_identity: str
    left_auto_mask_count: int
    right_auto_mask_count: int
    total_auto_mask_count: int
    total_review_mask_count: int
    retained_energy_bounds: tuple[float | None, float | None]
    status: PaddingStatus
    needs_review: bool


@dataclass(frozen=True, slots=True)
class EdgePaddingSummary:
    """Privacy-safe structural padding summary for a dataset."""

    algorithm_version: str
    spectra: tuple[SpectrumPaddingSummary, ...]
    total_auto_mask_count: int
    total_review_mask_count: int


@dataclass(frozen=True, slots=True)
class EdgePaddingDetectionResult:
    """Ordered edge-padding results for a reduced dataset."""

    spectra: tuple[SpectrumPaddingResult, ...]
    algorithm_version: str = ALGORITHM_VERSION

    def structural_summary(self, dataset: ReducedDataset) -> EdgePaddingSummary:
        """Return mask counts and retained bounds without array values."""

        if len(dataset.spectra) != len(self.spectra):
            raise ValueError("padding result does not match dataset spectrum count")
        summaries: list[SpectrumPaddingSummary] = []
        for spectrum, result in zip(dataset.spectra, self.spectra, strict=True):
            if (
                spectrum.group_index != result.group_index
                or spectrum.group_label != result.group_identity
            ):
                raise ValueError("padding result does not match dataset ordering")
            retained = spectrum.energy[np.isfinite(spectrum.energy) & ~result.auto_mask]
            retained_bounds: tuple[float | None, float | None]
            if retained.size:
                retained_bounds = (float(np.min(retained)), float(np.max(retained)))
            else:
                retained_bounds = (None, None)
            summaries.append(
                SpectrumPaddingSummary(
                    group_index=result.group_index,
                    group_identity=result.group_identity,
                    left_auto_mask_count=result.left_auto_mask_count,
                    right_auto_mask_count=result.right_auto_mask_count,
                    total_auto_mask_count=result.total_auto_mask_count,
                    total_review_mask_count=result.total_review_mask_count,
                    retained_energy_bounds=retained_bounds,
                    status=result.status,
                    needs_review=result.needs_review,
                )
            )
        return EdgePaddingSummary(
            algorithm_version=self.algorithm_version,
            spectra=tuple(summaries),
            total_auto_mask_count=sum(item.total_auto_mask_count for item in summaries),
            total_review_mask_count=sum(
                item.total_review_mask_count for item in summaries
            ),
        )


@dataclass(frozen=True, slots=True)
class _BoundaryCandidate:
    spectrum_index: int
    side: BoundarySide
    run_length: int
    energy_bounds: tuple[float, float] | None
    signature: tuple[float, float] | None
    has_valid_adjacent_interior: bool
    transition_is_clear: bool
    unavailable_reason: str | None = None


def _pair_matches(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return bool(
        np.isclose(
            first[0],
            second[0],
            rtol=_PAIR_RELATIVE_TOLERANCE,
            atol=_PAIR_ABSOLUTE_TOLERANCE,
        )
        and np.isclose(
            first[1],
            second[1],
            rtol=_PAIR_RELATIVE_TOLERANCE,
            atol=_PAIR_ABSOLUTE_TOLERANCE,
        )
    )


def _value_transition_is_clear(plateau: float, interior: float) -> bool:
    scale = max(abs(plateau), abs(interior))
    return abs(interior - plateau) > (
        _TRANSITION_ABSOLUTE_TOLERANCE + _TRANSITION_RELATIVE_TOLERANCE * scale
    )


def _inspect_boundary(
    spectrum: Spectrum,
    *,
    spectrum_index: int,
    side: BoundarySide,
) -> _BoundaryCandidate:
    point_count = spectrum.energy.size
    edge_index = 0 if side is BoundarySide.LEFT else point_count - 1
    edge = (
        float(spectrum.intensity[edge_index]),
        float(spectrum.uncertainty[edge_index]),
    )
    if (
        not np.isfinite(spectrum.energy[edge_index])
        or not np.isfinite(edge[0])
        or not np.isfinite(edge[1])
        or edge[1] <= 0.0
    ):
        return _BoundaryCandidate(
            spectrum_index=spectrum_index,
            side=side,
            run_length=0,
            energy_bounds=None,
            signature=None,
            has_valid_adjacent_interior=False,
            transition_is_clear=False,
            unavailable_reason="boundary_values_invalid",
        )

    step = 1 if side is BoundarySide.LEFT else -1
    indices = [edge_index]
    current = edge_index + step
    while 0 <= current < point_count:
        pair = (
            float(spectrum.intensity[current]),
            float(spectrum.uncertainty[current]),
        )
        if (
            not np.isfinite(spectrum.energy[current])
            or not np.isfinite(pair[0])
            or not np.isfinite(pair[1])
            or pair[1] <= 0.0
            or not _pair_matches(pair, edge)
        ):
            break
        indices.append(current)
        current += step

    if len(indices) < 2:
        return _BoundaryCandidate(
            spectrum_index=spectrum_index,
            side=side,
            run_length=len(indices),
            energy_bounds=None,
            signature=None,
            has_valid_adjacent_interior=False,
            transition_is_clear=False,
            unavailable_reason="no_repeated_boundary_pair",
        )

    ordered_indices = sorted(indices)
    boundary_energy = spectrum.energy[ordered_indices]
    energy_bounds = (
        float(np.min(boundary_energy)),
        float(np.max(boundary_energy)),
    )
    if not 0 <= current < point_count:
        return _BoundaryCandidate(
            spectrum_index=spectrum_index,
            side=side,
            run_length=len(indices),
            energy_bounds=energy_bounds,
            signature=edge,
            has_valid_adjacent_interior=False,
            transition_is_clear=False,
            unavailable_reason="adjacent_interior_missing",
        )

    adjacent = (
        float(spectrum.intensity[current]),
        float(spectrum.uncertainty[current]),
    )
    adjacent_is_valid = bool(
        np.isfinite(spectrum.energy[current])
        and np.isfinite(adjacent[0])
        and np.isfinite(adjacent[1])
        and adjacent[1] > 0.0
    )
    return _BoundaryCandidate(
        spectrum_index=spectrum_index,
        side=side,
        run_length=len(indices),
        energy_bounds=energy_bounds,
        signature=edge,
        has_valid_adjacent_interior=adjacent_is_valid,
        transition_is_clear=bool(
            adjacent_is_valid
            and (
                _value_transition_is_clear(edge[0], adjacent[0])
                or _value_transition_is_clear(edge[1], adjacent[1])
            )
        ),
        unavailable_reason=None if adjacent_is_valid else "adjacent_interior_invalid",
    )


def _long_support_count(
    candidate: _BoundaryCandidate,
    candidates: tuple[_BoundaryCandidate, ...],
) -> int:
    if candidate.signature is None:
        return 0
    return len(
        {
            other.spectrum_index
            for other in candidates
            if other.spectrum_index != candidate.spectrum_index
            and other.signature is not None
            and other.run_length >= _LONG_RUN_POINTS
            and other.has_valid_adjacent_interior
            and other.transition_is_clear
            and _pair_matches(candidate.signature, other.signature)
        }
    )


def _classify(
    candidate: _BoundaryCandidate,
    candidates: tuple[_BoundaryCandidate, ...],
) -> BoundaryPaddingResult:
    if candidate.signature is None:
        return BoundaryPaddingResult(
            side=candidate.side,
            run_length=candidate.run_length,
            energy_bounds=None,
            status=PaddingStatus.NONE,
            reason=candidate.unavailable_reason or "no_padding_candidate",
        )
    if not candidate.has_valid_adjacent_interior:
        return BoundaryPaddingResult(
            side=candidate.side,
            run_length=candidate.run_length,
            energy_bounds=candidate.energy_bounds,
            status=PaddingStatus.REVIEW,
            reason=candidate.unavailable_reason or "adjacent_interior_unavailable",
        )
    if not candidate.transition_is_clear:
        return BoundaryPaddingResult(
            side=candidate.side,
            run_length=candidate.run_length,
            energy_bounds=candidate.energy_bounds,
            status=PaddingStatus.REVIEW,
            reason="transition_not_clear",
        )
    if candidate.run_length >= _LONG_RUN_POINTS:
        return BoundaryPaddingResult(
            side=candidate.side,
            run_length=candidate.run_length,
            energy_bounds=candidate.energy_bounds,
            status=PaddingStatus.AUTO,
            reason="long_run_with_clear_transition",
        )
    if _long_support_count(candidate, candidates) >= _SHORT_RUN_SUPPORTING_SPECTRA:
        return BoundaryPaddingResult(
            side=candidate.side,
            run_length=candidate.run_length,
            energy_bounds=candidate.energy_bounds,
            status=PaddingStatus.AUTO,
            reason="short_run_with_cross_spectrum_support",
        )
    return BoundaryPaddingResult(
        side=candidate.side,
        run_length=candidate.run_length,
        energy_bounds=candidate.energy_bounds,
        status=PaddingStatus.REVIEW,
        reason="short_run_requires_review",
    )


def _apply_boundary(mask: BoolArray, boundary: BoundaryPaddingResult) -> None:
    if boundary.run_length == 0:
        return
    if boundary.side is BoundarySide.LEFT:
        mask[: boundary.run_length] = True
    else:
        mask[-boundary.run_length :] = True


def detect_edge_padding(dataset: ReducedDataset) -> EdgePaddingDetectionResult:
    """Detect point-exact ``AUTO`` and ``REVIEW`` boundary-padding masks."""

    candidates = tuple(
        _inspect_boundary(
            spectrum,
            spectrum_index=spectrum_index,
            side=side,
        )
        for spectrum_index, spectrum in enumerate(dataset.spectra)
        for side in BoundarySide
    )
    classified = {
        (candidate.spectrum_index, candidate.side): _classify(candidate, candidates)
        for candidate in candidates
    }

    results: list[SpectrumPaddingResult] = []
    for spectrum_index, spectrum in enumerate(dataset.spectra):
        left = classified[(spectrum_index, BoundarySide.LEFT)]
        right = classified[(spectrum_index, BoundarySide.RIGHT)]
        auto_mask = np.zeros(spectrum.energy.size, dtype=np.bool_)
        review_mask = np.zeros(spectrum.energy.size, dtype=np.bool_)
        for boundary in (left, right):
            if boundary.status is PaddingStatus.AUTO:
                _apply_boundary(auto_mask, boundary)
            elif boundary.status is PaddingStatus.REVIEW:
                _apply_boundary(review_mask, boundary)
        results.append(
            SpectrumPaddingResult(
                group_index=spectrum.group_index,
                group_identity=spectrum.group_label,
                auto_mask=auto_mask,
                review_mask=review_mask,
                left=left,
                right=right,
            )
        )
    return EdgePaddingDetectionResult(spectra=tuple(results))
