"""Small typed scientific models for reduced QENS spectra."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from enum import StrEnum
from numbers import Real

import numpy as np
import numpy.typing as npt

from ezqens.domain.diagnostics import DiagnosticSeverity, ImportDiagnostic

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


class SpectrumRole(StrEnum):
    """Scientific role of a reduced spectrum."""

    SAMPLE = "sample"
    RESOLUTION = "resolution"


class ReducedDataFormat(StrEnum):
    """Reduced text layouts recognized by the current importer."""

    DAVE_GROUP_BLOCKS = "dave_group_blocks"
    MANTID_XYE_BLOCKS = "mantid_xye_blocks"
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


class FractionalCoverageOrigin(StrEnum):
    """Origin of fractional Q-E coverage aligned with reduced spectra."""

    EXPLICIT_SOURCE = "explicit_source"
    CONFIRMED_UNREBINNED_SOURCE = "confirmed_unrebinned_source"
    PROPAGATED_Q_REBIN = "propagated_q_rebin"


class FractionalCoverageAvailability(StrEnum):
    """Whether fractional Q-E coverage is available for later operations."""

    AVAILABLE = "available"
    MISSING = "missing"


def _readonly_float_array(value: npt.ArrayLike, *, name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class QExclusionInterval:
    """One absolute-Q interval removed from Q-bin overlap geometry."""

    lower_q: float
    upper_q: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.lower_q, bool)
            or isinstance(self.upper_q, bool)
            or not isinstance(self.lower_q, Real)
            or not isinstance(self.upper_q, Real)
        ):
            raise ValueError("Q-exclusion bounds must be finite numbers")
        lower = float(self.lower_q)
        upper = float(self.upper_q)
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("Q-exclusion bounds must be finite")
        if lower >= upper:
            raise ValueError("Q-exclusion lower bound must be less than upper bound")
        object.__setattr__(self, "lower_q", lower)
        object.__setattr__(self, "upper_q", upper)


@dataclass(frozen=True, slots=True)
class QRebinSpecification:
    """Canonical target Q edges and absolute-Q exclusions for one rebin step."""

    target_edges: tuple[float, ...]
    exclusions: tuple[QExclusionInterval, ...] = ()

    def __post_init__(self) -> None:
        try:
            edges = tuple(float(value) for value in self.target_edges)
        except (TypeError, ValueError) as error:
            raise ValueError("target Q-bin edges must be finite numbers") from error
        if len(edges) < 2:
            raise ValueError("target Q-bin edges must contain at least two values")
        edge_array = np.asarray(edges, dtype=np.float64)
        if not np.all(np.isfinite(edge_array)):
            raise ValueError("target Q-bin edges must be finite")
        if not np.all(np.diff(edge_array) > 0.0):
            raise ValueError("target Q-bin edges must be strictly increasing")
        exclusions = tuple(self.exclusions)
        if any(not isinstance(item, QExclusionInterval) for item in exclusions):
            raise ValueError("Q exclusions must be QExclusionInterval values")
        object.__setattr__(self, "target_edges", edges)
        object.__setattr__(self, "exclusions", exclusions)


def _point_correspondence_digest(spectrum: Spectrum) -> str:
    """Return a compact exact-value binding for one ordered X/Y/E spectrum."""

    digest = hashlib.sha256()
    digest.update(str(spectrum.group_index).encode())
    digest.update(b"\0")
    digest.update(spectrum.group_label.encode())
    for values in (spectrum.energy, spectrum.intensity, spectrum.uncertainty):
        normalized = np.array(values, dtype=np.float64, copy=True)
        normalized[normalized == 0.0] = 0.0
        normalized[np.isnan(normalized)] = np.nan
        digest.update(normalized.size.to_bytes(8, byteorder="little", signed=False))
        digest.update(normalized.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FractionalCoverage:
    """Read-only fractional coverage arrays aligned with ordered spectra."""

    values: tuple[FloatArray, ...] = field(repr=False)
    origin: FractionalCoverageOrigin
    point_correspondence: tuple[str, ...] = field(repr=False)
    q_rebin_history: tuple[QRebinSpecification, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.origin, FractionalCoverageOrigin):
            raise ValueError("fractional-coverage origin must be typed")
        arrays = tuple(
            _readonly_float_array(value, name="fractional coverage")
            for value in self.values
        )
        if not arrays:
            raise ValueError("fractional coverage requires at least one spectrum")
        correspondence = tuple(self.point_correspondence)
        if len(correspondence) != len(arrays) or any(
            not item for item in correspondence
        ):
            raise ValueError(
                "fractional coverage requires one point-correspondence binding "
                "per spectrum"
            )
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("fractional coverage values must be finite")
        if any(np.any(array < 0.0) for array in arrays):
            raise ValueError("fractional coverage values must be nonnegative")
        if self.origin is FractionalCoverageOrigin.CONFIRMED_UNREBINNED_SOURCE and any(
            not np.all(array == 1.0) for array in arrays
        ):
            raise ValueError(
                "confirmed-unrebinned fractional coverage values must all equal one"
            )
        object.__setattr__(self, "values", arrays)
        object.__setattr__(self, "point_correspondence", correspondence)
        history = tuple(self.q_rebin_history)
        if any(not isinstance(item, QRebinSpecification) for item in history):
            raise ValueError("Q-rebin history must contain QRebinSpecification values")
        object.__setattr__(self, "q_rebin_history", history)

    @classmethod
    def aligned_with(
        cls,
        spectra: tuple[Spectrum, ...],
        *,
        values: tuple[npt.ArrayLike, ...],
        origin: FractionalCoverageOrigin,
        q_rebin_history: tuple[QRebinSpecification, ...] = (),
    ) -> FractionalCoverage:
        """Create coverage explicitly bound to ordered scientific point identity."""

        return cls(
            values=tuple(
                _readonly_float_array(value, name="fractional coverage")
                for value in values
            ),
            origin=origin,
            point_correspondence=tuple(
                _point_correspondence_digest(spectrum) for spectrum in spectra
            ),
            q_rebin_history=q_rebin_history,
        )


@dataclass(frozen=True, slots=True)
class QBins:
    """Ordered representative Q values and optional bin edges in inverse angstrom."""

    q_values: FloatArray = field(repr=False)
    edges: FloatArray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        q_values = _readonly_float_array(self.q_values, name="q_values")
        if q_values.size == 0:
            raise ValueError("q_values must contain at least one value")
        if not np.all(np.isfinite(q_values)):
            raise ValueError("q_values must be finite")

        edges: FloatArray | None = None
        if self.edges is not None:
            edges = _readonly_float_array(self.edges, name="edges")
            if edges.size < 2:
                raise ValueError("edges must contain at least two values")
            if not np.all(np.isfinite(edges)):
                raise ValueError("edges must be finite")
            if not np.all(np.diff(edges) > 0.0):
                raise ValueError("edges must be strictly increasing")
            if edges.size != q_values.size + 1:
                raise ValueError(
                    "edges must contain exactly one more value than q_values"
                )

        object.__setattr__(self, "q_values", q_values)
        object.__setattr__(self, "edges", edges)

    @classmethod
    def from_edges(cls, q_bin_edges: npt.ArrayLike) -> QBins:
        """Create bins whose Milestone-2 representatives are edge midpoints."""

        edges = _readonly_float_array(q_bin_edges, name="q_bin_edges")
        if edges.size < 2:
            raise ValueError("q_bin_edges must contain at least two values")
        if not np.all(np.isfinite(edges)):
            raise ValueError("q_bin_edges must be finite")
        if not np.all(np.diff(edges) > 0.0):
            raise ValueError("q_bin_edges must be strictly increasing")
        midpoint_values = edges[:-1] / 2.0 + edges[1:] / 2.0
        return cls(q_values=midpoint_values, edges=edges)

    @classmethod
    def from_q_values(cls, representative_q_values: npt.ArrayLike) -> QBins:
        """Create ordered representative Q values without inferring bin edges."""

        q_values = _readonly_float_array(
            representative_q_values, name="representative_q_values"
        )
        return cls(q_values=q_values)

    @classmethod
    def from_q_values_and_uniform_step(
        cls,
        representative_q_values: npt.ArrayLike,
        step: float,
    ) -> QBins:
        """Preserve representatives and propose centered edges for explicit step."""

        if isinstance(step, bool) or not isinstance(step, Real):
            raise ValueError("step must be a finite positive number")
        numeric_step = float(step)
        if not np.isfinite(numeric_step) or numeric_step <= 0.0:
            raise ValueError("step must be finite and strictly positive")
        half_step = numeric_step / 2.0
        if not np.isfinite(half_step) or half_step <= 0.0:
            raise ValueError("step half-width is not representable as a positive float")
        explicit = cls.from_q_values(representative_q_values)
        epsilon = np.finfo(np.float64).eps
        coordinate_ulp = float(np.max(np.spacing(np.abs(explicit.q_values))))
        if coordinate_ulp / numeric_step > np.sqrt(epsilon):
            raise ValueError(
                "representative Q coordinate precision is too coarse to verify "
                "the supplied uniform step"
            )
        step_roundoff = 8.0 * max(
            epsilon * numeric_step,
            float(np.spacing(numeric_step)),
        )
        q_spacing_tolerance = step_roundoff + 4.0 * coordinate_ulp

        def matches_step(
            values: FloatArray,
            expected: float,
            absolute_tolerance: float,
        ) -> bool:
            return bool(
                np.all(
                    np.isclose(
                        values,
                        expected,
                        rtol=8.0 * epsilon,
                        atol=absolute_tolerance,
                    )
                )
            )

        if explicit.group_count > 1 and not matches_step(
            np.diff(explicit.q_values),
            numeric_step,
            q_spacing_tolerance,
        ):
            raise ValueError(
                "adjacent representative Q differences must match the "
                "supplied uniform step"
            )
        first_edge = float(explicit.q_values[0]) - half_step
        with np.errstate(over="ignore", invalid="ignore"):
            edges = first_edge + numeric_step * np.arange(
                explicit.group_count + 1,
                dtype=np.float64,
            )
        if not np.all(np.isfinite(edges)):
            raise ValueError("centered Q-bin edges must be finite")
        if not np.all(np.diff(edges) > 0.0):
            raise ValueError(
                "uniform step is not representable as strictly increasing Q-bin edges"
            )
        edge_ulp = float(np.max(np.spacing(np.abs(edges))))
        geometry_tolerance = step_roundoff + 4.0 * max(coordinate_ulp, edge_ulp)
        left_distance = explicit.q_values - edges[:-1]
        right_distance = edges[1:] - explicit.q_values
        if (
            not np.all(left_distance > 0.0)
            or not np.all(right_distance > 0.0)
            or not matches_step(left_distance, half_step, geometry_tolerance)
            or not matches_step(right_distance, half_step, geometry_tolerance)
            or not matches_step(
                np.diff(edges),
                numeric_step,
                geometry_tolerance,
            )
        ):
            raise ValueError(
                "representative Q values and step do not define representable "
                "centered uniform edges"
            )
        return cls(q_values=explicit.q_values, edges=edges)

    @property
    def group_count(self) -> int:
        """Return the number of represented spectrum groups."""

        return int(self.q_values.size)

    @property
    def unit(self) -> str:
        """Return the fixed momentum-transfer unit."""

        return "Å^-1"


def uniform_q_bins(
    *, lower_q_edge: float, upper_q_edge: float, group_count: int
) -> QBins:
    """Generate count-driven uniform bins that exactly cover both outer edges."""

    if isinstance(group_count, bool) or not isinstance(group_count, int):
        raise ValueError("group_count must be an integer")
    if group_count < 1:
        raise ValueError("group_count must be at least one")
    if not np.isfinite(lower_q_edge) or not np.isfinite(upper_q_edge):
        raise ValueError("Q-bin edges must be finite")
    if lower_q_edge >= upper_q_edge:
        raise ValueError("lower_q_edge must be less than upper_q_edge")
    edges = np.linspace(lower_q_edge, upper_q_edge, group_count + 1)
    return QBins.from_edges(edges)


def _fixed_width_edge_tolerance(
    lower_q_edge: float,
    upper_q_limit: float,
    step: float,
    candidate_edge: float,
) -> float:
    """Return the locally capped tolerance for fixed-width edge arithmetic."""

    roundoff = (
        np.finfo(np.float64).eps
        * max(
            1.0,
            abs(lower_q_edge),
            abs(upper_q_limit),
            abs(candidate_edge),
        )
        * 16.0
    )
    # Never let coordinate-scale roundoff become a material fraction of a
    # Q bin. Normal binary noise is far below this local one-step cap.
    one_step_cap = float(np.sqrt(np.finfo(np.float64).eps) * abs(step))
    return min(roundoff, one_step_cap)


def _complete_fixed_width_bin_count(
    lower_q_edge: float,
    upper_q_limit: float,
    step: float,
) -> int:
    """Return the complete fixed-width bins contained by the requested limits."""

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        span = float(np.subtract(np.float64(upper_q_limit), lower_q_edge))
    if not np.isfinite(span):
        raise ValueError("fixed-width Q span must be finite")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        quotient = float(np.divide(np.float64(span), step))
    if not np.isfinite(quotient):
        raise ValueError("fixed-width Q-bin count must be finite")
    maximum_exact_float64_integer = 1 << (np.finfo(np.float64).nmant + 1)
    maximum_group_count = min(
        int(np.iinfo(np.intp).max),
        maximum_exact_float64_integer,
    )
    if quotient >= float(maximum_group_count):
        raise ValueError("fixed-width Q-bin count is not representable")
    group_count = int(np.floor(quotient))
    # Confirm the complete-edge condition directly. At most the bin adjacent
    # to floor(span / step) can be affected by ordinary floating-point noise.
    with np.errstate(over="ignore", invalid="ignore"):
        candidate_edge = float(np.add(lower_q_edge, np.multiply(group_count, step)))
    if group_count > 0 and candidate_edge > upper_q_limit + _fixed_width_edge_tolerance(
        lower_q_edge,
        upper_q_limit,
        step,
        candidate_edge,
    ):
        group_count -= 1
    next_count = group_count + 1
    with np.errstate(over="ignore", invalid="ignore"):
        next_edge = float(np.add(lower_q_edge, np.multiply(next_count, step)))
    if next_edge <= upper_q_limit + _fixed_width_edge_tolerance(
        lower_q_edge,
        upper_q_limit,
        step,
        next_edge,
    ):
        group_count += 1
    if group_count >= maximum_group_count:
        raise ValueError("fixed-width Q-bin count is not representable")
    return group_count


def fixed_width_q_bins(
    *, lower_q_edge: float, upper_q_limit: float, step: float
) -> QBins:
    """Generate every complete fixed-width bin within requested Q limits."""

    inputs = {
        "lower_q_edge": lower_q_edge,
        "upper_q_limit": upper_q_limit,
        "step": step,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, Real)
        for value in inputs.values()
    ):
        raise ValueError("fixed-width Q-bin inputs must be finite numbers")
    lower = float(lower_q_edge)
    upper = float(upper_q_limit)
    numeric_step = float(step)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("fixed-width Q limits must be finite")
    if lower >= upper:
        raise ValueError("lower_q_edge must be less than upper_q_limit")
    if not np.isfinite(numeric_step) or numeric_step <= 0.0:
        raise ValueError("step must be finite and strictly positive")

    group_count = _complete_fixed_width_bin_count(lower, upper, numeric_step)
    if group_count < 1:
        raise ValueError("requested Q range does not contain one complete bin")
    try:
        indices = np.arange(group_count + 1, dtype=np.float64)
    except (MemoryError, OverflowError, ValueError) as error:
        raise ValueError(
            "fixed-width Q-bin edge array cannot be represented"
        ) from error
    with np.errstate(over="ignore", invalid="ignore"):
        edges = lower + numeric_step * indices
    if not np.all(np.isfinite(edges)):
        raise ValueError("fixed-width Q-bin edges must be finite")
    represented_widths = np.diff(edges)
    largest_edge = max((float(edges[0]), float(edges[-1])), key=abs)
    width_tolerance = _fixed_width_edge_tolerance(
        lower,
        upper,
        numeric_step,
        largest_edge,
    )
    if not np.all(
        np.isclose(
            represented_widths,
            numeric_step,
            rtol=0.0,
            atol=width_tolerance,
        )
    ):
        raise ValueError(
            "step cannot be represented as a uniform float64 Q-bin grid at "
            "the supplied coordinate scale"
        )
    return QBins.from_edges(edges)


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
class SourceMetadata:
    """Small typed source provenance with lossless ordered header lines."""

    source_filename: str
    raw_header_lines: tuple[str, ...] = field(default=(), repr=False)
    instrument: str | None = None
    sample: str | None = None
    title: str | None = None
    temperature_kelvin: float | None = None
    wavelength_angstrom: float | None = None

    def __post_init__(self) -> None:
        if not self.source_filename:
            raise ValueError("source_filename must not be empty")
        if any(not isinstance(line, str) for line in self.raw_header_lines):
            raise ValueError("raw_header_lines must contain only strings")
        for name in ("instrument", "sample", "title"):
            value = getattr(self, name)
            if value is not None and not value:
                raise ValueError(f"{name} must be nonempty when supplied")
        if self.temperature_kelvin is not None and (
            not np.isfinite(self.temperature_kelvin) or self.temperature_kelvin < 0.0
        ):
            raise ValueError(
                "temperature_kelvin must be finite and nonnegative when supplied"
            )
        if self.wavelength_angstrom is not None and (
            not np.isfinite(self.wavelength_angstrom) or self.wavelength_angstrom <= 0.0
        ):
            raise ValueError(
                "wavelength_angstrom must be finite and positive when supplied"
            )


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
    source_metadata: SourceMetadata | None = None
    q_bins: QBins | None = None
    fractional_coverage: FractionalCoverage | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.role, SpectrumRole):
            raise ValueError("role must be a SpectrumRole")
        if self.source_layout is not None and not isinstance(
            self.source_layout, ReducedDataFormat
        ):
            raise ValueError("source_layout must be a ReducedDataFormat or None")
        if self.source_metadata is not None and not isinstance(
            self.source_metadata, SourceMetadata
        ):
            raise ValueError("source_metadata must be SourceMetadata or None")
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
        if self.q_bins is not None and self.q_bins.group_count != len(self.spectra):
            raise ValueError("Q-bin count must match spectrum count")
        if self.fractional_coverage is not None:
            if not isinstance(self.fractional_coverage, FractionalCoverage):
                raise ValueError(
                    "fractional_coverage must be FractionalCoverage or None"
                )
            if len(self.fractional_coverage.values) != len(self.spectra):
                raise ValueError(
                    "fractional-coverage spectrum count must match spectrum count"
                )
            if any(
                coverage.size != spectrum.energy.size
                for coverage, spectrum in zip(
                    self.fractional_coverage.values,
                    self.spectra,
                    strict=True,
                )
            ):
                raise ValueError(
                    "fractional coverage must align with every spectrum point"
                )
            expected_correspondence = tuple(
                _point_correspondence_digest(spectrum) for spectrum in self.spectra
            )
            if self.fractional_coverage.point_correspondence != expected_correspondence:
                raise ValueError(
                    "fractional coverage is bound to different ordered X/Y/E points"
                )

    def assign_q_bins(self, q_bins: QBins) -> ReducedDataset:
        """Return this dataset with validated dataset-level Q-bin identity."""

        return replace(self, q_bins=q_bins)

    def confirm_unrebinned_source(self) -> ReducedDataset:
        """Return a dataset with all-one coverage after explicit user assertion."""

        if self.fractional_coverage is not None:
            raise ValueError(
                "fractional coverage already exists and must not be reset to one"
            )
        coverage = FractionalCoverage.aligned_with(
            self.spectra,
            values=tuple(
                np.ones(spectrum.energy.size, dtype=np.float64)
                for spectrum in self.spectra
            ),
            origin=FractionalCoverageOrigin.CONFIRMED_UNREBINNED_SOURCE,
        )
        return replace(self, fractional_coverage=coverage)

    @property
    def fractional_coverage_availability(self) -> FractionalCoverageAvailability:
        """Report whether per-point fractional coverage is available."""

        if self.fractional_coverage is None:
            return FractionalCoverageAvailability.MISSING
        return FractionalCoverageAvailability.AVAILABLE

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
