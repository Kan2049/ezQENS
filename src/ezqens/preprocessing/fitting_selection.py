"""Per-group fitting ranges and derived point selection."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from ezqens.domain import ReducedDataset, Spectrum
from ezqens.preprocessing.edge_padding import EdgePaddingDetectionResult

BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class FittingRange:
    """Inclusive energy interval selected for one spectrum."""

    lower_energy: float
    upper_energy: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.lower_energy) or not np.isfinite(self.upper_energy):
            raise ValueError("fitting-range bounds must be finite")
        if self.lower_energy > self.upper_energy:
            raise ValueError("lower_energy must not exceed upper_energy")


def _readonly_mask(mask: npt.ArrayLike) -> BoolArray:
    result = np.array(mask, dtype=np.bool_, copy=True)
    result.setflags(write=False)
    return result


def _invalid_measurement_mask(spectrum: Spectrum) -> BoolArray:
    return _readonly_mask(
        spectrum.invalid_energy_mask
        | spectrum.invalid_intensity_mask
        | spectrum.invalid_uncertainty_mask
    )


@dataclass(frozen=True, slots=True)
class FittingSelection:
    """Immutable per-group ranges with masks derived from current data state."""

    dataset: ReducedDataset = field(repr=False)
    padding: EdgePaddingDetectionResult = field(repr=False)
    ranges: tuple[FittingRange, ...]

    def __post_init__(self) -> None:
        ranges = tuple(self.ranges)
        if any(not isinstance(item, FittingRange) for item in ranges):
            raise ValueError("ranges must contain FittingRange values")
        object.__setattr__(self, "ranges", ranges)
        spectrum_count = len(self.dataset.spectra)
        if len(ranges) != spectrum_count:
            raise ValueError("fitting-range count must match spectrum count")
        if len(self.padding.spectra) != spectrum_count:
            raise ValueError("padding result must match spectrum count")
        for spectrum, padding in zip(
            self.dataset.spectra, self.padding.spectra, strict=True
        ):
            if (
                spectrum.group_index != padding.group_index
                or spectrum.group_label != padding.group_identity
            ):
                raise ValueError("padding result must match dataset group order")
            if padding.auto_mask.size != spectrum.energy.size:
                raise ValueError("padding mask must match spectrum length")
        for group_index, spectrum in enumerate(self.dataset.spectra):
            if not np.any(self.retained_mask(group_index)):
                raise ValueError(
                    "selected fitting range contains no usable measured points "
                    f"for group {spectrum.group_label}"
                )

    @classmethod
    def uniform(
        cls,
        dataset: ReducedDataset,
        padding: EdgePaddingDetectionResult,
        *,
        lower_energy: float,
        upper_energy: float,
    ) -> FittingSelection:
        """Apply one initial inclusive energy range to every spectrum."""

        fitting_range = FittingRange(lower_energy, upper_energy)
        return cls(
            dataset=dataset,
            padding=padding,
            ranges=(fitting_range,) * len(dataset.spectra),
        )

    def with_group_range(
        self,
        group_index: int,
        *,
        lower_energy: float,
        upper_energy: float,
    ) -> FittingSelection:
        """Return a selection with one group's inclusive range replaced."""

        self._spectrum(group_index)
        ranges = list(self.ranges)
        ranges[group_index] = FittingRange(lower_energy, upper_energy)
        return replace(self, ranges=tuple(ranges))

    def _spectrum(self, group_index: int) -> Spectrum:
        if not 0 <= group_index < len(self.dataset.spectra):
            raise IndexError("group_index is outside the dataset")
        return self.dataset.spectra[group_index]

    def invalid_mask(self, group_index: int) -> BoolArray:
        """Return the derived invalid-measurement mask for one group."""

        return _invalid_measurement_mask(self._spectrum(group_index))

    def in_range_mask(self, group_index: int) -> BoolArray:
        """Return the inclusive selected-energy mask for one group."""

        spectrum = self._spectrum(group_index)
        fitting_range = self.ranges[group_index]
        return _readonly_mask(
            (spectrum.energy >= fitting_range.lower_energy)
            & (spectrum.energy <= fitting_range.upper_energy)
        )

    def excluded_mask(self, group_index: int) -> BoolArray:
        """Return invalid OR AUTO-padding OR outside-range points."""

        invalid = self.invalid_mask(group_index)
        in_range = self.in_range_mask(group_index)
        auto_padding = self.padding.spectra[group_index].auto_mask
        return _readonly_mask(invalid | auto_padding | ~in_range)

    def retained_mask(self, group_index: int) -> BoolArray:
        """Return points retained for later fitting."""

        return _readonly_mask(~self.excluded_mask(group_index))
