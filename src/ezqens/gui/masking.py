"""Thin GUI workflow state around the validated padding and selection core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from matplotlib.path import Path as MatplotlibPath

from ezqens.domain import ReducedDataset
from ezqens.preprocessing import (
    BoundarySide,
    EdgePaddingDetectionResult,
    FittingRange,
    FittingSelection,
    PaddingStatus,
    SpectrumPaddingResult,
    detect_edge_padding,
)
from ezqens.workflow import measured_point_correspondence_exactly_unchanged

BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class BoundaryCoordinates:
    """Canonical measured-energy coordinates owned by one Group's Boundary."""

    left_energy: float
    right_energy: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.left_energy) or not np.isfinite(self.right_energy):
            raise ValueError("boundary coordinates must be finite")
        if self.left_energy > self.right_energy:
            raise ValueError("left boundary cannot cross the right boundary")

    def energy(self, side: BoundarySide) -> float:
        """Return the coordinate for one typed Boundary side."""

        return self.left_energy if side is BoundarySide.LEFT else self.right_energy


@dataclass(frozen=True, slots=True)
class AutoMaskState:
    """The core AutoMask proposal and its validated effective selection."""

    padding: EdgePaddingDetectionResult
    selection: FittingSelection | None
    diagnostic: str | None = None
    manual_boundary_groups: frozenset[int] = frozenset()
    boundary_exclusion_masks: tuple[tuple[BoolArray, BoolArray], ...] = ()
    boundary_coordinates: tuple[BoundaryCoordinates, ...] = ()
    auto_baseline_absent: bool = False

    def __post_init__(self) -> None:
        if self.selection is None:
            object.__setattr__(self, "auto_baseline_absent", True)
        masks = self.boundary_exclusion_masks
        if not masks:
            masks = tuple(
                (
                    np.zeros(result.auto_mask.size, dtype=np.bool_),
                    np.zeros(result.auto_mask.size, dtype=np.bool_),
                )
                for result in self.padding.spectra
            )
        if len(masks) != len(self.padding.spectra):
            raise ValueError("boundary masks must match the dataset group count")
        normalized: list[tuple[BoolArray, BoolArray]] = []
        for pair, result in zip(masks, self.padding.spectra, strict=True):
            if len(pair) != 2:
                raise ValueError("boundary masks require left and right entries")
            sides: list[BoolArray] = []
            for value in pair:
                mask = np.array(value, dtype=np.bool_, copy=True)
                if mask.ndim != 1 or mask.size != result.auto_mask.size:
                    raise ValueError("boundary mask must match measured point count")
                mask.setflags(write=False)
                sides.append(mask)
            normalized.append((sides[0], sides[1]))
        object.__setattr__(self, "boundary_exclusion_masks", tuple(normalized))
        coordinates = tuple(self.boundary_coordinates)
        if coordinates and len(coordinates) != len(self.padding.spectra):
            raise ValueError("boundary coordinates must match the dataset group count")
        object.__setattr__(self, "boundary_coordinates", coordinates)


@dataclass(frozen=True, slots=True)
class _MaskDraftState:
    """One selection and its Manual-boundary intent in the shared undo history."""

    selection: FittingSelection
    manual_boundary_groups: frozenset[int]
    boundary_exclusion_masks: tuple[tuple[BoolArray, BoolArray], ...]
    boundary_coordinates: tuple[BoundaryCoordinates, ...]


def create_auto_mask_state(dataset: ReducedDataset) -> AutoMaskState:
    """Run the core detector and create its full-data selection when possible."""

    padding = detect_edge_padding(dataset)
    try:
        ranges = tuple(
            _full_energy_range(spectrum.energy) for spectrum in dataset.spectra
        )
        selection = FittingSelection(dataset=dataset, padding=padding, ranges=ranges)
    except ValueError as error:
        return AutoMaskState(
            padding=padding,
            selection=None,
            diagnostic=str(error),
            boundary_coordinates=_initial_boundary_coordinates(
                dataset,
                padding,
                auto_baseline_absent=True,
            ),
        )
    return AutoMaskState(
        padding=padding,
        selection=selection,
        boundary_coordinates=_initial_boundary_coordinates(
            dataset,
            padding,
            auto_baseline_absent=False,
        ),
    )


def rebind_auto_mask_state(
    state: AutoMaskState,
    dataset: ReducedDataset,
) -> AutoMaskState:
    """Attach unchanged selection masks to metadata-only dataset replacement."""

    if state.selection is None:
        return AutoMaskState(
            padding=state.padding,
            selection=None,
            diagnostic=state.diagnostic,
            manual_boundary_groups=state.manual_boundary_groups,
            boundary_exclusion_masks=state.boundary_exclusion_masks,
            boundary_coordinates=state.boundary_coordinates,
            auto_baseline_absent=state.auto_baseline_absent,
        )
    if not measured_point_correspondence_exactly_unchanged(
        state.selection.dataset,
        dataset,
    ):
        return AutoMaskState(
            padding=state.padding,
            selection=None,
            diagnostic=(
                "measured spectra changed; AutoMask and fitting selection must be "
                "initialized again"
            ),
        )
    try:
        selection = FittingSelection(
            dataset=dataset,
            padding=state.padding,
            ranges=state.selection.ranges,
            manual_exclusion_masks=state.selection.manual_exclusion_masks,
            manual_auto_reinclusion_masks=state.selection.manual_auto_reinclusion_masks,
        )
    except ValueError as error:
        return AutoMaskState(
            padding=state.padding,
            selection=None,
            diagnostic=str(error),
        )
    return AutoMaskState(
        padding=state.padding,
        selection=selection,
        diagnostic=state.diagnostic,
        manual_boundary_groups=state.manual_boundary_groups,
        boundary_exclusion_masks=state.boundary_exclusion_masks,
        boundary_coordinates=state.boundary_coordinates,
        auto_baseline_absent=state.auto_baseline_absent,
    )


class MaskTaskDraft:
    """Task-local, undoable edits delegated to ``FittingSelection`` semantics."""

    def __init__(
        self,
        state: AutoMaskState,
        dataset: ReducedDataset | None = None,
    ) -> None:
        selection = state.selection
        if selection is None:
            if dataset is None:
                message = state.diagnostic or "AutoMask selection is unavailable"
                raise ValueError(message)
            selection = _manual_selection_without_auto_baseline(state, dataset)
        self._padding = state.padding
        self._diagnostic = state.diagnostic
        self._auto_baseline_absent = state.auto_baseline_absent
        self._baseline_ranges = (
            tuple(
                _full_energy_range(spectrum.energy)
                for spectrum in selection.dataset.spectra
            )
            if state.auto_baseline_absent
            else selection.ranges
        )
        self._baseline_reinclusions = (
            tuple(padding.auto_mask for padding in state.padding.spectra)
            if state.auto_baseline_absent
            else tuple(
                np.zeros(spectrum.energy.size, dtype=np.bool_)
                for spectrum in selection.dataset.spectra
            )
        )
        boundary_coordinates = (
            state.boundary_coordinates
            if state.boundary_coordinates
            else _initial_boundary_coordinates(
                selection.dataset,
                state.padding,
                auto_baseline_absent=state.auto_baseline_absent,
            )
        )
        _validate_boundary_coordinates(selection.dataset, boundary_coordinates)
        self._baseline_boundary_coordinates = _initial_boundary_coordinates(
            selection.dataset,
            state.padding,
            auto_baseline_absent=state.auto_baseline_absent,
        )
        self._history = [
            _MaskDraftState(
                selection,
                state.manual_boundary_groups,
                state.boundary_exclusion_masks,
                boundary_coordinates,
            ),
        ]
        self._position = 0

    @property
    def selection(self) -> FittingSelection:
        """Return the current preview selection."""

        return self._history[self._position].selection

    @property
    def manual_boundary_groups(self) -> frozenset[int]:
        """Return Groups with a deliberate Manual Boundary in this history state."""

        return self._history[self._position].manual_boundary_groups

    @property
    def boundary_exclusion_masks(
        self,
    ) -> tuple[tuple[BoolArray, BoolArray], ...]:
        """Return boundary-owned manual exclusions for the current history state."""

        return self._history[self._position].boundary_exclusion_masks

    @property
    def boundary_coordinates(self) -> tuple[BoundaryCoordinates, ...]:
        """Return explicit canonical Boundary intent for every Group."""

        return self._history[self._position].boundary_coordinates

    def can_apply_boundary_to_all_groups(self, source_group_index: int) -> bool:
        """Return whether the current Group owns explicit Manual boundary intent."""

        return (
            len(self.selection.dataset.spectra) > 1
            and source_group_index in self.manual_boundary_groups
        )

    @property
    def can_undo(self) -> bool:
        return self._position > 0

    @property
    def can_redo(self) -> bool:
        return self._position + 1 < len(self._history)

    @property
    def is_dirty(self) -> bool:
        return self._position != 0

    @property
    def has_manual_edits(self) -> bool:
        """Return whether the current preview differs from its AutoMask baseline."""

        return self.selection.ranges != self._baseline_ranges or any(
            np.any(self.selection.manual_exclusion_mask(group_index))
            or np.any(
                self.selection.manual_auto_reinclusion_mask(group_index)
                != self._baseline_reinclusions[group_index]
            )
            for group_index in range(len(self.selection.dataset.spectra))
        )

    def undo(self) -> None:
        """Move one local edit backward without mutating the saved state."""

        if self.can_undo:
            self._position -= 1

    def redo(self) -> None:
        """Move one local edit forward without mutating the saved state."""

        if self.can_redo:
            self._position += 1

    def exclude_points(self, group_index: int, mask: npt.ArrayLike) -> bool:
        """Add explicit manual exclusions through the core selection model."""

        selected = _boolean_mask(mask, self.selection, group_index)
        manual = self.selection.manual_exclusion_mask(group_index).copy()
        reincluded = self.selection.manual_auto_reinclusion_mask(group_index).copy()
        manual[selected] = True
        reincluded[selected] = False
        return self._replace_group_masks(
            group_index,
            manual,
            reincluded,
            boundary_override=selected,
        )

    def restore_points(self, group_index: int, mask: npt.ArrayLike) -> bool:
        """Restore manual edits while leaving the core invalid-point rule intact."""

        selected = _boolean_mask(mask, self.selection, group_index)
        manual = self.selection.manual_exclusion_mask(group_index).copy()
        reincluded = self.selection.manual_auto_reinclusion_mask(group_index).copy()
        auto = self._padding.spectra[group_index].auto_mask
        manual[selected] = False
        reincluded[selected & auto] = True
        return self._replace_group_masks(
            group_index,
            manual,
            reincluded,
            boundary_override=selected,
        )

    def exclude_rectangle(
        self,
        group_index: int,
        *,
        lower_energy: float,
        upper_energy: float,
        lower_intensity: float,
        upper_intensity: float,
    ) -> bool:
        """Exclude original points inside a visual rectangle without changing data."""

        spectrum = self.selection.dataset.spectra[group_index]
        mask = (
            (spectrum.energy >= min(lower_energy, upper_energy))
            & (spectrum.energy <= max(lower_energy, upper_energy))
            & (spectrum.intensity >= min(lower_intensity, upper_intensity))
            & (spectrum.intensity <= max(lower_intensity, upper_intensity))
        )
        return self.exclude_points(group_index, mask)

    def exclude_lasso(
        self,
        group_index: int,
        vertices: object,
    ) -> bool:
        """Exclude original points selected by a visual lasso polygon."""

        spectrum = self.selection.dataset.spectra[group_index]
        points = np.column_stack((spectrum.energy, spectrum.intensity))
        mask = MatplotlibPath(np.asarray(vertices, dtype=np.float64)).contains_points(
            points,
        )
        return self.exclude_points(group_index, mask)

    def restore_rectangle(
        self,
        group_index: int,
        *,
        lower_energy: float,
        upper_energy: float,
        lower_intensity: float,
        upper_intensity: float,
    ) -> bool:
        """Restore points in a visual rectangle through the selection API."""

        spectrum = self.selection.dataset.spectra[group_index]
        mask = (
            (spectrum.energy >= min(lower_energy, upper_energy))
            & (spectrum.energy <= max(lower_energy, upper_energy))
            & (spectrum.intensity >= min(lower_intensity, upper_intensity))
            & (spectrum.intensity <= max(lower_intensity, upper_intensity))
        )
        return self.restore_points(group_index, mask)

    def restore_lasso(self, group_index: int, vertices: object) -> bool:
        """Restore lasso-selected points where core selection semantics permit."""

        spectrum = self.selection.dataset.spectra[group_index]
        points = np.column_stack((spectrum.energy, spectrum.intensity))
        mask = MatplotlibPath(np.asarray(vertices, dtype=np.float64)).contains_points(
            points,
        )
        return self.restore_points(group_index, mask)

    def set_auto_boundary(
        self,
        group_index: int,
        *,
        side: BoundarySide,
        energy: float,
    ) -> bool:
        """Apply reversible boundary intent using the core's AUTO re-inclusion API."""

        replacement, boundary_masks, canonical = self._boundary_replacement(
            self.selection,
            self.boundary_exclusion_masks,
            group_index,
            side=side,
            energy=energy,
        )
        previous_energy = self.boundary_coordinates[group_index].energy(side)
        manual_boundary_groups = self.manual_boundary_groups
        if canonical != previous_energy:
            manual_boundary_groups = manual_boundary_groups | {group_index}
        else:
            return False
        boundary_coordinates = _replace_boundary_coordinate(
            self.boundary_coordinates,
            group_index,
            side,
            canonical,
        )
        changed = self._push(
            replacement,
            manual_boundary_groups=frozenset(manual_boundary_groups),
            boundary_exclusion_masks=boundary_masks,
            boundary_coordinates=boundary_coordinates,
        )
        return changed

    def preview_auto_boundary(
        self,
        group_index: int,
        *,
        side: BoundarySide,
        energy: float,
    ) -> tuple[FittingSelection, tuple[BoundaryCoordinates, ...]]:
        """Return one live boundary preview without recording an undo entry."""

        replacement, _boundary_masks, canonical = self._boundary_replacement(
            self.selection,
            self.boundary_exclusion_masks,
            group_index,
            side=side,
            energy=energy,
        )
        coordinates = _replace_boundary_coordinate(
            self.boundary_coordinates,
            group_index,
            side,
            canonical,
        )
        return replacement, coordinates

    def apply_boundary_to_all_groups(self, source_group_index: int) -> bool:
        """Apply one Group's energy boundary to every Group in one history entry."""

        if not self.can_apply_boundary_to_all_groups(source_group_index):
            raise ValueError(
                "Adjust the current Group's Manual Boundary before applying it "
                "to all Groups"
            )
        source = self.boundary_coordinates[source_group_index]
        lower_energy = source.left_energy
        upper_energy = source.right_energy
        replacement = self.selection
        boundary_masks = self.boundary_exclusion_masks
        boundary_coordinates = list(self.boundary_coordinates)
        for group_index in range(len(replacement.dataset.spectra)):
            intended_positions = {
                BoundarySide.LEFT: _snap_boundary_coordinate(
                    replacement,
                    group_index,
                    BoundarySide.LEFT,
                    lower_energy,
                ),
                BoundarySide.RIGHT: _snap_boundary_coordinate(
                    replacement,
                    group_index,
                    BoundarySide.RIGHT,
                    upper_energy,
                ),
            }
            if (
                intended_positions[BoundarySide.LEFT]
                > intended_positions[BoundarySide.RIGHT]
            ):
                raise ValueError(
                    "the applied boundary does not retain a valid measured domain"
                )
            replacement, boundary_masks, _left = self._boundary_replacement(
                replacement,
                boundary_masks,
                group_index,
                side=BoundarySide.LEFT,
                energy=lower_energy,
                current_positions=intended_positions,
            )
            replacement, boundary_masks, _right = self._boundary_replacement(
                replacement,
                boundary_masks,
                group_index,
                side=BoundarySide.RIGHT,
                energy=upper_energy,
                current_positions=intended_positions,
            )
            boundary_coordinates[group_index] = BoundaryCoordinates(
                intended_positions[BoundarySide.LEFT],
                intended_positions[BoundarySide.RIGHT],
            )
        changed = self._push(
            replacement,
            manual_boundary_groups=frozenset(range(len(replacement.dataset.spectra))),
            boundary_exclusion_masks=boundary_masks,
            boundary_coordinates=tuple(boundary_coordinates),
        )
        return changed

    def _boundary_replacement(
        self,
        selection: FittingSelection,
        boundary_masks: tuple[tuple[BoolArray, BoolArray], ...],
        group_index: int,
        *,
        side: BoundarySide,
        energy: float,
        current_positions: dict[BoundarySide, float] | None = None,
    ) -> tuple[
        FittingSelection,
        tuple[tuple[BoolArray, BoolArray], ...],
        float,
    ]:
        """Compose one canonical boundary while retaining unrelated mask edits."""

        canonical = _canonical_boundary_coordinate(
            selection,
            group_index,
            side,
            energy,
            (
                _boundary_coordinate_mapping(self.boundary_coordinates[group_index])
                if current_positions is None
                else current_positions
            ),
        )
        spectrum = selection.dataset.spectra[group_index]
        result = self._padding.spectra[group_index]
        side_auto = _boundary_auto_mask(result.auto_mask, result, side)
        boundary = (
            spectrum.energy < canonical
            if side is BoundarySide.LEFT
            else spectrum.energy > canonical
        )
        side_index = 0 if side is BoundarySide.LEFT else 1
        manual = selection.manual_exclusion_mask(group_index).copy()
        reincluded = selection.manual_auto_reinclusion_mask(group_index).copy()
        manual[boundary_masks[group_index][side_index]] = False
        owned = boundary & ~side_auto & np.isfinite(spectrum.energy)
        manual[owned] = True
        reincluded[side_auto & boundary] = False
        reincluded[side_auto & ~boundary] = True
        replacement = selection.with_group_manual_exclusion(group_index, manual)
        replacement = replacement.with_group_manual_auto_reinclusion(
            group_index,
            reincluded,
        )
        updated_masks = _replace_boundary_exclusion_mask(
            boundary_masks,
            group_index,
            side_index,
            owned,
        )
        return replacement, updated_masks, canonical

    def reset_group(self, group_index: int) -> bool:
        """Restore one Group to its existing AutoMask proposal without recomputing."""

        selection = self.selection
        manual = np.zeros_like(selection.manual_exclusion_mask(group_index))
        reincluded = self._baseline_reinclusions[group_index]
        replacement = selection.with_group_manual_exclusion(group_index, manual)
        replacement = replacement.with_group_manual_auto_reinclusion(
            group_index,
            reincluded,
        )
        baseline = self._baseline_ranges[group_index]
        replacement = replacement.with_group_range(
            group_index,
            lower_energy=baseline.lower_energy,
            upper_energy=baseline.upper_energy,
        )
        manual_boundary_groups = self.manual_boundary_groups - {group_index}
        boundary_masks = _clear_group_boundary_exclusion_masks(
            self.boundary_exclusion_masks,
            group_index,
        )
        boundary_coordinates = list(self.boundary_coordinates)
        boundary_coordinates[group_index] = self._baseline_boundary_coordinates[
            group_index
        ]
        changed = self._push(
            replacement,
            manual_boundary_groups=frozenset(manual_boundary_groups),
            boundary_exclusion_masks=boundary_masks,
            boundary_coordinates=tuple(boundary_coordinates),
        )
        return changed

    @property
    def can_disable_auto_mask(self) -> bool:
        """Return whether any valid automatic exclusion can be re-included."""

        for group_index, padding in enumerate(self._padding.spectra):
            selection = self.selection
            candidate = (
                padding.auto_mask
                & ~selection.invalid_mask(group_index)
                & ~selection.manual_exclusion_mask(group_index)
                & ~selection.manual_auto_reinclusion_mask(group_index)
            )
            if np.any(candidate):
                return True
        return False

    def disable_auto_mask(self) -> bool:
        """Re-include valid automatic exclusions in one reversible draft edit.

        The automatic proposal remains intact: this records only explicit manual
        AUTO re-inclusions, so Reset Group and Reset All can restore the baseline.
        """

        selection = self.selection
        replacement = selection
        for group_index, padding in enumerate(self._padding.spectra):
            manual = selection.manual_exclusion_mask(group_index)
            reincluded = selection.manual_auto_reinclusion_mask(group_index)
            eligible = (
                padding.auto_mask & ~selection.invalid_mask(group_index) & ~manual
            )
            if not np.any(eligible & ~reincluded):
                continue
            replacement = replacement.with_group_manual_auto_reinclusion(
                group_index,
                reincluded | eligible,
            )
        return self._push(replacement)

    def reset_all(self) -> bool:
        """Restore every Group to its existing AutoMask proposal without recomputing."""

        selection = self.selection
        replacement = FittingSelection(
            dataset=selection.dataset,
            padding=selection.padding,
            ranges=self._baseline_ranges,
            manual_auto_reinclusion_masks=self._baseline_reinclusions,
        )
        changed = self._push(
            replacement,
            manual_boundary_groups=frozenset(),
            boundary_exclusion_masks=_empty_boundary_exclusion_masks(selection),
            boundary_coordinates=self._baseline_boundary_coordinates,
        )
        return changed

    def reset_to_auto_mask(self) -> bool:
        """Compatibility alias for restoring every Group to the current baseline."""

        return self.reset_all()

    def replace_auto_mask(self, state: AutoMaskState) -> None:
        """Use a newly confirmed core proposal as this task's fresh baseline."""

        selection = state.selection
        if selection is None:
            selection = _manual_selection_without_auto_baseline(
                state,
                self.selection.dataset,
            )
        self._padding = state.padding
        self._diagnostic = state.diagnostic
        self._auto_baseline_absent = state.auto_baseline_absent
        boundary_coordinates = (
            state.boundary_coordinates
            if state.boundary_coordinates
            else _initial_boundary_coordinates(
                selection.dataset,
                state.padding,
                auto_baseline_absent=state.auto_baseline_absent,
            )
        )
        _validate_boundary_coordinates(selection.dataset, boundary_coordinates)
        self._history = [
            _MaskDraftState(
                selection,
                state.manual_boundary_groups,
                state.boundary_exclusion_masks,
                boundary_coordinates,
            ),
        ]
        self._baseline_ranges = (
            tuple(
                _full_energy_range(spectrum.energy)
                for spectrum in selection.dataset.spectra
            )
            if state.auto_baseline_absent
            else selection.ranges
        )
        self._baseline_reinclusions = (
            tuple(padding.auto_mask for padding in state.padding.spectra)
            if state.auto_baseline_absent
            else tuple(
                np.zeros(spectrum.energy.size, dtype=np.bool_)
                for spectrum in selection.dataset.spectra
            )
        )
        self._baseline_boundary_coordinates = _initial_boundary_coordinates(
            selection.dataset,
            state.padding,
            auto_baseline_absent=state.auto_baseline_absent,
        )
        self._position = 0

    def saved_state(self) -> AutoMaskState:
        """Return the current task selection paired with its immutable proposal."""

        return AutoMaskState(
            padding=self._padding,
            selection=self.selection,
            diagnostic=self._diagnostic,
            manual_boundary_groups=self.manual_boundary_groups,
            boundary_exclusion_masks=self.boundary_exclusion_masks,
            boundary_coordinates=self.boundary_coordinates,
            auto_baseline_absent=self._auto_baseline_absent,
        )

    def _replace_group_masks(
        self,
        group_index: int,
        manual: BoolArray,
        reincluded: BoolArray,
        *,
        boundary_override: BoolArray | None = None,
    ) -> bool:
        selection = self.selection
        try:
            replacement = selection.with_group_manual_exclusion(group_index, manual)
            replacement = replacement.with_group_manual_auto_reinclusion(
                group_index,
                reincluded,
            )
        except ValueError:
            return False
        boundary_masks = self.boundary_exclusion_masks
        if boundary_override is not None:
            boundary_masks = _remove_boundary_exclusion_ownership(
                boundary_masks,
                group_index,
                boundary_override,
            )
        return self._push(
            replacement,
            boundary_exclusion_masks=boundary_masks,
        )

    def _push(
        self,
        selection: FittingSelection,
        *,
        manual_boundary_groups: frozenset[int] | None = None,
        boundary_exclusion_masks: (
            tuple[tuple[BoolArray, BoolArray], ...] | None
        ) = None,
        boundary_coordinates: tuple[BoundaryCoordinates, ...] | None = None,
    ) -> bool:
        if (
            selection is self.selection
            and manual_boundary_groups is None
            and boundary_exclusion_masks is None
            and boundary_coordinates is None
        ):
            return False
        self._history = self._history[: self._position + 1]
        self._history.append(
            _MaskDraftState(
                selection,
                (
                    self.manual_boundary_groups
                    if manual_boundary_groups is None
                    else manual_boundary_groups
                ),
                (
                    self.boundary_exclusion_masks
                    if boundary_exclusion_masks is None
                    else boundary_exclusion_masks
                ),
                (
                    self.boundary_coordinates
                    if boundary_coordinates is None
                    else boundary_coordinates
                ),
            )
        )
        self._position += 1
        return True


def _full_energy_range(energy: npt.ArrayLike) -> FittingRange:
    finite = np.asarray(energy, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("spectrum has no finite energy values for AutoMask selection")
    return FittingRange(float(np.min(finite)), float(np.max(finite)))


def _initial_boundary_coordinates(
    dataset: ReducedDataset,
    padding: EdgePaddingDetectionResult,
    *,
    auto_baseline_absent: bool,
) -> tuple[BoundaryCoordinates, ...]:
    """Create explicit initial Boundary intent from domain and AUTO provenance."""

    if len(dataset.spectra) != len(padding.spectra):
        raise ValueError("padding result must match dataset group count")
    coordinates: list[BoundaryCoordinates] = []
    for spectrum, result in zip(dataset.spectra, padding.spectra, strict=True):
        finite = np.isfinite(spectrum.energy)
        candidates = spectrum.energy[
            finite if auto_baseline_absent else finite & ~result.auto_mask
        ]
        if not candidates.size:
            candidates = spectrum.energy[finite]
        if not candidates.size:
            raise ValueError("spectrum has no finite energy values for a mask boundary")
        coordinates.append(
            BoundaryCoordinates(
                float(np.min(candidates)),
                float(np.max(candidates)),
            )
        )
    return tuple(coordinates)


def _validate_boundary_coordinates(
    dataset: ReducedDataset,
    coordinates: tuple[BoundaryCoordinates, ...],
) -> None:
    """Require every stored Boundary to be a measured coordinate in its Group."""

    if len(coordinates) != len(dataset.spectra):
        raise ValueError("boundary coordinates must match the dataset group count")
    for spectrum, boundary in zip(dataset.spectra, coordinates, strict=True):
        finite_energy = spectrum.energy[np.isfinite(spectrum.energy)]
        if not np.any(finite_energy == boundary.left_energy) or not np.any(
            finite_energy == boundary.right_energy
        ):
            raise ValueError("boundary coordinates must match measured energy values")


def _boundary_coordinate_mapping(
    coordinates: BoundaryCoordinates,
) -> dict[BoundarySide, float]:
    return {
        BoundarySide.LEFT: coordinates.left_energy,
        BoundarySide.RIGHT: coordinates.right_energy,
    }


def _replace_boundary_coordinate(
    coordinates: tuple[BoundaryCoordinates, ...],
    group_index: int,
    side: BoundarySide,
    energy: float,
) -> tuple[BoundaryCoordinates, ...]:
    groups = list(coordinates)
    current = groups[group_index]
    groups[group_index] = (
        BoundaryCoordinates(energy, current.right_energy)
        if side is BoundarySide.LEFT
        else BoundaryCoordinates(current.left_energy, energy)
    )
    return tuple(groups)


def mask_task_available(state: AutoMaskState, dataset: ReducedDataset) -> bool:
    """Return whether Auto or valid measured state can seed Manual Mask editing."""

    if state.selection is not None:
        return True
    try:
        _manual_selection_without_auto_baseline(state, dataset)
    except ValueError:
        return False
    return True


def _manual_selection_without_auto_baseline(
    state: AutoMaskState,
    dataset: ReducedDataset,
) -> FittingSelection:
    """Seed Manual editing while retaining an unavailable Auto proposal verbatim."""

    if len(state.padding.spectra) != len(dataset.spectra):
        raise ValueError("padding result must match dataset group count")
    ranges = tuple(_full_energy_range(spectrum.energy) for spectrum in dataset.spectra)
    reinclusions = tuple(padding.auto_mask for padding in state.padding.spectra)
    return FittingSelection(
        dataset=dataset,
        padding=state.padding,
        ranges=ranges,
        manual_auto_reinclusion_masks=reinclusions,
    )


def _boolean_mask(
    mask: npt.ArrayLike,
    selection: FittingSelection,
    group_index: int,
) -> BoolArray:
    result = np.asarray(mask, dtype=np.bool_)
    expected_size = selection.dataset.spectra[group_index].energy.size
    if result.ndim != 1 or result.size != expected_size:
        raise ValueError("mask must match the selected group's point count")
    return result


def _boundary_auto_mask(
    auto_mask: BoolArray,
    result: SpectrumPaddingResult,
    side: BoundarySide,
) -> BoolArray:
    """Limit a boundary edit to the corresponding core AUTO run."""

    boundary = getattr(result, side.value)
    side_mask = np.zeros(auto_mask.size, dtype=np.bool_)
    if boundary.status is not PaddingStatus.AUTO:
        return side_mask
    count = boundary.run_length
    if side is BoundarySide.LEFT:
        side_mask[:count] = auto_mask[:count]
    else:
        side_mask[-count:] = auto_mask[-count:]
    return side_mask


def _canonical_boundary_coordinate(
    selection: FittingSelection,
    group_index: int,
    side: BoundarySide,
    energy: float,
    current: dict[BoundarySide, float],
) -> float:
    """Snap one requested boundary to its measured coordinate before evaluation."""

    canonical = _snap_boundary_coordinate(
        selection,
        group_index,
        side,
        energy,
    )
    if side is BoundarySide.LEFT:
        if canonical > current[BoundarySide.RIGHT]:
            raise ValueError("left boundary cannot cross the right boundary")
    elif canonical < current[BoundarySide.LEFT]:
        raise ValueError("right boundary cannot cross the left boundary")
    return canonical


def _snap_boundary_coordinate(
    selection: FittingSelection,
    group_index: int,
    side: BoundarySide,
    energy: float,
) -> float:
    """Snap a finite request to the corresponding measured-point boundary."""

    if not np.isfinite(energy):
        raise ValueError("boundary energy must be finite")
    measured = selection.dataset.spectra[group_index].energy
    finite = np.unique(measured[np.isfinite(measured)])
    if not finite.size:
        raise ValueError("spectrum has no finite energy values for a mask boundary")
    if side is BoundarySide.LEFT:
        candidates = finite[finite >= energy]
        return float(candidates[0] if candidates.size else finite[-1])
    candidates = finite[finite <= energy]
    return float(candidates[-1] if candidates.size else finite[0])


def _replace_boundary_exclusion_mask(
    masks: tuple[tuple[BoolArray, BoolArray], ...],
    group_index: int,
    side_index: int,
    replacement: BoolArray,
) -> tuple[tuple[BoolArray, BoolArray], ...]:
    groups = list(masks)
    sides = list(groups[group_index])
    sides[side_index] = np.array(replacement, dtype=np.bool_, copy=True)
    groups[group_index] = (sides[0], sides[1])
    return tuple(groups)


def _remove_boundary_exclusion_ownership(
    masks: tuple[tuple[BoolArray, BoolArray], ...],
    group_index: int,
    selected: BoolArray,
) -> tuple[tuple[BoolArray, BoolArray], ...]:
    groups = list(masks)
    left, right = groups[group_index]
    groups[group_index] = (
        np.asarray(left & ~selected, dtype=np.bool_),
        np.asarray(right & ~selected, dtype=np.bool_),
    )
    return tuple(groups)


def _clear_group_boundary_exclusion_masks(
    masks: tuple[tuple[BoolArray, BoolArray], ...],
    group_index: int,
) -> tuple[tuple[BoolArray, BoolArray], ...]:
    groups = list(masks)
    left, right = groups[group_index]
    groups[group_index] = (np.zeros_like(left), np.zeros_like(right))
    return tuple(groups)


def _empty_boundary_exclusion_masks(
    selection: FittingSelection,
) -> tuple[tuple[BoolArray, BoolArray], ...]:
    return tuple(
        (
            np.zeros(spectrum.energy.size, dtype=np.bool_),
            np.zeros(spectrum.energy.size, dtype=np.bool_),
        )
        for spectrum in selection.dataset.spectra
    )
