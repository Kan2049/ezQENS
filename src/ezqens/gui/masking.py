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
class AutoMaskState:
    """The core AutoMask proposal and its validated effective selection."""

    padding: EdgePaddingDetectionResult
    selection: FittingSelection | None
    diagnostic: str | None = None
    manual_boundary_groups: frozenset[int] = frozenset()
    auto_baseline_absent: bool = False

    def __post_init__(self) -> None:
        if self.selection is None:
            object.__setattr__(self, "auto_baseline_absent", True)


@dataclass(frozen=True, slots=True)
class _MaskDraftState:
    """One selection and its Manual-boundary intent in the shared undo history."""

    selection: FittingSelection
    manual_boundary_groups: frozenset[int]


def create_auto_mask_state(dataset: ReducedDataset) -> AutoMaskState:
    """Run the core detector and create its full-data selection when possible."""

    padding = detect_edge_padding(dataset)
    try:
        ranges = tuple(
            _full_energy_range(spectrum.energy) for spectrum in dataset.spectra
        )
        selection = FittingSelection(dataset=dataset, padding=padding, ranges=ranges)
    except ValueError as error:
        return AutoMaskState(padding=padding, selection=None, diagnostic=str(error))
    return AutoMaskState(padding=padding, selection=selection)


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
        self._history = [
            _MaskDraftState(selection, state.manual_boundary_groups),
        ]
        self._position = 0
        self._boundary_positions = _effective_boundary_positions(selection)

    @property
    def selection(self) -> FittingSelection:
        """Return the current preview selection."""

        return self._history[self._position].selection

    @property
    def manual_boundary_groups(self) -> frozenset[int]:
        """Return Groups with a deliberate Manual Boundary in this history state."""

        return self._history[self._position].manual_boundary_groups

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
            self._boundary_positions = _effective_boundary_positions(self.selection)

    def redo(self) -> None:
        """Move one local edit forward without mutating the saved state."""

        if self.can_redo:
            self._position += 1
            self._boundary_positions = _effective_boundary_positions(self.selection)

    def exclude_points(self, group_index: int, mask: npt.ArrayLike) -> bool:
        """Add explicit manual exclusions through the core selection model."""

        selected = _boolean_mask(mask, self.selection, group_index)
        manual = self.selection.manual_exclusion_mask(group_index).copy()
        reincluded = self.selection.manual_auto_reinclusion_mask(group_index).copy()
        manual[selected] = True
        reincluded[selected] = False
        return self._replace_group_masks(group_index, manual, reincluded)

    def restore_points(self, group_index: int, mask: npt.ArrayLike) -> bool:
        """Restore manual edits while leaving the core invalid-point rule intact."""

        selected = _boolean_mask(mask, self.selection, group_index)
        manual = self.selection.manual_exclusion_mask(group_index).copy()
        reincluded = self.selection.manual_auto_reinclusion_mask(group_index).copy()
        auto = self._padding.spectra[group_index].auto_mask
        manual[selected] = False
        reincluded[selected & auto] = True
        return self._replace_group_masks(group_index, manual, reincluded)

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

        replacement = self.preview_auto_boundary(
            group_index,
            side=side,
            energy=energy,
        )
        previous_energy = self._boundary_positions[group_index][side]
        manual_boundary_groups = self.manual_boundary_groups
        if energy != previous_energy:
            manual_boundary_groups = manual_boundary_groups | {group_index}
        changed = self._push(
            replacement,
            manual_boundary_groups=frozenset(manual_boundary_groups),
        )
        if changed:
            self._boundary_positions[group_index][side] = energy
        return changed

    def preview_auto_boundary(
        self,
        group_index: int,
        *,
        side: BoundarySide,
        energy: float,
    ) -> FittingSelection:
        """Return one live boundary preview without recording an undo entry."""

        selection = self.selection
        spectrum = selection.dataset.spectra[group_index]
        result = self._padding.spectra[group_index]
        side_auto = _boundary_auto_mask(result.auto_mask, result, side)
        boundary = (
            spectrum.energy < energy
            if side is BoundarySide.LEFT
            else spectrum.energy > energy
        )

        manual = selection.manual_exclusion_mask(group_index).copy()
        reincluded = selection.manual_auto_reinclusion_mask(group_index).copy()
        previous_energy = self._boundary_positions[group_index][side]
        if side is BoundarySide.LEFT and energy < previous_energy:
            restored = (spectrum.energy >= energy) & (spectrum.energy < previous_energy)
            manual[restored & ~side_auto] = False
        elif side is BoundarySide.RIGHT and energy > previous_energy:
            restored = (spectrum.energy > previous_energy) & (spectrum.energy <= energy)
            manual[restored & ~side_auto] = False
        manual[boundary & ~side_auto] = True
        manual[side_auto] = False
        reincluded[side_auto & boundary] = False
        reincluded[side_auto & ~boundary] = True
        replacement = selection.with_group_manual_exclusion(group_index, manual)
        return replacement.with_group_manual_auto_reinclusion(group_index, reincluded)

    def apply_boundary_to_all_groups(self, source_group_index: int) -> bool:
        """Apply one Group's energy boundary to every Group in one history entry."""

        if not self.can_apply_boundary_to_all_groups(source_group_index):
            raise ValueError(
                "Adjust the current Group's Manual Boundary before applying it "
                "to all Groups"
            )
        source = self._boundary_positions[source_group_index]
        lower_energy = source[BoundarySide.LEFT]
        upper_energy = source[BoundarySide.RIGHT]
        replacement = self.selection
        for group_index in range(len(replacement.dataset.spectra)):
            replacement = replacement.with_group_range(
                group_index,
                lower_energy=lower_energy,
                upper_energy=upper_energy,
            )
        changed = self._push(
            replacement,
            manual_boundary_groups=frozenset(range(len(replacement.dataset.spectra))),
        )
        if changed:
            self._boundary_positions = _effective_boundary_positions(replacement)
        return changed

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
        changed = self._push(
            replacement,
            manual_boundary_groups=frozenset(manual_boundary_groups),
        )
        if changed:
            self._boundary_positions = _effective_boundary_positions(self.selection)
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
        )
        if changed:
            self._boundary_positions = _effective_boundary_positions(replacement)
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
        self._history = [
            _MaskDraftState(selection, state.manual_boundary_groups),
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
        self._position = 0
        self._boundary_positions = _effective_boundary_positions(selection)

    def saved_state(self) -> AutoMaskState:
        """Return the current task selection paired with its immutable proposal."""

        return AutoMaskState(
            padding=self._padding,
            selection=self.selection,
            diagnostic=self._diagnostic,
            manual_boundary_groups=self.manual_boundary_groups,
            auto_baseline_absent=self._auto_baseline_absent,
        )

    def _replace_group_masks(
        self,
        group_index: int,
        manual: BoolArray,
        reincluded: BoolArray,
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
        return self._push(replacement)

    def _push(
        self,
        selection: FittingSelection,
        *,
        manual_boundary_groups: frozenset[int] | None = None,
    ) -> bool:
        if selection is self.selection:
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


def _effective_boundary_positions(
    selection: FittingSelection,
) -> list[dict[BoundarySide, float]]:
    """Locate the current retained edge points for draggable boundary handles."""

    positions: list[dict[BoundarySide, float]] = []
    for group_index, spectrum in enumerate(selection.dataset.spectra):
        retained = ~selection.excluded_mask(group_index) & np.isfinite(spectrum.energy)
        retained_energy = spectrum.energy[retained]
        if not retained_energy.size:
            raise ValueError("selection has no retained energy for a mask boundary")
        positions.append(
            {
                BoundarySide.LEFT: float(np.min(retained_energy)),
                BoundarySide.RIGHT: float(np.max(retained_energy)),
            },
        )
    return positions
