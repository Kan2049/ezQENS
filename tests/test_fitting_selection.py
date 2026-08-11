"""Tests for per-group fitting ranges and derived point masks."""

from collections.abc import Sequence

import numpy as np
import pytest

from ezqens.domain import QBins, ReducedDataset, Spectrum, SpectrumRole
from ezqens.preprocessing import (
    FittingRange,
    FittingSelection,
    PaddingStatus,
    detect_edge_padding,
)


def make_dataset(
    intensities: Sequence[Sequence[float]],
    *,
    energies: Sequence[Sequence[float]] | None = None,
    uncertainties: Sequence[Sequence[float]] | None = None,
) -> ReducedDataset:
    spectra = []
    for index, values in enumerate(intensities):
        intensity = np.asarray(values, dtype=np.float64)
        energy = (
            np.arange(intensity.size, dtype=np.float64) - intensity.size // 2
            if energies is None
            else np.asarray(energies[index], dtype=np.float64)
        )
        uncertainty = (
            np.full(intensity.size, 0.2)
            if uncertainties is None
            else np.asarray(uncertainties[index], dtype=np.float64)
        )
        spectra.append(
            Spectrum(
                role=SpectrumRole.SAMPLE,
                group_index=index,
                group_label=f"group-{index}",
                energy=energy,
                intensity=intensity,
                uncertainty=uncertainty,
                energy_unit="meV",
                intensity_unit="counts",
                uncertainty_unit="counts",
            )
        )
    return ReducedDataset(role=SpectrumRole.SAMPLE, spectra=tuple(spectra))


def test_fitting_range_bounds_are_validated() -> None:
    with pytest.raises(ValueError, match="finite"):
        FittingRange(np.nan, 1.0)
    with pytest.raises(ValueError, match="must not exceed"):
        FittingRange(2.0, 1.0)


def test_uniform_initial_range_and_independent_override_are_inclusive() -> None:
    dataset = make_dataset(
        [[1.0, 2.0, 4.0, 3.0, 5.0], [2.0, 3.0, 5.0, 4.0, 6.0]],
        energies=[[-2.0, -1.0, 0.0, 1.0, 2.0]] * 2,
    )
    padding = detect_edge_padding(dataset)

    initial = FittingSelection.uniform(
        dataset,
        padding,
        lower_energy=-1.0,
        upper_energy=1.0,
    )
    overridden = initial.with_group_range(
        1,
        lower_energy=0.0,
        upper_energy=2.0,
    )

    assert initial.ranges == (FittingRange(-1.0, 1.0),) * 2
    assert overridden.ranges == (
        FittingRange(-1.0, 1.0),
        FittingRange(0.0, 2.0),
    )
    np.testing.assert_array_equal(
        overridden.in_range_mask(0), [False, True, True, True, False]
    )
    np.testing.assert_array_equal(
        overridden.in_range_mask(1), [False, False, True, True, True]
    )


def test_effective_mask_combines_invalid_auto_and_outside_but_not_review() -> None:
    dataset = make_dataset(
        [[-2.0] * 5 + [3.0, 4.0, 2.0] + [-1.0] * 2],
        energies=[np.arange(-5.0, 5.0).tolist()],
        uncertainties=[[0.2] * 6 + [0.0] + [0.2] * 3],
    )
    originals = tuple(
        array.copy()
        for array in (
            dataset.spectra[0].energy,
            dataset.spectra[0].intensity,
            dataset.spectra[0].uncertainty,
        )
    )
    padding = detect_edge_padding(dataset)
    assert padding.spectra[0].left.status is PaddingStatus.AUTO
    assert padding.spectra[0].right.status is PaddingStatus.REVIEW

    selection = FittingSelection.uniform(
        dataset,
        padding,
        lower_energy=-3.0,
        upper_energy=3.0,
    )

    np.testing.assert_array_equal(
        selection.excluded_mask(0),
        [True, True, True, True, True, False, True, False, False, True],
    )
    np.testing.assert_array_equal(
        selection.retained_mask(0),
        [False, False, False, False, False, True, False, True, True, False],
    )
    assert padding.spectra[0].review_mask[8]
    assert selection.retained_mask(0)[8]
    for current, original in zip(
        (
            dataset.spectra[0].energy,
            dataset.spectra[0].intensity,
            dataset.spectra[0].uncertainty,
        ),
        originals,
        strict=True,
    ):
        np.testing.assert_array_equal(current, original)
    for mask in (
        selection.invalid_mask(0),
        selection.in_range_mask(0),
        selection.excluded_mask(0),
        selection.retained_mask(0),
    ):
        assert not mask.flags.writeable


def test_range_count_and_padding_alignment_are_validated() -> None:
    dataset = make_dataset([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    padding = detect_edge_padding(dataset)

    with pytest.raises(ValueError, match="range count"):
        FittingSelection(
            dataset=dataset,
            padding=padding,
            ranges=(FittingRange(-1.0, 1.0),),
        )

    with pytest.raises(ValueError, match="padding result"):
        FittingSelection(
            dataset=dataset,
            padding=type(padding)(spectra=()),
            ranges=(FittingRange(-1.0, 1.0),) * 2,
        )


def test_range_without_usable_measured_points_fails_clearly() -> None:
    dataset = make_dataset([[1.0, 2.0, 4.0, 3.0]])
    padding = detect_edge_padding(dataset)

    with pytest.raises(ValueError, match="no usable measured points"):
        FittingSelection.uniform(
            dataset,
            padding,
            lower_energy=10.0,
            upper_energy=20.0,
        )


def test_group_index_outside_dataset_fails_clearly() -> None:
    dataset = make_dataset([[1.0, 2.0, 4.0, 3.0]])
    selection = FittingSelection.uniform(
        dataset,
        detect_edge_padding(dataset),
        lower_energy=-2.0,
        upper_energy=2.0,
    )

    with pytest.raises(IndexError, match="outside the dataset"):
        selection.retained_mask(1)


def test_future_source_independent_dataset_runs_milestone_2_operations() -> None:
    dataset = make_dataset(
        [[1.0, 2.0, 4.0, 3.0], [2.0, 3.0, 5.0, 4.0]],
        energies=[[-1.5, -0.5, 0.5, 1.5]] * 2,
    )
    q_assigned = dataset.assign_q_bins(QBins.from_edges([0.2, 0.6, 1.4]))
    padding = detect_edge_padding(q_assigned)
    selection = FittingSelection.uniform(
        q_assigned,
        padding,
        lower_energy=-0.5,
        upper_energy=0.5,
    )

    assert q_assigned.source_layout is None
    assert q_assigned.source_columns == ()
    assert q_assigned.q_bins is not None
    np.testing.assert_allclose(q_assigned.q_bins.q_values, [0.4, 1.0])
    np.testing.assert_array_equal(
        selection.retained_mask(0), [False, True, True, False]
    )
