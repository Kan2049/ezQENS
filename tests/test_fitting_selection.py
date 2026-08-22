"""Tests for per-group fitting ranges and derived point masks."""

from collections.abc import Sequence
from dataclasses import replace

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


def test_manual_exclusion_defaults_to_read_only_all_false_masks() -> None:
    dataset = make_dataset(
        [[1.0, 2.0, 4.0, 3.0], [2.0, 3.0, 5.0, 4.0]],
    )
    padding = detect_edge_padding(dataset)

    selection = FittingSelection(
        dataset=dataset,
        padding=padding,
        ranges=(FittingRange(-2.0, 2.0),) * 2,
    )

    for group_index in range(2):
        manual = selection.manual_exclusion_mask(group_index)
        auto_reinclusion = selection.manual_auto_reinclusion_mask(group_index)
        assert manual.dtype == np.bool_
        assert manual.size == dataset.spectra[group_index].energy.size
        assert not manual.flags.writeable
        assert not np.any(manual)
        assert auto_reinclusion.dtype == np.bool_
        assert auto_reinclusion.size == dataset.spectra[group_index].energy.size
        assert not auto_reinclusion.flags.writeable
        assert not np.any(auto_reinclusion)


def test_manual_exclusion_composes_without_overriding_other_mask_states() -> None:
    dataset = make_dataset(
        [[-2.0] * 5 + [3.0, 4.0, 2.0] + [-1.0] * 2],
        energies=[np.arange(-5.0, 5.0).tolist()],
        uncertainties=[[0.2] * 6 + [0.0] + [0.2] * 3],
    )
    padding = detect_edge_padding(dataset)
    selection = FittingSelection.uniform(
        dataset,
        padding,
        lower_energy=-3.0,
        upper_energy=3.0,
    )
    manual = np.zeros(10, dtype=np.bool_)
    manual[8] = True

    updated = selection.with_group_manual_exclusion(0, manual)

    assert selection.retained_mask(0)[8]
    assert padding.spectra[0].review_mask[8]
    assert updated.excluded_mask(0)[8]
    assert updated.excluded_mask(0)[0]  # AUTO remains excluded.
    assert updated.excluded_mask(0)[6]  # Invalid uncertainty remains excluded.
    assert updated.excluded_mask(0)[9]  # Outside the range remains excluded.
    assert not np.any(selection.manual_exclusion_mask(0))
    assert not updated.manual_exclusion_mask(0).flags.writeable
    cleared = updated.clear_group_manual_exclusion(0)
    assert cleared.retained_mask(0)[8]
    assert not np.any(cleared.manual_exclusion_mask(0))


def test_manual_exclusion_is_boolean_length_validated_and_cannot_empty_group() -> None:
    dataset = make_dataset([[1.0, 2.0, 4.0, 3.0]])
    selection = FittingSelection.uniform(
        dataset,
        detect_edge_padding(dataset),
        lower_energy=-2.0,
        upper_energy=2.0,
    )

    with pytest.raises(ValueError, match="boolean vectors"):
        selection.with_group_manual_exclusion(0, [0, 1, 0, 0])
    with pytest.raises(ValueError, match="spectrum length"):
        selection.with_group_manual_exclusion(
            0,
            np.zeros(3, dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="no usable measured points"):
        selection.with_group_manual_exclusion(
            0,
            np.ones(4, dtype=np.bool_),
        )


def test_auto_reinclusion_is_per_group_read_only_and_preserves_proposal() -> None:
    dataset = make_dataset(
        [[-2.0] * 5 + [3.0, 4.0, 2.0], [-3.0] * 5 + [2.0, 5.0, 3.0]],
    )
    originals = tuple(spectrum.intensity.copy() for spectrum in dataset.spectra)
    padding = detect_edge_padding(dataset)
    original_auto = tuple(item.auto_mask.copy() for item in padding.spectra)
    selection = FittingSelection.uniform(
        dataset,
        padding,
        lower_energy=-4.0,
        upper_energy=3.0,
    )
    group_zero = np.zeros(8, dtype=np.bool_)
    group_zero[4] = True
    one_reincluded = selection.with_group_manual_auto_reinclusion(0, group_zero)
    group_one = np.zeros(8, dtype=np.bool_)
    group_one[[3, 4]] = True
    independently_reincluded = one_reincluded.with_group_manual_auto_reinclusion(
        1,
        group_one,
    )

    assert not selection.retained_mask(0)[4]
    assert one_reincluded.retained_mask(0)[4]
    np.testing.assert_array_equal(
        one_reincluded.manual_auto_reinclusion_mask(1),
        np.zeros(8, dtype=np.bool_),
    )
    assert independently_reincluded.retained_mask(1)[3]
    assert independently_reincluded.retained_mask(1)[4]
    assert not independently_reincluded.manual_auto_reinclusion_mask(0).flags.writeable
    for spectrum, original, padding_result, auto in zip(
        dataset.spectra,
        originals,
        padding.spectra,
        original_auto,
        strict=True,
    ):
        np.testing.assert_array_equal(spectrum.intensity, original)
        np.testing.assert_array_equal(padding_result.auto_mask, auto)


def test_invalid_measurement_remains_excluded_when_auto_reinclusion_is_requested() -> (
    None
):
    dataset = make_dataset(
        [[-2.0] * 5 + [3.0, 4.0, 2.0]],
        uncertainties=[[0.2] * 5 + [0.0, 0.2, 0.2]],
    )
    detected = detect_edge_padding(dataset)
    auto = detected.spectra[0].auto_mask.copy()
    auto[5] = True
    padding = replace(
        detected,
        spectra=(replace(detected.spectra[0], auto_mask=auto),),
    )
    selection = FittingSelection.uniform(
        dataset,
        padding,
        lower_energy=-4.0,
        upper_energy=3.0,
    )
    reinclusion = np.zeros(8, dtype=np.bool_)
    reinclusion[5] = True

    updated = selection.with_group_manual_auto_reinclusion(0, reinclusion)

    assert updated.manual_auto_reinclusion_mask(0)[5]
    assert updated.invalid_mask(0)[5]
    assert updated.excluded_mask(0)[5]
    assert not updated.retained_mask(0)[5]


def test_manual_exclusion_and_auto_reinclusion_conflict_is_rejected() -> None:
    dataset = make_dataset([[-2.0] * 5 + [3.0, 4.0, 2.0]])
    selection = FittingSelection.uniform(
        dataset,
        detect_edge_padding(dataset),
        lower_energy=-4.0,
        upper_energy=3.0,
    )
    reinclusion = np.zeros(8, dtype=np.bool_)
    reinclusion[4] = True
    updated = selection.with_group_manual_auto_reinclusion(0, reinclusion)
    manual = np.zeros(8, dtype=np.bool_)
    manual[4] = True

    with pytest.raises(ValueError, match="mutually exclusive"):
        updated.with_group_manual_exclusion(0, manual)

    manual[4] = False
    manual[6] = True
    ordinary_exclusion = updated.with_group_manual_exclusion(0, manual)
    assert ordinary_exclusion.retained_mask(0)[4]
    assert ordinary_exclusion.excluded_mask(0)[6]


def test_auto_reinclusion_validation_and_clear_workflows() -> None:
    dataset = make_dataset([[-2.0] * 5 + [3.0, 4.0, 2.0]])
    padding = detect_edge_padding(dataset)
    selection = FittingSelection.uniform(
        dataset,
        padding,
        lower_energy=-4.0,
        upper_energy=3.0,
    )
    reinclusion = np.zeros(8, dtype=np.bool_)
    reinclusion[4] = True
    manual = np.zeros(8, dtype=np.bool_)
    manual[6] = True
    edited = selection.with_group_manual_auto_reinclusion(
        0,
        reinclusion,
    ).with_group_manual_exclusion(0, manual)

    clear_manual = edited.clear_group_manual_exclusion(0)
    assert clear_manual.retained_mask(0)[4]
    assert clear_manual.retained_mask(0)[6]
    reset_to_auto = clear_manual.clear_group_manual_auto_reinclusion(0)
    np.testing.assert_array_equal(
        reset_to_auto.retained_mask(0), selection.retained_mask(0)
    )
    clear_reversible = edited.clear_group_manual_exclusion(
        0
    ).with_group_manual_auto_reinclusion(0, padding.spectra[0].auto_mask)
    assert np.all(clear_reversible.retained_mask(0))

    with pytest.raises(ValueError, match="boolean vectors"):
        selection.with_group_manual_auto_reinclusion(0, [0, 0, 0, 0, 1, 0, 0, 0])
    with pytest.raises(ValueError, match="spectrum length"):
        selection.with_group_manual_auto_reinclusion(
            0,
            np.zeros(7, dtype=np.bool_),
        )
    outside_auto = np.zeros(8, dtype=np.bool_)
    outside_auto[6] = True
    with pytest.raises(ValueError, match="only target AUTO"):
        selection.with_group_manual_auto_reinclusion(0, outside_auto)


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
