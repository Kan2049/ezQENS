"""Scientific behavior tests for boundary-padding v2."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from ezqens.domain import ReducedDataset, Spectrum, SpectrumRole
from ezqens.preprocessing import PaddingStatus, detect_edge_padding


def make_dataset(
    intensities: Sequence[Sequence[float]],
    *,
    uncertainties: Sequence[Sequence[float]] | None = None,
    energies: Sequence[Sequence[float]] | None = None,
) -> ReducedDataset:
    """Build an independently specified synthetic reduced dataset."""

    spectra: list[Spectrum] = []
    for index, intensity_values in enumerate(intensities):
        intensity = np.asarray(intensity_values, dtype=np.float64)
        energy = (
            np.linspace(-5.0, 5.0, intensity.size)
            if energies is None
            else np.asarray(energies[index], dtype=np.float64)
        )
        uncertainty = (
            np.full(intensity.size, 0.2, dtype=np.float64)
            if uncertainties is None
            else np.asarray(uncertainties[index], dtype=np.float64)
        )
        spectra.append(
            Spectrum(
                role=SpectrumRole.SAMPLE,
                group_index=index,
                group_label=f"group-{index + 1}",
                energy=energy,
                intensity=intensity,
                uncertainty=uncertainty,
                energy_unit="meV",
                intensity_unit="counts",
                uncertainty_unit="counts",
            )
        )
    return ReducedDataset(
        role=SpectrumRole.SAMPLE,
        spectra=tuple(spectra),
        source_reference="synthetic-padding.txt",
    )


def test_long_left_boundary_plateau_is_auto() -> None:
    dataset = make_dataset([[-0.5] * 5 + [3.0, 4.0, 2.0, 5.0, 1.0]])

    result = detect_edge_padding(dataset).spectra[0]

    assert result.left.status is PaddingStatus.AUTO
    assert result.status is PaddingStatus.AUTO
    assert result.left_auto_mask_count == 5
    assert result.right_auto_mask_count == 0


def test_fixed_long_run_remains_auto_when_spectrum_length_changes() -> None:
    dataset = make_dataset(
        [
            [-0.5] * 5 + [3.0, 4.0, 2.0, 5.0, 1.0],
            [-0.5] * 5 + np.linspace(3.0, 30.0, 50).tolist(),
        ]
    )

    results = detect_edge_padding(dataset).spectra

    assert [result.left.status for result in results] == [
        PaddingStatus.AUTO,
        PaddingStatus.AUTO,
    ]
    assert [result.left.run_length for result in results] == [5, 5]
    assert [result.left_auto_mask_count for result in results] == [5, 5]


def test_long_right_boundary_plateau_is_auto() -> None:
    dataset = make_dataset([[1.0, 4.0, 2.0, 5.0, 3.0] + [-0.4] * 5])

    result = detect_edge_padding(dataset).spectra[0]

    assert result.right.status is PaddingStatus.AUTO
    np.testing.assert_array_equal(result.auto_mask, [False] * 5 + [True] * 5)


def test_plateaus_on_both_boundaries_are_masked_independently() -> None:
    dataset = make_dataset([[-0.5] * 5 + [2.0, 4.0, 3.0, 5.0, 2.0] + [-0.25] * 5])

    result = detect_edge_padding(dataset).spectra[0]

    assert result.left_auto_mask_count == 5
    assert result.right_auto_mask_count == 5
    assert result.total_auto_mask_count == 10


def test_boundary_lengths_are_independent_between_groups() -> None:
    dataset = make_dataset(
        [
            [-0.7] * 5 + [2.0, 3.0, 4.0, 5.0, 3.0, 2.0],
            [2.0, 3.0, 4.0, 5.0, 3.0] + [-0.7] * 6,
        ]
    )

    results = detect_edge_padding(dataset).spectra

    assert results[0].left_auto_mask_count == 5
    assert results[0].right_auto_mask_count == 0
    assert results[1].left_auto_mask_count == 0
    assert results[1].right_auto_mask_count == 6


def test_two_point_run_is_promoted_by_long_cross_spectrum_support() -> None:
    dataset = make_dataset(
        [
            [-2.0] * 5 + [3.0, 4.0, 2.0, 5.0],
            [3.0, 4.0, 2.0, 5.0] + [-2.0] * 5,
            [-2.0] * 2 + [3.0, 4.0, 2.0, 5.0, 3.0],
        ]
    )

    short_result = detect_edge_padding(dataset).spectra[2]

    assert short_result.left.status is PaddingStatus.AUTO
    assert short_result.left.reason == "short_run_with_cross_spectrum_support"
    assert short_result.left_auto_mask_count == 2


def test_isolated_negative_interior_point_is_not_masked() -> None:
    dataset = make_dataset([[1.0, 2.0, 3.0, -4.0, 2.5, 1.5]])

    result = detect_edge_padding(dataset).spectra[0]

    assert result.total_auto_mask_count == 0
    assert not result.auto_mask[3]


def test_noisy_negative_physical_tail_is_not_masked() -> None:
    dataset = make_dataset([[-3.0, -2.9, -3.1, -2.8, 2.0, 3.0, 4.0]])

    result = detect_edge_padding(dataset).spectra[0]

    assert result.left.status is PaddingStatus.NONE
    assert result.total_auto_mask_count == 0


def test_constant_internal_segment_is_not_masked() -> None:
    dataset = make_dataset([[1.0, 2.0, -0.7, -0.7, -0.7, -0.7, 3.0, 2.0]])

    result = detect_edge_padding(dataset).spectra[0]

    assert not np.any(result.auto_mask)
    assert not np.any(result.review_mask)


def test_spectrum_without_padding_has_empty_masks() -> None:
    dataset = make_dataset([[1.0, 2.0, 4.0, 3.0, 5.0, 2.5]])

    result = detect_edge_padding(dataset).spectra[0]

    assert result.status is PaddingStatus.NONE
    assert result.total_auto_mask_count == 0
    assert result.total_review_mask_count == 0


def test_finite_positive_padding_uncertainty_is_supported() -> None:
    dataset = make_dataset(
        [[-0.8] * 5 + [3.0, 4.0, 2.0, 5.0]],
        uncertainties=[[0.4] * 9],
    )

    result = detect_edge_padding(dataset).spectra[0]

    assert result.left.status is PaddingStatus.AUTO


def test_invalid_uncertainty_remains_separate_from_padding() -> None:
    dataset = make_dataset(
        [[-0.8] * 5 + [3.0, 4.0, 2.0, 5.0]],
        uncertainties=[[0.0] * 5 + [0.2] * 4],
    )

    result = detect_edge_padding(dataset).spectra[0]

    np.testing.assert_array_equal(
        dataset.spectra[0].invalid_uncertainty_mask,
        [True] * 5 + [False] * 4,
    )
    assert result.left.status is PaddingStatus.NONE
    assert result.left.reason == "boundary_values_invalid"
    assert result.total_auto_mask_count == 0


def test_near_equal_plateau_values_use_declared_tolerance() -> None:
    dataset = make_dataset(
        [
            [
                -1.0,
                -1.0 + 1.0e-9,
                -1.0 - 1.0e-9,
                -1.0 + 5.0e-10,
                -1.0,
                3.0,
                4.0,
                2.0,
                5.0,
            ]
        ],
        uncertainties=[
            [
                0.3,
                0.3 + 1.0e-9,
                0.3 - 1.0e-9,
                0.3,
                0.3,
                0.2,
                0.2,
                0.2,
                0.2,
            ]
        ],
    )

    result = detect_edge_padding(dataset).spectra[0]

    assert result.left.run_length == 5
    assert result.left.status is PaddingStatus.AUTO


def test_weak_transition_is_review_not_auto() -> None:
    dataset = make_dataset(
        [[-1.0] * 5 + [-0.99, -0.97, -0.95, -0.92]],
        uncertainties=[[10.0] * 9],
    )

    result = detect_edge_padding(dataset).spectra[0]

    assert result.left.status is PaddingStatus.REVIEW
    assert result.left.reason == "transition_not_clear"
    assert np.count_nonzero(result.review_mask) == 5
    assert result.total_auto_mask_count == 0
    assert result.needs_review


def test_short_clear_run_without_cross_spectrum_support_is_review() -> None:
    dataset = make_dataset([[-1.0] * 4 + [3.0, 4.0, 2.0, 5.0]])

    result = detect_edge_padding(dataset).spectra[0]

    assert result.left.status is PaddingStatus.REVIEW
    assert result.left.reason == "short_run_requires_review"
    assert result.total_review_mask_count == 4


@pytest.mark.parametrize("plateau", [-2.0, 0.0, 2.0])
def test_padding_classification_does_not_depend_on_intensity_sign(
    plateau: float,
) -> None:
    dataset = make_dataset(
        [[plateau] * 5 + [plateau + 4.0, plateau + 3.0, plateau + 5.0]]
    )

    result = detect_edge_padding(dataset).spectra[0]

    assert result.left.status is PaddingStatus.AUTO
    assert result.left_auto_mask_count == 5


def test_detection_leaves_all_original_arrays_and_invalid_masks_unchanged() -> None:
    dataset = make_dataset([[-0.6] * 5 + [3.0, 4.0, 2.0, 5.0]])
    spectrum = dataset.spectra[0]
    originals = tuple(
        array.copy()
        for array in (
            spectrum.energy,
            spectrum.intensity,
            spectrum.uncertainty,
            spectrum.invalid_energy_mask,
            spectrum.invalid_intensity_mask,
            spectrum.invalid_uncertainty_mask,
        )
    )

    detect_edge_padding(dataset)

    for current, original in zip(
        (
            spectrum.energy,
            spectrum.intensity,
            spectrum.uncertainty,
            spectrum.invalid_energy_mask,
            spectrum.invalid_intensity_mask,
            spectrum.invalid_uncertainty_mask,
        ),
        originals,
        strict=True,
    ):
        np.testing.assert_array_equal(current, original)


def test_masks_match_spectrum_length_are_read_only_and_do_not_overlap() -> None:
    dataset = make_dataset([[-0.3] * 5 + [3.0, 4.0, 2.0, 5.0]])

    result = detect_edge_padding(dataset).spectra[0]

    assert result.auto_mask.size == dataset.spectra[0].energy.size
    assert result.review_mask.size == dataset.spectra[0].energy.size
    assert not result.auto_mask.flags.writeable
    assert not result.review_mask.flags.writeable
    assert not np.any(result.auto_mask & result.review_mask)
    with pytest.raises(ValueError):
        result.auto_mask[0] = False


def test_plateau_point_adjacent_to_interior_is_included_exactly() -> None:
    dataset = make_dataset([[-0.5] * 5 + [3.0, 4.0, 2.0, 5.0]])

    result = detect_edge_padding(dataset).spectra[0]

    assert result.auto_mask[4]
    assert not result.auto_mask[5]
    assert result.left.energy_bounds == pytest.approx((-5.0, 0.0))


def test_structural_summary_is_privacy_safe() -> None:
    dataset = make_dataset([[-0.5] * 5 + [3.0, 4.0, 2.0, 5.0]])

    detection = detect_edge_padding(dataset)
    summary = detection.structural_summary(dataset)
    representation = repr(summary)

    assert summary.total_auto_mask_count == 5
    assert summary.spectra[0].left_auto_mask_count == 5
    assert summary.spectra[0].retained_energy_bounds[0] == pytest.approx(1.25)
    assert "array(" not in representation
    assert "intensity=" not in representation
    assert "uncertainty=" not in representation
