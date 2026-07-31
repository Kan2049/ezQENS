"""Independent synthetic tests for boundary-padding detection."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from qensfit.domain import (
    DetectionConfidence,
    FormatDetectionResult,
    ImportedDataset,
    ReducedDataFormat,
    SourceColumnMetadata,
    Spectrum,
    SpectrumRole,
)
from qensfit.preprocessing import PaddingConfidence, detect_edge_padding


def make_dataset(
    intensities: Sequence[Sequence[float]],
    *,
    uncertainties: Sequence[Sequence[float]] | None = None,
    energies: Sequence[Sequence[float]] | None = None,
) -> ImportedDataset:
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
            Spectrum.from_imported_arrays(
                role=SpectrumRole.SAMPLE,
                group_index=index,
                group_label=f"group-{index + 1}",
                energy=energy,
                intensity=intensity,
                uncertainty=uncertainty,
                energy_unit="meV",
                intensity_unit="counts",
                uncertainty_unit="counts",
                source_row_numbers=tuple(range(1, intensity.size + 1)),
                source_columns=SourceColumnMetadata("x", "y", "yerr"),
                source_layout=ReducedDataFormat.DAVE_GROUP_BLOCKS,
            )
        )
    ordered = tuple(spectra)
    first_energy = ordered[0].energy
    shared = all(
        np.array_equal(spectrum.energy, first_energy, equal_nan=True)
        for spectrum in ordered[1:]
    )
    detection = FormatDetectionResult(
        proposed_format=ReducedDataFormat.DAVE_GROUP_BLOCKS,
        confidence=DetectionConfidence.HIGH,
        evidence=("Synthetic test dataset",),
        detected_required_columns=("x", "y", "yerr"),
        detected_extra_columns=(),
        detected_count=len(ordered),
    )
    return ImportedDataset(
        role=SpectrumRole.SAMPLE,
        source_layout=ReducedDataFormat.DAVE_GROUP_BLOCKS,
        spectra=ordered,
        source_reference="synthetic-padding.txt",
        diagnostics=(),
        detected_extra_columns=(),
        shared_energy_grid=shared,
        shared_energy_axis=first_energy if shared else None,
        format_detection=detection,
    )


def test_long_left_boundary_plateau_is_default_on() -> None:
    dataset = make_dataset([[-0.5] * 5 + [3.0, 4.0, 2.0, 5.0, 1.0]])

    result = detect_edge_padding(dataset).spectra[0]

    assert result.confidence is PaddingConfidence.HIGH
    assert result.left_masked_point_count == 5
    assert result.right_masked_point_count == 0
    assert result.recommended_default_on


def test_long_right_boundary_plateau_is_default_on() -> None:
    dataset = make_dataset([[1.0, 4.0, 2.0, 5.0, 3.0] + [-0.4] * 5])

    result = detect_edge_padding(dataset).spectra[0]

    assert result.right.confidence is PaddingConfidence.HIGH
    assert result.right_masked_point_count == 5
    np.testing.assert_array_equal(
        result.padding_mask,
        [False] * 5 + [True] * 5,
    )


def test_plateaus_on_both_boundaries_are_masked_independently() -> None:
    dataset = make_dataset(
        [[-0.5] * 5 + [2.0, 4.0, 3.0, 5.0, 2.0] + [-0.25] * 5]
    )

    result = detect_edge_padding(dataset).spectra[0]

    assert result.left_masked_point_count == 5
    assert result.right_masked_point_count == 5
    assert result.total_masked_point_count == 10


def test_boundary_lengths_are_independent_between_groups() -> None:
    dataset = make_dataset(
        [
            [-0.7] * 5 + [2.0, 3.0, 4.0, 5.0, 3.0, 2.0],
            [2.0, 3.0, 4.0, 5.0, 3.0] + [-0.7] * 6,
        ]
    )

    results = detect_edge_padding(dataset).spectra

    assert results[0].left_masked_point_count == 5
    assert results[0].right_masked_point_count == 0
    assert results[1].left_masked_point_count == 0
    assert results[1].right_masked_point_count == 6


def test_matching_sentinel_signature_is_recorded_across_groups() -> None:
    dataset = make_dataset(
        [
            [-1.2] * 5 + [3.0, 4.0, 2.0, 5.0],
            [2.0, 4.0, 3.0, 5.0] + [-1.2] * 5,
            [-1.2] * 5 + [4.0, 3.0, 5.0, 2.0],
        ]
    )

    results = detect_edge_padding(dataset).spectra

    assert all(
        "matching_boundary_signature_multiple_spectra"
        in (
            result.left.evidence_codes
            if result.left.run_length >= 5
            else result.right.evidence_codes
        )
        for result in results
    )


def test_two_point_run_is_promoted_by_long_cross_group_support() -> None:
    dataset = make_dataset(
        [
            [-2.0] * 5 + [3.0, 4.0, 2.0, 5.0],
            [3.0, 4.0, 2.0, 5.0] + [-2.0] * 5,
            [-2.0] * 2 + [3.0, 4.0, 2.0, 5.0, 3.0],
        ]
    )

    short_result = detect_edge_padding(dataset).spectra[2]

    assert short_result.left.confidence is PaddingConfidence.HIGH
    assert short_result.left.supporting_long_run_spectrum_count == 2
    assert short_result.left_masked_point_count == 2
    assert (
        "short_run_promoted_by_cross_group_support"
        in short_result.left.evidence_codes
    )


def test_isolated_negative_interior_point_is_not_masked() -> None:
    dataset = make_dataset([[1.0, 2.0, 3.0, -4.0, 2.5, 1.5]])

    result = detect_edge_padding(dataset).spectra[0]

    assert result.total_masked_point_count == 0
    assert not result.padding_mask[3]


def test_noisy_negative_physical_tail_is_not_masked() -> None:
    dataset = make_dataset([[-3.0, -2.9, -3.1, -2.8, 2.0, 3.0, 4.0]])

    result = detect_edge_padding(dataset).spectra[0]

    assert result.left.confidence is PaddingConfidence.NONE
    assert result.total_masked_point_count == 0


def test_constant_internal_segment_is_not_masked() -> None:
    dataset = make_dataset([[1.0, 2.0, -0.7, -0.7, -0.7, -0.7, 3.0, 2.0]])

    result = detect_edge_padding(dataset).spectra[0]

    assert not np.any(result.padding_mask)
    assert not np.any(result.suggested_padding_mask)


def test_spectrum_without_padding_has_empty_masks() -> None:
    dataset = make_dataset([[1.0, 2.0, 4.0, 3.0, 5.0, 2.5]])

    result = detect_edge_padding(dataset).spectra[0]

    assert result.confidence is PaddingConfidence.NONE
    assert result.total_masked_point_count == 0


def test_finite_positive_padding_uncertainty_is_supported() -> None:
    dataset = make_dataset(
        [[-0.8] * 5 + [3.0, 4.0, 2.0, 5.0]],
        uncertainties=[[0.4] * 9],
    )

    result = detect_edge_padding(dataset).spectra[0]

    assert result.left.plateau_uncertainty == pytest.approx(0.4)
    assert result.left.confidence is PaddingConfidence.HIGH


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
    assert result.left.confidence is PaddingConfidence.NONE
    assert result.total_masked_point_count == 0
    assert "boundary_invalid_value_separate" in result.left.evidence_codes


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
    assert result.left.confidence is PaddingConfidence.HIGH


def test_weak_transition_is_medium_suggestion_not_default_on() -> None:
    dataset = make_dataset(
        [[-1.0] * 5 + [-0.99, -0.97, -0.95, -0.92]],
        uncertainties=[[10.0] * 9],
    )

    result = detect_edge_padding(dataset).spectra[0]

    assert result.left.confidence is PaddingConfidence.MEDIUM
    assert np.count_nonzero(result.suggested_padding_mask) == 5
    assert result.total_masked_point_count == 0
    assert not result.recommended_default_on


def test_detection_leaves_all_original_arrays_unchanged() -> None:
    dataset = make_dataset([[-0.6] * 5 + [3.0, 4.0, 2.0, 5.0]])
    spectrum = dataset.spectra[0]
    original_energy = spectrum.energy.copy()
    original_intensity = spectrum.intensity.copy()
    original_uncertainty = spectrum.uncertainty.copy()
    original_invalid_masks = (
        spectrum.invalid_energy_mask.copy(),
        spectrum.invalid_intensity_mask.copy(),
        spectrum.invalid_uncertainty_mask.copy(),
    )

    detect_edge_padding(dataset)

    np.testing.assert_array_equal(spectrum.energy, original_energy)
    np.testing.assert_array_equal(spectrum.intensity, original_intensity)
    np.testing.assert_array_equal(spectrum.uncertainty, original_uncertainty)
    np.testing.assert_array_equal(
        spectrum.invalid_energy_mask,
        original_invalid_masks[0],
    )
    np.testing.assert_array_equal(
        spectrum.invalid_intensity_mask,
        original_invalid_masks[1],
    )
    np.testing.assert_array_equal(
        spectrum.invalid_uncertainty_mask,
        original_invalid_masks[2],
    )


def test_padding_masks_match_spectrum_length_and_are_read_only() -> None:
    dataset = make_dataset([[-0.3] * 5 + [3.0, 4.0, 2.0, 5.0]])

    result = detect_edge_padding(dataset).spectra[0]

    assert result.padding_mask.size == dataset.spectra[0].energy.size
    assert result.suggested_padding_mask.size == dataset.spectra[0].energy.size
    assert not result.padding_mask.flags.writeable
    assert not result.suggested_padding_mask.flags.writeable
    with pytest.raises(ValueError):
        result.padding_mask[0] = False


def test_plateau_point_adjacent_to_interior_is_included_exactly() -> None:
    dataset = make_dataset([[-0.5] * 5 + [3.0, 4.0, 2.0, 5.0]])

    result = detect_edge_padding(dataset).spectra[0]

    assert result.padding_mask[4]
    assert not result.padding_mask[5]
    assert result.left.adjacent_interior_index == 5
    assert result.left.energy_bounds == pytest.approx((-5.0, 0.0))


def test_structural_summary_is_privacy_safe() -> None:
    dataset = make_dataset([[-0.5] * 5 + [3.0, 4.0, 2.0, 5.0]])

    detection = detect_edge_padding(dataset)
    summary = detection.structural_summary(dataset)
    representation = repr(summary)

    assert summary.total_auto_padding_mask_count == 5
    assert summary.spectra[0].left_auto_masked_point_count == 5
    assert summary.spectra[0].derived_valid_energy_range[0] == pytest.approx(1.25)
    assert "array(" not in representation
    assert "plateau_intensity=" not in representation
    assert "plateau_uncertainty=" not in representation
