"""Scientific tests for measured-resolution preparation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest

from ezqens.domain import QBins, ReducedDataset, Spectrum, SpectrumRole
from ezqens.preprocessing import FittingSelection, PaddingStatus, detect_edge_padding
from ezqens.resolution import (
    NORMALIZATION_METHOD,
    Q_MATCH_ATOL,
    PreparedResolution,
    ResolutionPreparationError,
    ResolutionSupport,
    ResolutionSupportSource,
    prepare_measured_resolution,
)


def make_dataset(
    role: SpectrumRole,
    intensities: Sequence[npt.ArrayLike],
    *,
    energies: Sequence[npt.ArrayLike] | None = None,
    uncertainties: Sequence[npt.ArrayLike] | None = None,
    q_bins: QBins | None = None,
) -> ReducedDataset:
    spectra: list[Spectrum] = []
    for index, values in enumerate(intensities):
        intensity = np.asarray(values, dtype=np.float64)
        energy = (
            np.linspace(-2.0, 2.0, intensity.size)
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
                role=role,
                group_index=index,
                group_label=f"{role.value}-{index}",
                energy=energy,
                intensity=intensity,
                uncertainty=uncertainty,
                energy_unit="meV",
                intensity_unit="counts",
                uncertainty_unit="counts",
            )
        )
    dataset = ReducedDataset(role=role, spectra=tuple(spectra))
    if q_bins is not None:
        dataset = dataset.assign_q_bins(q_bins)
    return dataset


def matched_pair(
    *,
    resolution_intensity: npt.ArrayLike = (1.0, 3.0, 5.0, 3.0, 1.0),
    energy: npt.ArrayLike = (-2.0, -1.0, 0.0, 1.0, 2.0),
    resolution_uncertainty: npt.ArrayLike | None = None,
) -> tuple[ReducedDataset, ReducedDataset]:
    q_bins = QBins.from_edges([0.4, 0.6])
    energy_values = np.asarray(energy, dtype=np.float64)
    sample_intensity = np.linspace(1.0, 2.0, energy_values.size)
    sample = make_dataset(
        SpectrumRole.SAMPLE,
        [sample_intensity],
        energies=[energy_values],
        q_bins=q_bins,
    )
    resolution = make_dataset(
        SpectrumRole.RESOLUTION,
        [resolution_intensity],
        energies=[energy_values],
        uncertainties=None
        if resolution_uncertainty is None
        else [resolution_uncertainty],
        q_bins=q_bins,
    )
    return sample, resolution


def diagnostic_code(error: ResolutionPreparationError) -> str:
    return error.diagnostics[0].code


def test_matching_q_bins_associate_successfully_with_small_float_tolerance() -> None:
    sample, resolution = matched_pair()
    resolution_q = QBins(
        q_values=np.array([0.5 + Q_MATCH_ATOL / 2.0]),
        edges=np.array([0.4, 0.6 + Q_MATCH_ATOL / 2.0]),
    )
    resolution = resolution.assign_q_bins(resolution_q)

    prepared = prepare_measured_resolution(sample, resolution)

    assert isinstance(prepared, PreparedResolution)
    assert prepared.q_value(0) == pytest.approx(0.5)
    assert prepared.spectra[0].source_spectrum is resolution.spectra[0]


def test_q_count_mismatch_fails_without_repair() -> None:
    sample = make_dataset(
        SpectrumRole.SAMPLE,
        [[1.0, 2.0, 1.0]],
        q_bins=QBins.from_q_values([0.5]),
    )
    resolution = make_dataset(
        SpectrumRole.RESOLUTION,
        [[1.0, 2.0, 1.0], [1.0, 3.0, 1.0]],
        q_bins=QBins.from_q_values([0.5, 0.8]),
    )

    with pytest.raises(ResolutionPreparationError) as caught:
        prepare_measured_resolution(sample, resolution)

    assert diagnostic_code(caught.value) == "q_group_count_mismatch"


def test_q_order_or_value_mismatch_fails_without_nearest_q_repair() -> None:
    sample = make_dataset(
        SpectrumRole.SAMPLE,
        [[1.0, 2.0, 1.0], [1.0, 3.0, 1.0]],
        q_bins=QBins.from_q_values([0.5, 0.8]),
    )
    resolution = make_dataset(
        SpectrumRole.RESOLUTION,
        [[1.0, 2.0, 1.0], [1.0, 3.0, 1.0]],
        q_bins=QBins.from_q_values([0.8, 0.5]),
    )

    with pytest.raises(ResolutionPreparationError) as caught:
        prepare_measured_resolution(sample, resolution)

    assert diagnostic_code(caught.value) == "q_value_mismatch"


def test_known_q_edge_mismatch_fails_even_when_representatives_match() -> None:
    sample = make_dataset(
        SpectrumRole.SAMPLE,
        [[1.0, 2.0, 1.0]],
        q_bins=QBins(q_values=np.array([0.5]), edges=np.array([0.4, 0.6])),
    )
    resolution = make_dataset(
        SpectrumRole.RESOLUTION,
        [[1.0, 2.0, 1.0]],
        q_bins=QBins(q_values=np.array([0.5]), edges=np.array([0.39, 0.61])),
    )

    with pytest.raises(ResolutionPreparationError) as caught:
        prepare_measured_resolution(sample, resolution)

    assert diagnostic_code(caught.value) == "q_edge_mismatch"


def test_resolution_padding_is_detected_independently_from_sample() -> None:
    energy = np.arange(-5.0, 5.0)
    q_bins = QBins.from_q_values([0.5])
    sample = make_dataset(
        SpectrumRole.SAMPLE,
        [[-2.0] * 5 + [2.0, 3.0, 4.0, 2.0, 1.0]],
        energies=[energy],
        q_bins=q_bins,
    )
    resolution = make_dataset(
        SpectrumRole.RESOLUTION,
        [[1.0, 2.0, 4.0, 3.0, 2.0] + [-1.0] * 5],
        energies=[energy],
        q_bins=q_bins,
    )

    prepared = prepare_measured_resolution(sample, resolution)
    sample_padding = prepared.sample_padding.spectra[0]
    resolution_padding = prepared.resolution_padding.spectra[0]

    assert sample_padding.left.status is PaddingStatus.AUTO
    assert sample_padding.right.status is PaddingStatus.NONE
    assert resolution_padding.left.status is PaddingStatus.NONE
    assert resolution_padding.right.status is PaddingStatus.AUTO
    assert not np.array_equal(sample_padding.auto_mask, resolution_padding.auto_mask)


def test_padding_comparison_is_consistent_or_warns_without_mask_changes() -> None:
    energy = np.arange(-5.0, 5.0)
    q_bins = QBins.from_q_values([0.5])
    matching_sample = make_dataset(
        SpectrumRole.SAMPLE,
        [[-2.0] * 5 + [2.0, 3.0, 4.0, 2.0, 1.0]],
        energies=[energy],
        q_bins=q_bins,
    )
    matching_resolution = make_dataset(
        SpectrumRole.RESOLUTION,
        [[-1.0] * 5 + [1.0, 2.0, 4.0, 2.0, 1.0]],
        energies=[energy],
        q_bins=q_bins,
    )
    matching = prepare_measured_resolution(matching_sample, matching_resolution)
    assert matching.padding_comparisons[0].is_consistent
    assert matching.diagnostics == ()

    different_resolution = make_dataset(
        SpectrumRole.RESOLUTION,
        [[1.0, 2.0, 4.0, 3.0, 2.0] + [-1.0] * 5],
        energies=[energy],
        q_bins=q_bins,
    )
    different = prepare_measured_resolution(matching_sample, different_resolution)

    assert not different.padding_comparisons[0].is_consistent
    assert [item.code for item in different.diagnostics] == [
        "padding_boundary_mismatch"
    ]
    np.testing.assert_array_equal(
        different.sample_padding.spectra[0].auto_mask,
        [True] * 5 + [False] * 5,
    )
    np.testing.assert_array_equal(
        different.resolution_padding.spectra[0].auto_mask,
        [False] * 5 + [True] * 5,
    )


def test_auto_padding_is_zero_on_kernel_grid_and_excluded_from_area() -> None:
    energy = np.arange(-5.0, 5.0)
    sample, _ = matched_pair(
        energy=energy,
        resolution_intensity=[1.0] * 10,
    )
    resolution = make_dataset(
        SpectrumRole.RESOLUTION,
        [[100.0] * 5 + [1.0, 3.0, 4.0, 2.0, 1.0]],
        energies=[energy],
        q_bins=sample.q_bins,
    )
    original_intensity = resolution.spectra[0].intensity.copy()

    prepared = prepare_measured_resolution(sample, resolution).spectra[0]

    assert prepared.padding.left.status is PaddingStatus.AUTO
    assert prepared.auto_padding_applied
    expected_area = float(np.trapezoid([1.0, 3.0, 4.0, 2.0, 1.0], energy[5:]))
    assert prepared.normalization_integral == pytest.approx(expected_area)
    np.testing.assert_array_equal(
        prepared.normalized_intensity_on_source_grid[:5], np.zeros(5)
    )
    np.testing.assert_array_equal(resolution.spectra[0].intensity, original_intensity)


def test_explicit_auto_disable_restores_valid_points_without_mutating_source() -> None:
    energy = np.arange(-5.0, 5.0)
    sample, _ = matched_pair(energy=energy, resolution_intensity=[1.0] * 10)
    resolution = make_dataset(
        SpectrumRole.RESOLUTION,
        [[2.0] * 5 + [3.0, 5.0, 4.0, 3.0, 1.0]],
        energies=[energy],
        q_bins=sample.q_bins,
    )
    source = resolution.spectra[0]
    originals = tuple(
        value.copy() for value in (source.energy, source.intensity, source.uncertainty)
    )

    default = prepare_measured_resolution(sample, resolution).spectra[0]
    restored = prepare_measured_resolution(
        sample,
        resolution,
        apply_auto_padding={0: False},
    ).spectra[0]

    assert default.auto_padding_applied
    assert not restored.auto_padding_applied
    assert default.support == restored.support
    assert np.count_nonzero(default.accepted_mask) == 5
    assert np.count_nonzero(restored.accepted_mask) == 10
    np.testing.assert_array_equal(restored.energy, source.energy)
    assert restored.normalization_integral == pytest.approx(
        float(np.trapezoid(source.intensity, source.energy))
    )
    for current, original in zip(
        (source.energy, source.intensity, source.uncertainty),
        originals,
        strict=True,
    ):
        np.testing.assert_array_equal(current, original)


def test_support_change_does_not_silently_disable_auto_padding() -> None:
    energy = np.arange(-5.0, 5.0)
    sample, _ = matched_pair(energy=energy, resolution_intensity=[1.0] * 10)
    resolution = make_dataset(
        SpectrumRole.RESOLUTION,
        [[2.0] * 5 + [3.0, 5.0, 4.0, 3.0, 1.0]],
        energies=[energy],
        q_bins=sample.q_bins,
    )

    prepared = prepare_measured_resolution(
        sample,
        resolution,
        support_overrides={0: ResolutionSupport(-5.0, 4.0)},
    ).spectra[0]

    assert prepared.auto_padding_applied
    assert np.all(~prepared.accepted_mask[:5])
    assert np.all(prepared.accepted_mask[5:])


def test_review_padding_is_retained_by_default() -> None:
    energy = np.arange(-4.0, 4.0)
    q_bins = QBins.from_q_values([0.5])
    sample = make_dataset(
        SpectrumRole.SAMPLE,
        [[1.0, 2.0, 4.0, 3.0, 2.0, 1.0, 0.8, 0.5]],
        energies=[energy],
        q_bins=q_bins,
    )
    resolution = make_dataset(
        SpectrumRole.RESOLUTION,
        [[0.5] * 3 + [3.0, 5.0, 3.0, 1.0, 0.5]],
        energies=[energy],
        q_bins=q_bins,
    )

    prepared = prepare_measured_resolution(sample, resolution).spectra[0]

    assert prepared.padding.left.status is PaddingStatus.REVIEW
    assert np.all(prepared.accepted_mask[:3])


def test_nonuniform_grid_normalizes_to_unit_integrated_area_not_peak_height() -> None:
    energy = [-2.0, -0.8, -0.1, 0.35, 1.4, 2.7]
    sample, resolution = matched_pair(
        energy=energy,
        resolution_intensity=[0.2, 1.0, 4.0, 3.0, 0.7, 0.1],
    )

    prepared = prepare_measured_resolution(sample, resolution).spectra[0]

    assert prepared.normalization_method == NORMALIZATION_METHOD
    assert prepared.normalized_integral == pytest.approx(1.0, abs=1.0e-12)
    assert np.max(prepared.normalized_intensity) != pytest.approx(1.0)
    np.testing.assert_array_equal(prepared.energy, np.asarray(energy))
    assert not prepared.energy.flags.writeable
    assert not prepared.normalized_intensity.flags.writeable


@pytest.mark.parametrize(
    ("intensity", "code"),
    [
        ([0.0, 0.0, 0.0], "resolution_integral_nonpositive"),
        ([-1.0, -2.0, -1.0], "resolution_integral_nonpositive"),
        ([1.0e308, 1.0e308, 1.0e308], "resolution_integral_nonfinite"),
    ],
)
def test_nonpositive_or_nonfinite_normalization_area_fails(
    intensity: list[float], code: str
) -> None:
    q_bins = QBins.from_q_values([0.5])
    sample = make_dataset(
        SpectrumRole.SAMPLE,
        [[1.0, 2.0, 1.0]],
        energies=[[0.0, 1.0, 2.0]],
        q_bins=q_bins,
    )
    resolution = make_dataset(
        SpectrumRole.RESOLUTION,
        [intensity],
        energies=[[0.0, 1.0, 2.0]],
        q_bins=q_bins,
    )

    with pytest.raises(ResolutionPreparationError) as caught:
        prepare_measured_resolution(sample, resolution)

    assert diagnostic_code(caught.value) == code


def test_preparation_preserves_sources_and_derives_normalized_uncertainty() -> None:
    sample, resolution = matched_pair(resolution_uncertainty=[0.1, 0.2, 0.0, 0.2, 0.1])
    source = resolution.spectra[0]
    originals = tuple(
        value.copy()
        for value in (
            source.energy,
            source.intensity,
            source.uncertainty,
            source.invalid_energy_mask,
            source.invalid_intensity_mask,
            source.invalid_uncertainty_mask,
        )
    )

    prepared = prepare_measured_resolution(sample, resolution).spectra[0]

    assert prepared.source_spectrum is source
    assert prepared.normalized_uncertainty[0] == pytest.approx(
        source.uncertainty[0] * prepared.normalization_factor
    )
    assert prepared.normalized_uncertainty[2] == 0.0
    assert not prepared.normalized_uncertainty.flags.writeable
    assert [item.code for item in prepared.diagnostics] == [
        "invalid_resolution_uncertainty_retained"
    ]
    for current, original in zip(
        (
            source.energy,
            source.intensity,
            source.uncertainty,
            source.invalid_energy_mask,
            source.invalid_intensity_mask,
            source.invalid_uncertainty_mask,
        ),
        originals,
        strict=True,
    ):
        np.testing.assert_array_equal(current, original)


def test_explicit_support_is_distinct_from_sample_fitting_range() -> None:
    energy = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    sample, resolution = matched_pair(
        energy=energy,
        resolution_intensity=[0.2, 0.5, 2.0, 4.0, 2.0, 0.5, 0.2],
    )
    sample_selection = FittingSelection.uniform(
        sample,
        detect_edge_padding(sample),
        lower_energy=-1.0,
        upper_energy=1.0,
    )

    default = prepare_measured_resolution(sample_selection.dataset, resolution)
    overridden = prepare_measured_resolution(
        sample_selection.dataset,
        resolution,
        support_overrides={0: ResolutionSupport(-2.0, 2.0)},
    )

    assert default.spectra[0].support == ResolutionSupport(
        -3.0,
        3.0,
        ResolutionSupportSource.DEFAULT_VALID_DATA,
    )
    assert overridden.spectra[0].support == ResolutionSupport(-2.0, 2.0)
    np.testing.assert_array_equal(
        overridden.spectra[0].energy,
        [-2.0, -1.0, 0.0, 1.0, 2.0],
    )
    assert sample_selection.ranges[0].lower_energy == -1.0


@pytest.mark.parametrize(
    "energy",
    [
        [-1.0, 0.0, 0.0, 1.0],
        [-1.0, 0.5, 0.25, 1.0],
    ],
)
def test_ambiguous_or_unordered_energy_grid_fails_without_reordering(
    energy: list[float],
) -> None:
    q_bins = QBins.from_q_values([0.5])
    sample = make_dataset(
        SpectrumRole.SAMPLE,
        [[1.0, 2.0, 2.0, 1.0]],
        energies=[energy],
        q_bins=q_bins,
    )
    resolution = make_dataset(
        SpectrumRole.RESOLUTION,
        [[1.0, 3.0, 2.0, 1.0]],
        energies=[energy],
        q_bins=q_bins,
    )

    with pytest.raises(ResolutionPreparationError) as caught:
        prepare_measured_resolution(sample, resolution)

    assert diagnostic_code(caught.value) == "resolution_energy_not_strictly_increasing"


def test_internal_invalid_intensity_hole_fails_without_trapezoid_bridge() -> None:
    sample, resolution = matched_pair(
        energy=[-0.2, -0.1, 0.0, 0.1, 0.2],
        resolution_intensity=[1.0, 10.0, np.nan, 9.0, 1.0],
    )

    with pytest.raises(ResolutionPreparationError) as caught:
        prepare_measured_resolution(sample, resolution)

    assert diagnostic_code(caught.value) == (
        "resolution_support_contains_internal_invalid_hole"
    )


def test_internal_invalid_energy_hole_fails_without_trapezoid_bridge() -> None:
    sample, resolution = matched_pair(
        energy=[-0.2, -0.1, np.nan, 0.1, 0.2],
        resolution_intensity=[1.0, 10.0, 8.0, 9.0, 1.0],
    )

    with pytest.raises(ResolutionPreparationError) as caught:
        prepare_measured_resolution(sample, resolution)

    assert diagnostic_code(caught.value) == (
        "resolution_support_contains_internal_invalid_hole"
    )


def test_invalid_uncertainty_does_not_change_valid_intensity_kernel() -> None:
    sample, resolution = matched_pair(
        resolution_uncertainty=[0.1, 0.2, np.nan, 0.2, 0.1]
    )
    source = resolution.spectra[0]

    prepared = prepare_measured_resolution(sample, resolution).spectra[0]

    np.testing.assert_array_equal(prepared.energy, source.energy)
    np.testing.assert_allclose(
        prepared.normalized_intensity,
        source.intensity * prepared.normalization_factor,
    )
    assert np.isnan(prepared.normalized_uncertainty[2])
    assert [item.code for item in prepared.diagnostics] == [
        "invalid_resolution_uncertainty_retained"
    ]


def test_support_excluding_invalid_boundary_region_prepares_contiguous_data() -> None:
    energy = [-0.2, -0.1, 0.0, 0.1, 0.2]
    sample, resolution = matched_pair(
        energy=energy,
        resolution_intensity=[np.nan, 2.0, 5.0, 2.0, np.nan],
    )

    prepared = prepare_measured_resolution(
        sample,
        resolution,
        support_overrides={0: ResolutionSupport(-0.1, 0.1)},
    ).spectra[0]

    np.testing.assert_array_equal(prepared.energy, [-0.1, 0.0, 0.1])
    assert prepared.normalized_integral == pytest.approx(1.0)
