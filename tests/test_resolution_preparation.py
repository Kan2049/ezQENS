"""Scientific tests for measured-resolution preparation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pytest

from ezqens.domain import QBins, ReducedDataset, Spectrum, SpectrumRole
from ezqens.preprocessing import FittingSelection, PaddingStatus, detect_edge_padding
from ezqens.resolution import (
    NORMALIZATION_METHOD,
    Q_MATCH_ATOL,
    PreparedResolution,
    ResolutionAcceptance,
    ResolutionAcceptanceDecision,
    ResolutionAcceptanceWarning,
    ResolutionPreparationError,
    ResolutionSupport,
    ResolutionSupportSource,
    prepare_measured_resolution,
    preview_measured_resolution,
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


def prepare_confirmed_resolution(
    sample: ReducedDataset,
    resolution: ReducedDataset,
    *,
    support_overrides: Mapping[int, ResolutionSupport] | None = None,
    apply_auto_padding: Mapping[int, bool] | None = None,
) -> PreparedResolution:
    """Exercise frozen M3 paths with the new explicit acceptance contract."""

    override_indices = set((support_overrides or {}).keys())
    acceptances = {
        index: ResolutionAcceptance(
            decision=(
                ResolutionAcceptanceDecision.EXCLUDE_BY_CONTIGUOUS_SUPPORT
                if index in override_indices
                else ResolutionAcceptanceDecision.KEEP
            ),
            confirmed=True,
        )
        for index in range(len(resolution.spectra))
    }
    return prepare_measured_resolution(
        sample,
        resolution,
        acceptance_decisions=acceptances,
        support_overrides=support_overrides,
        apply_auto_padding=apply_auto_padding,
    )


def test_matching_q_bins_associate_successfully_with_small_float_tolerance() -> None:
    sample, resolution = matched_pair()
    resolution_q = QBins(
        q_values=np.array([0.5 + Q_MATCH_ATOL / 2.0]),
        edges=np.array([0.4, 0.6 + Q_MATCH_ATOL / 2.0]),
    )
    resolution = resolution.assign_q_bins(resolution_q)

    prepared = prepare_confirmed_resolution(sample, resolution)

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
        prepare_confirmed_resolution(sample, resolution)

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
        prepare_confirmed_resolution(sample, resolution)

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
        prepare_confirmed_resolution(sample, resolution)

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

    prepared = prepare_confirmed_resolution(sample, resolution)
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
    matching = prepare_confirmed_resolution(matching_sample, matching_resolution)
    assert matching.padding_comparisons[0].is_consistent
    assert matching.diagnostics == ()

    different_resolution = make_dataset(
        SpectrumRole.RESOLUTION,
        [[1.0, 2.0, 4.0, 3.0, 2.0] + [-1.0] * 5],
        energies=[energy],
        q_bins=q_bins,
    )
    different = prepare_confirmed_resolution(matching_sample, different_resolution)

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

    prepared = prepare_confirmed_resolution(sample, resolution).spectra[0]

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

    default = prepare_confirmed_resolution(sample, resolution).spectra[0]
    restored = prepare_confirmed_resolution(
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

    prepared = prepare_confirmed_resolution(
        sample,
        resolution,
        support_overrides={0: ResolutionSupport(-4.0, 4.0)},
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

    prepared = prepare_confirmed_resolution(sample, resolution).spectra[0]

    assert prepared.padding.left.status is PaddingStatus.REVIEW
    assert np.all(prepared.accepted_mask[:3])


def test_nonuniform_grid_normalizes_to_unit_integrated_area_not_peak_height() -> None:
    energy = [-2.0, -0.8, -0.1, 0.35, 1.4, 2.7]
    sample, resolution = matched_pair(
        energy=energy,
        resolution_intensity=[0.2, 1.0, 4.0, 3.0, 0.7, 0.1],
    )

    prepared = prepare_confirmed_resolution(sample, resolution).spectra[0]

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
        prepare_confirmed_resolution(sample, resolution)

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

    prepared = prepare_confirmed_resolution(sample, resolution).spectra[0]

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

    default = prepare_confirmed_resolution(sample_selection.dataset, resolution)
    overridden = prepare_confirmed_resolution(
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
        prepare_confirmed_resolution(sample, resolution)

    assert diagnostic_code(caught.value) == "resolution_energy_not_strictly_increasing"


def test_internal_invalid_intensity_hole_fails_without_trapezoid_bridge() -> None:
    sample, resolution = matched_pair(
        energy=[-0.2, -0.1, 0.0, 0.1, 0.2],
        resolution_intensity=[1.0, 10.0, np.nan, 9.0, 1.0],
    )

    with pytest.raises(ResolutionPreparationError) as caught:
        prepare_confirmed_resolution(sample, resolution)

    assert diagnostic_code(caught.value) == (
        "resolution_support_contains_internal_invalid_hole"
    )


def test_internal_invalid_energy_hole_fails_without_trapezoid_bridge() -> None:
    sample, resolution = matched_pair(
        energy=[-0.2, -0.1, np.nan, 0.1, 0.2],
        resolution_intensity=[1.0, 10.0, 8.0, 9.0, 1.0],
    )

    with pytest.raises(ResolutionPreparationError) as caught:
        prepare_confirmed_resolution(sample, resolution)

    assert diagnostic_code(caught.value) == (
        "resolution_support_contains_internal_invalid_hole"
    )


def test_invalid_uncertainty_does_not_change_valid_intensity_kernel() -> None:
    sample, resolution = matched_pair(
        resolution_uncertainty=[0.1, 0.2, np.nan, 0.2, 0.1]
    )
    source = resolution.spectra[0]

    prepared = prepare_confirmed_resolution(sample, resolution).spectra[0]

    np.testing.assert_array_equal(prepared.energy, source.energy)
    np.testing.assert_allclose(
        prepared.normalized_intensity,
        source.intensity * prepared.normalization_factor,
    )
    assert np.isnan(prepared.normalized_uncertainty[2])
    assert [item.code for item in prepared.diagnostics] == [
        "invalid_resolution_uncertainty_retained"
    ]


def test_default_support_excludes_invalid_boundary_region_contiguously() -> None:
    energy = [-0.2, -0.1, 0.0, 0.1, 0.2]
    sample, resolution = matched_pair(
        energy=energy,
        resolution_intensity=[np.nan, 2.0, 5.0, 2.0, np.nan],
    )

    prepared = prepare_confirmed_resolution(sample, resolution).spectra[0]

    np.testing.assert_array_equal(prepared.energy, [-0.1, 0.0, 0.1])
    assert prepared.normalized_integral == pytest.approx(1.0)


def test_preview_exposes_pending_keep_data_but_preparation_requires_confirmation() -> (
    None
):
    sample, resolution = matched_pair()

    preview = preview_measured_resolution(sample, resolution)
    spectrum = preview.spectra[0]

    assert spectrum.acceptance == ResolutionAcceptance()
    assert not spectrum.acceptance.confirmed
    assert spectrum.original_support == ResolutionSupport(
        -2.0,
        2.0,
        ResolutionSupportSource.DEFAULT_VALID_DATA,
    )
    assert spectrum.support == spectrum.original_support
    np.testing.assert_array_equal(spectrum.energy, resolution.spectra[0].energy)
    np.testing.assert_array_equal(spectrum.intensity, resolution.spectra[0].intensity)
    np.testing.assert_array_equal(
        spectrum.uncertainty,
        resolution.spectra[0].uncertainty,
    )
    assert not spectrum.energy.flags.writeable
    assert not spectrum.intensity.flags.writeable
    assert not spectrum.uncertainty.flags.writeable
    assert spectrum.normalization_factor == pytest.approx(
        1.0
        / np.trapezoid(
            resolution.spectra[0].intensity,
            resolution.spectra[0].energy,
        )
    )

    with pytest.raises(ResolutionPreparationError) as missing:
        prepare_measured_resolution(sample, resolution)
    assert diagnostic_code(missing.value) == "resolution_acceptance_required"

    with pytest.raises(ResolutionPreparationError) as unconfirmed:
        prepare_measured_resolution(
            sample,
            resolution,
            acceptance_decisions={
                0: ResolutionAcceptance(
                    decision=ResolutionAcceptanceDecision.KEEP,
                    confirmed=False,
                )
            },
        )
    assert diagnostic_code(unconfirmed.value) == "resolution_acceptance_unconfirmed"


def test_confirmed_keep_exactly_preserves_m3_normalization_semantics() -> None:
    sample, resolution = matched_pair(
        energy=[-2.0, -0.7, -0.1, 0.4, 1.5],
        resolution_intensity=[0.3, 2.0, 5.0, 1.5, 0.2],
        resolution_uncertainty=[0.03, 0.04, 0.05, 0.04, 0.03],
    )
    source = resolution.spectra[0]
    originals = (
        source.energy.copy(),
        source.intensity.copy(),
        source.uncertainty.copy(),
    )
    expected_integral = float(np.trapezoid(source.intensity, source.energy))

    prepared = prepare_measured_resolution(
        sample,
        resolution,
        acceptance_decisions={
            0: ResolutionAcceptance(
                decision=ResolutionAcceptanceDecision.KEEP,
                confirmed=True,
            )
        },
    ).spectra[0]

    assert prepared.acceptance.decision is ResolutionAcceptanceDecision.KEEP
    assert prepared.acceptance.confirmed
    assert prepared.support == prepared.original_support
    assert prepared.normalization_integral == expected_integral
    assert prepared.normalization_factor == 1.0 / expected_integral
    assert prepared.signed_area_ratio == pytest.approx(1.0)
    np.testing.assert_array_equal(prepared.energy, source.energy)
    np.testing.assert_allclose(
        prepared.normalized_intensity,
        source.intensity * (1.0 / expected_integral),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        prepared.normalized_uncertainty,
        source.uncertainty * (1.0 / expected_integral),
        rtol=0.0,
        atol=0.0,
    )
    for current, original in zip(
        (source.energy, source.intensity, source.uncertainty),
        originals,
        strict=True,
    ):
        np.testing.assert_array_equal(current, original)


@pytest.mark.parametrize(
    ("bounds", "expected_energy"),
    [
        ((-2.0, 3.0), [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]),
        ((-3.0, 2.0), [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]),
        ((-2.0, 2.0), [-2.0, -1.0, 0.0, 1.0, 2.0]),
    ],
)
def test_confirmed_contiguous_exclusion_trims_boundaries_before_normalization(
    bounds: tuple[float, float], expected_energy: list[float]
) -> None:
    energy = np.arange(-3.0, 4.0)
    intensity = np.array([0.2, 0.5, 2.0, 5.0, 2.0, 0.5, 0.2])
    uncertainty = np.linspace(0.02, 0.08, energy.size)
    sample, resolution = matched_pair(
        energy=energy,
        resolution_intensity=intensity,
        resolution_uncertainty=uncertainty,
    )
    support = ResolutionSupport(*bounds)

    prepared = prepare_measured_resolution(
        sample,
        resolution,
        acceptance_decisions={
            0: ResolutionAcceptance(
                decision=(ResolutionAcceptanceDecision.EXCLUDE_BY_CONTIGUOUS_SUPPORT),
                confirmed=True,
            )
        },
        support_overrides={0: support},
    ).spectra[0]

    expected_mask = (energy >= bounds[0]) & (energy <= bounds[1])
    expected_integral = float(
        np.trapezoid(intensity[expected_mask], energy[expected_mask])
    )
    full_integral = float(np.trapezoid(intensity, energy))
    np.testing.assert_array_equal(prepared.energy, expected_energy)
    np.testing.assert_array_equal(
        np.flatnonzero(prepared.accepted_mask), np.flatnonzero(expected_mask)
    )
    assert prepared.normalized_integral == pytest.approx(1.0, abs=1.0e-15)
    np.testing.assert_allclose(
        prepared.normalized_uncertainty,
        uncertainty[expected_mask] / expected_integral,
    )
    np.testing.assert_array_equal(
        prepared.normalized_intensity_on_source_grid[~expected_mask],
        np.zeros(np.count_nonzero(~expected_mask)),
    )
    assert prepared.signed_area_ratio == pytest.approx(
        expected_integral / full_integral
    )


def test_signed_area_ratio_above_one_is_diagnostic_only() -> None:
    energy = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
    intensity = np.array([-1.0, -0.5, 1.0, 4.0, 1.0, 0.5])
    uncertainty = np.linspace(0.02, 0.07, energy.size)
    sample, resolution = matched_pair(
        energy=energy,
        resolution_intensity=intensity,
        resolution_uncertainty=uncertainty,
    )
    acceptance = ResolutionAcceptance(
        decision=ResolutionAcceptanceDecision.EXCLUDE_BY_CONTIGUOUS_SUPPORT,
        confirmed=True,
    )
    support = ResolutionSupport(-1.0, 2.0)
    source = resolution.spectra[0]
    originals = (
        source.energy.copy(),
        source.intensity.copy(),
        source.uncertainty.copy(),
    )

    with pytest.raises(ResolutionPreparationError) as unconfirmed:
        prepare_measured_resolution(
            sample,
            resolution,
            acceptance_decisions={
                0: ResolutionAcceptance(
                    decision=(
                        ResolutionAcceptanceDecision.EXCLUDE_BY_CONTIGUOUS_SUPPORT
                    ),
                    confirmed=False,
                )
            },
            support_overrides={0: support},
            apply_auto_padding={0: False},
        )
    assert diagnostic_code(unconfirmed.value) == "resolution_acceptance_unconfirmed"

    result = prepare_measured_resolution(
        sample,
        resolution,
        acceptance_decisions={0: acceptance},
        support_overrides={0: support},
        apply_auto_padding={0: False},
    )
    prepared = result.spectra[0]

    expected_mask = np.array([False, False, True, True, True, True])
    accepted_area = float(np.trapezoid(intensity[expected_mask], energy[expected_mask]))
    original_area = float(np.trapezoid(intensity, energy))
    expected_ratio = accepted_area / original_area
    assert original_area > 0.0
    assert prepared.acceptance == acceptance
    ratio = prepared.signed_area_ratio
    assert ratio is not None
    assert ratio == pytest.approx(expected_ratio)
    assert ratio > 1.0
    assert result.diagnostics == ()
    assert prepared.diagnostics == ()
    np.testing.assert_array_equal(prepared.accepted_mask, expected_mask)
    np.testing.assert_array_equal(prepared.energy, energy[expected_mask])
    np.testing.assert_array_equal(prepared.intensity, intensity[expected_mask])
    assert np.all(np.diff(np.flatnonzero(prepared.accepted_mask)) == 1)
    assert prepared.normalized_integral == pytest.approx(1.0, abs=1.0e-15)
    np.testing.assert_allclose(
        prepared.normalized_uncertainty,
        uncertainty[expected_mask] / accepted_area,
    )
    np.testing.assert_array_equal(
        prepared.normalized_intensity_on_source_grid[~expected_mask],
        np.zeros(np.count_nonzero(~expected_mask)),
    )
    for current, original in zip(
        (source.energy, source.intensity, source.uncertainty),
        originals,
        strict=True,
    ):
        np.testing.assert_array_equal(current, original)


def test_exclusion_requires_a_real_valid_contiguous_boundary_trim() -> None:
    sample, resolution = matched_pair()
    exclusion = ResolutionAcceptance(
        decision=ResolutionAcceptanceDecision.EXCLUDE_BY_CONTIGUOUS_SUPPORT,
        confirmed=True,
    )

    with pytest.raises(ResolutionPreparationError) as missing:
        prepare_measured_resolution(
            sample,
            resolution,
            acceptance_decisions={0: exclusion},
        )
    assert diagnostic_code(missing.value) == "resolution_exclusion_support_required"

    with pytest.raises(ResolutionPreparationError) as unchanged:
        prepare_measured_resolution(
            sample,
            resolution,
            acceptance_decisions={0: exclusion},
            support_overrides={0: ResolutionSupport(-2.0, 2.0)},
        )
    assert diagnostic_code(unchanged.value) == "resolution_exclusion_not_narrower"

    with pytest.raises(ResolutionPreparationError) as outside:
        prepare_measured_resolution(
            sample,
            resolution,
            acceptance_decisions={0: exclusion},
            support_overrides={0: ResolutionSupport(-3.0, 1.0)},
        )
    assert diagnostic_code(outside.value) == (
        "resolution_exclusion_outside_original_support"
    )

    with pytest.raises(ResolutionPreparationError) as insufficient:
        prepare_measured_resolution(
            sample,
            resolution,
            acceptance_decisions={0: exclusion},
            support_overrides={0: ResolutionSupport(0.0, 0.0)},
        )
    assert diagnostic_code(insufficient.value) == "insufficient_resolution_points"

    with pytest.raises(ValueError, match="lower_energy"):
        ResolutionSupport(1.0, -1.0)


def test_keep_with_warning_preserves_kernel_and_neutral_provenance() -> None:
    sample, resolution = matched_pair()
    warning = ResolutionAcceptanceWarning.SUSPICIOUS_STRUCTURE_RETAINED

    prepared = prepare_measured_resolution(
        sample,
        resolution,
        acceptance_decisions={
            0: ResolutionAcceptance(
                decision=ResolutionAcceptanceDecision.KEEP,
                confirmed=True,
                warnings=(warning,),
            )
        },
    )
    spectrum = prepared.spectra[0]
    provenance = prepared.acceptance_provenance(0)

    np.testing.assert_array_equal(spectrum.energy, resolution.spectra[0].energy)
    assert spectrum.support == spectrum.original_support
    assert spectrum.acceptance.warnings == (warning,)
    assert [item.code for item in spectrum.diagnostics] == [
        "suspicious_resolution_structure_retained_by_user"
    ]
    assert provenance.decision is ResolutionAcceptanceDecision.KEEP
    assert provenance.confirmed
    assert provenance.warnings == (warning,)


def test_resolution_acceptance_and_support_are_independent_per_q_group() -> None:
    energy = np.arange(-2.0, 3.0)
    q_bins = QBins.from_q_values([0.5, 0.8])
    sample = make_dataset(
        SpectrumRole.SAMPLE,
        [np.ones(5), np.ones(5)],
        energies=[energy, energy],
        q_bins=q_bins,
    )
    resolution = make_dataset(
        SpectrumRole.RESOLUTION,
        [[1.0, 3.0, 5.0, 3.0, 1.0], [2.0, 4.0, 6.0, 4.0, 2.0]],
        energies=[energy, energy],
        q_bins=q_bins,
    )

    prepared = prepare_measured_resolution(
        sample,
        resolution,
        acceptance_decisions={
            0: ResolutionAcceptance(
                decision=ResolutionAcceptanceDecision.KEEP,
                confirmed=True,
            ),
            1: ResolutionAcceptance(
                decision=(ResolutionAcceptanceDecision.EXCLUDE_BY_CONTIGUOUS_SUPPORT),
                confirmed=True,
            ),
        },
        support_overrides={1: ResolutionSupport(-1.0, 1.0)},
    )

    np.testing.assert_array_equal(prepared.spectra[0].energy, energy)
    np.testing.assert_array_equal(prepared.spectra[1].energy, [-1.0, 0.0, 1.0])
    assert prepared.spectra[0].signed_area_ratio == pytest.approx(1.0)
    assert prepared.spectra[1].signed_area_ratio != pytest.approx(1.0)
    assert prepared.q_value(0) == pytest.approx(0.5)
    assert prepared.q_value(1) == pytest.approx(0.8)
