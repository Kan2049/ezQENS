"""Scientific and numerical tests for the Milestone-4 convolution core."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import numpy.typing as npt
import pytest

from ezqens.convolution import (
    ConvolutionError,
    ConvolutionPlan,
    automatic_grid_spacing,
    build_convolution_plan,
    cell_integrated_lorentzian,
)
from ezqens.domain import QBins, ReducedDataset, Spectrum, SpectrumRole
from ezqens.resolution import (
    PreparedResolution,
    ResolutionAcceptance,
    ResolutionAcceptanceDecision,
    prepare_measured_resolution,
)

FloatArray = npt.NDArray[np.float64]


def gaussian(
    energy: npt.ArrayLike,
    *,
    center: float,
    sigma: float,
) -> FloatArray:
    coordinates = np.asarray(energy, dtype=np.float64)
    return np.asarray(
        np.exp(-0.5 * ((coordinates - center) / sigma) ** 2)
        / (sigma * math.sqrt(2.0 * math.pi)),
        dtype=np.float64,
    )


def lorentzian(
    energy: npt.ArrayLike,
    *,
    center: float,
    fwhm: float,
) -> FloatArray:
    coordinates = np.asarray(energy, dtype=np.float64)
    gamma = fwhm / 2.0
    return np.asarray(
        gamma / (math.pi * ((coordinates - center) ** 2 + gamma**2)),
        dtype=np.float64,
    )


def make_prepared(
    sample_energy: npt.ArrayLike,
    resolution_energy: npt.ArrayLike,
    resolution_intensity: npt.ArrayLike,
    *,
    sample_energy_unit: str = "meV",
    resolution_energy_unit: str = "meV",
) -> PreparedResolution:
    sample_coordinates = np.asarray(sample_energy, dtype=np.float64)
    resolution_coordinates = np.asarray(resolution_energy, dtype=np.float64)
    q_bins = QBins.from_q_values([0.5])
    sample_spectrum = Spectrum(
        role=SpectrumRole.SAMPLE,
        group_index=0,
        group_label="sample-0",
        energy=sample_coordinates,
        intensity=np.asarray(
            1.0 + 0.01 * np.arange(sample_coordinates.size),
            dtype=np.float64,
        ),
        uncertainty=np.full(sample_coordinates.size, 0.1),
        energy_unit=sample_energy_unit,
        intensity_unit="counts",
        uncertainty_unit="counts",
    )
    resolution_spectrum = Spectrum(
        role=SpectrumRole.RESOLUTION,
        group_index=0,
        group_label="resolution-0",
        energy=resolution_coordinates,
        intensity=np.asarray(resolution_intensity, dtype=np.float64),
        uncertainty=np.full(resolution_coordinates.size, 0.01),
        energy_unit=resolution_energy_unit,
        intensity_unit="counts",
        uncertainty_unit="counts",
    )
    sample = ReducedDataset(
        role=SpectrumRole.SAMPLE,
        spectra=(sample_spectrum,),
        q_bins=q_bins,
    )
    resolution = ReducedDataset(
        role=SpectrumRole.RESOLUTION,
        spectra=(resolution_spectrum,),
        q_bins=q_bins,
    )
    return prepare_measured_resolution(
        sample,
        resolution,
        acceptance_decisions={
            0: ResolutionAcceptance(
                decision=ResolutionAcceptanceDecision.KEEP,
                confirmed=True,
            )
        },
    )


def gaussian_plan(
    *,
    sample_step: float = 0.01,
    resolution_step: float = 0.005,
    sample_bounds: tuple[float, float] = (-0.8, 0.8),
    resolution_bounds: tuple[float, float] = (-0.35, 0.35),
    resolution_center: float = 0.013,
    resolution_sigma: float = 0.04,
) -> tuple[PreparedResolution, ConvolutionPlan]:
    sample_energy = np.arange(
        sample_bounds[0],
        sample_bounds[1] + sample_step / 2.0,
        sample_step,
    )
    resolution_energy = np.arange(
        resolution_bounds[0],
        resolution_bounds[1] + resolution_step / 2.0,
        resolution_step,
    )
    prepared = make_prepared(
        sample_energy,
        resolution_energy,
        gaussian(
            resolution_energy,
            center=resolution_center,
            sigma=resolution_sigma,
        ),
    )
    return prepared, build_convolution_plan(prepared, 0)


def relative_l2(actual: FloatArray, expected: FloatArray) -> float:
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def centroid(energy: FloatArray, values: FloatArray) -> float:
    area = float(np.trapezoid(values, energy))
    return float(np.trapezoid(energy * values, energy) / area)


def fwhm(energy: FloatArray, values: FloatArray) -> float:
    peak_index = int(np.argmax(values))
    half = float(values[peak_index]) / 2.0
    left_values = values[: peak_index + 1]
    right_values = values[peak_index:]
    left_index = int(np.flatnonzero(left_values >= half)[0])
    right_index = peak_index + int(np.flatnonzero(right_values < half)[0])

    left_lower = max(0, left_index - 1)
    left = float(
        np.interp(
            half,
            values[left_lower : left_index + 1],
            energy[left_lower : left_index + 1],
        )
    )
    right = float(
        np.interp(
            half,
            values[right_index - 1 : right_index + 1][::-1],
            energy[right_index - 1 : right_index + 1][::-1],
        )
    )
    return right - left


def test_automatic_spacing_uses_finer_median_spacing_divided_by_four() -> None:
    sample = np.array([-0.4, -0.2, 0.0, 0.21, 0.4])
    resolution = np.array([-0.2, -0.13, -0.05, 0.01, 0.1])

    spacing = automatic_grid_spacing(sample, resolution)

    assert spacing == pytest.approx(min(np.median(np.diff(sample)), 0.075) / 4.0)


def test_gaussian_times_gaussian_matches_analytic_center_width_and_profile() -> None:
    _, plan = gaussian_plan()
    model_center = -0.017
    model_sigma = 0.055
    model = gaussian(plan.model_energy, center=model_center, sigma=model_sigma)

    actual = plan.evaluate_on_sample(model)
    expected_sigma = math.sqrt(model_sigma**2 + 0.04**2)
    expected = gaussian(
        plan.target_energy,
        center=model_center + 0.013,
        sigma=expected_sigma,
    )

    assert relative_l2(actual, expected) < 5.0e-4
    assert centroid(plan.target_energy, actual) == pytest.approx(
        model_center + 0.013,
        abs=5.0e-4 * expected_sigma,
    )
    assert fwhm(plan.target_energy, actual) == pytest.approx(
        fwhm(plan.target_energy, expected),
        rel=5.0e-4,
    )


def test_lorentzian_times_lorentzian_has_summed_fwhm_behavior() -> None:
    sample_energy = np.arange(-3.0, 3.0001, 0.005)
    resolution_energy = np.arange(-8.0, 8.0001, 0.00125)
    resolution_fwhm = 0.025
    model_fwhm = 0.075
    prepared = make_prepared(
        sample_energy,
        resolution_energy,
        lorentzian(resolution_energy, center=0.0, fwhm=resolution_fwhm),
    )
    plan = build_convolution_plan(prepared, 0)
    model = cell_integrated_lorentzian(
        plan.model_energy,
        fwhm=model_fwhm,
        spacing=plan.spacing,
    )

    result = plan.evaluate_on_sample(model)

    assert centroid(plan.target_energy, result) == pytest.approx(0.0, abs=1.0e-8)
    assert fwhm(plan.target_energy, result) == pytest.approx(
        resolution_fwhm + model_fwhm,
        rel=2.0e-3,
    )


def test_gaussian_resolution_lorentzian_matches_direct_reference() -> None:
    _, plan = gaussian_plan(sample_step=0.008, resolution_step=0.005)
    model = cell_integrated_lorentzian(
        plan.model_energy,
        fwhm=0.031,
        spacing=plan.spacing,
    )

    actual = plan.convolve(model)
    direct = np.convolve(model, plan.resolution_values, mode="full") * plan.spacing

    np.testing.assert_allclose(actual.values, direct, rtol=2.0e-13, atol=2.0e-13)


def test_resolution_representation_and_convolution_preserve_integrated_area() -> None:
    _, plan = gaussian_plan()
    model = cell_integrated_lorentzian(
        plan.model_energy,
        fwhm=0.012,
        spacing=plan.spacing,
    )
    profile = plan.convolve(model)
    model_area = float(np.sum(model) * plan.spacing)

    assert plan.resolution_grid_integral == pytest.approx(1.0, abs=2.0e-15)
    assert float(np.trapezoid(profile.values, profile.energy)) == pytest.approx(
        model_area,
        rel=2.0e-13,
        abs=2.0e-13,
    )


def test_differing_sample_and_resolution_spacings_use_the_finer_median() -> None:
    prepared, plan = gaussian_plan(sample_step=0.013, resolution_step=0.007)

    assert plan.spacing == pytest.approx(0.007 / 4.0)
    np.testing.assert_array_equal(
        plan.target_energy,
        prepared.sample_dataset.spectra[0].energy,
    )


def test_nonuniform_resolution_is_linearly_represented_and_normalized() -> None:
    sample = np.linspace(-0.5, 0.5, 101)
    resolution = np.array([-0.3, -0.19, -0.08, -0.015, 0.04, 0.17, 0.31])
    prepared = make_prepared(
        sample,
        resolution,
        gaussian(resolution, center=0.02, sigma=0.07),
    )

    plan = build_convolution_plan(prepared, 0)
    source = prepared.spectra[0]
    expected_before_correction = np.interp(
        plan.resolution_energy,
        source.energy,
        source.normalized_intensity,
        left=0.0,
        right=0.0,
    )

    assert plan.resolution_grid_integral_before_correction == pytest.approx(
        float(np.trapezoid(expected_before_correction, plan.resolution_energy))
    )
    assert plan.resolution_grid_integral == pytest.approx(1.0, abs=2.0e-15)


def test_asymmetric_resolution_support_and_sample_interval_are_supported() -> None:
    prepared, plan = gaussian_plan(
        sample_bounds=(-0.31, 0.72),
        resolution_bounds=(-0.18, 0.39),
        resolution_center=0.08,
    )

    exact_model_min = float(plan.target_energy[0] - prepared.spectra[0].energy[-1])
    exact_model_max = float(plan.target_energy[-1] - prepared.spectra[0].energy[0])

    assert plan.model_energy[0] <= exact_model_min
    assert plan.model_energy[-1] >= exact_model_max
    assert exact_model_min - plan.model_energy[0] < plan.spacing
    assert plan.model_energy[-1] - exact_model_max < plan.spacing
    assert plan.target_energy[0] < 0.0 < plan.target_energy[-1]


def test_shifted_resolution_peak_is_preserved_without_hidden_recentering() -> None:
    resolution_center = 0.071
    _, plan = gaussian_plan(resolution_center=resolution_center)
    model = cell_integrated_lorentzian(
        plan.model_energy,
        fwhm=plan.spacing * 0.05,
        spacing=plan.spacing,
    )

    result = plan.evaluate_on_sample(model)

    assert centroid(plan.target_energy, result) == pytest.approx(
        resolution_center,
        abs=5.0e-4 * 0.04,
    )


def test_full_convolution_coordinates_are_the_sum_of_input_origins() -> None:
    _, plan = gaussian_plan()

    assert plan.convolution_energy[0] == pytest.approx(
        plan.model_energy[0] + plan.resolution_energy[0]
    )
    np.testing.assert_allclose(np.diff(plan.convolution_energy), plan.spacing)
    assert plan.full_length == (
        plan.model_energy.size + plan.resolution_energy.size - 1
    )


def test_fft_padding_prevents_circular_wraparound() -> None:
    _, plan = gaussian_plan()
    model = np.zeros(plan.model_energy.size)
    model[2] = 1.0 / plan.spacing

    profile = plan.convolve(model)
    direct = np.convolve(model, plan.resolution_values, mode="full") * plan.spacing

    assert plan.fft_length >= plan.full_length
    np.testing.assert_allclose(profile.values, direct, rtol=2.0e-13, atol=2.0e-13)
    assert np.max(np.abs(profile.values[-20:])) < 1.0e-12


def test_single_spacing_factor_and_refinement_preserve_amplitude() -> None:
    peaks: list[float] = []
    areas: list[float] = []
    for sample_step in (0.012, 0.006):
        _, plan = gaussian_plan(
            sample_step=sample_step,
            resolution_step=0.02,
        )
        model = gaussian(plan.model_energy, center=0.0, sigma=0.05)
        profile = plan.convolve(model)
        direct = np.convolve(model, plan.resolution_values, mode="full") * plan.spacing
        np.testing.assert_allclose(
            profile.values,
            direct,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        peaks.append(float(profile.evaluate([0.013])[0]))
        areas.append(float(np.trapezoid(profile.values, profile.energy)))

    assert peaks[0] == pytest.approx(peaks[1], rel=5.0e-4)
    assert areas[0] == pytest.approx(areas[1], rel=5.0e-4)


def test_power_of_two_fft_agrees_with_independent_exact_direct_convolution() -> None:
    _, plan = gaussian_plan(sample_step=0.011, resolution_step=0.007)
    model = gaussian(plan.model_energy, center=-0.02, sigma=0.08)

    profile = plan.convolve(model)
    direct = np.convolve(model, plan.resolution_values, mode="full") * plan.spacing

    assert plan.fft_length == 1 << (plan.full_length - 1).bit_length()
    np.testing.assert_allclose(profile.values, direct, rtol=2.0e-13, atol=2.0e-13)


def test_only_convolved_model_is_interpolated_to_original_sample_coordinates() -> None:
    sample_energy = np.array([-0.41, -0.27, -0.11, -0.025, 0.04, 0.19, 0.36, 0.58])
    resolution_energy = np.linspace(-0.22, 0.25, 81)
    prepared = make_prepared(
        sample_energy,
        resolution_energy,
        gaussian(resolution_energy, center=0.015, sigma=0.05),
    )
    plan = build_convolution_plan(prepared, 0)
    model = gaussian(plan.model_energy, center=-0.01, sigma=0.06)
    profile = plan.convolve(model)

    evaluated = plan.evaluate_on_sample(model)

    np.testing.assert_array_equal(plan.target_energy, sample_energy)
    np.testing.assert_allclose(
        evaluated,
        np.interp(sample_energy, profile.energy, profile.values),
    )


def test_source_sample_and_resolution_arrays_and_masks_remain_unchanged() -> None:
    prepared, plan = gaussian_plan()
    sample = prepared.sample_dataset.spectra[0]
    resolution = prepared.resolution_dataset.spectra[0]
    arrays: Sequence[FloatArray | npt.NDArray[np.bool_]] = (
        sample.energy,
        sample.intensity,
        sample.uncertainty,
        sample.invalid_energy_mask,
        sample.invalid_intensity_mask,
        sample.invalid_uncertainty_mask,
        resolution.energy,
        resolution.intensity,
        resolution.uncertainty,
        resolution.invalid_energy_mask,
        resolution.invalid_intensity_mask,
        resolution.invalid_uncertainty_mask,
    )
    originals = tuple(value.copy() for value in arrays)
    model = cell_integrated_lorentzian(
        plan.model_energy,
        fwhm=0.02,
        spacing=plan.spacing,
    )

    plan.evaluate_on_sample(model)

    for current, original in zip(arrays, originals, strict=True):
        np.testing.assert_array_equal(current, original)
        assert not current.flags.writeable


@pytest.mark.parametrize("width_ratio", [4.0, 0.5, 0.05])
def test_cell_integrated_lorentzian_matches_exact_finite_cell_area(
    width_ratio: float,
) -> None:
    spacing = 0.01
    energy = np.arange(-2.0, 2.0 + spacing / 2.0, spacing)
    fwhm_value = width_ratio * spacing
    values = cell_integrated_lorentzian(
        energy,
        fwhm=fwhm_value,
        spacing=spacing,
    )
    gamma = fwhm_value / 2.0
    lower = float(energy[0] - spacing / 2.0)
    upper = float(energy[-1] + spacing / 2.0)
    expected_area = (math.atan(upper / gamma) - math.atan(lower / gamma)) / math.pi

    assert float(np.sum(values) * spacing) == pytest.approx(
        expected_area,
        rel=2.0e-13,
        abs=2.0e-13,
    )
    assert not values.flags.writeable


def test_very_narrow_intrinsic_line_converges_to_measured_resolution_profile() -> None:
    _, plan = gaussian_plan(sample_step=0.004, resolution_step=0.003)
    model = cell_integrated_lorentzian(
        plan.model_energy,
        fwhm=0.05 * plan.spacing,
        spacing=plan.spacing,
    )
    actual = plan.evaluate_on_sample(model)
    expected = np.interp(
        plan.target_energy,
        plan.resolution_energy,
        plan.resolution_values,
        left=0.0,
        right=0.0,
    )

    assert relative_l2(actual, expected) < 5.0e-4


def test_subcell_energy_shifts_use_one_fixed_grid_with_stable_profile_metrics() -> None:
    _, plan = gaussian_plan(sample_step=0.002, resolution_step=0.003)
    model = cell_integrated_lorentzian(
        plan.model_energy,
        fwhm=0.05 * plan.spacing,
        spacing=plan.spacing,
    )
    profile = plan.convolve(model)
    model_axis_before = plan.model_energy.copy()
    resolution_axis_before = plan.resolution_energy.copy()
    phases = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9)
    centroids: list[float] = []
    widths: list[float] = []
    areas: list[float] = []
    peaks: list[float] = []
    aligned_profiles: list[FloatArray] = []

    for phase in phases:
        shift = phase * plan.spacing
        values = profile.evaluate(plan.target_energy, energy_shift=shift)
        centroids.append(centroid(plan.target_energy, values) - shift)
        widths.append(fwhm(plan.target_energy, values))
        areas.append(float(np.trapezoid(values, plan.target_energy)))
        peaks.append(float(np.max(values)))
        common_energy = plan.target_energy[2:-2]
        aligned_profiles.append(
            np.interp(
                common_energy,
                plan.target_energy - shift,
                values,
            )
        )

    np.testing.assert_array_equal(plan.model_energy, model_axis_before)
    np.testing.assert_array_equal(plan.resolution_energy, resolution_axis_before)
    assert max(centroids) - min(centroids) < 5.0e-4 * 0.04
    assert (max(widths) - min(widths)) / np.mean(widths) < 5.0e-4
    assert (max(areas) - min(areas)) / np.mean(areas) < 5.0e-4
    assert (max(peaks) - min(peaks)) / np.mean(peaks) < 5.0e-4
    for aligned in aligned_profiles[1:]:
        assert relative_l2(aligned, aligned_profiles[0]) < 5.0e-4


def test_manual_plan_rejects_inconsistent_spacing_origin_or_unit_area() -> None:
    _, plan = gaussian_plan()

    with pytest.raises(ConvolutionError, match="increments"):
        replace(plan, spacing=2.0 * plan.spacing)
    shifted_convolution_energy = plan.convolution_energy + plan.spacing
    with pytest.raises(ConvolutionError, match="origin"):
        replace(plan, convolution_energy=shifted_convolution_energy)
    with pytest.raises(ConvolutionError, match="unit area"):
        replace(plan, resolution_values=np.zeros_like(plan.resolution_values))


def test_replaced_resolution_values_rederive_the_cached_fft() -> None:
    _, plan = gaussian_plan()
    reversed_resolution = plan.resolution_values[::-1]
    reversed_resolution = reversed_resolution / float(
        np.trapezoid(reversed_resolution, plan.resolution_energy)
    )
    replaced = replace(plan, resolution_values=reversed_resolution)
    model = gaussian(plan.model_energy, center=0.0, sigma=0.05)

    actual = replaced.convolve(model).values
    direct = np.convolve(model, reversed_resolution, mode="full") * plan.spacing

    np.testing.assert_allclose(actual, direct, rtol=2.0e-13, atol=2.0e-13)


def test_convolved_profile_rejects_coordinates_inconsistent_with_spacing() -> None:
    _, plan = gaussian_plan()
    model = gaussian(plan.model_energy, center=0.0, sigma=0.05)
    profile = plan.convolve(model)

    with pytest.raises(ConvolutionError, match="increments"):
        replace(profile, spacing=2.0 * profile.spacing)


@pytest.mark.parametrize("scale", [1.0e-3, 1.0e-2, 1.0e-1, 1.0])
def test_dimensionally_equivalent_physical_scales_are_invariant(scale: float) -> None:
    dimensionless_sample = np.arange(-0.8, 0.8001, 0.01)
    dimensionless_resolution = np.arange(-0.35, 0.3501, 0.006)
    sample = dimensionless_sample * scale
    resolution = dimensionless_resolution * scale
    prepared = make_prepared(
        sample,
        resolution,
        gaussian(resolution, center=0.013 * scale, sigma=0.04 * scale),
    )
    plan = build_convolution_plan(prepared, 0)
    model = cell_integrated_lorentzian(
        plan.model_energy,
        fwhm=0.025 * scale,
        spacing=plan.spacing,
    )
    dimensionless_result = plan.evaluate_on_sample(model) * scale

    baseline_prepared, baseline_plan = gaussian_plan(resolution_step=0.006)
    baseline_model = cell_integrated_lorentzian(
        baseline_plan.model_energy,
        fwhm=0.025,
        spacing=baseline_plan.spacing,
    )
    baseline = baseline_plan.evaluate_on_sample(baseline_model)

    del baseline_prepared
    np.testing.assert_allclose(
        dimensionless_result,
        baseline,
        rtol=5.0e-4,
        atol=5.0e-8,
    )


def test_explicit_microelectronvolt_to_mev_conversion_matches_mev_problem() -> None:
    energy_mev = np.arange(-0.4, 0.4001, 0.005)
    resolution_mev = np.arange(-0.2, 0.2001, 0.004)
    width_mev = 0.025
    energy_microev = energy_mev * 1000.0
    resolution_microev = resolution_mev * 1000.0

    mev_prepared = make_prepared(
        energy_mev,
        resolution_mev,
        gaussian(resolution_mev, center=0.0, sigma=0.03),
    )
    converted_prepared = make_prepared(
        energy_microev * 1.0e-3,
        resolution_microev * 1.0e-3,
        gaussian(resolution_microev * 1.0e-3, center=0.0, sigma=30.0e-3),
    )
    mev_plan = build_convolution_plan(mev_prepared, 0)
    converted_plan = build_convolution_plan(converted_prepared, 0)
    mev_model = cell_integrated_lorentzian(
        mev_plan.model_energy,
        fwhm=width_mev,
        spacing=mev_plan.spacing,
    )
    converted_model = cell_integrated_lorentzian(
        converted_plan.model_energy,
        fwhm=25.0 * 1.0e-3,
        spacing=converted_plan.spacing,
    )

    np.testing.assert_allclose(
        mev_plan.target_energy,
        converted_plan.target_energy,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        mev_plan.evaluate_on_sample(mev_model),
        converted_plan.evaluate_on_sample(converted_model),
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_repeated_convolution_evaluation_is_deterministic() -> None:
    _, plan = gaussian_plan()
    model = cell_integrated_lorentzian(
        plan.model_energy,
        fwhm=0.018,
        spacing=plan.spacing,
    )

    first = plan.evaluate_on_sample(model, energy_shift=0.37 * plan.spacing)
    second = plan.evaluate_on_sample(model, energy_shift=0.37 * plan.spacing)

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "sample_energy",
    [
        [0.0],
        [-0.1, np.nan, 0.1],
        [-0.1, 0.0, 0.0, 0.1],
        [-0.1, 0.05, 0.0, 0.1],
    ],
)
def test_unusable_sample_coordinates_fail_without_sorting_or_repair(
    sample_energy: list[float],
) -> None:
    resolution_energy = np.linspace(-0.2, 0.2, 41)
    prepared = make_prepared(
        sample_energy,
        resolution_energy,
        gaussian(resolution_energy, center=0.0, sigma=0.04),
    )

    with pytest.raises(ConvolutionError, match="sample energy"):
        build_convolution_plan(prepared, 0)


@pytest.mark.parametrize("unit", ["unknown", "µeV", "eV"])
def test_unknown_or_noncanonical_physical_energy_unit_is_blocking(unit: str) -> None:
    sample_energy = np.linspace(-0.4, 0.4, 81)
    resolution_energy = np.linspace(-0.2, 0.2, 41)
    prepared = make_prepared(
        sample_energy,
        resolution_energy,
        gaussian(resolution_energy, center=0.0, sigma=0.04),
        sample_energy_unit=unit,
    )

    with pytest.raises(ConvolutionError, match="canonicalized to meV"):
        build_convolution_plan(prepared, 0)


def test_invalid_model_and_out_of_domain_evaluation_fail_clearly() -> None:
    _, plan = gaussian_plan()

    with pytest.raises(ConvolutionError, match="one value"):
        plan.convolve(np.ones(plan.model_energy.size - 1))
    invalid_model = np.ones(plan.model_energy.size)
    invalid_model[3] = np.nan
    with pytest.raises(ConvolutionError, match="finite"):
        plan.convolve(invalid_model)

    valid_model = gaussian(plan.model_energy, center=0.0, sigma=0.05)
    profile = plan.convolve(valid_model)
    with pytest.raises(ConvolutionError, match="outside"):
        profile.evaluate([profile.energy[-1] + 10.0 * plan.spacing])
