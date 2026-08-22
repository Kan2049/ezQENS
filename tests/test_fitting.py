"""Scientific tests for the production single-Q fitting core."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pytest

import ezqens.fitting.core as fitting_core
from ezqens.convolution import build_convolution_plan, cell_integrated_lorentzian
from ezqens.domain import QBins, ReducedDataset, Spectrum, SpectrumRole
from ezqens.fitting import (
    BackgroundModel,
    FittingError,
    LorentzianComponent,
    ParameterConfiguration,
    SpectralModelDefinition,
    StandardModelCandidate,
    evaluate_spectral_model,
    evaluate_standard_candidates,
    fit_single_q,
    fit_standard_candidate,
    generate_standard_candidates,
)
from ezqens.preprocessing import FittingSelection
from ezqens.resolution import (
    PreparedResolution,
    ResolutionAcceptance,
    ResolutionAcceptanceDecision,
    prepare_measured_resolution,
)

FloatArray = npt.NDArray[np.float64]


def gaussian(energy: npt.ArrayLike, *, sigma: float, center: float = 0.0) -> FloatArray:
    coordinates = np.asarray(energy, dtype=np.float64)
    return np.asarray(
        np.exp(-0.5 * np.square((coordinates - center) / sigma))
        / (sigma * math.sqrt(2.0 * math.pi)),
        dtype=np.float64,
    )


def parameter(
    initial: float,
    lower: float = -math.inf,
    upper: float = math.inf,
    *,
    free: bool = True,
) -> ParameterConfiguration:
    return ParameterConfiguration(initial, lower, upper, free)


def model_definition(
    *,
    energy_shift: float = 0.012,
    elastic_area: float = 0.8,
    lorentzians: Sequence[tuple[float, float]] = (),
    background: BackgroundModel = BackgroundModel.NONE,
    b0: float = 0.0,
    b1: float = 0.0,
    free: bool = True,
) -> SpectralModelDefinition:
    return SpectralModelDefinition(
        energy_shift=parameter(energy_shift, -0.15, 0.15, free=free),
        elastic_area=parameter(elastic_area, 0.0, free=free),
        lorentzians=tuple(
            LorentzianComponent(
                area=parameter(area, 0.0, free=free),
                fwhm=parameter(fwhm, 1.0e-8, free=free),
            )
            for area, fwhm in lorentzians
        ),
        background=background,
        b0=(
            parameter(b0, free=free) if background is not BackgroundModel.NONE else None
        ),
        b1=(parameter(b1, free=free) if background is BackgroundModel.LINEAR else None),
    )


def make_dataset_pair(
    sample_energy: FloatArray,
    sample_intensity: FloatArray,
    uncertainty: FloatArray,
    *,
    resolution_energy: FloatArray,
    resolution_values: FloatArray,
) -> tuple[ReducedDataset, ReducedDataset]:
    q_bins = QBins.from_q_values([0.75])
    sample = ReducedDataset(
        role=SpectrumRole.SAMPLE,
        spectra=(
            Spectrum(
                role=SpectrumRole.SAMPLE,
                group_index=0,
                group_label="sample-0",
                energy=sample_energy,
                intensity=sample_intensity,
                uncertainty=uncertainty,
                energy_unit="meV",
                intensity_unit="arb",
                uncertainty_unit="arb",
            ),
        ),
        q_bins=q_bins,
    )
    resolution = ReducedDataset(
        role=SpectrumRole.RESOLUTION,
        spectra=(
            Spectrum(
                role=SpectrumRole.RESOLUTION,
                group_index=0,
                group_label="resolution-0",
                energy=resolution_energy,
                intensity=resolution_values,
                uncertainty=np.full(resolution_energy.size, 1.0e-3),
                energy_unit="meV",
                intensity_unit="arb",
                uncertainty_unit="arb",
            ),
        ),
        q_bins=q_bins,
    )
    return sample, resolution


def synthetic_problem(
    truth: SpectralModelDefinition,
    *,
    sample_energy: FloatArray | None = None,
    resolution_energy: FloatArray | None = None,
    sigma: float = 0.003,
    noise_seed: int | None = None,
) -> tuple[PreparedResolution, FittingSelection]:
    sample_coordinates = (
        np.linspace(-0.75, 0.75, 151)
        if sample_energy is None
        else np.asarray(sample_energy, dtype=np.float64)
    )
    resolution_coordinates = (
        np.linspace(-0.3, 0.3, 121)
        if resolution_energy is None
        else np.asarray(resolution_energy, dtype=np.float64)
    )
    resolution_values = gaussian(resolution_coordinates, sigma=0.035)
    placeholder, resolution = make_dataset_pair(
        sample_coordinates,
        np.ones(sample_coordinates.size),
        np.full(sample_coordinates.size, sigma),
        resolution_energy=resolution_coordinates,
        resolution_values=resolution_values,
    )
    acceptance = {
        0: ResolutionAcceptance(
            decision=ResolutionAcceptanceDecision.KEEP,
            confirmed=True,
        )
    }
    provisional = prepare_measured_resolution(
        placeholder,
        resolution,
        acceptance_decisions=acceptance,
    )
    plan = build_convolution_plan(provisional, 0)
    values = evaluate_spectral_model(plan, truth).total.copy()
    if noise_seed is not None:
        values += np.random.default_rng(noise_seed).normal(0.0, sigma, values.size)
    sample, resolution = make_dataset_pair(
        sample_coordinates,
        values,
        np.full(sample_coordinates.size, sigma),
        resolution_energy=resolution_coordinates,
        resolution_values=resolution_values,
    )
    prepared = prepare_measured_resolution(
        sample,
        resolution,
        acceptance_decisions=acceptance,
    )
    selection = FittingSelection.uniform(
        sample,
        prepared.sample_padding,
        lower_energy=float(sample_coordinates[0]),
        upper_energy=float(sample_coordinates[-1]),
    )
    return prepared, selection


def perturbed(model: SpectralModelDefinition) -> SpectralModelDefinition:
    return model_definition(
        energy_shift=model.energy_shift.initial_value + 0.004,
        elastic_area=model.elastic_area.initial_value * 0.9,
        lorentzians=tuple(
            (component.area.initial_value * 1.1, component.fwhm.initial_value * 1.15)
            for component in model.lorentzians
        ),
        background=model.background,
        b0=0.8 * (model.b0.initial_value if model.b0 is not None else 0.0),
        b1=0.8 * (model.b1.initial_value if model.b1 is not None else 0.0),
    )


@pytest.mark.parametrize(
    ("truth", "expected_count"),
    [
        (model_definition(), 0),
        (model_definition(lorentzians=((0.55, 0.11),)), 1),
        (
            model_definition(
                lorentzians=((0.35, 0.075), (0.28, 0.31)),
                background=BackgroundModel.LINEAR,
                b0=0.03,
                b1=-0.015,
            ),
            2,
        ),
    ],
)
def test_production_fits_zero_one_and_two_lorentzians(
    truth: SpectralModelDefinition,
    expected_count: int,
) -> None:
    prepared, selection = synthetic_problem(truth)

    result = fit_single_q(prepared, selection, 0, perturbed(truth))

    assert result.diagnostics.optimizer_success
    assert result.model.lorentzian_count == expected_count
    assert result.statistics.reduced_chi_square < 1.0e-8
    reconstructed = result.evaluation.elastic + result.evaluation.background
    for contribution in result.evaluation.lorentzians:
        reconstructed = reconstructed + contribution
    np.testing.assert_allclose(result.evaluation.total, reconstructed)


@pytest.mark.parametrize(
    "background",
    [BackgroundModel.NONE, BackgroundModel.CONSTANT, BackgroundModel.LINEAR],
)
def test_none_b0_and_b1_backgrounds_are_additive(
    background: BackgroundModel,
) -> None:
    truth = model_definition(
        lorentzians=((0.45, 0.12),),
        background=background,
        b0=0.04,
        b1=-0.02,
    )
    prepared, selection = synthetic_problem(truth)

    result = fit_single_q(prepared, selection, 0, perturbed(truth))

    expected = np.zeros(result.evaluation.energy.size)
    if background is not BackgroundModel.NONE:
        expected += result.parameter("b0").value
    if background is BackgroundModel.LINEAR:
        expected += result.parameter("b1").value * result.evaluation.energy
    np.testing.assert_allclose(result.evaluation.background, expected)


def test_shared_e0_shifts_elastic_and_every_lorentzian_component() -> None:
    truth = model_definition(
        energy_shift=0.027,
        lorentzians=((0.4, 0.09), (0.25, 0.28)),
    )
    prepared, selection = synthetic_problem(truth)

    result = fit_single_q(prepared, selection, 0, perturbed(truth))

    assert result.parameter("energy_shift").value == pytest.approx(0.027, abs=1.0e-7)
    assert not any(
        parameter.name.startswith("lorentzian_1_e0") for parameter in result.parameters
    )
    assert sum(parameter.name == "energy_shift" for parameter in result.parameters) == 1


def test_independent_component_centers_evaluate_separately_from_elastic_center() -> (
    None
):
    independent = SpectralModelDefinition(
        energy_shift=parameter(0.02, -0.15, 0.15, free=False),
        elastic_area=parameter(0.65, 0.0, free=False),
        lorentzians=(
            LorentzianComponent(
                area=parameter(0.35, 0.0, free=False),
                fwhm=parameter(0.08, 1.0e-8, free=False),
                center=parameter(-0.09, -0.15, 0.15, free=False),
            ),
            LorentzianComponent(
                area=parameter(0.25, 0.0, free=False),
                fwhm=parameter(0.24, 1.0e-8, free=False),
                center=parameter(0.11, -0.15, 0.15, free=False),
            ),
        ),
    )
    prepared, _ = synthetic_problem(independent)
    plan = build_convolution_plan(prepared, 0)
    evaluated = evaluate_spectral_model(plan, independent)

    shared_elastic = model_definition(
        energy_shift=0.02,
        elastic_area=0.65,
        free=False,
    )
    np.testing.assert_allclose(
        evaluated.elastic,
        evaluate_spectral_model(plan, shared_elastic).elastic,
    )
    for index, component in enumerate(independent.lorentzians):
        assert component.center is not None
        shared_equivalent = model_definition(
            energy_shift=component.center.initial_value,
            elastic_area=0.0,
            lorentzians=((component.area.initial_value, component.fwhm.initial_value),),
            free=False,
        )
        np.testing.assert_allclose(
            evaluated.lorentzians[index],
            evaluate_spectral_model(plan, shared_equivalent).lorentzians[0],
        )


def test_one_independent_center_can_start_at_shared_e0_and_fit_separately() -> None:
    truth = SpectralModelDefinition(
        energy_shift=parameter(0.0, -0.15, 0.15, free=False),
        elastic_area=parameter(0.70, 0.0, free=False),
        lorentzians=(
            LorentzianComponent(
                area=parameter(0.42, 0.0, free=False),
                fwhm=parameter(0.11, 1.0e-8, free=False),
                center=parameter(0.065, -0.15, 0.15, free=False),
            ),
        ),
    )
    prepared, selection = synthetic_problem(truth)
    configured = SpectralModelDefinition(
        energy_shift=parameter(0.0, -0.15, 0.15, free=False),
        elastic_area=parameter(0.66, 0.0, 1.2),
        lorentzians=(
            LorentzianComponent(
                area=parameter(0.38, 0.0, 0.8),
                fwhm=parameter(0.13, 0.04, 0.25),
                center=parameter(0.0, -0.12, 0.12),
            ),
        ),
    )

    result = fit_single_q(prepared, selection, 0, configured)

    assert configured.lorentzians[0].center is not None
    assert (
        configured.lorentzians[0].center.initial_value
        == configured.energy_shift.initial_value
    )
    center = result.parameter("lorentzian_1_center")
    assert center.free
    assert center.lower_bound == -0.12
    assert center.upper_bound == 0.12
    assert center.value == pytest.approx(0.065, abs=2.0e-7)
    assert center.standard_error is not None
    assert result.statistics.free_parameters == 4
    assert result.statistics.nominal_degrees_of_freedom == 151 - 4


def test_canonicalization_keeps_optional_center_with_its_component() -> None:
    configured = SpectralModelDefinition(
        energy_shift=parameter(0.01, -0.15, 0.15, free=False),
        elastic_area=parameter(0.70, 0.0, free=False),
        lorentzians=(
            LorentzianComponent(
                area=parameter(0.24, 0.0, free=False),
                fwhm=parameter(0.26, 1.0e-8, free=False),
                center=parameter(0.09, -0.15, 0.15, free=False),
            ),
            LorentzianComponent(
                area=parameter(0.36, 0.0, free=False),
                fwhm=parameter(0.08, 1.0e-8, free=False),
            ),
        ),
    )
    prepared, selection = synthetic_problem(configured)

    result = fit_single_q(prepared, selection, 0, configured)

    assert tuple(parameter.name for parameter in result.parameters) == (
        "energy_shift",
        "elastic_area",
        "lorentzian_1_area",
        "lorentzian_1_fwhm",
        "lorentzian_2_area",
        "lorentzian_2_fwhm",
        "lorentzian_2_center",
    )
    assert result.parameter("lorentzian_1_fwhm").value == 0.08
    with pytest.raises(KeyError):
        result.parameter("lorentzian_1_center")
    assert result.parameter("lorentzian_2_fwhm").value == 0.26
    assert result.parameter("lorentzian_2_center").value == 0.09
    assert result.diagnostics.alternative_starts[0].canonical_component_order == (1, 0)


def test_independent_centers_fit_with_bounds_covariance_and_canonical_components() -> (
    None
):
    truth = SpectralModelDefinition(
        energy_shift=parameter(0.015, -0.15, 0.15, free=False),
        elastic_area=parameter(0.72, 0.0, free=False),
        lorentzians=(
            LorentzianComponent(
                area=parameter(0.24, 0.0, free=False),
                fwhm=parameter(0.27, 1.0e-8, free=False),
                center=parameter(0.10, -0.15, 0.15, free=False),
            ),
            LorentzianComponent(
                area=parameter(0.38, 0.0, free=False),
                fwhm=parameter(0.075, 1.0e-8, free=False),
                center=parameter(-0.085, -0.15, 0.15, free=False),
            ),
        ),
    )
    prepared, selection = synthetic_problem(truth)
    configured = SpectralModelDefinition(
        energy_shift=parameter(0.015, -0.15, 0.15, free=False),
        elastic_area=parameter(0.68, 0.0, 1.2),
        lorentzians=(
            LorentzianComponent(
                area=parameter(0.22, 0.0, 0.8),
                fwhm=parameter(0.24, 0.1, 0.45),
                center=parameter(0.08, 0.04, 0.14),
            ),
            LorentzianComponent(
                area=parameter(0.35, 0.0, 0.8),
                fwhm=parameter(0.09, 0.03, 0.15),
                center=parameter(-0.085, -0.085, -0.085, free=False),
            ),
        ),
    )

    result = fit_single_q(prepared, selection, 0, configured)

    assert result.diagnostics.optimizer_success
    assert tuple(parameter.name for parameter in result.parameters) == (
        "energy_shift",
        "elastic_area",
        "lorentzian_1_area",
        "lorentzian_1_fwhm",
        "lorentzian_1_center",
        "lorentzian_2_area",
        "lorentzian_2_fwhm",
        "lorentzian_2_center",
    )
    assert result.statistics.free_parameters == 6
    assert result.statistics.nominal_degrees_of_freedom == 151 - 6
    assert result.covariance is not None
    assert result.correlation is not None
    assert result.covariance.shape == (8, 8)
    assert result.correlation.shape == (8, 8)
    assert result.parameter("lorentzian_1_fwhm").value == pytest.approx(
        0.075,
        rel=2.0e-5,
    )
    assert result.parameter("lorentzian_1_center").value == pytest.approx(-0.085)
    assert not result.parameter("lorentzian_1_center").free
    assert result.parameter("lorentzian_1_center").standard_error is None
    assert result.parameter("lorentzian_2_fwhm").value == pytest.approx(
        0.27,
        rel=2.0e-5,
    )
    assert result.parameter("lorentzian_2_center").value == pytest.approx(
        0.10,
        abs=2.0e-7,
    )
    assert result.parameter("lorentzian_2_center").free
    assert result.parameter("lorentzian_2_center").standard_error is not None
    assert result.diagnostics.alternative_starts[0].canonical_component_order == (1, 0)
    assert result.parameter("lorentzian_1_center").lower_bound == -0.085
    assert result.parameter("lorentzian_1_center").upper_bound == -0.085
    assert result.parameter("lorentzian_2_center").lower_bound == 0.04
    assert result.parameter("lorentzian_2_center").upper_bound == 0.14


def test_integrated_area_and_fwhm_semantics_are_preserved() -> None:
    truth = model_definition(
        elastic_area=0.9,
        lorentzians=((0.6, 0.14),),
    )
    prepared, selection = synthetic_problem(truth)
    plan = build_convolution_plan(prepared, 0)

    evaluation = evaluate_spectral_model(plan, truth, plan.convolution_energy[20:-20])
    result = fit_single_q(prepared, selection, 0, perturbed(truth))

    assert result.parameter("elastic_area").value == pytest.approx(0.9, rel=2.0e-5)
    assert result.parameter("lorentzian_1_area").value == pytest.approx(0.6, rel=2.0e-5)
    assert result.parameter("lorentzian_1_fwhm").value == pytest.approx(
        0.14, rel=2.0e-5
    )
    assert float(np.trapezoid(evaluation.elastic, evaluation.energy)) <= 0.9 * 1.001
    assert result.diagnostics.lorentzian_full_convolution_areas[0] <= 0.6 * 1.001


def test_fixed_parameter_and_bounds_are_honored() -> None:
    truth = model_definition(lorentzians=((0.5, 0.12),))
    prepared, selection = synthetic_problem(truth)
    configured = SpectralModelDefinition(
        energy_shift=parameter(0.012, -0.05, 0.05, free=False),
        elastic_area=parameter(0.7, 0.65, 0.95),
        lorentzians=(
            LorentzianComponent(
                area=parameter(0.4, 0.1, 0.8),
                fwhm=parameter(0.10, 0.05, 0.2),
            ),
        ),
    )

    result = fit_single_q(prepared, selection, 0, configured)

    e0 = result.parameter("energy_shift")
    assert e0.value == 0.012
    assert not e0.free
    assert e0.standard_error is None
    for estimate in result.parameters:
        assert estimate.lower_bound <= estimate.value <= estimate.upper_bound


def test_active_bound_is_reported_without_misleading_symmetric_error() -> None:
    truth = model_definition(energy_shift=0.0, elastic_area=0.8)
    prepared, selection = synthetic_problem(truth)
    constrained = SpectralModelDefinition(
        energy_shift=parameter(0.0, free=False),
        elastic_area=parameter(0.4, 0.0, 0.5),
    )

    result = fit_single_q(prepared, selection, 0, constrained)

    estimate = result.parameter("elastic_area")
    assert estimate.active_upper_bound
    assert estimate.standard_error is None
    assert "elastic_area:upper" in result.diagnostics.active_bounds
    assert not result.diagnostics.covariance_available
    assert result.covariance is None
    assert result.correlation is None
    assert all(parameter.standard_error is None for parameter in result.parameters)


def test_physical_positivity_constraints_are_validated() -> None:
    with pytest.raises(ValueError, match="elastic integrated area"):
        SpectralModelDefinition(
            energy_shift=parameter(0.0),
            elastic_area=parameter(-0.1, -1.0),
        )
    with pytest.raises(ValueError, match="Lorentzian integrated area"):
        LorentzianComponent(
            area=parameter(-0.1, -1.0),
            fwhm=parameter(0.1, 1.0e-8),
        )
    with pytest.raises(ValueError, match="FWHM"):
        LorentzianComponent(
            area=parameter(0.1, 0.0),
            fwhm=parameter(0.1, 0.0),
        )


def test_lorentzians_are_canonicalized_by_increasing_fwhm() -> None:
    truth = model_definition(lorentzians=((0.28, 0.30), (0.42, 0.08)))
    prepared, selection = synthetic_problem(truth)

    result = fit_single_q(prepared, selection, 0, perturbed(truth))

    assert result.diagnostics.lorentzian_fwhm[0] < result.diagnostics.lorentzian_fwhm[1]
    assert result.parameter("lorentzian_1_area").value == pytest.approx(
        0.42, rel=1.0e-4
    )
    assert result.parameter("lorentzian_2_area").value == pytest.approx(
        0.28, rel=1.0e-4
    )


def test_arbitrary_three_lorentzian_manual_fit_has_no_engine_ceiling() -> None:
    truth = model_definition(
        lorentzians=((0.22, 0.055), (0.31, 0.16), (0.19, 0.43)),
        background=BackgroundModel.CONSTANT,
        b0=0.025,
    )
    prepared, selection = synthetic_problem(
        truth, sample_energy=np.linspace(-1.0, 1.0, 241)
    )
    configured = model_definition(
        energy_shift=truth.energy_shift.initial_value,
        elastic_area=truth.elastic_area.initial_value,
        lorentzians=((0.20, 0.06), (0.29, 0.15), (0.20, 0.40)),
        background=BackgroundModel.CONSTANT,
        b0=0.02,
    )

    result = fit_single_q(prepared, selection, 0, configured)

    assert result.diagnostics.optimizer_success
    assert result.model.lorentzian_count == 3
    assert result.diagnostics.lorentzian_fwhm == tuple(
        sorted(result.diagnostics.lorentzian_fwhm)
    )
    assert result.parameter("energy_shift").value == pytest.approx(
        truth.energy_shift.initial_value,
        abs=2.0e-7,
    )
    assert result.parameter("b0").value == pytest.approx(0.025, abs=2.0e-7)
    for index, (expected_area, expected_fwhm) in enumerate(
        ((0.22, 0.055), (0.31, 0.16), (0.19, 0.43)),
        start=1,
    ):
        assert result.parameter(f"lorentzian_{index}_area").value == pytest.approx(
            expected_area,
            rel=2.0e-5,
        )
        assert result.parameter(f"lorentzian_{index}_fwhm").value == pytest.approx(
            expected_fwhm,
            rel=2.0e-5,
        )
    reconstructed = result.evaluation.elastic + result.evaluation.background
    for contribution in result.evaluation.lorentzians:
        reconstructed = reconstructed + contribution
    np.testing.assert_allclose(result.evaluation.total, reconstructed, rtol=1.0e-13)


def test_arbitrary_3l_mixed_centers_canonicalize_with_their_components() -> None:
    configured = SpectralModelDefinition(
        energy_shift=parameter(0.01, -0.15, 0.15, free=False),
        elastic_area=parameter(0.70, 0.0, free=False),
        lorentzians=(
            LorentzianComponent(
                area=parameter(0.20, 0.0, free=False),
                fwhm=parameter(0.40, 1.0e-8, free=False),
                center=parameter(0.12, -0.15, 0.15, free=False),
            ),
            LorentzianComponent(
                area=parameter(0.30, 0.0, free=False),
                fwhm=parameter(0.06, 1.0e-8, free=False),
            ),
            LorentzianComponent(
                area=parameter(0.25, 0.0, free=False),
                fwhm=parameter(0.18, 1.0e-8, free=False),
                center=parameter(-0.10, -0.15, 0.15, free=False),
            ),
        ),
    )
    prepared, selection = synthetic_problem(configured)

    result = fit_single_q(prepared, selection, 0, configured)

    assert result.model.lorentzian_count == 3
    assert result.diagnostics.alternative_starts[0].canonical_component_order == (
        1,
        2,
        0,
    )
    assert result.parameter("lorentzian_1_fwhm").value == 0.06
    assert result.parameter("lorentzian_1_area").value == 0.30
    with pytest.raises(KeyError):
        result.parameter("lorentzian_1_center")
    assert result.parameter("lorentzian_2_fwhm").value == 0.18
    assert result.parameter("lorentzian_2_area").value == 0.25
    assert result.parameter("lorentzian_2_center").value == -0.10
    assert result.parameter("lorentzian_3_fwhm").value == 0.40
    assert result.parameter("lorentzian_3_area").value == 0.20
    assert result.parameter("lorentzian_3_center").value == 0.12


def test_manual_initial_configuration_survives_separately_from_fitted_values() -> None:
    truth = model_definition(lorentzians=((0.5, 0.12),))
    prepared, selection = synthetic_problem(truth)
    configured = perturbed(truth)

    result = fit_single_q(prepared, selection, 0, configured)

    assert result.configuration is configured
    assert (
        result.configuration.elastic_area.initial_value
        == configured.elastic_area.initial_value
    )
    assert result.configuration.lorentzians[0].fwhm.initial_value == (
        configured.lorentzians[0].fwhm.initial_value
    )
    assert (
        result.parameter("elastic_area").value != configured.elastic_area.initial_value
    )
    start = result.diagnostics.alternative_starts[0]
    assert start.start_parameter_values == (
        configured.energy_shift.initial_value,
        configured.elastic_area.initial_value,
        configured.lorentzians[0].area.initial_value,
        configured.lorentzians[0].fwhm.initial_value,
    )
    assert start.fitted_parameter_values == tuple(
        parameter.value for parameter in result.parameters
    )


def test_canonicalization_reorders_all_component_linked_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth = model_definition(lorentzians=((0.2, 0.30), (0.4, 0.08)))
    prepared, selection = synthetic_problem(truth)
    configured = model_definition(
        energy_shift=0.0,
        elastic_area=0.7,
        lorentzians=((0.25, 0.28), (0.35, 0.09)),
    )
    fitted_original_order = np.array([0.01, 0.8, 0.2, 0.30, 0.4, 0.08])
    covariance_original_order = np.diag(
        np.array([1.0, 4.0, 9.0, 16.0, 25.0, 36.0]) * 1.0e-6
    )
    covariance_original_order[2, 3] = covariance_original_order[3, 2] = 3.0e-6
    covariance_original_order[4, 5] = covariance_original_order[5, 4] = -6.0e-6

    def fake_least_squares(
        function: Callable[[npt.ArrayLike], FloatArray],
        x0: npt.ArrayLike,
        **_: object,
    ) -> object:
        del x0
        residual = function(fitted_original_order)
        information = np.linalg.inv(covariance_original_order)
        square_jacobian = np.linalg.cholesky(information).T
        jacobian = np.vstack(
            (
                square_jacobian,
                np.zeros((residual.size - square_jacobian.shape[0], 6)),
            )
        )
        return SimpleNamespace(
            x=fitted_original_order,
            fun=residual,
            jac=jacobian,
            success=True,
            status=1,
            message="controlled success",
            nfev=2,
            njev=1,
            active_mask=np.zeros(6, dtype=np.int8),
        )

    monkeypatch.setattr(fitting_core, "least_squares", fake_least_squares)

    result = fit_single_q(prepared, selection, 0, configured)

    permutation = (0, 1, 4, 5, 2, 3)
    expected_covariance = covariance_original_order[np.ix_(permutation, permutation)]
    assert result.covariance is not None
    assert result.correlation is not None
    np.testing.assert_allclose(result.covariance, expected_covariance)
    expected_scales = np.sqrt(np.diag(expected_covariance))
    expected_correlation = expected_covariance / np.outer(
        expected_scales,
        expected_scales,
    )
    np.testing.assert_allclose(result.correlation, expected_correlation)
    expected_errors = np.sqrt(np.diag(expected_covariance))
    actual_errors = [parameter.standard_error for parameter in result.parameters]
    assert all(error is not None for error in actual_errors)
    np.testing.assert_allclose(
        np.asarray(actual_errors, dtype=np.float64),
        expected_errors,
    )
    assert result.parameter("lorentzian_1_area").value == 0.4
    assert result.parameter("lorentzian_1_fwhm").value == 0.08
    assert result.parameter("lorentzian_2_area").value == 0.2
    assert result.parameter("lorentzian_2_fwhm").value == 0.30
    assert result.diagnostics.lorentzian_areas == (0.4, 0.2)
    assert result.diagnostics.lorentzian_fwhm == (0.08, 0.30)
    start = result.diagnostics.alternative_starts[0]
    assert start.canonical_component_order == (1, 0)
    assert start.start_parameter_values[2:] == (0.25, 0.28, 0.35, 0.09)
    assert start.fitted_parameter_values == tuple(
        fitted_original_order[np.asarray(permutation)]
    )
    reconstructed = result.evaluation.elastic + result.evaluation.background
    for contribution in result.evaluation.lorentzians:
        reconstructed = reconstructed + contribution
    np.testing.assert_allclose(result.evaluation.total, reconstructed)
    plan = build_convolution_plan(prepared, 0)
    expected_narrow = evaluate_spectral_model(
        plan,
        model_definition(
            energy_shift=0.01,
            elastic_area=0.0,
            lorentzians=((0.4, 0.08),),
            free=False,
        ),
        result.evaluation.energy,
    )
    np.testing.assert_allclose(
        result.evaluation.lorentzians[0],
        expected_narrow.lorentzians[0],
    )


def test_rank_deficient_fit_suppresses_covariance_and_standard_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth = model_definition(lorentzians=((0.3, 0.12), (0.4, 0.12)))
    prepared, selection = synthetic_problem(truth)
    configured = SpectralModelDefinition(
        energy_shift=parameter(truth.energy_shift.initial_value, free=False),
        elastic_area=parameter(truth.elastic_area.initial_value, 0.0, free=False),
        lorentzians=(
            LorentzianComponent(
                area=parameter(0.25, 0.0),
                fwhm=parameter(0.12, 1.0e-8, free=False),
            ),
            LorentzianComponent(
                area=parameter(0.45, 0.0),
                fwhm=parameter(0.12, 1.0e-8, free=False),
            ),
        ),
    )

    def fake_least_squares(
        function: Callable[[npt.ArrayLike], FloatArray],
        x0: npt.ArrayLike,
        **_: object,
    ) -> object:
        values = np.asarray(x0, dtype=np.float64)
        residual = function(values)
        jacobian = np.ones((residual.size, values.size), dtype=np.float64)
        return SimpleNamespace(
            x=values,
            fun=residual,
            jac=jacobian,
            success=True,
            status=1,
            message="controlled rank deficiency",
            nfev=1,
            njev=1,
            active_mask=np.zeros(values.size, dtype=np.int8),
        )

    monkeypatch.setattr(fitting_core, "least_squares", fake_least_squares)

    result = fit_single_q(prepared, selection, 0, configured)

    assert result.diagnostics.optimizer_success
    assert result.diagnostics.jacobian_rank < result.statistics.free_parameters
    assert not result.diagnostics.covariance_available
    assert result.covariance is None
    assert result.correlation is None
    assert all(parameter.standard_error is None for parameter in result.parameters)


def test_successful_multistart_wins_over_lower_chi_square_failed_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth = model_definition(lorentzians=((0.5, 0.12),))
    prepared, selection = synthetic_problem(truth)
    call_index = 0

    def fake_least_squares(
        function: Callable[[npt.ArrayLike], FloatArray],
        x0: npt.ArrayLike,
        **_: object,
    ) -> object:
        nonlocal call_index
        current = call_index
        call_index += 1
        values = np.asarray(x0, dtype=np.float64)
        actual = function(values)
        controlled_residual = np.full_like(actual, float(current))
        success = current > 0
        return SimpleNamespace(
            x=values,
            fun=controlled_residual,
            jac=np.eye(actual.size, values.size),
            success=success,
            status=1 if success else 0,
            message="success" if success else "iteration limit",
            nfev=1,
            njev=1,
            active_mask=np.zeros(values.size, dtype=np.int8),
        )

    monkeypatch.setattr(fitting_core, "least_squares", fake_least_squares)

    result = fit_standard_candidate(
        prepared,
        selection,
        0,
        StandardModelCandidate(1, BackgroundModel.NONE),
    )

    alternatives = result.diagnostics.alternative_starts
    assert alternatives[0].chi_square < alternatives[1].chi_square
    assert not alternatives[0].success
    assert alternatives[1].success
    assert result.diagnostics.optimizer_success
    assert result.diagnostics.selected_start_index == 1
    assert (
        result.model.lorentzians[0].fwhm.initial_value
        == (alternatives[1].start_parameter_values[3])
    )


def test_generated_multistart_seeds_and_fitted_values_are_both_recoverable() -> None:
    truth = model_definition(lorentzians=((0.5, 0.12),))
    prepared, selection = synthetic_problem(truth)

    result = fit_standard_candidate(
        prepared,
        selection,
        0,
        StandardModelCandidate(1, BackgroundModel.NONE),
    )

    alternatives = result.diagnostics.alternative_starts
    assert len(alternatives) == 3
    assert len({start.start_parameter_values[3] for start in alternatives}) == 3
    assert all(
        len(start.start_parameter_values) == len(start.fitted_parameter_values) == 4
        for start in alternatives
    )
    assert alternatives[
        result.diagnostics.selected_start_index
    ].fitted_parameter_values == (
        tuple(parameter.value for parameter in result.parameters)
    )


def test_absolute_sigma_residual_and_unscaled_covariance_match_hand_case() -> None:
    truth = model_definition(energy_shift=0.0, elastic_area=0.7)
    prepared, selection = synthetic_problem(truth, sigma=0.02, noise_seed=701)
    configured = SpectralModelDefinition(
        energy_shift=parameter(0.0, free=False),
        elastic_area=parameter(0.6, 0.0),
    )
    plan = build_convolution_plan(prepared, 0)
    retained = selection.retained_mask(0)
    energy = prepared.sample_dataset.spectra[0].energy[retained]
    sigma = prepared.sample_dataset.spectra[0].uncertainty[retained]
    elastic_shape = np.interp(
        energy,
        plan.resolution_energy,
        plan.resolution_values,
        left=0.0,
        right=0.0,
    )
    expected_variance = 1.0 / float(np.sum(np.square(elastic_shape / sigma)))

    result = fit_single_q(prepared, selection, 0, configured)

    np.testing.assert_allclose(
        result.standardized_residuals,
        result.raw_residuals / sigma,
    )
    assert result.covariance is not None
    assert result.covariance[1, 1] == pytest.approx(expected_variance, rel=2.0e-5)
    assert result.parameter("elastic_area").standard_error == pytest.approx(
        math.sqrt(expected_variance),
        rel=2.0e-5,
    )
    k = result.statistics.free_parameters
    n = result.statistics.observations
    expected_aic = result.statistics.chi_square + 2.0 * k
    assert result.statistics.aic == pytest.approx(expected_aic)
    assert result.statistics.aicc == pytest.approx(
        expected_aic + 2.0 * k * (k + 1) / (n - k - 1)
    )
    assert result.statistics.bic == pytest.approx(
        result.statistics.chi_square + k * math.log(n)
    )


def test_optimizer_nonconvergence_is_returned_without_covariance() -> None:
    truth = model_definition(
        lorentzians=((0.35, 0.07), (0.3, 0.28)),
        background=BackgroundModel.LINEAR,
        b0=0.03,
        b1=-0.01,
    )
    prepared, selection = synthetic_problem(truth)

    result = fit_single_q(
        prepared,
        selection,
        0,
        perturbed(truth),
        max_nfev=1,
    )

    assert not result.diagnostics.optimizer_success
    assert result.diagnostics.optimizer_status == 0
    assert result.covariance is None
    assert all(parameter.standard_error is None for parameter in result.parameters)


def test_aicc_is_infinite_at_small_sample_boundary() -> None:
    truth = model_definition(energy_shift=0.0, elastic_area=0.7)
    prepared, selection = synthetic_problem(
        truth,
        sample_energy=np.array([-0.1, 0.0, 0.1]),
    )

    result = fit_single_q(prepared, selection, 0, perturbed(truth))

    assert result.statistics.observations == 3
    assert result.statistics.free_parameters == 2
    assert result.statistics.nominal_degrees_of_freedom == 1
    assert math.isinf(result.statistics.aicc)


def test_single_retained_energy_coordinate_fails_before_fitting() -> None:
    truth = model_definition()
    prepared, selection = synthetic_problem(
        truth,
        sample_energy=np.array([-0.1, 0.0, 0.1]),
    )
    one_point = selection.with_group_range(
        0,
        lower_energy=0.0,
        upper_energy=0.0,
    )

    with pytest.raises(FittingError, match="at least two retained"):
        fit_single_q(prepared, one_point, 0, model_definition(free=False))


def test_positive_degrees_of_freedom_and_dataset_match_are_preconditions() -> None:
    truth = model_definition()
    sample_energy = np.linspace(-0.1, 0.1, 3)
    prepared, selection = synthetic_problem(truth, sample_energy=sample_energy)
    too_complex = model_definition(lorentzians=((0.1, 0.1),))

    with pytest.raises(FittingError, match="degrees of freedom"):
        fit_single_q(prepared, selection, 0, too_complex)

    equivalent_selection = FittingSelection.uniform(
        prepared.sample_dataset,
        prepared.sample_padding,
        lower_energy=-0.1,
        upper_energy=0.1,
    )
    different_sample = ReducedDataset(
        role=SpectrumRole.SAMPLE,
        spectra=prepared.sample_dataset.spectra,
        q_bins=prepared.sample_dataset.q_bins,
    )
    mismatched = FittingSelection.uniform(
        different_sample,
        prepared.sample_padding,
        lower_energy=-0.1,
        upper_energy=0.1,
    )
    assert equivalent_selection.dataset is prepared.sample_dataset
    with pytest.raises(FittingError, match="prepared sample dataset"):
        fit_single_q(prepared, mismatched, 0, model_definition(free=False))


def test_manual_exclusion_is_applied_end_to_end_without_source_mutation() -> None:
    truth = model_definition(lorentzians=((0.4, 0.13),))
    prepared, selection = synthetic_problem(truth)
    spectrum = prepared.sample_dataset.spectra[0]
    originals = tuple(
        array.copy()
        for array in (spectrum.energy, spectrum.intensity, spectrum.uncertainty)
    )
    excluded_index = 83
    excluded_energy = float(spectrum.energy[excluded_index])
    manual = np.zeros(spectrum.energy.size, dtype=np.bool_)
    manual[excluded_index] = True
    selection = selection.with_group_manual_exclusion(0, manual)

    result = fit_single_q(prepared, selection, 0, perturbed(truth))

    assert result.statistics.observations == spectrum.energy.size - 1
    assert result.evaluation.energy.size == result.statistics.observations
    assert result.raw_residuals.size == result.statistics.observations
    assert result.standardized_residuals.size == result.statistics.observations
    assert excluded_energy not in result.evaluation.energy
    np.testing.assert_array_equal(
        result.evaluation.energy,
        spectrum.energy[selection.retained_mask(0)],
    )
    for current, original in zip(
        (spectrum.energy, spectrum.intensity, spectrum.uncertainty),
        originals,
        strict=True,
    ):
        np.testing.assert_array_equal(current, original)


def test_different_sample_resolution_grids_and_retained_selection_are_used() -> None:
    truth = model_definition(lorentzians=((0.4, 0.13),))
    sample_energy = np.linspace(-0.8, 0.8, 129)
    resolution_energy = np.linspace(-0.27, 0.31, 97)
    prepared, selection = synthetic_problem(
        truth,
        sample_energy=sample_energy,
        resolution_energy=resolution_energy,
    )
    selection = selection.with_group_range(0, lower_energy=-0.45, upper_energy=0.5)

    result = fit_single_q(prepared, selection, 0, perturbed(truth))

    retained = selection.retained_mask(0)
    np.testing.assert_array_equal(
        result.evaluation.energy,
        prepared.sample_dataset.spectra[0].energy[retained],
    )
    assert result.provenance.convolution_spacing < np.median(np.diff(sample_energy))
    resolution_provenance = result.provenance.resolution_acceptance
    assert resolution_provenance.group_index == 0
    assert resolution_provenance.group_label == "resolution-0"
    assert resolution_provenance.q_value == pytest.approx(0.75)
    assert resolution_provenance.decision is ResolutionAcceptanceDecision.KEEP
    assert resolution_provenance.confirmed
    assert resolution_provenance.original_support == (
        prepared.spectra[0].original_support.lower_energy,
        prepared.spectra[0].original_support.upper_energy,
    )
    assert resolution_provenance.accepted_support == (
        prepared.spectra[0].support.lower_energy,
        prepared.spectra[0].support.upper_energy,
    )
    assert resolution_provenance.signed_area_ratio == (
        prepared.spectra[0].signed_area_ratio
    )
    assert resolution_provenance.normalization_factor == (
        prepared.spectra[0].normalization_factor
    )


def test_candidate_generation_counts_and_has_no_structural_count_ceiling() -> None:
    nine = generate_standard_candidates()
    six = generate_standard_candidates(allow_linear_background=False)
    larger = generate_standard_candidates(max_lorentzians=5)

    assert len(nine) == 9
    assert len(six) == 6
    assert len(larger) == 18
    assert larger[-1].lorentzian_count == 5
    assert larger[-1].name == "E+L1+L2+L3+L4+L5+B1"


def test_standard_candidate_results_expose_evidence_without_a_winner_rule() -> None:
    truth = model_definition(
        lorentzians=((0.5, 0.12),),
        background=BackgroundModel.CONSTANT,
        b0=0.02,
    )
    prepared, selection = synthetic_problem(truth, noise_seed=88)

    results = evaluate_standard_candidates(prepared, selection, 0)

    assert len(results) == 9
    assert all(item.fit is not None for item in results)
    for item in results:
        assert item.fit is not None
        assert np.isfinite(item.fit.statistics.chi_square)
        assert np.isfinite(item.fit.statistics.bic)
        assert not hasattr(item, "recommended")
        assert all(component.center is None for component in item.fit.model.lorentzians)


def test_standard_initialization_above_two_lorentzians_is_explicitly_unvalidated() -> (
    None
):
    truth = model_definition()
    prepared, selection = synthetic_problem(truth)

    with pytest.raises(FittingError, match="above 2L"):
        fit_standard_candidate(
            prepared,
            selection,
            0,
            StandardModelCandidate(3, BackgroundModel.NONE),
        )


def sampled_fwhm(energy: FloatArray, values: FloatArray) -> float:
    peak = int(np.argmax(values))
    half = float(values[peak]) / 2.0
    left = int(np.flatnonzero(values[:peak] <= half)[-1])
    right = peak + 1 + int(np.flatnonzero(values[peak + 1 :] <= half)[0])
    left_crossing = float(
        np.interp(half, values[left : left + 2], energy[left : left + 2])
    )
    right_crossing = float(
        np.interp(
            half,
            values[right - 1 : right + 1][::-1],
            energy[right - 1 : right + 1][::-1],
        )
    )
    return right_crossing - left_crossing


@pytest.mark.parametrize("irregular", [False, True])
def test_dense_narrow_subbin_observable_is_stable_without_hidden_spikes(
    irregular: bool,
) -> None:
    regular_energy = np.arange(-0.6, 0.6001, 0.03)
    sample_energy = regular_energy.copy()
    if irregular:
        sample_energy[1:-1] += 0.003 * np.sin(np.arange(1, sample_energy.size - 1))
    truth = model_definition(elastic_area=0.0, lorentzians=((1.0, 0.01),), free=False)
    prepared, _ = synthetic_problem(truth, sample_energy=sample_energy)
    plan = build_convolution_plan(prepared, 0)
    narrow_fwhm = 0.05 * plan.spacing
    dense = np.arange(-0.25, 0.25001, plan.spacing / 10.0)
    phases = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9)
    metrics: list[tuple[float, float, float, float]] = []
    intrinsic = None
    for phase in phases:
        current = model_definition(
            energy_shift=phase * plan.spacing,
            elastic_area=0.0,
            lorentzians=((1.0, narrow_fwhm),),
            free=False,
        )
        evaluation = evaluate_spectral_model(plan, current, dense)
        values = evaluation.lorentzians[0]
        area = float(np.trapezoid(values, dense))
        centroid = float(np.trapezoid(dense * values, dense) / area)
        metrics.append(
            (float(np.max(values)), sampled_fwhm(dense, values), area, centroid)
        )
        shifted_resolution = np.interp(
            dense - phase * plan.spacing,
            plan.resolution_energy,
            plan.resolution_values,
            left=0.0,
            right=0.0,
        )
        relative_l2 = float(
            np.linalg.norm(values - shifted_resolution)
            / np.linalg.norm(shifted_resolution)
        )
        assert relative_l2 < 2.0e-3
        intrinsic = current
    metric_array = np.asarray(metrics)
    assert float(np.ptp(metric_array[:, 0]) / np.mean(metric_array[:, 0])) < 2.0e-4
    assert float(np.ptp(metric_array[:, 1]) / np.mean(metric_array[:, 1])) < 2.0e-4
    assert float(np.ptp(metric_array[:, 2]) / np.mean(metric_array[:, 2])) < 2.0e-4
    centered_centroids = metric_array[:, 3] - np.asarray(phases) * plan.spacing
    assert float(np.ptp(centered_centroids)) < plan.spacing * 2.0e-4
    assert intrinsic is not None
    working = plan.convolve(
        cell_integrated_lorentzian(
            plan.model_energy,
            fwhm=narrow_fwhm,
            spacing=plan.spacing,
        )
    )
    dense_values = evaluate_spectral_model(plan, intrinsic, dense).lorentzians[0]
    assert float(np.max(dense_values)) <= float(np.max(working.values)) * (
        1.0 + 1.0e-12
    )


def test_coarse_sampling_diagnostic_is_separate_from_intrinsic_width_ratio() -> None:
    sample_energy = np.arange(-0.6, 0.6001, 0.05)
    provisional = model_definition(
        elastic_area=0.0, lorentzians=((1.0, 0.01),), free=False
    )
    prepared, selection = synthetic_problem(provisional, sample_energy=sample_energy)
    plan = build_convolution_plan(prepared, 0)
    narrow = model_definition(
        elastic_area=0.0,
        lorentzians=((1.0, 0.05 * plan.spacing),),
        free=False,
    )
    prepared, selection = synthetic_problem(narrow, sample_energy=sample_energy)

    result = fit_single_q(prepared, selection, 0, narrow)

    linewidth_ratio = result.diagnostics.fwhm_to_resolution_fwhm[0]
    sampling_ratio = result.diagnostics.resolution_fwhm_to_sample_spacing
    assert linewidth_ratio is not None and linewidth_ratio < 0.01
    assert sampling_ratio is not None and sampling_ratio > 1.0
    assert result.diagnostics.median_sample_spacing == pytest.approx(0.05)
