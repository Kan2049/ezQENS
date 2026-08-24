"""Scientific tests for the thin single-Q Manual Fit bridge APIs."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

import ezqens.fitting.core as fitting_core
import ezqens.fitting.manual as manual_bridge
from ezqens.convolution import build_convolution_plan, cell_integrated_lorentzian
from ezqens.domain import QBins, ReducedDataset, Spectrum, SpectrumRole
from ezqens.fitting import (
    BackgroundModel,
    CenterGroup,
    LorentzianComponent,
    ManualInitializationError,
    ManualMaterializationError,
    ManualParameterIntent,
    ManualParameterKind,
    ParameterConfiguration,
    SpectralModelDefinition,
    evaluate_spectral_model,
    initialize_elastic_from_interaction,
    initialize_lorentzian_from_interaction,
    materialize_manual_parameter,
    preview_manual_model,
)
from ezqens.preprocessing import FittingSelection
from ezqens.resolution import PreparedResolution, prepare_measured_resolution

FloatArray = npt.NDArray[np.float64]


def gaussian(energy: npt.ArrayLike, *, sigma: float) -> FloatArray:
    coordinates = np.asarray(energy, dtype=np.float64)
    return np.asarray(
        np.exp(-0.5 * np.square(coordinates / sigma))
        / (sigma * math.sqrt(2.0 * math.pi)),
        dtype=np.float64,
    )


def manual_problem(
    *,
    resolution_sigma: float = 0.035,
    resolution_values: npt.ArrayLike | None = None,
) -> tuple[PreparedResolution, FittingSelection]:
    sample_energy = np.linspace(-0.75, 0.75, 151)
    resolution_energy = np.linspace(-0.3, 0.3, 121)
    sample_intensity = 0.25 + 0.04 * np.cos(3.0 * sample_energy)
    measured_resolution = (
        gaussian(resolution_energy, sigma=resolution_sigma)
        if resolution_values is None
        else np.asarray(resolution_values, dtype=np.float64)
    )
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
                uncertainty=np.linspace(0.02, 0.04, sample_energy.size),
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
                intensity=measured_resolution,
                uncertainty=np.full(resolution_energy.size, 1.0e-3),
                energy_unit="meV",
                intensity_unit="arb",
                uncertainty_unit="arb",
            ),
        ),
        q_bins=q_bins,
    )
    prepared = prepare_measured_resolution(sample, resolution)
    selection = FittingSelection.uniform(
        sample,
        prepared.sample_padding,
        lower_energy=-0.7,
        upper_energy=0.7,
    )
    return prepared, selection


def parameter(
    value: float,
    lower: float = -math.inf,
    upper: float = math.inf,
    *,
    free: bool = False,
) -> ParameterConfiguration:
    return ParameterConfiguration(value, lower, upper, free)


def test_elastic_interaction_height_uses_actual_measured_resolution() -> None:
    prepared, _ = manual_problem()
    desired_center = 0.04
    desired_height = 2.5

    seed = initialize_elastic_from_interaction(
        prepared,
        0,
        component_peak_center=desired_center,
        component_peak_height=desired_height,
    )
    plan = build_convolution_plan(prepared, 0)
    model = SpectralModelDefinition(
        energy_shift=parameter(seed.center_parameter_value),
        elastic_area=parameter(seed.integrated_area, 0.0),
    )
    evaluated = evaluate_spectral_model(plan, model, [desired_center])

    assert seed.integrated_area == pytest.approx(
        desired_height / float(np.max(plan.resolution_values))
    )
    assert evaluated.elastic[0] == pytest.approx(desired_height, rel=1.0e-12)
    assert evaluated.total[0] == evaluated.elastic[0]


def test_lorentzian_interaction_uses_convolved_width_and_component_height() -> None:
    prepared, _ = manual_problem()
    desired_center = -0.06
    desired_height = 1.7
    observed_fwhm = 0.22

    seed = initialize_lorentzian_from_interaction(
        prepared,
        0,
        component_peak_center=desired_center,
        component_peak_height=desired_height,
        observed_fwhm=observed_fwhm,
    )
    plan = build_convolution_plan(prepared, 0)
    intrinsic = cell_integrated_lorentzian(
        plan.model_energy,
        fwhm=seed.intrinsic_fwhm,
        spacing=plan.spacing,
    )
    profile = plan.convolve(intrinsic)
    model = SpectralModelDefinition(
        center_groups=(CenterGroup("new", parameter(seed.center_parameter_value)),),
        lorentzians=(
            LorentzianComponent(
                area=parameter(seed.integrated_area, 0.0),
                fwhm=parameter(seed.intrinsic_fwhm, 1.0e-12),
                center_group="new",
            ),
        ),
    )
    evaluated = evaluate_spectral_model(plan, model, [desired_center])

    forward_width = fitting_core._profile_fwhm(  # noqa: SLF001
        profile.energy,
        profile.values,
    )
    assert seed.intrinsic_fwhm > 0.0
    assert seed.forward_observed_fwhm == pytest.approx(forward_width)
    assert seed.absolute_width_mismatch == pytest.approx(
        abs(forward_width - observed_fwhm)
    )
    assert seed.absolute_width_mismatch < plan.spacing
    assert seed.integrated_area == pytest.approx(
        desired_height / float(np.max(profile.values))
    )
    assert evaluated.lorentzians[0][0] == pytest.approx(
        desired_height,
        rel=1.0e-10,
    )


def test_broader_measured_resolution_requires_smaller_intrinsic_width_seed() -> None:
    narrow, _ = manual_problem(resolution_sigma=0.025)
    broad, _ = manual_problem(resolution_sigma=0.070)

    narrow_seed = initialize_lorentzian_from_interaction(
        narrow,
        0,
        component_peak_center=0.0,
        component_peak_height=1.0,
        observed_fwhm=0.30,
    )
    broad_seed = initialize_lorentzian_from_interaction(
        broad,
        0,
        component_peak_center=0.0,
        component_peak_height=1.0,
        observed_fwhm=0.30,
    )

    assert broad_seed.intrinsic_fwhm < narrow_seed.intrinsic_fwhm


def test_asymmetric_resolution_produces_a_positive_approximate_width_seed() -> None:
    energy = np.linspace(-0.3, 0.3, 121)
    asymmetric = 0.82 * gaussian(energy + 0.012, sigma=0.030) + 0.18 * gaussian(
        energy - 0.055, sigma=0.065
    )
    prepared, _ = manual_problem(resolution_values=asymmetric)

    seed = initialize_lorentzian_from_interaction(
        prepared,
        0,
        component_peak_center=0.03,
        component_peak_height=1.2,
        observed_fwhm=0.19,
    )

    assert np.isfinite(seed.intrinsic_fwhm)
    assert seed.intrinsic_fwhm > 0.0
    assert np.isfinite(seed.forward_observed_fwhm)
    assert seed.forward_observed_fwhm > 0.0
    assert seed.absolute_width_mismatch < 0.01


def test_structured_two_peak_resolution_uses_deterministic_approximate_match() -> None:
    energy = np.linspace(-0.3, 0.3, 121)
    structured = gaussian(energy + 0.065, sigma=0.018) + 0.85 * gaussian(
        energy - 0.060, sigma=0.020
    )
    prepared, _ = manual_problem(resolution_values=structured)
    hint = 0.11

    first = initialize_lorentzian_from_interaction(
        prepared,
        0,
        component_peak_center=0.0,
        component_peak_height=1.0,
        observed_fwhm=hint,
    )
    second = initialize_lorentzian_from_interaction(
        prepared,
        0,
        component_peak_center=0.0,
        component_peak_height=1.0,
        observed_fwhm=hint,
    )
    plan = build_convolution_plan(prepared, 0)
    probe_widths = np.geomspace(0.04, 0.09, 401)
    probe_candidates = tuple(
        candidate
        for intrinsic_width in probe_widths
        if (
            candidate := manual_bridge._evaluate_width_candidate(  # noqa: SLF001
                plan,
                float(intrinsic_width),
                hint,
            )
        )
        is not None
    )
    forward_widths = np.asarray(
        [candidate.forward_observed_fwhm for candidate in probe_candidates]
    )

    assert first == second
    assert first.intrinsic_fwhm > 0.0
    assert first.absolute_width_mismatch > 1.0e-4
    assert (
        first.absolute_width_mismatch
        <= min(candidate.absolute_mismatch for candidate in probe_candidates) + 1.0e-4
    )
    assert np.max(np.diff(forward_widths)) > 0.05
    assert first.absolute_width_mismatch == pytest.approx(
        abs(first.forward_observed_fwhm - hint)
    )


def test_width_search_handles_a_nonmonotonic_forward_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, _ = manual_problem()
    plan = build_convolution_plan(prepared, 0)
    hint = 0.19

    def nonmonotonic_candidate(
        candidate_plan: object,
        intrinsic_fwhm: float,
        observed_fwhm: float,
    ) -> manual_bridge._WidthCandidate:  # noqa: SLF001
        assert candidate_plan is plan
        forward_width = 0.18 + 0.025 * np.sin(
            7.0 * np.log10(intrinsic_fwhm / plan.spacing)
        )
        return manual_bridge._WidthCandidate(  # noqa: SLF001
            intrinsic_fwhm=intrinsic_fwhm,
            forward_observed_fwhm=float(forward_width),
            absolute_mismatch=float(abs(forward_width - observed_fwhm)),
        )

    monkeypatch.setattr(
        manual_bridge,
        "_evaluate_width_candidate",
        nonmonotonic_candidate,
    )
    first = manual_bridge._intrinsic_fwhm_seed(plan, hint)  # noqa: SLF001
    second = manual_bridge._intrinsic_fwhm_seed(plan, hint)  # noqa: SLF001
    probe = np.geomspace(plan.spacing * 1.0e-6, plan.spacing * 256.0, 200)
    forward = np.asarray(
        [
            nonmonotonic_candidate(plan, float(width), hint).forward_observed_fwhm
            for width in probe
        ]
    )

    assert np.any(np.diff(forward) < 0.0)
    assert np.any(np.diff(forward) > 0.0)
    assert first == second
    assert first.intrinsic_fwhm > 0.0
    assert first.absolute_mismatch < 1.0e-4


def test_width_candidate_tie_break_prefers_smaller_intrinsic_width() -> None:
    wider = manual_bridge._WidthCandidate(  # noqa: SLF001
        intrinsic_fwhm=0.2,
        forward_observed_fwhm=0.3,
        absolute_mismatch=0.01,
    )
    narrower = manual_bridge._WidthCandidate(  # noqa: SLF001
        intrinsic_fwhm=0.1,
        forward_observed_fwhm=0.28,
        absolute_mismatch=0.01,
    )

    selected = manual_bridge._select_width_candidate((wider, narrower))  # noqa: SLF001

    assert selected is narrower


def test_width_hint_narrower_than_resolution_still_returns_best_positive_seed() -> None:
    prepared, _ = manual_problem()

    seed = initialize_lorentzian_from_interaction(
        prepared,
        0,
        component_peak_center=0.0,
        component_peak_height=0.0,
        observed_fwhm=0.01,
    )

    assert seed.integrated_area == 0.0
    assert np.isfinite(seed.intrinsic_fwhm)
    assert seed.intrinsic_fwhm > 0.0
    assert seed.forward_observed_fwhm > 0.0
    assert seed.absolute_width_mismatch > 0.0


@pytest.mark.parametrize(
    ("initializer", "arguments"),
    [
        (
            "elastic",
            {"component_peak_center": math.nan, "component_peak_height": 1.0},
        ),
        (
            "elastic",
            {"component_peak_center": 0.0, "component_peak_height": -1.0},
        ),
        (
            "lorentzian",
            {
                "component_peak_center": 0.0,
                "component_peak_height": math.inf,
                "observed_fwhm": 0.2,
            },
        ),
        (
            "lorentzian",
            {
                "component_peak_center": 0.0,
                "component_peak_height": 1.0,
                "observed_fwhm": 0.0,
            },
        ),
        (
            "lorentzian",
            {
                "component_peak_center": 0.0,
                "component_peak_height": 1.0,
                "observed_fwhm": math.nan,
            },
        ),
    ],
)
def test_invalid_interaction_hints_fail_without_physical_classification(
    initializer: str,
    arguments: dict[str, float],
) -> None:
    prepared, _ = manual_problem()

    with pytest.raises(ManualInitializationError) as error:
        if initializer == "elastic":
            initialize_elastic_from_interaction(prepared, 0, **arguments)
        else:
            initialize_lorentzian_from_interaction(prepared, 0, **arguments)

    message = str(error.value).lower()
    assert "inelastic" not in message
    assert "phonon" not in message
    assert "mechanism" not in message


def test_materialization_keeps_preview_value_separate_from_fit_limits() -> None:
    intent = ManualParameterIntent(
        current_value=0.30,
        user_upper_limit=0.20,
        free=True,
    )

    result = materialize_manual_parameter(
        intent,
        ManualParameterKind.POSITIVE_FWHM,
    )

    assert result.intent is intent
    assert result.intent.user_lower_limit is None
    assert result.preview_configuration.initial_value == 0.30
    assert np.isinf(result.preview_configuration.upper_bound)
    assert result.fit_configuration.initial_value < 0.20
    assert result.fit_configuration.initial_value > 0.0
    assert result.fit_configuration.upper_bound == 0.20
    assert result.fit_start_adjusted
    assert intent.current_value == 0.30


def test_materialization_without_or_with_valid_user_limits() -> None:
    unlimited = materialize_manual_parameter(
        ManualParameterIntent(0.8),
        ManualParameterKind.NONNEGATIVE_AREA,
    )
    limited = materialize_manual_parameter(
        ManualParameterIntent(0.2, user_lower_limit=-0.5, user_upper_limit=0.5),
        ManualParameterKind.UNCONSTRAINED,
    )

    assert unlimited.intent.user_lower_limit is None
    assert unlimited.intent.user_upper_limit is None
    assert unlimited.fit_configuration.lower_bound == 0.0
    assert np.isposinf(unlimited.fit_configuration.upper_bound)
    assert not unlimited.fit_start_adjusted
    assert limited.fit_configuration.lower_bound == -0.5
    assert limited.fit_configuration.upper_bound == 0.5
    assert limited.fit_configuration.initial_value == 0.2


@pytest.mark.parametrize(
    ("intent", "kind", "message"),
    [
        (
            ManualParameterIntent(0.2, user_lower_limit=1.0, user_upper_limit=0.0),
            ManualParameterKind.UNCONSTRAINED,
            "lower limit",
        ),
        (
            ManualParameterIntent(0.2, user_upper_limit=-0.1),
            ManualParameterKind.NONNEGATIVE_AREA,
            "no intersection",
        ),
        (
            ManualParameterIntent(0.2, user_lower_limit=0.1, user_upper_limit=0.1),
            ManualParameterKind.UNCONSTRAINED,
            "usable interval",
        ),
        (
            ManualParameterIntent(0.0),
            ManualParameterKind.POSITIVE_FWHM,
            "strictly positive",
        ),
    ],
)
def test_contradictory_or_intrinsically_invalid_materialization_blocks(
    intent: ManualParameterIntent,
    kind: ManualParameterKind,
    message: str,
) -> None:
    with pytest.raises(ManualMaterializationError, match=message):
        materialize_manual_parameter(intent, kind)


def test_center_group_intention_materializes_once_for_all_tied_members() -> None:
    prepared, selection = manual_problem()
    materialized = materialize_manual_parameter(
        ManualParameterIntent(0.015, user_lower_limit=-0.1, user_upper_limit=0.1),
        ManualParameterKind.CENTER,
        prepared_resolution=prepared,
        selection=selection,
        group_index=0,
    )
    group = CenterGroup("shared", materialized.fit_configuration)
    component = LorentzianComponent(
        area=parameter(0.3, 0.0),
        fwhm=parameter(0.12, 1.0e-12),
        center_group="shared",
    )
    model = SpectralModelDefinition(
        elastic_area=parameter(0.5, 0.0),
        center_groups=(group,),
        elastic_center_group="shared",
        lorentzians=(component,),
    )

    assert model.elastic_center() is materialized.fit_configuration
    assert model.lorentzian_center(component) is materialized.fit_configuration
    assert len(model.center_groups) == 1


def test_manual_preview_uses_exact_retained_points_and_existing_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, selection = manual_problem()
    spectrum = prepared.sample_dataset.spectra[0]
    manual_mask = np.zeros(spectrum.energy.size, dtype=np.bool_)
    manual_mask[[40, 72, 110]] = True
    selection = selection.with_group_manual_exclusion(0, manual_mask)
    model = SpectralModelDefinition(
        energy_shift=parameter(0.01),
        elastic_area=parameter(0.35, 0.0),
        lorentzians=(
            LorentzianComponent(
                area=parameter(0.4, 0.0),
                fwhm=parameter(0.16, 1.0e-12),
            ),
        ),
        background=BackgroundModel.CONSTANT,
        b0=parameter(0.02),
    )
    display_energy = np.linspace(-0.7, 0.7, 401)
    sample_originals = (
        spectrum.energy.copy(),
        spectrum.intensity.copy(),
        spectrum.uncertainty.copy(),
    )
    resolution_source = prepared.spectra[0].source_spectrum
    resolution_originals = (
        resolution_source.energy.copy(),
        resolution_source.intensity.copy(),
        resolution_source.uncertainty.copy(),
    )
    monkeypatch.setattr(
        fitting_core,
        "least_squares",
        lambda *args, **kwargs: pytest.fail("Manual preview invoked the optimizer"),
    )

    preview = preview_manual_model(
        prepared,
        selection,
        0,
        model,
        display_energy=display_energy,
    )
    retained = selection.retained_mask(0)
    plan = build_convolution_plan(prepared, 0)
    expected_display = evaluate_spectral_model(plan, model, display_energy)
    expected_retained = evaluate_spectral_model(
        plan,
        model,
        spectrum.energy[retained],
    )

    np.testing.assert_array_equal(preview.retained_energy, spectrum.energy[retained])
    np.testing.assert_array_equal(
        preview.retained_intensity,
        spectrum.intensity[retained],
    )
    np.testing.assert_array_equal(
        preview.retained_sigma, spectrum.uncertainty[retained]
    )
    assert not np.any(np.isin(spectrum.energy[manual_mask], preview.retained_energy))
    np.testing.assert_allclose(preview.display_evaluation.total, expected_display.total)
    np.testing.assert_allclose(
        preview.display_evaluation.elastic, expected_display.elastic
    )
    for actual, expected in zip(
        preview.display_evaluation.lorentzians,
        expected_display.lorentzians,
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(
        preview.retained_evaluation.total,
        expected_retained.total,
    )
    np.testing.assert_allclose(
        preview.raw_residuals,
        expected_retained.total - spectrum.intensity[retained],
    )
    np.testing.assert_allclose(
        preview.standardized_residuals,
        (expected_retained.total - spectrum.intensity[retained])
        / spectrum.uncertainty[retained],
    )
    assert not hasattr(preview, "recommendation")
    for current, original in zip(
        (spectrum.energy, spectrum.intensity, spectrum.uncertainty),
        sample_originals,
        strict=True,
    ):
        np.testing.assert_array_equal(current, original)
    for current, original in zip(
        (
            resolution_source.energy,
            resolution_source.intensity,
            resolution_source.uncertainty,
        ),
        resolution_originals,
        strict=True,
    ):
        np.testing.assert_array_equal(current, original)


def test_manual_preview_supports_models_without_elastic() -> None:
    prepared, selection = manual_problem()
    model = SpectralModelDefinition(
        background=BackgroundModel.CONSTANT,
        b0=parameter(0.18),
    )

    preview = preview_manual_model(prepared, selection, 0, model)

    assert np.all(preview.display_evaluation.elastic == 0.0)
    assert preview.display_evaluation.lorentzians == ()
    np.testing.assert_allclose(preview.display_evaluation.total, 0.18)


def test_materialization_rejects_nonfinite_intent_values() -> None:
    with pytest.raises(ManualMaterializationError, match="finite"):
        materialize_manual_parameter(
            ManualParameterIntent(math.nan),
            ManualParameterKind.UNCONSTRAINED,
        )
