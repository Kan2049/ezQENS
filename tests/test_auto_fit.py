"""Behavior tests for the deterministic production AutoFit policy."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

import ezqens.fitting.auto as auto_module
import ezqens.fitting.core as fitting_core
from ezqens.fitting import (
    AdditionalComplexityStatus,
    AlternativeStartResult,
    BackgroundModel,
    CandidateFitResult,
    FitDiagnostics,
    FitProvenance,
    FitResult,
    FitStatistics,
    InterpretationLimitationCode,
    LorentzianComponent,
    ModelEvaluation,
    ParameterConfiguration,
    ParameterEstimate,
    PrimaryFamilySupport,
    ResidualAdequacy,
    ResidualDiagnostics,
    ScientificWarningCode,
    SpectralModelDefinition,
    StandardModelCandidate,
    recommend_standard_candidates,
)
from ezqens.preprocessing import FittingSelection
from ezqens.resolution import (
    NORMALIZATION_METHOD,
    PreparedResolution,
    ResolutionAcceptanceDecision,
    ResolutionAcceptanceProvenance,
)

ALL_BACKGROUNDS = (
    BackgroundModel.NONE,
    BackgroundModel.CONSTANT,
    BackgroundModel.LINEAR,
)


def _configuration(candidate: StandardModelCandidate) -> SpectralModelDefinition:
    parameter = ParameterConfiguration
    return SpectralModelDefinition(
        energy_shift=parameter(0.0, -1.0, 1.0),
        elastic_area=parameter(1.0, 0.0, math.inf),
        lorentzians=tuple(
            LorentzianComponent(
                area=parameter(1.0, 0.0, math.inf),
                fwhm=parameter(0.1 * (index + 1), 1.0e-8, math.inf),
            )
            for index in range(candidate.lorentzian_count)
        ),
        background=candidate.background,
        b0=(
            parameter(0.0) if candidate.background is not BackgroundModel.NONE else None
        ),
        b1=(parameter(0.0) if candidate.background is BackgroundModel.LINEAR else None),
    )


def _parameter_names(candidate: StandardModelCandidate) -> tuple[str, ...]:
    names = ["energy_shift", "elastic_area"]
    for index in range(1, candidate.lorentzian_count + 1):
        names.extend((f"lorentzian_{index}_area", f"lorentzian_{index}_fwhm"))
    if candidate.background is not BackgroundModel.NONE:
        names.append("b0")
    if candidate.background is BackgroundModel.LINEAR:
        names.append("b1")
    return tuple(names)


def _candidate_result(
    candidate: StandardModelCandidate,
    *,
    aicc: float,
    bic: float,
    residual_rms: float = 1.0,
    residual_lag: float = 0.0,
    residual_run: int = 3,
    residual_trend: float = 0.0,
    residual_maximum: float = 3.0,
    area_to_se: tuple[float | None, ...] | None = None,
    covariance_available: bool = True,
    jacobian_rank_deficit: int = 0,
    active_bounds: tuple[str, ...] = (),
    condition_number: float = 100.0,
    adjacent_fwhm_ratios: tuple[float, ...] | None = None,
    nuisance_correlation: float | None = None,
    component_correlation: float | None = None,
    missing_standard_error: str | None = None,
    multistart_span: float = 0.0,
    nuisance_multistart_span: float = 0.0,
    component_area_multistart_span: float = 0.0,
    component_fwhm_multistart_span: float = 0.0,
    reverse_alternative_components: bool = False,
    equal_component_fwhm: bool = False,
    nonfinite_alternative_component: float | None = None,
) -> CandidateFitResult:
    names = _parameter_names(candidate)
    parameter_count = len(names)
    parameters = tuple(
        ParameterEstimate(
            name=name,
            value=1.0,
            standard_error=None if name == missing_standard_error else 0.1,
            lower_bound=-math.inf,
            upper_bound=math.inf,
            free=True,
            active_lower_bound=name in active_bounds,
        )
        for name in names
    )
    covariance = np.eye(parameter_count) if covariance_available else None
    correlation = np.eye(parameter_count) if covariance_available else None
    maximum_correlation: float | None = 0.0 if covariance_available else None
    if nuisance_correlation is not None:
        assert candidate.background is BackgroundModel.LINEAR
        assert correlation is not None
        first = names.index("b0")
        second = names.index("b1")
        correlation[first, second] = nuisance_correlation
        correlation[second, first] = nuisance_correlation
        maximum_correlation = abs(nuisance_correlation)
    if component_correlation is not None:
        assert candidate.lorentzian_count > 0
        assert correlation is not None
        first = names.index("lorentzian_1_area")
        second = names.index("lorentzian_1_fwhm")
        correlation[first, second] = component_correlation
        correlation[second, first] = component_correlation
        maximum_correlation = abs(component_correlation)
    fitted_values = tuple(
        (
            float(int(name.split("_")[1]))
            if name.startswith("lorentzian_") and name.endswith("_area")
            else (0.1 if equal_component_fwhm else 0.1 * float(int(name.split("_")[1])))
            if name.startswith("lorentzian_") and name.endswith("_fwhm")
            else 1.0
        )
        for name in names
    )
    alternative_values_list: list[float] = []
    for name, value in zip(names, fitted_values, strict=True):
        component = name.startswith("lorentzian_")
        span = multistart_span
        if not component:
            span += nuisance_multistart_span
        elif name.endswith("_area"):
            span += component_area_multistart_span
        else:
            span += component_fwhm_multistart_span
        alternative_values_list.append(value * (1.0 + span))
    if reverse_alternative_components:
        component_values = tuple(
            (
                alternative_values_list[2 + 2 * index],
                alternative_values_list[3 + 2 * index],
            )
            for index in range(candidate.lorentzian_count)
        )
        for index, (area, fwhm) in enumerate(reversed(component_values)):
            alternative_values_list[2 + 2 * index] = area
            alternative_values_list[3 + 2 * index] = fwhm
    if nonfinite_alternative_component is not None:
        assert candidate.lorentzian_count > 0
        alternative_values_list[2] = nonfinite_alternative_component
    alternative_values = tuple(alternative_values_list)
    starts = (
        AlternativeStartResult(
            start_index=0,
            success=True,
            status=1,
            chi_square=10.0,
            evaluations=10,
            elapsed_seconds=0.01,
            start_parameter_values=fitted_values,
            fitted_parameter_values=fitted_values,
            canonical_component_order=tuple(range(candidate.lorentzian_count)),
        ),
        AlternativeStartResult(
            start_index=1,
            success=True,
            status=1,
            chi_square=11.0,
            evaluations=10,
            elapsed_seconds=0.01,
            start_parameter_values=fitted_values,
            fitted_parameter_values=alternative_values,
            canonical_component_order=(
                tuple(reversed(range(candidate.lorentzian_count)))
                if reverse_alternative_components
                else tuple(range(candidate.lorentzian_count))
            ),
        ),
    )
    energy = np.asarray([-1.0, 0.0, 1.0])
    zeros = np.zeros(energy.size)
    model = ModelEvaluation(
        energy=energy,
        total=zeros,
        elastic=zeros,
        lorentzians=tuple(zeros for _ in range(candidate.lorentzian_count)),
        background=zeros,
    )
    free_parameters = candidate.nominal_parameter_count
    fit = FitResult(
        configuration=_configuration(candidate),
        parameters=parameters,
        covariance=covariance,
        correlation=correlation,
        evaluation=model,
        raw_residuals=zeros,
        standardized_residuals=zeros,
        statistics=FitStatistics(
            chi_square=10.0,
            reduced_chi_square=1.0,
            observations=40,
            free_parameters=free_parameters,
            nominal_degrees_of_freedom=40 - free_parameters,
            aic=aicc - 1.0,
            aicc=aicc,
            bic=bic,
        ),
        diagnostics=FitDiagnostics(
            optimizer_success=True,
            optimizer_status=1,
            optimizer_message="converged",
            function_evaluations=10,
            jacobian_evaluations=5,
            jacobian_rank=free_parameters - jacobian_rank_deficit,
            jacobian_singular_values=np.ones(free_parameters),
            condition_number=condition_number,
            covariance_available=covariance_available,
            maximum_absolute_correlation=maximum_correlation,
            active_bounds=active_bounds,
            relative_standard_errors=tuple(0.1 for _ in names),
            lorentzian_areas=tuple(1.0 for _ in range(candidate.lorentzian_count)),
            lorentzian_fwhm=tuple(
                0.1 * (index + 1) for index in range(candidate.lorentzian_count)
            ),
            adjacent_fwhm_ratios=(
                adjacent_fwhm_ratios
                if adjacent_fwhm_ratios is not None
                else tuple(2.0 for _ in range(max(candidate.lorentzian_count - 1, 0)))
            ),
            component_area_to_standard_error=(
                area_to_se
                if area_to_se is not None
                else tuple(10.0 for _ in range(candidate.lorentzian_count))
            ),
            lorentzian_full_convolution_areas=tuple(
                1.0 for _ in range(candidate.lorentzian_count)
            ),
            lorentzian_retained_sampled_trapezoid_areas=tuple(
                0.9 for _ in range(candidate.lorentzian_count)
            ),
            fwhm_to_resolution_fwhm=tuple(
                2.0 for _ in range(candidate.lorentzian_count)
            ),
            fwhm_to_fitting_window=tuple(
                0.1 for _ in range(candidate.lorentzian_count)
            ),
            resolution_fwhm=0.05,
            median_sample_spacing=0.01,
            resolution_fwhm_to_sample_spacing=5.0,
            residual=ResidualDiagnostics(
                mean=0.0,
                rms=residual_rms,
                maximum_absolute=residual_maximum,
                linear_trend=residual_trend,
                lag1_correlation=residual_lag,
                longest_same_sign_run=residual_run,
            ),
            alternative_starts=starts,
            selected_start_index=0,
            total_elapsed_seconds=0.02,
        ),
        provenance=FitProvenance(
            group_index=0,
            group_label="synthetic-group",
            q_value=1.0,
            energy_unit="meV",
            optimizer="scipy.optimize.least_squares",
            optimizer_method="trf",
            residual_definition="(model - data) / sigma",
            sigma_interpretation="absolute",
            convolution_spacing=0.01,
            retained_energy_bounds=(-1.0, 1.0),
            model_energy_bounds=(-2.0, 2.0),
            convolution_energy_bounds=(-2.0, 2.0),
            resolution_acceptance=ResolutionAcceptanceProvenance(
                group_index=0,
                group_label="synthetic-resolution",
                q_value=1.0,
                source_reference=None,
                original_support=(-0.2, 0.2),
                accepted_support=(-0.2, 0.2),
                decision=ResolutionAcceptanceDecision.KEEP,
                retained_pre_normalization_area=1.0,
                signed_area_ratio=1.0,
                normalization_method=NORMALIZATION_METHOD,
                normalization_factor=1.0,
                confirmed=True,
                warnings=(),
                auto_padding_applied=True,
            ),
        ),
    )
    return CandidateFitResult(candidate, fit)


def _results(
    aicc: Mapping[tuple[int, BackgroundModel], float],
    *,
    bic: Mapping[tuple[int, BackgroundModel], float] | None = None,
    overrides: Mapping[tuple[int, BackgroundModel], Mapping[str, Any]] | None = None,
    failed_counts: tuple[int, ...] = (),
) -> tuple[CandidateFitResult, ...]:
    results: list[CandidateFitResult] = []
    for count in (0, 1, 2):
        for background in ALL_BACKGROUNDS:
            candidate = StandardModelCandidate(count, background)
            if count in failed_counts:
                results.append(
                    CandidateFitResult(
                        candidate,
                        None,
                        error_type="SyntheticFailure",
                        error_message="synthetic candidate failure",
                    )
                )
                continue
            key = (count, background)
            options = dict(overrides.get(key, {})) if overrides else {}
            results.append(
                _candidate_result(
                    candidate,
                    aicc=aicc[key],
                    bic=(bic or aicc)[key],
                    **options,
                )
            )
    return tuple(results)


def _scores(
    rows: tuple[tuple[float, float, float], ...],
) -> dict[tuple[int, BackgroundModel], float]:
    return {
        (count, background): rows[count][index]
        for count in (0, 1, 2)
        for index, background in enumerate(ALL_BACKGROUNDS)
    }


def _fail_candidates(
    evidence: tuple[CandidateFitResult, ...],
    candidates: set[StandardModelCandidate],
) -> tuple[CandidateFitResult, ...]:
    return tuple(
        CandidateFitResult(
            item.candidate,
            None,
            error_type="SyntheticFailure",
            error_message="synthetic candidate failure",
        )
        if item.candidate in candidates
        else item
        for item in evidence
    )


def _replace_aicc(
    evidence: tuple[CandidateFitResult, ...],
    *,
    lorentzian_count: int,
    aicc: float,
) -> tuple[CandidateFitResult, ...]:
    updated: list[CandidateFitResult] = []
    for item in evidence:
        if item.candidate.lorentzian_count != lorentzian_count:
            updated.append(item)
            continue
        assert item.fit is not None
        updated.append(
            replace(
                item,
                fit=replace(
                    item.fit,
                    statistics=replace(item.fit.statistics, aicc=aicc),
                ),
            )
        )
    return tuple(updated)


def _reference_name(result: CandidateFitResult | None) -> str | None:
    return result.candidate.name if result is not None else None


def _recommendation_semantics(
    recommendation: auto_module.AutoFitRecommendation,
) -> tuple[object, ...]:
    additional = recommendation.additional_complexity
    return (
        recommendation.recommended_candidate.name
        if recommendation.recommended_candidate is not None
        else None,
        recommendation.recommended_lorentzian_count,
        recommendation.primary_family_support,
        recommendation.primary_residual_adequacy,
        recommendation.primary_identifiability,
        (
            additional.simpler_lorentzian_count,
            additional.complex_lorentzian_count,
            additional.status,
            additional.information_criteria,
            additional.background_robustness,
            additional.matched_backgrounds,
            _reference_name(additional.proposed_candidate),
            tuple(item.code for item in additional.interpretation_limitations),
            additional.reason,
        ),
        tuple(
            (
                item.simpler_lorentzian_count,
                item.complex_lorentzian_count,
                item.status,
                _reference_name(item.proposed_candidate),
                item.reason,
            )
            for item in recommendation.transition_assessments
        ),
        _reference_name(recommendation.strong_alternative),
        _reference_name(recommendation.comparator),
        tuple(
            (item.code, item.message, _reference_name(item.candidate))
            for item in recommendation.scientific_warnings
        ),
        tuple(
            (item.code, item.message, _reference_name(item.candidate))
            for item in recommendation.interpretation_limitations
        ),
        recommendation.search_scope,
        recommendation.resolution_reliability,
    )


def test_marginal_discrimination_keeps_primary_and_strong_alternative() -> None:
    scores = _scores(((20.0, 10.0, 0.0), (20.0, 15.0, -3.0), (30.0, 30.0, 30.0)))
    evidence = _results(scores)

    first = recommend_standard_candidates(evidence)
    second = recommend_standard_candidates(evidence)
    reversed_order = recommend_standard_candidates(tuple(reversed(evidence)))

    assert first.recommended_candidate == StandardModelCandidate(
        0, BackgroundModel.LINEAR
    )
    assert first.primary_family_support is PrimaryFamilySupport.SUPPORTED
    assert first.additional_complexity.status is AdditionalComplexityStatus.MARGINAL
    assert first.strong_alternative is not None
    assert first.strong_alternative.candidate == StandardModelCandidate(
        1, BackgroundModel.LINEAR
    )
    assert second.recommended_candidate == first.recommended_candidate
    assert second.additional_complexity == first.additional_complexity
    assert reversed_order.recommended_candidate == first.recommended_candidate
    assert (
        reversed_order.additional_complexity.status
        is first.additional_complexity.status
    )
    assert first.candidate_results is evidence
    assert first.resolution_reliability.structured_containment_assessment_available is (
        False
    )
    assert first.resolution_reliability.provenance_gap is not None


def test_clear_identifiable_and_background_robust_complexity_upgrades() -> None:
    scores = _scores(
        (
            (120.0, 100.0, 101.0),
            (100.0, 80.0, 85.0),
            (110.0, 95.0, 96.0),
        )
    )

    recommendation = recommend_standard_candidates(_results(scores))

    assert recommendation.recommended_candidate == StandardModelCandidate(
        1, BackgroundModel.CONSTANT
    )
    assert recommendation.transition_assessments[0].status is (
        AdditionalComplexityStatus.SUPPORTED_TRANSITION
    )
    assert recommendation.additional_complexity.status is (
        AdditionalComplexityStatus.NOT_SUPPORTED
    )
    assert recommendation.primary_family_support is PrimaryFamilySupport.SUPPORTED
    assert not recommendation.auto_search_exhausted
    assert not recommendation.higher_complexity_manual_fit_available


def test_favored_but_unidentifiable_higher_family_limits_only_complexity() -> None:
    scores = _scores(
        (
            (125.0, 120.0, 125.0),
            (105.0, 100.0, 105.0),
            (85.0, 80.0, 85.0),
        )
    )
    evidence = _results(
        scores,
        overrides={
            (2, BackgroundModel.CONSTANT): {
                "area_to_se": (10.0, 0.5),
                "active_bounds": ("energy_shift:lower",),
            },
        },
    )

    recommendation = recommend_standard_candidates(evidence)

    assert recommendation.recommended_lorentzian_count == 1
    assert recommendation.primary_family_support is PrimaryFamilySupport.SUPPORTED
    assert recommendation.additional_complexity.status is (
        AdditionalComplexityStatus.SUPPORTED_BUT_UNINTERPRETABLE
    )
    additional_limitations = (
        recommendation.additional_complexity.interpretation_limitations
    )
    assert any(
        limitation.code is InterpretationLimitationCode.COMPONENT_AREA_UNCONSTRAINED
        for limitation in additional_limitations
    )
    assert any(
        limitation.code is InterpretationLimitationCode.MATERIAL_ACTIVE_BOUND
        for limitation in additional_limitations
    )


def test_background_substitution_rejects_transition_without_demoting_primary() -> None:
    scores = _scores(
        (
            (100.0, 80.0, 90.0),
            (81.0, 90.0, 95.0),
            (110.0, 100.0, 105.0),
        )
    )

    recommendation = recommend_standard_candidates(
        _results(
            scores,
            overrides={
                (1, BackgroundModel.NONE): {"area_to_se": (0.5,)},
            },
        )
    )

    assert recommendation.recommended_candidate == StandardModelCandidate(
        0, BackgroundModel.CONSTANT
    )
    assert recommendation.primary_family_support is PrimaryFamilySupport.SUPPORTED
    assert recommendation.additional_complexity.status is (
        AdditionalComplexityStatus.BACKGROUND_CONFOUNDED
    )
    assert recommendation.additional_complexity.interpretation_limitations
    assert any(
        warning.code is ScientificWarningCode.BACKGROUND_CONFOUNDED_COMPLEXITY
        for warning in recommendation.scientific_warnings
    )


def test_no_adequate_model_and_numerical_failure_are_distinct() -> None:
    scores = _scores(
        (
            (10.0, 0.0, 20.0),
            (30.0, 20.0, 40.0),
            (50.0, 40.0, 60.0),
        )
    )
    inadequate = {key: {"residual_rms": 1.7} for key in scores}
    no_adequate = recommend_standard_candidates(_results(scores, overrides=inadequate))
    unavailable = recommend_standard_candidates(_results(scores, failed_counts=(0,)))

    assert no_adequate.most_recommended is None
    assert no_adequate.primary_family_support is PrimaryFamilySupport.NO_ADEQUATE_MODEL
    assert unavailable.most_recommended is None
    assert unavailable.primary_family_support is (
        PrimaryFamilySupport.NUMERICAL_RECOMMENDATION_UNAVAILABLE
    )
    assert unavailable.additional_complexity.status is (
        AdditionalComplexityStatus.NOT_EVALUABLE
    )


def test_same_family_uses_residual_eligible_candidate_not_ic_envelope() -> None:
    scores = _scores(
        (
            (5.0, 10.0, 10.0),
            (30.0, 35.0, 40.0),
            (50.0, 55.0, 60.0),
        )
    )
    evidence = _results(
        scores,
        overrides={(0, BackgroundModel.NONE): {"residual_rms": 1.7}},
    )

    recommendation = recommend_standard_candidates(evidence)
    reversed_recommendation = recommend_standard_candidates(tuple(reversed(evidence)))

    assert recommendation.recommended_candidate == StandardModelCandidate(
        0, BackgroundModel.CONSTANT
    )
    assert recommendation.primary_residual_adequacy is ResidualAdequacy.ADEQUATE
    assert recommendation.primary_family_support is (
        PrimaryFamilySupport.SUPPORTED_WITH_CAUTION
    )
    assert reversed_recommendation.recommended_candidate == (
        recommendation.recommended_candidate
    )
    assert recommendation.additional_complexity.information_criteria.delta_aicc == (
        pytest.approx(5.0 - 30.0)
    )


def test_ic_supported_inadequate_higher_family_does_not_replace_adequate_primary() -> (
    None
):
    scores = _scores(
        (
            (100.0, 80.0, 90.0),
            (60.0, 55.0, 50.0),
            (120.0, 115.0, 110.0),
        )
    )
    inadequate_one_l = {key: {"residual_rms": 1.7} for key in scores if key[0] == 1}

    recommendation = recommend_standard_candidates(
        _results(scores, overrides=inadequate_one_l)
    )

    assert recommendation.recommended_candidate == StandardModelCandidate(
        0, BackgroundModel.CONSTANT
    )
    assert recommendation.primary_residual_adequacy is ResidualAdequacy.ADEQUATE
    assert recommendation.primary_family_support is PrimaryFamilySupport.SUPPORTED
    assert recommendation.additional_complexity.status is (
        AdditionalComplexityStatus.SUPPORTED_BUT_RESIDUALLY_INADEQUATE
    )
    assert recommendation.additional_complexity.information_criteria.evidence in {
        auto_module.ComplexityEvidence.CLEAR,
        auto_module.ComplexityEvidence.STRONG,
    }
    assert (
        recommendation.comparator
        is recommendation.additional_complexity.proposed_candidate
    )
    with pytest.raises(ValueError, match="recommendation-eligible"):
        replace(
            recommendation.additional_complexity,
            status=AdditionalComplexityStatus.SUPPORTED_TRANSITION,
        )


def test_inadequate_higher_family_does_not_replace_questionable_primary() -> None:
    scores = _scores(
        (
            (100.0, 80.0, 90.0),
            (60.0, 55.0, 50.0),
            (120.0, 115.0, 110.0),
        )
    )
    overrides: dict[tuple[int, BackgroundModel], Mapping[str, Any]] = {
        (0, BackgroundModel.CONSTANT): {
            "residual_rms": 1.2,
            "residual_lag": 0.2,
        }
    }
    overrides.update({key: {"residual_rms": 1.7} for key in scores if key[0] == 1})

    recommendation = recommend_standard_candidates(
        _results(scores, overrides=overrides)
    )

    assert recommendation.recommended_candidate == StandardModelCandidate(
        0, BackgroundModel.CONSTANT
    )
    assert recommendation.primary_residual_adequacy is ResidualAdequacy.QUESTIONABLE
    assert recommendation.primary_family_support is (
        PrimaryFamilySupport.SUPPORTED_WITH_CAUTION
    )
    assert recommendation.additional_complexity.status is (
        AdditionalComplexityStatus.SUPPORTED_BUT_RESIDUALLY_INADEQUATE
    )


def test_inadequate_lower_family_can_progress_to_adequate_supported_family() -> None:
    scores = _scores(
        (
            (140.0, 135.0, 130.0),
            (115.0, 110.0, 105.0),
            (150.0, 145.0, 140.0),
        )
    )
    inadequate_zero_l = {key: {"residual_rms": 1.7} for key in scores if key[0] == 0}

    recommendation = recommend_standard_candidates(
        _results(scores, overrides=inadequate_zero_l)
    )

    assert recommendation.recommended_candidate == StandardModelCandidate(
        1, BackgroundModel.LINEAR
    )
    assert recommendation.primary_residual_adequacy is ResidualAdequacy.ADEQUATE
    assert recommendation.transition_assessments[0].status is (
        AdditionalComplexityStatus.SUPPORTED_TRANSITION
    )
    assert recommendation.primary_family_support is PrimaryFamilySupport.SUPPORTED
    with pytest.raises(ValueError, match="requires an inadequate"):
        replace(
            recommendation.transition_assessments[0],
            status=(AdditionalComplexityStatus.SUPPORTED_BUT_RESIDUALLY_INADEQUATE),
        )


def test_marginal_transition_reaches_eligible_higher_family_with_caution() -> None:
    scores = _scores(
        (
            (100.0, 95.0, 90.0),
            (97.0, 92.0, 87.0),
            (110.0, 105.0, 100.0),
        )
    )
    inadequate_zero_l = {key: {"residual_rms": 1.7} for key in scores if key[0] == 0}

    recommendation = recommend_standard_candidates(
        _results(scores, overrides=inadequate_zero_l)
    )

    assert recommendation.recommended_candidate == StandardModelCandidate(
        1, BackgroundModel.LINEAR
    )
    assert recommendation.primary_residual_adequacy is ResidualAdequacy.ADEQUATE
    assert recommendation.primary_family_support is (
        PrimaryFamilySupport.SUPPORTED_WITH_CAUTION
    )
    assert recommendation.transition_assessments[0].status is (
        AdditionalComplexityStatus.MARGINAL
    )
    assert recommendation.transition_assessments[0].information_criteria.evidence is (
        auto_module.ComplexityEvidence.MARGINAL_OR_CONFLICTING
    )
    assert recommendation.additional_complexity.status is (
        AdditionalComplexityStatus.NOT_SUPPORTED
    )


def test_inadequate_two_l_exhausts_auto_scope_without_promoting_a_model() -> None:
    scores = _scores(
        (
            (140.0, 135.0, 130.0),
            (115.0, 110.0, 105.0),
            (90.0, 85.0, 80.0),
        )
    )
    inadequate = {key: {"residual_rms": 1.7} for key in scores}

    recommendation = recommend_standard_candidates(
        _results(scores, overrides=inadequate)
    )

    assert len(recommendation.candidate_results) == 9
    assert recommendation.most_recommended is None
    assert recommendation.recommended_lorentzian_count is None
    assert recommendation.primary_family_support is (
        PrimaryFamilySupport.NO_ADEQUATE_MODEL
    )
    assert recommendation.primary_residual_adequacy is ResidualAdequacy.INADEQUATE
    assert recommendation.auto_search_exhausted
    assert recommendation.additional_complexity.status is (
        AdditionalComplexityStatus.SEARCH_LIMIT_REACHED
    )
    assert recommendation.higher_complexity_manual_fit_available
    assert recommendation.comparator is not None
    assert recommendation.comparator.candidate.lorentzian_count == 2
    assert tuple(
        (item.simpler_lorentzian_count, item.complex_lorentzian_count, item.status)
        for item in recommendation.transition_assessments
    ) == (
        (
            0,
            1,
            AdditionalComplexityStatus.SUPPORTED_BUT_RESIDUALLY_INADEQUATE,
        ),
        (
            1,
            2,
            AdditionalComplexityStatus.SUPPORTED_BUT_RESIDUALLY_INADEQUATE,
        ),
    )
    other_two_l = next(
        item
        for item in recommendation.candidate_results
        if item.candidate.lorentzian_count == 2
        and item is not recommendation.comparator
    )
    with pytest.raises(ValueError, match="final evaluated 2L candidate"):
        replace(recommendation, comparator=other_two_l)


def test_advisory_condition_width_and_nuisance_correlation_are_not_hard_gates() -> None:
    scores = _scores(
        (
            (140.0, 135.0, 130.0),
            (115.0, 110.0, 105.0),
            (90.0, 85.0, 80.0),
        )
    )
    evidence = _results(
        scores,
        overrides={
            (2, BackgroundModel.LINEAR): {
                "condition_number": 1.0e9,
                "adjacent_fwhm_ratios": (1.1,),
                "nuisance_correlation": 0.999,
            }
        },
    )

    recommendation = recommend_standard_candidates(evidence)
    warning_codes = {warning.code for warning in recommendation.scientific_warnings}

    assert recommendation.recommended_candidate == StandardModelCandidate(
        2, BackgroundModel.LINEAR
    )
    assert recommendation.additional_complexity.status is (
        AdditionalComplexityStatus.SEARCH_LIMIT_REACHED
    )
    assert recommendation.primary_family_support is PrimaryFamilySupport.SUPPORTED
    assert ScientificWarningCode.HIGH_FINITE_CONDITION_NUMBER in warning_codes
    assert ScientificWarningCode.SIMILAR_INTRINSIC_LINEWIDTHS in warning_codes
    assert recommendation.primary_identifiability.value == "interpretable"
    assert recommendation.search_scope.lorentzian_counts == (0, 1, 2)
    assert recommendation.search_scope.backgrounds == ALL_BACKGROUNDS
    assert len(recommendation.search_scope.candidates) == 9
    assert recommendation.auto_search_exhausted
    assert recommendation.higher_complexity_manual_fit_available


def test_complete_standard_candidate_lattice_is_required() -> None:
    scores = _scores(((20.0, 10.0, 0.0), (30.0, 20.0, 10.0), (40.0, 30.0, 20.0)))
    evidence = _results(scores)

    with pytest.raises(
        ValueError, match="complete nine-candidate.*missing: E\\+L1\\+L2\\+B1"
    ):
        recommend_standard_candidates(evidence[:-1])

    with pytest.raises(ValueError, match="duplicate candidates"):
        recommend_standard_candidates((*evidence, evidence[0]))

    out_of_scope = CandidateFitResult(
        StandardModelCandidate(3, BackgroundModel.LINEAR),
        None,
        error_type="SyntheticFailure",
        error_message="out of scope",
    )
    with pytest.raises(ValueError, match="out-of-scope candidates"):
        recommend_standard_candidates((*evidence[:-1], out_of_scope))


def test_complete_lattice_accepts_explicit_candidate_failures() -> None:
    scores = _scores(
        (
            (120.0, 100.0, 101.0),
            (100.0, 80.0, 85.0),
            (110.0, 95.0, 96.0),
        )
    )
    evidence = _results(scores)
    failed = _fail_candidates(
        evidence,
        {StandardModelCandidate(2, BackgroundModel.LINEAR)},
    )

    recommendation = recommend_standard_candidates(failed)

    assert len(recommendation.candidate_results) == 9
    assert sum(not item.success for item in recommendation.candidate_results) == 1
    assert recommendation.recommended_candidate == StandardModelCandidate(
        1, BackgroundModel.CONSTANT
    )


@pytest.mark.parametrize(
    "wrong_configuration",
    (
        StandardModelCandidate(0, BackgroundModel.CONSTANT),
        StandardModelCandidate(1, BackgroundModel.LINEAR),
    ),
)
def test_successful_fit_configuration_must_match_candidate_key(
    wrong_configuration: StandardModelCandidate,
) -> None:
    scores = _scores(((30.0, 20.0, 10.0), (20.0, 10.0, 0.0), (40.0, 30.0, 20.0)))
    evidence = list(_results(scores))
    index = next(
        index
        for index, item in enumerate(evidence)
        if item.candidate == StandardModelCandidate(1, BackgroundModel.CONSTANT)
    )
    result = evidence[index]
    assert result.fit is not None
    evidence[index] = replace(
        result,
        fit=replace(result.fit, configuration=_configuration(wrong_configuration)),
    )

    with pytest.raises(ValueError, match="fit configuration disagrees"):
        recommend_standard_candidates(tuple(evidence))


def _with_independent_center_parameter(
    fit: FitResult,
    component_number: int,
) -> FitResult:
    insertion = next(
        index + 1
        for index, parameter in enumerate(fit.parameters)
        if parameter.name == f"lorentzian_{component_number}_fwhm"
    )
    parameters = list(fit.parameters)
    parameters.insert(
        insertion,
        ParameterEstimate(
            name=f"lorentzian_{component_number}_center",
            value=0.03,
            standard_error=0.01,
            lower_bound=-0.2,
            upper_bound=0.2,
            free=True,
        ),
    )
    parameter_count = len(parameters)
    return replace(
        fit,
        parameters=tuple(parameters),
        covariance=np.eye(parameter_count),
        correlation=np.eye(parameter_count),
    )


def test_independently_centered_1l_fit_is_rejected_as_standard_auto_evidence() -> None:
    scores = _scores(((30.0, 31.0, 32.0), (10.0, 11.0, 12.0), (20.0, 21.0, 22.0)))
    evidence = list(_results(scores))
    target = StandardModelCandidate(1, BackgroundModel.NONE)
    index = next(i for i, item in enumerate(evidence) if item.candidate == target)
    item = evidence[index]
    assert item.fit is not None
    component = item.fit.configuration.lorentzians[0]
    independent_configuration = replace(
        item.fit.configuration,
        lorentzians=(
            replace(
                component,
                center=ParameterConfiguration(0.03, -0.2, 0.2),
            ),
        ),
    )
    independent_fit = replace(
        _with_independent_center_parameter(item.fit, 1),
        configuration=independent_configuration,
    )
    evidence[index] = replace(item, fit=independent_fit)

    with pytest.raises(ValueError, match="must use the shared energy_shift"):
        recommend_standard_candidates(tuple(evidence))


def test_2l_center_parameter_schema_is_rejected_before_auto_interpretation() -> None:
    scores = _scores(((30.0, 31.0, 32.0), (20.0, 21.0, 22.0), (10.0, 11.0, 12.0)))
    evidence = list(_results(scores))
    target = StandardModelCandidate(2, BackgroundModel.CONSTANT)
    index = next(i for i, item in enumerate(evidence) if item.candidate == target)
    item = evidence[index]
    assert item.fit is not None
    evidence[index] = replace(
        item,
        fit=_with_independent_center_parameter(item.fit, 2),
    )

    with pytest.raises(ValueError, match="parameter schema.*independent Lorentzian"):
        recommend_standard_candidates(tuple(evidence))


def test_complete_shared_center_lattice_remains_accepted_unchanged() -> None:
    scores = _scores(((20.0, 10.0, 0.0), (40.0, 30.0, 20.0), (50.0, 40.0, 30.0)))
    evidence = _results(scores)

    recommendation = recommend_standard_candidates(evidence)

    assert recommendation.recommended_candidate == StandardModelCandidate(
        0,
        BackgroundModel.LINEAR,
    )
    assert recommendation.candidate_results is evidence
    assert len(recommendation.candidate_results) == 9
    for item in recommendation.candidate_results:
        assert item.fit is not None
        assert all(
            component.center is None for component in item.fit.configuration.lorentzians
        )
        assert not any(
            parameter.name.startswith("lorentzian_")
            and parameter.name.endswith("_center")
            for parameter in item.fit.parameters
        )


def test_tied_background_alternative_is_fully_input_order_invariant() -> None:
    scores = _scores(
        (
            (110.0, 111.0, 111.0),
            (80.0, 81.0, 81.0),
            (100.0, 101.0, 101.0),
        )
    )
    evidence = _results(scores)

    forward = recommend_standard_candidates(evidence)
    reverse = recommend_standard_candidates(tuple(reversed(evidence)))

    assert forward.strong_alternative is not None
    assert forward.strong_alternative.candidate == StandardModelCandidate(
        1, BackgroundModel.CONSTANT
    )
    assert _recommendation_semantics(reverse) == _recommendation_semantics(forward)


def test_public_recommendation_records_reject_contradictory_states() -> None:
    scores = _scores(((20.0, 10.0, 0.0), (20.0, 15.0, -3.0), (30.0, 30.0, 30.0)))
    recommendation = recommend_standard_candidates(_results(scores))
    assert recommendation.most_recommended is not None

    with pytest.raises(ValueError, match="count must match"):
        replace(recommendation, recommended_lorentzian_count=2)
    with pytest.raises(ValueError, match="ordinary supported primary"):
        replace(
            recommendation,
            most_recommended=None,
            recommended_lorentzian_count=None,
        )
    with pytest.raises(ValueError, match="Strong Alternative must differ"):
        replace(
            recommendation,
            strong_alternative=recommendation.most_recommended,
        )
    with pytest.raises(ValueError, match="next searched family"):
        replace(
            recommendation.additional_complexity,
            complex_lorentzian_count=2,
        )
    with pytest.raises(ValueError, match="stated higher family"):
        replace(
            recommendation.additional_complexity,
            proposed_candidate=recommendation.most_recommended,
        )
    with pytest.raises(ValueError, match="production AutoFit search scope"):
        auto_module.AutoFitSearchScope(lorentzian_counts=(0, 1))


def test_search_limit_and_unevaluated_complexity_invariants_are_enforced() -> None:
    scores = _scores(
        (
            (140.0, 135.0, 130.0),
            (115.0, 110.0, 105.0),
            (90.0, 85.0, 80.0),
        )
    )
    two_l = recommend_standard_candidates(_results(scores))
    assert two_l.additional_complexity.status is (
        AdditionalComplexityStatus.SEARCH_LIMIT_REACHED
    )

    with pytest.raises(ValueError, match="terminal 2L"):
        replace(two_l.additional_complexity, simpler_lorentzian_count=1)
    assert two_l.most_recommended is not None
    with pytest.raises(ValueError, match="must not contain"):
        replace(
            two_l.additional_complexity,
            proposed_candidate=two_l.most_recommended,
        )

    unavailable_family = recommend_standard_candidates(
        _results(scores, failed_counts=(1,))
    )
    assert unavailable_family.additional_complexity.status is (
        AdditionalComplexityStatus.NOT_EVALUABLE
    )
    with pytest.raises(ValueError, match="must not invent"):
        replace(
            unavailable_family.additional_complexity,
            proposed_candidate=unavailable_family.candidate_results[3],
        )


def test_terminal_complexity_states_reject_stale_transition_evidence() -> None:
    scores = _scores(
        (
            (140.0, 135.0, 130.0),
            (115.0, 110.0, 105.0),
            (90.0, 85.0, 80.0),
        )
    )
    two_l = recommend_standard_candidates(_results(scores))
    supported_transition = two_l.transition_assessments[-1]
    assert supported_transition.status is (
        AdditionalComplexityStatus.SUPPORTED_TRANSITION
    )
    assert supported_transition.matched_backgrounds
    assert supported_transition.proposed_candidate is not None
    stale_limitation = auto_module.InterpretationLimitation(
        code=InterpretationLimitationCode.COVARIANCE_UNAVAILABLE,
        message="Synthetic stale transition limitation.",
        candidate=supported_transition.proposed_candidate,
    )

    with pytest.raises(ValueError, match="must not retain"):
        replace(
            two_l.additional_complexity,
            matched_backgrounds=supported_transition.matched_backgrounds,
        )
    with pytest.raises(ValueError, match="must not retain"):
        replace(
            two_l.additional_complexity,
            interpretation_limitations=(stale_limitation,),
        )

    unavailable = recommend_standard_candidates(_results(scores, failed_counts=(1,)))
    assert unavailable.additional_complexity.status is (
        AdditionalComplexityStatus.NOT_EVALUABLE
    )
    with pytest.raises(ValueError, match="must not retain"):
        replace(
            unavailable.additional_complexity,
            matched_backgrounds=supported_transition.matched_backgrounds,
        )
    with pytest.raises(ValueError, match="must not retain"):
        replace(
            unavailable.additional_complexity,
            interpretation_limitations=(stale_limitation,),
        )

    assert two_l.transition_assessments[-1] is supported_transition
    assert two_l.transition_assessments[-1].matched_backgrounds


def test_numerical_unavailability_is_not_residual_inadequacy() -> None:
    scores = _scores(((10.0, 0.0, 20.0), (30.0, 20.0, 40.0), (50.0, 40.0, 60.0)))
    unavailable = recommend_standard_candidates(_results(scores, failed_counts=(0,)))

    assert unavailable.primary_residual_adequacy is ResidualAdequacy.NOT_EVALUABLE
    with pytest.raises(ValueError, match="unevaluated residual adequacy"):
        replace(
            unavailable,
            primary_residual_adequacy=ResidualAdequacy.INADEQUATE,
        )

    inadequate = {key: {"residual_rms": 1.7} for key in scores}
    evaluated = recommend_standard_candidates(_results(scores, overrides=inadequate))
    assert evaluated.primary_residual_adequacy is ResidualAdequacy.INADEQUATE
    assert evaluated.primary_family_support is PrimaryFamilySupport.NO_ADEQUATE_MODEL


@pytest.mark.parametrize("failed_count", (1, 2))
def test_wholly_failed_higher_family_is_reported_as_not_evaluable(
    failed_count: int,
) -> None:
    scores = _scores(
        (
            (120.0, 100.0, 101.0),
            (100.0, 80.0, 85.0),
            (80.0, 60.0, 65.0),
        )
    )
    recommendation = recommend_standard_candidates(
        _results(scores, failed_counts=(failed_count,))
    )

    assert recommendation.most_recommended is not None
    assert recommendation.additional_complexity.status is (
        AdditionalComplexityStatus.NOT_EVALUABLE
    )
    assert recommendation.additional_complexity.complex_lorentzian_count == (
        failed_count
    )


def test_undefined_family_aicc_is_reported_as_not_evaluable() -> None:
    scores = _scores(((30.0, 20.0, 10.0), (20.0, 10.0, 0.0), (40.0, 30.0, 20.0)))
    evidence = _replace_aicc(_results(scores), lorentzian_count=1, aicc=math.inf)

    recommendation = recommend_standard_candidates(evidence)

    assert recommendation.recommended_lorentzian_count == 0
    assert recommendation.additional_complexity.status is (
        AdditionalComplexityStatus.NOT_EVALUABLE
    )


def test_component_correlation_is_severe_but_nuisance_correlation_is_not() -> None:
    scores = _scores(
        (
            (140.0, 135.0, 130.0),
            (115.0, 110.0, 105.0),
            (90.0, 85.0, 80.0),
        )
    )
    nuisance = recommend_standard_candidates(
        _results(
            scores,
            overrides={
                (2, BackgroundModel.LINEAR): {"nuisance_correlation": 0.999},
            },
        )
    )
    component = recommend_standard_candidates(
        _results(
            scores,
            overrides={
                (2, BackgroundModel.LINEAR): {"component_correlation": 0.999},
            },
        )
    )

    assert nuisance.recommended_lorentzian_count == 2
    assert component.recommended_lorentzian_count == 1
    assert component.additional_complexity.status is (
        AdditionalComplexityStatus.SUPPORTED_BUT_UNINTERPRETABLE
    )
    assert any(
        item.code is InterpretationLimitationCode.COMPONENT_RELEVANT_DEGENERACY
        for item in component.additional_complexity.interpretation_limitations
    )


@pytest.mark.parametrize(
    ("override", "expected_code"),
    (
        (
            {
                "covariance_available": False,
                "missing_standard_error": "lorentzian_1_area",
            },
            InterpretationLimitationCode.COVARIANCE_UNAVAILABLE,
        ),
        (
            {"jacobian_rank_deficit": 1},
            InterpretationLimitationCode.JACOBIAN_RANK_DEFICIENT,
        ),
        (
            {"missing_standard_error": "lorentzian_2_area"},
            InterpretationLimitationCode.FREE_PARAMETER_UNCERTAINTY_UNAVAILABLE,
        ),
        (
            {"multistart_span": 1.0},
            InterpretationLimitationCode.MULTISTART_DECOMPOSITION_INCONSISTENT,
        ),
    ),
)
def test_severe_identifiability_diagnostics_limit_added_complexity(
    override: Mapping[str, Any],
    expected_code: InterpretationLimitationCode,
) -> None:
    scores = _scores(
        (
            (140.0, 135.0, 130.0),
            (115.0, 110.0, 105.0),
            (90.0, 85.0, 80.0),
        )
    )
    recommendation = recommend_standard_candidates(
        _results(
            scores,
            overrides={(2, BackgroundModel.LINEAR): override},
        )
    )

    assert recommendation.recommended_lorentzian_count == 1
    assert recommendation.additional_complexity.status is (
        AdditionalComplexityStatus.SUPPORTED_BUT_UNINTERPRETABLE
    )
    assert any(
        item.code is expected_code
        for item in recommendation.additional_complexity.interpretation_limitations
    )


def test_nuisance_only_multistart_variation_does_not_limit_decomposition() -> None:
    scores = _scores(
        (
            (140.0, 135.0, 130.0),
            (115.0, 110.0, 105.0),
            (90.0, 85.0, 80.0),
        )
    )

    recommendation = recommend_standard_candidates(
        _results(
            scores,
            overrides={
                (2, BackgroundModel.LINEAR): {
                    "nuisance_multistart_span": 2.0,
                }
            },
        )
    )

    assert recommendation.recommended_lorentzian_count == 2
    assert all(
        limitation.code
        is not InterpretationLimitationCode.MULTISTART_DECOMPOSITION_INCONSISTENT
        for limitation in recommendation.interpretation_limitations
    )


@pytest.mark.parametrize(
    "override",
    (
        {"component_area_multistart_span": 1.0},
        {"component_fwhm_multistart_span": 1.0},
    ),
)
def test_component_multistart_disagreement_limits_decomposition(
    override: Mapping[str, Any],
) -> None:
    scores = _scores(
        (
            (140.0, 135.0, 130.0),
            (115.0, 110.0, 105.0),
            (90.0, 85.0, 80.0),
        )
    )

    recommendation = recommend_standard_candidates(
        _results(
            scores,
            overrides={(2, BackgroundModel.LINEAR): override},
        )
    )

    assert recommendation.recommended_lorentzian_count == 1
    assert recommendation.additional_complexity.status is (
        AdditionalComplexityStatus.SUPPORTED_BUT_UNINTERPRETABLE
    )
    assert any(
        limitation.code
        is InterpretationLimitationCode.MULTISTART_DECOMPOSITION_INCONSISTENT
        for limitation in (
            recommendation.additional_complexity.interpretation_limitations
        )
    )


def test_permuted_multistart_components_are_canonicalized_before_comparison() -> None:
    scores = _scores(
        (
            (140.0, 135.0, 130.0),
            (115.0, 110.0, 105.0),
            (90.0, 85.0, 80.0),
        )
    )

    recommendation = recommend_standard_candidates(
        _results(
            scores,
            overrides={
                (2, BackgroundModel.LINEAR): {
                    "reverse_alternative_components": True,
                }
            },
        )
    )

    assert recommendation.recommended_lorentzian_count == 2
    assert all(
        limitation.code
        is not InterpretationLimitationCode.MULTISTART_DECOMPOSITION_INCONSISTENT
        for limitation in recommendation.interpretation_limitations
    )


def test_equal_fwhm_permuted_components_use_area_tie_break() -> None:
    scores = _scores(
        (
            (140.0, 135.0, 130.0),
            (115.0, 110.0, 105.0),
            (90.0, 85.0, 80.0),
        )
    )

    recommendation = recommend_standard_candidates(
        _results(
            scores,
            overrides={
                (2, BackgroundModel.LINEAR): {
                    "equal_component_fwhm": True,
                    "reverse_alternative_components": True,
                }
            },
        )
    )

    assert recommendation.recommended_lorentzian_count == 2
    assert all(
        limitation.code
        is not InterpretationLimitationCode.MULTISTART_DECOMPOSITION_INCONSISTENT
        for limitation in recommendation.interpretation_limitations
    )


@pytest.mark.parametrize("nonfinite_value", (math.nan, math.inf, -math.inf))
def test_nonfinite_successful_start_component_is_severe(
    nonfinite_value: float,
) -> None:
    scores = _scores(
        (
            (140.0, 135.0, 130.0),
            (115.0, 110.0, 105.0),
            (90.0, 85.0, 80.0),
        )
    )

    recommendation = recommend_standard_candidates(
        _results(
            scores,
            overrides={
                (2, BackgroundModel.LINEAR): {
                    "nonfinite_alternative_component": nonfinite_value,
                }
            },
        )
    )

    assert recommendation.recommended_lorentzian_count == 1
    assert recommendation.additional_complexity.status is (
        AdditionalComplexityStatus.SUPPORTED_BUT_UNINTERPRETABLE
    )
    assert any(
        limitation.code
        is InterpretationLimitationCode.MULTISTART_DECOMPOSITION_INCONSISTENT
        for limitation in (
            recommendation.additional_complexity.interpretation_limitations
        )
    )


def test_warning_records_are_deduplicated_and_reference_candidate_results() -> None:
    scores = _scores(
        (
            (100.0, 80.0, 90.0),
            (81.0, 90.0, 95.0),
            (110.0, 100.0, 105.0),
        )
    )
    recommendation = recommend_standard_candidates(_results(scores))
    assert recommendation.scientific_warnings
    warning = recommendation.scientific_warnings[0]

    with pytest.raises(ValueError, match="must not contain duplicates"):
        replace(recommendation, scientific_warnings=(warning, warning))
    assert recommendation.most_recommended is not None
    foreign = replace(recommendation.most_recommended)
    with pytest.raises(ValueError, match="must reference candidate_results"):
        replace(
            recommendation,
            scientific_warnings=(replace(warning, candidate=foreign),),
        )


def test_auto_fit_single_q_stops_at_2l_and_applies_policy_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = _scores(
        (
            (140.0, 135.0, 130.0),
            (115.0, 110.0, 105.0),
            (90.0, 85.0, 80.0),
        )
    )
    evidence = _results(
        scores,
        overrides={key: {"residual_rms": 1.7} for key in scores},
    )
    calls = {"evaluate": 0, "recommend": 0}

    def evaluate(*args: object, **kwargs: object) -> tuple[CandidateFitResult, ...]:
        calls["evaluate"] += 1
        assert kwargs == {
            "max_lorentzians": 2,
            "allow_linear_background": True,
            "max_nfev": 321,
        }
        return evidence

    original_recommend = auto_module.recommend_standard_candidates

    def recommend(
        results: tuple[CandidateFitResult, ...],
    ) -> auto_module.AutoFitRecommendation:
        calls["recommend"] += 1
        assert results is evidence
        return original_recommend(results)

    monkeypatch.setattr(fitting_core, "evaluate_standard_candidates", evaluate)
    monkeypatch.setattr(auto_module, "recommend_standard_candidates", recommend)

    result = auto_module.auto_fit_single_q(
        cast(PreparedResolution, object()),
        cast(FittingSelection, object()),
        4,
        max_nfev=321,
    )

    assert result.most_recommended is None
    assert result.primary_family_support is PrimaryFamilySupport.NO_ADEQUATE_MODEL
    assert result.auto_search_exhausted
    assert result.higher_complexity_manual_fit_available
    assert calls == {"evaluate": 1, "recommend": 1}
