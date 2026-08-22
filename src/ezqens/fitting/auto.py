"""Deterministic M5 AutoFit policy for the standard single-Q search space.

The policy consumes existing candidate-fit evidence. It does not fit models,
change candidate optimization, or infer microscopic truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from ezqens.fitting.models import (
    BackgroundModel,
    CandidateFitResult,
    FitResult,
    StandardModelCandidate,
)
from ezqens.preprocessing import FittingSelection
from ezqens.resolution import PreparedResolution


class ResidualAdequacy(StrEnum):
    """Calibrated M5 classification of standardized-residual structure."""

    NOT_EVALUABLE = "not_evaluable"
    ADEQUATE = "adequate"
    QUESTIONABLE = "questionable"
    INADEQUATE = "inadequate"


class ComplexityEvidence(StrEnum):
    """Family-envelope AICc/BIC evidence for one added-L transition."""

    STRONG = "strong"
    CLEAR = "clear"
    MARGINAL_OR_CONFLICTING = "marginal_or_conflicting"
    LITTLE_OR_NONE = "little_or_none"
    NOT_EVALUABLE = "not_evaluable"


class BackgroundRobustness(StrEnum):
    """Matched-background interpretation of an added-L transition."""

    ROBUST_ACROSS_B0_B1 = "robust_across_B0_B1"
    SUPPORTED_IN_ONE_ALLOWED_BACKGROUND = "supported_in_one_allowed_background"
    BACKGROUND_SENSITIVE = "background_sensitive"
    BACKGROUND_SUBSTITUTION = "background_substitution"
    NO_ADDITIONAL_COMPLEXITY = "no_additional_complexity"
    NOT_EVALUABLE = "not_evaluable"


class ComponentIdentifiability(StrEnum):
    """Interpretability of the fitted Lorentzian decomposition."""

    NOT_APPLICABLE = "not_applicable"
    INTERPRETABLE = "interpretable"
    SEVERE_LIMITATION = "severe_limitation"


class PrimaryFamilySupport(StrEnum):
    """Support for the selected observable family, separate from the next one."""

    SUPPORTED = "supported"
    SUPPORTED_WITH_CAUTION = "supported_with_caution"
    INTERPRETATION_LIMITED = "interpretation_limited"
    NO_ADEQUATE_MODEL = "no_adequate_model_within_search_scope"
    NUMERICAL_RECOMMENDATION_UNAVAILABLE = "numerical_recommendation_unavailable"


class AdditionalComplexityStatus(StrEnum):
    """Disposition of the next searched Lorentzian-family transition."""

    SUPPORTED_TRANSITION = "supported_transition"
    NOT_SUPPORTED = "not_supported"
    MARGINAL = "marginal_discrimination"
    BACKGROUND_CONFOUNDED = "background_confounded"
    SUPPORTED_BUT_RESIDUALLY_INADEQUATE = "supported_but_residually_inadequate"
    SUPPORTED_BUT_UNINTERPRETABLE = "supported_but_uninterpretable"
    NOT_EVALUABLE = "not_evaluable"
    SEARCH_LIMIT_REACHED = "search_limit_reached"


class InterpretationLimitationCode(StrEnum):
    """Severe reasons why fitted Lorentzian components are not interpretable."""

    COVARIANCE_UNAVAILABLE = "covariance_unavailable"
    JACOBIAN_RANK_DEFICIENT = "jacobian_rank_deficient"
    FREE_PARAMETER_UNCERTAINTY_UNAVAILABLE = "free_parameter_uncertainty_unavailable"
    MATERIAL_ACTIVE_BOUND = "material_active_bound"
    COMPONENT_AREA_UNCONSTRAINED = "component_area_unconstrained"
    COMPONENT_RELEVANT_DEGENERACY = "component_relevant_degeneracy"
    MULTISTART_UNAVAILABLE = "multistart_unavailable"
    MULTISTART_DECOMPOSITION_INCONSISTENT = "multistart_decomposition_inconsistent"


class ScientificWarningCode(StrEnum):
    """Small structured set of advisory M5 AutoFit warning meanings."""

    INTRINSIC_FWHM_BELOW_RESOLUTION = "intrinsic_fwhm_below_resolution_fwhm"
    HIGH_FINITE_CONDITION_NUMBER = "high_finite_condition_number"
    SIMILAR_INTRINSIC_LINEWIDTHS = "similar_intrinsic_linewidths"
    NUISANCE_PARAMETER_BOUND = "nuisance_parameter_bound"
    BACKGROUND_SENSITIVE_COMPLEXITY = "background_sensitive_complexity"
    BACKGROUND_CONFOUNDED_COMPLEXITY = "background_confounded_complexity"
    WITHIN_FAMILY_BACKGROUND_AMBIGUITY = "within_family_background_ambiguity"
    WITHIN_FAMILY_AICC_BIC_DISAGREEMENT = "within_family_aicc_bic_disagreement"


class ResolutionReliabilityStatus(StrEnum):
    """Structured AutoFit interpretation of resolution-containment provenance."""

    NOT_ASSESSED = "not_assessed"
    NO_LIMITATION_RECORDED = "no_limitation_recorded"
    POSSIBLE_RELEVANT_STRUCTURE_TRUNCATION = "possible_relevant_structure_truncation"


@dataclass(frozen=True, slots=True)
class AutoFitSearchScope:
    """The complete standard model-family scope searched by production AutoFit."""

    lorentzian_counts: tuple[int, ...] = (0, 1, 2)
    backgrounds: tuple[BackgroundModel, ...] = (
        BackgroundModel.NONE,
        BackgroundModel.CONSTANT,
        BackgroundModel.LINEAR,
    )

    def __post_init__(self) -> None:
        if self.lorentzian_counts != (0, 1, 2) or self.backgrounds != (
            BackgroundModel.NONE,
            BackgroundModel.CONSTANT,
            BackgroundModel.LINEAR,
        ):
            raise ValueError(
                "production AutoFit search scope must be exactly 0L/1L/2L x NONE/B0/B1"
            )

    @property
    def candidates(self) -> tuple[StandardModelCandidate, ...]:
        """Return the nine concrete candidates in deterministic search order."""

        return tuple(
            StandardModelCandidate(count, background)
            for count in self.lorentzian_counts
            for background in self.backgrounds
        )


@dataclass(frozen=True, slots=True)
class InformationCriterionComparison:
    """AICc/BIC improvement when moving to one more Lorentzian."""

    delta_aicc: float | None
    delta_bic: float | None
    evidence: ComplexityEvidence


@dataclass(frozen=True, slots=True)
class MatchedBackgroundComparison:
    """One same-background comparison across adjacent Lorentzian families."""

    background: BackgroundModel
    delta_aicc: float
    delta_bic: float
    evidence: ComplexityEvidence


@dataclass(frozen=True, slots=True)
class InterpretationLimitation:
    """One severe, candidate-specific component-interpretation limitation."""

    code: InterpretationLimitationCode
    message: str
    candidate: CandidateFitResult


@dataclass(frozen=True, slots=True)
class ScientificWarning:
    """One advisory observation that does not by itself reject a candidate."""

    code: ScientificWarningCode
    message: str
    candidate: CandidateFitResult | None = None


@dataclass(frozen=True, slots=True)
class AdditionalComplexityAssessment:
    """Evidence and disposition for one adjacent Lorentzian-family transition."""

    simpler_lorentzian_count: int
    complex_lorentzian_count: int | None
    status: AdditionalComplexityStatus
    information_criteria: InformationCriterionComparison
    background_robustness: BackgroundRobustness
    matched_backgrounds: tuple[MatchedBackgroundComparison, ...]
    proposed_candidate: CandidateFitResult | None
    interpretation_limitations: tuple[InterpretationLimitation, ...]
    reason: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.simpler_lorentzian_count, bool)
            or not isinstance(self.simpler_lorentzian_count, int)
            or self.simpler_lorentzian_count not in (0, 1, 2)
        ):
            raise ValueError("simpler_lorentzian_count must be 0, 1, or 2")
        if not isinstance(self.status, AdditionalComplexityStatus):
            raise ValueError("status must be an AdditionalComplexityStatus")
        if not isinstance(self.information_criteria, InformationCriterionComparison):
            raise ValueError(
                "information_criteria must be an InformationCriterionComparison"
            )
        if not isinstance(self.background_robustness, BackgroundRobustness):
            raise ValueError("background_robustness must be a BackgroundRobustness")
        matched = tuple(self.matched_backgrounds)
        limitations = tuple(self.interpretation_limitations)
        if any(not isinstance(item, MatchedBackgroundComparison) for item in matched):
            raise ValueError(
                "matched_backgrounds must contain MatchedBackgroundComparison values"
            )
        if len({item.background for item in matched}) != len(matched):
            raise ValueError("matched_backgrounds must not repeat a background")
        if any(not isinstance(item, InterpretationLimitation) for item in limitations):
            raise ValueError(
                "interpretation_limitations must contain InterpretationLimitation "
                "values"
            )
        if not self.reason:
            raise ValueError("additional-complexity reason must not be empty")

        unevaluated_status = self.status in {
            AdditionalComplexityStatus.NOT_EVALUABLE,
            AdditionalComplexityStatus.SEARCH_LIMIT_REACHED,
        }
        if unevaluated_status:
            if (
                self.information_criteria.evidence
                is not ComplexityEvidence.NOT_EVALUABLE
                or self.background_robustness is not BackgroundRobustness.NOT_EVALUABLE
            ):
                raise ValueError(
                    "unevaluated complexity states require unevaluated evidence"
                )
            if matched or limitations:
                raise ValueError(
                    "terminal unevaluated complexity states must not retain "
                    "matched-background evidence or interpretation limitations"
                )
        elif (
            self.information_criteria.evidence is ComplexityEvidence.NOT_EVALUABLE
            or self.background_robustness is BackgroundRobustness.NOT_EVALUABLE
        ):
            raise ValueError("evaluated complexity states require evaluated evidence")

        if self.status is AdditionalComplexityStatus.SEARCH_LIMIT_REACHED:
            if self.simpler_lorentzian_count != 2:
                raise ValueError("SEARCH_LIMIT_REACHED requires the terminal 2L family")
            if self.complex_lorentzian_count is not None:
                raise ValueError(
                    "SEARCH_LIMIT_REACHED must not invent a higher searched family"
                )
            if self.proposed_candidate is not None:
                raise ValueError(
                    "SEARCH_LIMIT_REACHED must not contain a proposed candidate"
                )
        else:
            expected_complex = self.simpler_lorentzian_count + 1
            if (
                expected_complex > 2
                or self.complex_lorentzian_count != expected_complex
            ):
                raise ValueError(
                    "additional complexity must describe the next searched family"
                )
            requires_candidate = self.status is not (
                AdditionalComplexityStatus.NOT_EVALUABLE
            )
            if requires_candidate and self.proposed_candidate is None:
                raise ValueError(f"{self.status.value} requires a proposed candidate")
            if not requires_candidate and self.proposed_candidate is not None:
                raise ValueError("NOT_EVALUABLE must not invent a proposed candidate")
        if self.proposed_candidate is not None:
            if (
                self.proposed_candidate.candidate.lorentzian_count
                != self.complex_lorentzian_count
            ):
                raise ValueError(
                    "proposed candidate must belong to the stated higher family"
                )
            if not self.proposed_candidate.success:
                raise ValueError("proposed candidate must be a successful result")
            _validate_successful_candidate_configuration(self.proposed_candidate)
            if any(
                limitation.candidate.candidate != self.proposed_candidate.candidate
                for limitation in limitations
            ):
                raise ValueError(
                    "additional-complexity limitations must reference the proposed "
                    "candidate"
                )
            proposed_adequacy = _residual_adequacy(self.proposed_candidate)
            if (
                self.status
                is AdditionalComplexityStatus.SUPPORTED_BUT_RESIDUALLY_INADEQUATE
                and proposed_adequacy is not ResidualAdequacy.INADEQUATE
            ):
                raise ValueError(
                    "SUPPORTED_BUT_RESIDUALLY_INADEQUATE requires an inadequate "
                    "proposed candidate"
                )
            if (
                self.status is AdditionalComplexityStatus.SUPPORTED_TRANSITION
                and proposed_adequacy is ResidualAdequacy.INADEQUATE
            ):
                raise ValueError(
                    "SUPPORTED_TRANSITION requires a recommendation-eligible "
                    "proposed candidate"
                )
        limitation_keys = {
            (item.code, item.message, item.candidate.candidate) for item in limitations
        }
        if len(limitation_keys) != len(limitations):
            raise ValueError(
                "additional-complexity limitations must not contain duplicates"
            )
        object.__setattr__(self, "matched_backgrounds", matched)
        object.__setattr__(self, "interpretation_limitations", limitations)


@dataclass(frozen=True, slots=True)
class ResolutionReliabilityAssessment:
    """Current availability of structured resolution-containment provenance."""

    status: ResolutionReliabilityStatus
    provenance_gap: str | None
    reason: str

    @property
    def structured_containment_assessment_available(self) -> bool:
        """Return whether upstream provenance supplied a scientific assessment."""

        return self.status is not ResolutionReliabilityStatus.NOT_ASSESSED


@dataclass(frozen=True, slots=True)
class AutoFitRecommendation:
    """Production AutoFit recommendation with inspectable candidate references."""

    most_recommended: CandidateFitResult | None
    recommended_lorentzian_count: int | None
    primary_family_support: PrimaryFamilySupport
    primary_residual_adequacy: ResidualAdequacy
    primary_identifiability: ComponentIdentifiability
    additional_complexity: AdditionalComplexityAssessment
    transition_assessments: tuple[AdditionalComplexityAssessment, ...]
    strong_alternative: CandidateFitResult | None
    comparator: CandidateFitResult | None
    interpretation_limitations: tuple[InterpretationLimitation, ...]
    scientific_warnings: tuple[ScientificWarning, ...]
    resolution_reliability: ResolutionReliabilityAssessment
    search_scope: AutoFitSearchScope
    candidate_results: tuple[CandidateFitResult, ...]

    def __post_init__(self) -> None:
        candidates = tuple(self.candidate_results)
        transitions = tuple(self.transition_assessments)
        limitations = tuple(self.interpretation_limitations)
        warnings = tuple(self.scientific_warnings)
        _validate_complete_candidate_results(candidates)
        successful = _successful_results(candidates)

        def is_marginal_eligibility_traversal(
            assessment: AdditionalComplexityAssessment,
        ) -> bool:
            return (
                assessment.status is AdditionalComplexityStatus.MARGINAL
                and _best_recommendation_candidate(
                    successful,
                    assessment.simpler_lorentzian_count,
                )
                is None
                and assessment.proposed_candidate is not None
                and _residual_adequacy(assessment.proposed_candidate)
                is not ResidualAdequacy.INADEQUATE
                and not assessment.interpretation_limitations
            )

        if not isinstance(self.primary_family_support, PrimaryFamilySupport):
            raise ValueError("primary_family_support must be a PrimaryFamilySupport")
        if not isinstance(self.primary_residual_adequacy, ResidualAdequacy):
            raise ValueError("primary_residual_adequacy must be a ResidualAdequacy")
        if not isinstance(self.primary_identifiability, ComponentIdentifiability):
            raise ValueError(
                "primary_identifiability must be a ComponentIdentifiability"
            )
        if not isinstance(
            self.additional_complexity,
            AdditionalComplexityAssessment,
        ):
            raise ValueError(
                "additional_complexity must be an AdditionalComplexityAssessment"
            )
        if not isinstance(
            self.resolution_reliability,
            ResolutionReliabilityAssessment,
        ):
            raise ValueError(
                "resolution_reliability must be a ResolutionReliabilityAssessment"
            )
        if not isinstance(self.search_scope, AutoFitSearchScope):
            raise ValueError("search_scope must be an AutoFitSearchScope")
        if any(
            not isinstance(item, AdditionalComplexityAssessment) for item in transitions
        ):
            raise ValueError(
                "transition_assessments must contain "
                "AdditionalComplexityAssessment values"
            )
        if not transitions or transitions[0].simpler_lorentzian_count != 0:
            raise ValueError("transition history must start from the 0L family")
        for previous, following in zip(
            transitions,
            transitions[1:],
            strict=False,
        ):
            if previous.status not in {
                AdditionalComplexityStatus.SUPPORTED_TRANSITION,
                AdditionalComplexityStatus.SUPPORTED_BUT_RESIDUALLY_INADEQUATE,
            } and not is_marginal_eligibility_traversal(previous):
                raise ValueError(
                    "only a supported transition or eligibility-only marginal "
                    "traversal may be followed by another"
                )
            if following.simpler_lorentzian_count != previous.complex_lorentzian_count:
                raise ValueError("transition history must be contiguous")
        if any(not isinstance(item, InterpretationLimitation) for item in limitations):
            raise ValueError(
                "interpretation_limitations must contain InterpretationLimitation "
                "values"
            )
        if any(not isinstance(item, ScientificWarning) for item in warnings):
            raise ValueError(
                "scientific_warnings must contain ScientificWarning values"
            )

        terminal_support = {
            PrimaryFamilySupport.NO_ADEQUATE_MODEL,
            PrimaryFamilySupport.NUMERICAL_RECOMMENDATION_UNAVAILABLE,
        }
        if self.most_recommended is None:
            if self.recommended_lorentzian_count is not None:
                raise ValueError(
                    "no recommendation requires recommended_lorentzian_count=None"
                )
            if self.primary_family_support not in terminal_support:
                raise ValueError(
                    "no recommendation cannot have an ordinary supported primary state"
                )
            if self.strong_alternative is not None:
                raise ValueError(
                    "no recommendation must not retain a Strong Alternative"
                )
        else:
            _require_candidate_reference(
                self.most_recommended,
                candidates,
                field_name="most_recommended",
            )
            if not self.most_recommended.success:
                raise ValueError("Most Recommended must be a successful candidate")
            if self.primary_family_support in terminal_support:
                raise ValueError(
                    "a concrete recommendation cannot have a terminal support state"
                )
            if (
                self.recommended_lorentzian_count
                != self.most_recommended.candidate.lorentzian_count
            ):
                raise ValueError(
                    "recommended Lorentzian count must match Most Recommended"
                )
            if (
                self.additional_complexity.simpler_lorentzian_count
                != self.recommended_lorentzian_count
            ):
                raise ValueError(
                    "additional complexity must start from the selected family"
                )
            if self.primary_residual_adequacy in {
                ResidualAdequacy.NOT_EVALUABLE,
                ResidualAdequacy.INADEQUATE,
            }:
                raise ValueError(
                    "a concrete recommendation requires evaluated non-inadequate "
                    "residuals"
                )
            selected_count = self.most_recommended.candidate.lorentzian_count
            if (
                selected_count == 0
                and self.primary_identifiability
                is not ComponentIdentifiability.NOT_APPLICABLE
            ):
                raise ValueError("0L primary identifiability must be NOT_APPLICABLE")
            if (
                selected_count > 0
                and self.primary_identifiability
                is ComponentIdentifiability.NOT_APPLICABLE
            ):
                raise ValueError("Lorentzian primary identifiability must be evaluated")
            if (
                self.primary_family_support
                is PrimaryFamilySupport.INTERPRETATION_LIMITED
            ) != (
                self.primary_identifiability
                is ComponentIdentifiability.SEVERE_LIMITATION
            ):
                raise ValueError(
                    "primary support and identifiability limitation must agree"
                )

        if (
            self.primary_family_support
            is PrimaryFamilySupport.NUMERICAL_RECOMMENDATION_UNAVAILABLE
        ):
            if self.primary_residual_adequacy is not ResidualAdequacy.NOT_EVALUABLE:
                raise ValueError(
                    "numerical unavailability requires unevaluated residual adequacy"
                )
            if self.comparator is not None:
                raise ValueError(
                    "numerical unavailability must not retain a Comparator"
                )
            if (
                self.primary_identifiability
                is not ComponentIdentifiability.NOT_APPLICABLE
                or limitations
                or warnings
                or len(transitions) != 1
                or transitions[0] is not self.additional_complexity
            ):
                raise ValueError(
                    "numerical unavailability must not retain evaluated diagnostics"
                )
            if (
                self.additional_complexity.status
                is not AdditionalComplexityStatus.NOT_EVALUABLE
            ):
                raise ValueError(
                    "numerical unavailability requires unevaluated complexity"
                )
        if self.primary_family_support is PrimaryFamilySupport.NO_ADEQUATE_MODEL:
            if self.primary_residual_adequacy is not ResidualAdequacy.INADEQUATE:
                raise ValueError(
                    "NO_ADEQUATE_MODEL requires evaluated inadequate residuals"
                )

        for field_name, reference in (
            ("strong_alternative", self.strong_alternative),
            ("comparator", self.comparator),
        ):
            if reference is not None:
                _require_candidate_reference(
                    reference,
                    candidates,
                    field_name=field_name,
                )
                if not reference.success:
                    raise ValueError(f"{field_name} must be a successful candidate")
        if (
            self.most_recommended is not None
            and self.strong_alternative is not None
            and self.strong_alternative.candidate == self.most_recommended.candidate
        ):
            raise ValueError("Strong Alternative must differ from Most Recommended")

        if (
            self.additional_complexity.status
            is AdditionalComplexityStatus.SEARCH_LIMIT_REACHED
        ):
            selected_two_l = self.recommended_lorentzian_count == 2
            inadequate_two_l = (
                self.most_recommended is None
                and self.primary_family_support
                is PrimaryFamilySupport.NO_ADEQUATE_MODEL
                and self.comparator is not None
                and self.comparator.candidate.lorentzian_count == 2
            )
            if not (selected_two_l or inadequate_two_l):
                raise ValueError(
                    "SEARCH_LIMIT_REACHED requires either a selected 2L family or "
                    "an evaluated inadequate 2L comparator"
                )
            if not transitions or transitions[-1].complex_lorentzian_count != 2:
                raise ValueError(
                    "2L search-limit result requires a recorded 1L->2L transition"
                )
            supported_selected_two_l = selected_two_l and (
                transitions[-1].status
                is AdditionalComplexityStatus.SUPPORTED_TRANSITION
                or is_marginal_eligibility_traversal(transitions[-1])
            )
            supported_inadequate_two_l = inadequate_two_l and (
                transitions[-1].status
                is AdditionalComplexityStatus.SUPPORTED_BUT_RESIDUALLY_INADEQUATE
            )
            if not (supported_selected_two_l or supported_inadequate_two_l):
                raise ValueError(
                    "2L search-limit transition status must match whether the final "
                    "2L candidate is recommendation-eligible"
                )
            if (
                inadequate_two_l
                and self.comparator is not transitions[-1].proposed_candidate
            ):
                raise ValueError(
                    "an inadequate 2L search-limit result must retain the final "
                    "evaluated 2L candidate as Comparator"
                )
            if (
                selected_two_l
                and self.most_recommended is not transitions[-1].proposed_candidate
            ):
                raise ValueError(
                    "a selected 2L search-limit result must retain the supported "
                    "transition candidate as Most Recommended"
                )
        elif not transitions or transitions[-1] is not self.additional_complexity:
            raise ValueError(
                "additional_complexity must be the final attempted transition"
            )

        for limitation in limitations:
            _require_candidate_reference(
                limitation.candidate,
                candidates,
                field_name="interpretation limitation",
            )
        for warning in warnings:
            if warning.candidate is not None:
                _require_candidate_reference(
                    warning.candidate,
                    candidates,
                    field_name="scientific warning",
                )
        limitation_keys = {
            (item.code, item.message, item.candidate.candidate) for item in limitations
        }
        if len(limitation_keys) != len(limitations):
            raise ValueError("interpretation limitations must not contain duplicates")
        warning_keys = {
            (
                item.code,
                item.message,
                item.candidate.candidate if item.candidate is not None else None,
            )
            for item in warnings
        }
        if len(warning_keys) != len(warnings):
            raise ValueError("scientific warnings must not contain duplicates")
        object.__setattr__(self, "candidate_results", candidates)
        object.__setattr__(self, "transition_assessments", transitions)
        object.__setattr__(self, "interpretation_limitations", limitations)
        object.__setattr__(self, "scientific_warnings", warnings)

    @property
    def recommended_candidate(self) -> StandardModelCandidate | None:
        """Return the concrete candidate identifier without duplicating its fit."""

        return (
            self.most_recommended.candidate
            if self.most_recommended is not None
            else None
        )

    @property
    def auto_search_exhausted(self) -> bool:
        """Return whether the fixed automatic search reached its 2L boundary."""

        return (
            self.additional_complexity.status
            is AdditionalComplexityStatus.SEARCH_LIMIT_REACHED
        )

    @property
    def higher_complexity_manual_fit_available(self) -> bool:
        """Return whether an exhausted Auto search can continue manually above 2L.

        Availability records a workflow capability, not a recommendation to add
        Lorentzians. Any continuation uses the existing user-configured arbitrary-N
        manual fitting path and remains outside AutoFit candidate evidence.
        """

        return self.auto_search_exhausted


# Reviewed H1/H1.2 M5 calibration values. They are policy choices, not
# universal physical constants and intentionally are not user-tunable in M5.
_RMS_MODERATE = 1.12
_LAG_MODERATE = 0.12
_RUN_MODERATE = 12
_TREND_MODERATE = 0.35
_MAXIMUM_MODERATE = 3.8
_RMS_STRONG = 1.35
_LAG_STRONG = 0.25
_RUN_STRONG = 18
_TREND_STRONG = 0.8
_MAXIMUM_STRONG = 5.0
_RMS_EXTREME_INADEQUATE = 1.6
_LAG_EXTREME_INADEQUATE = 0.45
_IC_STRONG_AICC = 10.0
_IC_STRONG_BIC = 6.0
_IC_CLEAR_AICC = 6.0
_IC_CLEAR_BIC = 2.0
_IC_MARGINAL_AICC = 2.0
_IC_MARGINAL_BIC = 0.0
_AREA_TO_SE_SEVERE = 1.0
_CORRELATION_SEVERE = 0.995
_MULTISTART_SPAN_SEVERE = 0.5
_EQUIVALENT_START_DELTA_CHI_SQUARE = 2.0
_BACKGROUND_NEAR_AICC = 2.0
_BACKGROUND_NEAR_BIC = 4.0
_ADVISORY_CONDITION_NUMBER = 1.0e8
_ADVISORY_ADJACENT_FWHM_RATIO = 1.2
_BACKGROUND_RANK = {
    BackgroundModel.NONE: 0,
    BackgroundModel.CONSTANT: 1,
    BackgroundModel.LINEAR: 2,
}


def _validate_successful_candidate_configuration(
    result: CandidateFitResult,
) -> None:
    if not result.success:
        return
    fit = result.fit
    if fit is None:  # guarded by CandidateFitResult.success
        raise ValueError("successful candidate evidence must contain a fit")
    if (
        fit.configuration.lorentzian_count != result.candidate.lorentzian_count
        or fit.configuration.background is not result.candidate.background
    ):
        raise ValueError(
            "successful fit configuration disagrees with candidate "
            f"{result.candidate.name}"
        )


def _validate_complete_candidate_results(
    candidate_results: tuple[CandidateFitResult, ...],
) -> None:
    if any(not isinstance(result, CandidateFitResult) for result in candidate_results):
        raise ValueError("candidate_results must contain CandidateFitResult values")
    expected_order = AutoFitSearchScope().candidates
    expected = set(expected_order)
    provided = tuple(result.candidate for result in candidate_results)
    if len(set(provided)) != len(provided):
        raise ValueError("candidate_results must not contain duplicate candidates")
    out_of_scope = tuple(
        candidate.name for candidate in provided if candidate not in expected
    )
    if out_of_scope:
        raise ValueError(
            "production AutoFit accepts only the 0L/1L/2L x NONE/B0/B1 scope; "
            "out-of-scope candidates: " + ", ".join(out_of_scope)
        )
    provided_set = set(provided)
    missing = tuple(
        candidate.name for candidate in expected_order if candidate not in provided_set
    )
    if missing:
        raise ValueError(
            "candidate_results must contain the complete nine-candidate standard "
            "scope; missing: " + ", ".join(missing)
        )
    for result in candidate_results:
        _validate_successful_candidate_configuration(result)


def _require_candidate_reference(
    reference: CandidateFitResult,
    candidate_results: tuple[CandidateFitResult, ...],
    *,
    field_name: str,
) -> None:
    if not any(reference is result for result in candidate_results):
        raise ValueError(f"{field_name} must reference candidate_results")
    _validate_successful_candidate_configuration(reference)


def _successful_results(
    candidate_results: tuple[CandidateFitResult, ...],
) -> tuple[CandidateFitResult, ...]:
    return tuple(
        result
        for result in candidate_results
        if result.success
        and result.fit is not None
        and math.isfinite(result.fit.statistics.aicc)
        and math.isfinite(result.fit.statistics.bic)
    )


def _by_key(
    successful: tuple[CandidateFitResult, ...],
) -> dict[tuple[int, BackgroundModel], CandidateFitResult]:
    return {
        (result.candidate.lorentzian_count, result.candidate.background): result
        for result in successful
    }


def _best_family(
    successful: tuple[CandidateFitResult, ...],
    count: int,
    criterion: str = "aicc",
) -> CandidateFitResult | None:
    family = tuple(
        result for result in successful if result.candidate.lorentzian_count == count
    )
    if not family:
        return None
    return min(
        family,
        key=lambda result: (
            getattr(result.fit.statistics, criterion)
            if result.fit is not None
            else math.inf,
            _BACKGROUND_RANK[result.candidate.background],
        ),
    )


def _residual_adequacy(result: CandidateFitResult) -> ResidualAdequacy:
    fit = result.fit
    if fit is None:
        return ResidualAdequacy.NOT_EVALUABLE
    residual = fit.diagnostics.residual
    lag = (
        abs(residual.lag1_correlation) if residual.lag1_correlation is not None else 0.0
    )
    moderate = sum(
        (
            residual.rms > _RMS_MODERATE,
            lag > _LAG_MODERATE,
            residual.longest_same_sign_run >= _RUN_MODERATE,
            abs(residual.linear_trend) > _TREND_MODERATE,
            residual.maximum_absolute > _MAXIMUM_MODERATE,
        )
    )
    strong = sum(
        (
            residual.rms > _RMS_STRONG,
            lag > _LAG_STRONG,
            residual.longest_same_sign_run >= _RUN_STRONG,
            abs(residual.linear_trend) > _TREND_STRONG,
            residual.maximum_absolute > _MAXIMUM_STRONG,
        )
    )
    if (
        strong >= 2
        or residual.rms > _RMS_EXTREME_INADEQUATE
        or lag > _LAG_EXTREME_INADEQUATE
    ):
        return ResidualAdequacy.INADEQUATE
    if strong >= 1 or moderate >= 2:
        return ResidualAdequacy.QUESTIONABLE
    return ResidualAdequacy.ADEQUATE


def _best_recommendation_candidate(
    successful: tuple[CandidateFitResult, ...],
    count: int,
    criterion: str = "aicc",
) -> CandidateFitResult | None:
    """Return the minimum-criterion residually eligible candidate in one family."""

    eligible = tuple(
        result
        for result in successful
        if result.candidate.lorentzian_count == count
        and _residual_adequacy(result) is not ResidualAdequacy.INADEQUATE
    )
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda result: (
            getattr(result.fit.statistics, criterion)
            if result.fit is not None
            else math.inf,
            _BACKGROUND_RANK[result.candidate.background],
        ),
    )


def _classify_information_criteria(
    delta_aicc: float | None,
    delta_bic: float | None,
) -> ComplexityEvidence:
    if (
        delta_aicc is None
        or delta_bic is None
        or not math.isfinite(delta_aicc)
        or not math.isfinite(delta_bic)
    ):
        return ComplexityEvidence.NOT_EVALUABLE
    if delta_aicc >= _IC_STRONG_AICC and delta_bic >= _IC_STRONG_BIC:
        return ComplexityEvidence.STRONG
    if delta_aicc >= _IC_CLEAR_AICC and delta_bic >= _IC_CLEAR_BIC:
        return ComplexityEvidence.CLEAR
    if (
        delta_aicc >= _IC_MARGINAL_AICC
        or delta_bic >= _IC_MARGINAL_BIC
        or (delta_aicc > 0.0) != (delta_bic > 0.0)
    ):
        return ComplexityEvidence.MARGINAL_OR_CONFLICTING
    return ComplexityEvidence.LITTLE_OR_NONE


def _information_comparison(
    successful: tuple[CandidateFitResult, ...],
    simpler_count: int,
    complex_count: int,
) -> InformationCriterionComparison:
    simple_aicc = _best_family(successful, simpler_count, "aicc")
    complex_aicc = _best_family(successful, complex_count, "aicc")
    simple_bic = _best_family(successful, simpler_count, "bic")
    complex_bic = _best_family(successful, complex_count, "bic")
    if None in (simple_aicc, complex_aicc, simple_bic, complex_bic):
        return InformationCriterionComparison(
            None,
            None,
            ComplexityEvidence.NOT_EVALUABLE,
        )
    assert simple_aicc is not None
    assert complex_aicc is not None
    assert simple_bic is not None
    assert complex_bic is not None
    assert simple_aicc.fit is not None
    assert complex_aicc.fit is not None
    assert simple_bic.fit is not None
    assert complex_bic.fit is not None
    delta_aicc = simple_aicc.fit.statistics.aicc - complex_aicc.fit.statistics.aicc
    delta_bic = simple_bic.fit.statistics.bic - complex_bic.fit.statistics.bic
    return InformationCriterionComparison(
        delta_aicc,
        delta_bic,
        _classify_information_criteria(delta_aicc, delta_bic),
    )


def _matched_background_comparisons(
    successful: tuple[CandidateFitResult, ...],
    simpler_count: int,
    complex_count: int,
) -> tuple[MatchedBackgroundComparison, ...]:
    by_key = _by_key(successful)
    comparisons: list[MatchedBackgroundComparison] = []
    for background in AutoFitSearchScope().backgrounds:
        simpler = by_key.get((simpler_count, background))
        complex_result = by_key.get((complex_count, background))
        if simpler is None or complex_result is None:
            continue
        assert simpler.fit is not None
        assert complex_result.fit is not None
        delta_aicc = simpler.fit.statistics.aicc - complex_result.fit.statistics.aicc
        delta_bic = simpler.fit.statistics.bic - complex_result.fit.statistics.bic
        comparisons.append(
            MatchedBackgroundComparison(
                background,
                delta_aicc,
                delta_bic,
                _classify_information_criteria(delta_aicc, delta_bic),
            )
        )
    return tuple(comparisons)


def _background_robustness(
    information: InformationCriterionComparison,
    matched: tuple[MatchedBackgroundComparison, ...],
) -> BackgroundRobustness:
    if information.evidence is ComplexityEvidence.NOT_EVALUABLE:
        return BackgroundRobustness.NOT_EVALUABLE
    by_background = {item.background: item.evidence for item in matched}
    positive = {ComplexityEvidence.CLEAR, ComplexityEvidence.STRONG}
    no_background = by_background.get(BackgroundModel.NONE)
    b0 = by_background.get(BackgroundModel.CONSTANT)
    b1 = by_background.get(BackgroundModel.LINEAR)
    if (
        information.evidence is ComplexityEvidence.LITTLE_OR_NONE
        and no_background in positive
        and b0 not in positive
        and b1 not in positive
    ):
        return BackgroundRobustness.BACKGROUND_SUBSTITUTION
    if b0 in positive and b1 in positive:
        return BackgroundRobustness.ROBUST_ACROSS_B0_B1
    if information.evidence in positive and ((b0 in positive) != (b1 in positive)):
        return BackgroundRobustness.SUPPORTED_IN_ONE_ALLOWED_BACKGROUND
    if information.evidence in positive and no_background in positive:
        return BackgroundRobustness.BACKGROUND_SENSITIVE
    return BackgroundRobustness.NO_ADDITIONAL_COMPLEXITY


def _component_parameter(name: str) -> bool:
    return name.split(":", maxsplit=1)[0].startswith("lorentzian_")


def _material_bound_parameter(name: str) -> bool:
    base_name = name.split(":", maxsplit=1)[0]
    return _component_parameter(base_name) or base_name in {
        "energy_shift",
        "elastic_area",
    }


def _canonical_lorentzian_values(
    values: tuple[float, ...],
    lorentzian_count: int,
) -> tuple[tuple[float, float], ...]:
    components = tuple(
        (values[2 + 2 * index], values[3 + 2 * index])
        for index in range(lorentzian_count)
    )
    return tuple(sorted(components, key=lambda component: (component[1], component[0])))


def _successful_start_span(fit: FitResult) -> tuple[int, float]:
    starts = tuple(
        item
        for item in fit.diagnostics.alternative_starts
        if item.success
        and item.chi_square
        <= fit.statistics.chi_square + _EQUIVALENT_START_DELTA_CHI_SQUARE
    )
    components = tuple(
        _canonical_lorentzian_values(
            item.fitted_parameter_values,
            fit.configuration.lorentzian_count,
        )
        for item in starts
    )
    if not components or not components[0]:
        return len(starts), 0.0
    areas = np.asarray(
        [[component[0] for component in start] for start in components],
        dtype=np.float64,
    )
    linewidths = np.asarray(
        [[component[1] for component in start] for start in components],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(areas)) or not np.all(np.isfinite(linewidths)):
        return len(starts), math.inf
    if len(starts) < 2:
        return len(starts), 0.0
    total_areas = np.sum(np.abs(areas), axis=1)
    area_scale = max(float(np.mean(total_areas)), np.finfo(np.float64).eps)
    area_allocation_span = float(np.max(np.ptp(areas, axis=0) / area_scale))
    total_area_span = float(np.ptp(total_areas) / area_scale)
    linewidth_scale = np.maximum(
        np.mean(np.abs(linewidths), axis=0),
        np.finfo(np.float64).eps,
    )
    linewidth_span = float(np.max(np.ptp(linewidths, axis=0) / linewidth_scale))
    return len(starts), max(
        area_allocation_span,
        total_area_span,
        linewidth_span,
    )


def _component_correlation_limitation(
    result: CandidateFitResult,
) -> InterpretationLimitation | None:
    fit = result.fit
    if fit is None or fit.correlation is None:
        return None
    names = tuple(parameter.name for parameter in fit.parameters)
    strongest: tuple[float, str, str] | None = None
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            if not (
                _component_parameter(names[first])
                or _component_parameter(names[second])
            ):
                continue
            value = abs(float(fit.correlation[first, second]))
            if math.isfinite(value) and (strongest is None or value > strongest[0]):
                strongest = (value, names[first], names[second])
    if strongest is None or strongest[0] < _CORRELATION_SEVERE:
        return None
    return InterpretationLimitation(
        InterpretationLimitationCode.COMPONENT_RELEVANT_DEGENERACY,
        "Component-relevant parameter correlation "
        f"|r|={strongest[0]:.6g} for {strongest[1]} and {strongest[2]} "
        f"meets the calibrated severe boundary {_CORRELATION_SEVERE:g}.",
        result,
    )


def _identifiability(
    result: CandidateFitResult,
) -> tuple[ComponentIdentifiability, tuple[InterpretationLimitation, ...]]:
    if result.candidate.lorentzian_count == 0:
        return ComponentIdentifiability.NOT_APPLICABLE, ()
    fit = result.fit
    if fit is None:
        return ComponentIdentifiability.SEVERE_LIMITATION, ()
    limitations: list[InterpretationLimitation] = []

    def add(code: InterpretationLimitationCode, message: str) -> None:
        limitations.append(InterpretationLimitation(code, message, result))

    if not fit.diagnostics.covariance_available:
        add(
            InterpretationLimitationCode.COVARIANCE_UNAVAILABLE,
            "Candidate covariance is unavailable.",
        )
    if fit.diagnostics.jacobian_rank < fit.statistics.free_parameters:
        add(
            InterpretationLimitationCode.JACOBIAN_RANK_DEFICIENT,
            "Candidate Jacobian rank is below the free-parameter count.",
        )
    missing_uncertainties = tuple(
        parameter.name
        for parameter in fit.parameters
        if parameter.free
        and (
            parameter.standard_error is None
            or not math.isfinite(parameter.standard_error)
        )
    )
    if missing_uncertainties:
        add(
            InterpretationLimitationCode.FREE_PARAMETER_UNCERTAINTY_UNAVAILABLE,
            "Free-parameter uncertainty is unavailable for "
            + ", ".join(missing_uncertainties)
            + ".",
        )
    material_bounds = tuple(
        name
        for name in fit.diagnostics.active_bounds
        if _material_bound_parameter(name)
    )
    if material_bounds:
        add(
            InterpretationLimitationCode.MATERIAL_ACTIVE_BOUND,
            "Material model parameter is active at a bound: "
            + ", ".join(material_bounds)
            + ".",
        )
    ratios = fit.diagnostics.component_area_to_standard_error
    if len(ratios) != result.candidate.lorentzian_count or any(
        value is None or not math.isfinite(value) or value < _AREA_TO_SE_SEVERE
        for value in ratios
    ):
        add(
            InterpretationLimitationCode.COMPONENT_AREA_UNCONSTRAINED,
            "At least one Lorentzian integrated area has area/SE below "
            f"{_AREA_TO_SE_SEVERE:g} or unavailable evidence.",
        )
    correlation_limitation = _component_correlation_limitation(result)
    if correlation_limitation is not None:
        limitations.append(correlation_limitation)
    start_count, start_span = _successful_start_span(fit)
    if start_count == 0:
        add(
            InterpretationLimitationCode.MULTISTART_UNAVAILABLE,
            "No successful optimizer start is available for interpretation.",
        )
    elif start_span >= _MULTISTART_SPAN_SEVERE:
        add(
            InterpretationLimitationCode.MULTISTART_DECOMPOSITION_INCONSISTENT,
            "Statistically equivalent successful starts have relative Lorentzian-"
            f"decomposition span {start_span:.6g}, at or above "
            f"{_MULTISTART_SPAN_SEVERE:g}.",
        )
    if limitations:
        return ComponentIdentifiability.SEVERE_LIMITATION, tuple(limitations)
    return ComponentIdentifiability.INTERPRETABLE, ()


def _candidate_warnings(result: CandidateFitResult) -> tuple[ScientificWarning, ...]:
    fit = result.fit
    if fit is None:
        return ()
    warnings: list[ScientificWarning] = []
    for index, ratio in enumerate(
        fit.diagnostics.fwhm_to_resolution_fwhm,
        start=1,
    ):
        if ratio is not None and math.isfinite(ratio) and ratio < 1.0:
            warnings.append(
                ScientificWarning(
                    ScientificWarningCode.INTRINSIC_FWHM_BELOW_RESOLUTION,
                    f"Lorentzian {index} intrinsic FWHM is {ratio:.6g} times "
                    "the measured-resolution FWHM; this is advisory only.",
                    result,
                )
            )
    condition = fit.diagnostics.condition_number
    if math.isfinite(condition) and condition >= _ADVISORY_CONDITION_NUMBER:
        warnings.append(
            ScientificWarning(
                ScientificWarningCode.HIGH_FINITE_CONDITION_NUMBER,
                f"Finite raw Jacobian condition number is {condition:.6g}; "
                "the calibrated 1e8 comparison is advisory, not a hard gate.",
                result,
            )
        )
    ratios = fit.diagnostics.adjacent_fwhm_ratios
    if ratios and min(ratios) <= _ADVISORY_ADJACENT_FWHM_RATIO:
        warnings.append(
            ScientificWarning(
                ScientificWarningCode.SIMILAR_INTRINSIC_LINEWIDTHS,
                f"Minimum adjacent intrinsic FWHM ratio is {min(ratios):.6g}; "
                "the calibrated 1.2 comparison is advisory, not a hard gate.",
                result,
            )
        )
    nuisance_bounds = tuple(
        name
        for name in fit.diagnostics.active_bounds
        if not _material_bound_parameter(name)
    )
    if nuisance_bounds:
        warnings.append(
            ScientificWarning(
                ScientificWarningCode.NUISANCE_PARAMETER_BOUND,
                "Nuisance/background parameter is active at a bound: "
                + ", ".join(nuisance_bounds)
                + ".",
                result,
            )
        )
    return tuple(warnings)


def _transition(
    successful: tuple[CandidateFitResult, ...],
    simpler_count: int,
    complex_count: int,
) -> tuple[
    CandidateFitResult | None,
    InformationCriterionComparison,
    tuple[MatchedBackgroundComparison, ...],
    BackgroundRobustness,
]:
    target = _best_family(successful, complex_count)
    information = _information_comparison(successful, simpler_count, complex_count)
    matched = _matched_background_comparisons(
        successful,
        simpler_count,
        complex_count,
    )
    return target, information, matched, _background_robustness(information, matched)


def _assessment(
    simpler_count: int,
    complex_count: int | None,
    status: AdditionalComplexityStatus,
    information: InformationCriterionComparison,
    robustness: BackgroundRobustness,
    matched: tuple[MatchedBackgroundComparison, ...],
    candidate: CandidateFitResult | None,
    limitations: tuple[InterpretationLimitation, ...],
    reason: str,
) -> AdditionalComplexityAssessment:
    return AdditionalComplexityAssessment(
        simpler_count,
        complex_count,
        status,
        information,
        robustness,
        matched,
        candidate,
        limitations,
        reason,
    )


def _background_alternative(
    successful: tuple[CandidateFitResult, ...],
    selected: CandidateFitResult,
) -> tuple[CandidateFitResult | None, bool]:
    count = selected.candidate.lorentzian_count
    family = tuple(
        result
        for result in successful
        if result.candidate.lorentzian_count == count and result is not selected
    )
    best_bic = _best_recommendation_candidate(successful, count, "bic")
    bic_disagrees = (
        best_bic is not None
        and best_bic.candidate.background is not selected.candidate.background
    )
    assert selected.fit is not None
    near: list[CandidateFitResult] = []
    for result in family:
        assert result.fit is not None
        identifiability, _ = _identifiability(result)
        if (
            result.fit.statistics.aicc - selected.fit.statistics.aicc
            <= _BACKGROUND_NEAR_AICC
            and result.fit.statistics.bic - selected.fit.statistics.bic
            <= _BACKGROUND_NEAR_BIC
            and _residual_adequacy(result) is not ResidualAdequacy.INADEQUATE
            and identifiability is not ComponentIdentifiability.SEVERE_LIMITATION
        ):
            near.append(result)

    def aicc(result: CandidateFitResult) -> float:
        assert result.fit is not None
        return result.fit.statistics.aicc

    return (
        min(
            near,
            key=lambda result: (
                aicc(result),
                _BACKGROUND_RANK[result.candidate.background],
            ),
        )
        if near
        else None,
        bic_disagrees,
    )


def _not_evaluable_assessment(
    simpler_count: int,
    complex_count: int | None,
    reason: str,
) -> AdditionalComplexityAssessment:
    return _assessment(
        simpler_count,
        complex_count,
        AdditionalComplexityStatus.NOT_EVALUABLE,
        InformationCriterionComparison(None, None, ComplexityEvidence.NOT_EVALUABLE),
        BackgroundRobustness.NOT_EVALUABLE,
        (),
        None,
        (),
        reason,
    )


def recommend_standard_candidates(
    candidate_results: tuple[CandidateFitResult, ...],
) -> AutoFitRecommendation:
    """Recommend the simplest adequately supported standard observable model.

    The input is existing 0L/1L/2L x NONE/B0/B1 candidate evidence. No fit is
    run and the returned object retains references to these results.
    """

    candidate_results = tuple(candidate_results)
    scope = AutoFitSearchScope()
    _validate_complete_candidate_results(candidate_results)
    successful = _successful_results(candidate_results)
    family_envelope = {count: _best_family(successful, count) for count in (0, 1, 2)}
    family_recommendation = {
        count: _best_recommendation_candidate(successful, count) for count in (0, 1, 2)
    }
    resolution_reliability = ResolutionReliabilityAssessment(
        status=ResolutionReliabilityStatus.NOT_ASSESSED,
        provenance_gap=(
            "Current FitProvenance records accepted resolution support, decision, "
            "confirmation, signed-area diagnostics, and neutral acceptance warnings, "
            "but no structured scientific assessment of whether relevant measured-"
            "resolution structure is truncated. AutoFit therefore applies no "
            "resolution-containment gate."
        ),
        reason=(
            "No structured resolution-containment assessment is present; no "
            "resolution-reliability limitation or clearance is inferred."
        ),
    )
    if family_envelope[0] is None:
        assessment = _not_evaluable_assessment(
            0,
            1,
            "No numerically usable 0L baseline candidate is available.",
        )
        return AutoFitRecommendation(
            most_recommended=None,
            recommended_lorentzian_count=None,
            primary_family_support=(
                PrimaryFamilySupport.NUMERICAL_RECOMMENDATION_UNAVAILABLE
            ),
            primary_residual_adequacy=ResidualAdequacy.NOT_EVALUABLE,
            primary_identifiability=ComponentIdentifiability.NOT_APPLICABLE,
            additional_complexity=assessment,
            transition_assessments=(assessment,),
            strong_alternative=None,
            comparator=None,
            interpretation_limitations=(),
            scientific_warnings=(),
            resolution_reliability=resolution_reliability,
            search_scope=scope,
            candidate_results=candidate_results,
        )

    current_count = 0
    current = family_recommendation[0] or family_envelope[0]
    assert current is not None
    comparator: CandidateFitResult | None = None
    strong_alternative: CandidateFitResult | None = None
    limitations: list[InterpretationLimitation] = []
    warnings: list[ScientificWarning] = []
    transitions: list[AdditionalComplexityAssessment] = []
    additional: AdditionalComplexityAssessment | None = None
    primary_caution = False

    for target_count in (1, 2):
        target_envelope, information, matched, robustness = _transition(
            successful,
            current_count,
            target_count,
        )
        if (
            target_envelope is None
            or information.evidence is ComplexityEvidence.NOT_EVALUABLE
        ):
            additional = _not_evaluable_assessment(
                current_count,
                target_count,
                f"The {target_count}L family has no complete numerically usable "
                "AICc/BIC evidence.",
            )
            transitions.append(additional)
            break
        target = family_recommendation[target_count] or target_envelope
        current_adequacy = _residual_adequacy(current)
        target_adequacy = _residual_adequacy(target)
        target_identifiability, target_limitations = _identifiability(target)

        if robustness is BackgroundRobustness.BACKGROUND_SUBSTITUTION:
            if target_identifiability is ComponentIdentifiability.SEVERE_LIMITATION:
                limitations.extend(target_limitations)
            additional = _assessment(
                current_count,
                target_count,
                AdditionalComplexityStatus.BACKGROUND_CONFOUNDED,
                information,
                robustness,
                matched,
                target,
                target_limitations
                if target_identifiability is ComponentIdentifiability.SEVERE_LIMITATION
                else (),
                f"Rejected {current_count}L->{target_count}L: apparent added-"
                "Lorentzian support without background disappears when B0/B1 is "
                "allowed. The retained simpler primary family is not demoted.",
            )
            transitions.append(additional)
            comparator = target
            warnings.append(
                ScientificWarning(
                    ScientificWarningCode.BACKGROUND_CONFOUNDED_COMPLEXITY,
                    additional.reason,
                    target,
                )
            )
            break
        if information.evidence is ComplexityEvidence.MARGINAL_OR_CONFLICTING:
            if target_identifiability is ComponentIdentifiability.SEVERE_LIMITATION:
                limitations.extend(target_limitations)
            eligibility_traversal = (
                current_adequacy is ResidualAdequacy.INADEQUATE
                and target_adequacy is not ResidualAdequacy.INADEQUATE
                and target_identifiability
                is not ComponentIdentifiability.SEVERE_LIMITATION
            )
            additional = _assessment(
                current_count,
                target_count,
                AdditionalComplexityStatus.MARGINAL,
                information,
                robustness,
                matched,
                target,
                target_limitations
                if target_identifiability is ComponentIdentifiability.SEVERE_LIMITATION
                else (),
                (
                    f"Family-envelope AICc/BIC evidence for {current_count}L->"
                    f"{target_count}L is marginal or conflicting. The higher "
                    "family is retained only because the lower family has no "
                    "recommendation-eligible candidate."
                    if eligibility_traversal
                    else f"The {current_count}L family remains adequate while "
                    f"family-envelope AICc/BIC evidence for {target_count}L is "
                    "marginal or conflicting."
                ),
            )
            transitions.append(additional)
            comparator = target
            if eligibility_traversal:
                current_count = target_count
                current = target
                primary_caution = True
                strong_alternative = None
                additional = None
                continue
            if (
                target_adequacy is not ResidualAdequacy.INADEQUATE
                and target_identifiability
                is not ComponentIdentifiability.SEVERE_LIMITATION
            ):
                strong_alternative = target
            break
        if information.evidence is ComplexityEvidence.LITTLE_OR_NONE:
            if target_identifiability is ComponentIdentifiability.SEVERE_LIMITATION:
                limitations.extend(target_limitations)
            additional = _assessment(
                current_count,
                target_count,
                AdditionalComplexityStatus.NOT_SUPPORTED,
                information,
                robustness,
                matched,
                target,
                target_limitations
                if target_identifiability is ComponentIdentifiability.SEVERE_LIMITATION
                else (),
                "Little or no family-envelope evidence supports "
                f"{current_count}L->{target_count}L.",
            )
            transitions.append(additional)
            if current_count > 0:
                comparator = target
            break
        if target_adequacy is ResidualAdequacy.INADEQUATE:
            residual_inadequacy = _assessment(
                current_count,
                target_count,
                AdditionalComplexityStatus.SUPPORTED_BUT_RESIDUALLY_INADEQUATE,
                information,
                robustness,
                matched,
                target,
                target_limitations
                if target_identifiability is ComponentIdentifiability.SEVERE_LIMITATION
                else (),
                f"Descriptive information-criterion evidence supports the "
                f"{current_count}L->{target_count}L transition, but every "
                f"recommendation candidate in the {target_count}L family has "
                "inadequate standardized-residual structure.",
            )
            transitions.append(residual_inadequacy)
            if target_identifiability is ComponentIdentifiability.SEVERE_LIMITATION:
                limitations.extend(target_limitations)
            comparator = target
            if current_adequacy is ResidualAdequacy.INADEQUATE:
                current_count = target_count
                current = target
                additional = None
                continue
            additional = residual_inadequacy
            break
        if target_identifiability is ComponentIdentifiability.SEVERE_LIMITATION:
            additional = _assessment(
                current_count,
                target_count,
                AdditionalComplexityStatus.SUPPORTED_BUT_UNINTERPRETABLE,
                information,
                robustness,
                matched,
                target,
                target_limitations,
                f"The {target_count}L family has {information.evidence.value} "
                "information-criterion evidence, but its Lorentzian decomposition "
                "has severe interpretation limitations.",
            )
            transitions.append(additional)
            limitations.extend(target_limitations)
            comparator = target
            break

        robust_enough = robustness in {
            BackgroundRobustness.ROBUST_ACROSS_B0_B1,
            BackgroundRobustness.SUPPORTED_IN_ONE_ALLOWED_BACKGROUND,
        }
        upgrade = current_adequacy is not ResidualAdequacy.ADEQUATE or (
            information.evidence is ComplexityEvidence.STRONG and robust_enough
        )
        if upgrade:
            previous = current
            by_key = _by_key(successful)
            current_count = target_count
            current = target
            comparator = by_key.get(
                (current_count - 1, target.candidate.background),
                previous,
            )
            primary_caution = primary_caution or (
                target_adequacy is ResidualAdequacy.QUESTIONABLE
                or robustness
                is BackgroundRobustness.SUPPORTED_IN_ONE_ALLOWED_BACKGROUND
            )
            supported_transition = _assessment(
                current_count - 1,
                current_count,
                AdditionalComplexityStatus.SUPPORTED_TRANSITION,
                information,
                robustness,
                matched,
                target,
                (),
                "Descriptive evidence supports the "
                f"{current_count - 1}L->{current_count}L transition: the simpler "
                f"family is {current_adequacy.value}, information-criterion "
                f"evidence is {information.evidence.value}, and components pass "
                "the calibrated severe-identifiability checks.",
            )
            transitions.append(supported_transition)
            if robustness in {
                BackgroundRobustness.SUPPORTED_IN_ONE_ALLOWED_BACKGROUND,
                BackgroundRobustness.BACKGROUND_SENSITIVE,
            }:
                warnings.append(
                    ScientificWarning(
                        ScientificWarningCode.BACKGROUND_SENSITIVE_COMPLEXITY,
                        f"The descriptively supported {current_count - 1}L->"
                        f"{current_count}L transition is not supported consistently "
                        "across both B0 and B1.",
                        target,
                    )
                )
            continue

        additional = _assessment(
            current_count,
            target_count,
            AdditionalComplexityStatus.MARGINAL,
            information,
            robustness,
            matched,
            target,
            (),
            f"The adequate {current_count}L family remains preferred: "
            f"{target_count}L evidence is not jointly strong and sufficiently "
            "background-robust to require an upgrade.",
        )
        transitions.append(additional)
        strong_alternative = target
        comparator = target
        break

    if additional is None:
        additional = _assessment(
            current_count,
            None,
            AdditionalComplexityStatus.SEARCH_LIMIT_REACHED,
            InformationCriterionComparison(
                None,
                None,
                ComplexityEvidence.NOT_EVALUABLE,
            ),
            BackgroundRobustness.NOT_EVALUABLE,
            (),
            None,
            (),
            "The automatic search was evaluated through its 2L boundary; absence "
            "of higher searched families is not evidence that higher-complexity "
            "physics is absent.",
        )

    final_adequacy = _residual_adequacy(current)
    final_identifiability, final_limitations = _identifiability(current)
    limitations.extend(final_limitations)
    if final_adequacy is ResidualAdequacy.INADEQUATE:
        comparator = current
        strong_alternative = None
        most_recommended: CandidateFitResult | None = None
        primary_support = PrimaryFamilySupport.NO_ADEQUATE_MODEL
    else:
        most_recommended = current
        if final_identifiability is ComponentIdentifiability.SEVERE_LIMITATION:
            primary_support = PrimaryFamilySupport.INTERPRETATION_LIMITED
        elif primary_caution or final_adequacy is ResidualAdequacy.QUESTIONABLE:
            primary_support = PrimaryFamilySupport.SUPPORTED_WITH_CAUTION
        else:
            primary_support = PrimaryFamilySupport.SUPPORTED

    if most_recommended is not None:
        background_alternative, bic_disagrees = _background_alternative(
            successful,
            most_recommended,
        )
        if background_alternative is not None:
            if strong_alternative is None:
                strong_alternative = background_alternative
            warnings.append(
                ScientificWarning(
                    ScientificWarningCode.WITHIN_FAMILY_BACKGROUND_AMBIGUITY,
                    "A scientifically adequate, interpretable background variant "
                    "within the selected Lorentzian family is near-competitive in "
                    "both AICc and BIC.",
                    background_alternative,
                )
            )
            if primary_support is PrimaryFamilySupport.SUPPORTED:
                primary_support = PrimaryFamilySupport.SUPPORTED_WITH_CAUTION
        if bic_disagrees:
            warnings.append(
                ScientificWarning(
                    ScientificWarningCode.WITHIN_FAMILY_AICC_BIC_DISAGREEMENT,
                    "AICc and BIC prefer different backgrounds within the selected "
                    "Lorentzian family.",
                    most_recommended,
                )
            )
        warnings.extend(_candidate_warnings(most_recommended))

    return AutoFitRecommendation(
        most_recommended=most_recommended,
        recommended_lorentzian_count=(
            current.candidate.lorentzian_count if most_recommended is not None else None
        ),
        primary_family_support=primary_support,
        primary_residual_adequacy=final_adequacy,
        primary_identifiability=final_identifiability,
        additional_complexity=additional,
        transition_assessments=tuple(transitions),
        strong_alternative=strong_alternative,
        comparator=comparator,
        interpretation_limitations=tuple(limitations),
        scientific_warnings=tuple(warnings),
        resolution_reliability=resolution_reliability,
        search_scope=scope,
        candidate_results=candidate_results,
    )


def auto_fit_single_q(
    prepared_resolution: PreparedResolution,
    selection: FittingSelection,
    group_index: int,
    *,
    max_nfev: int = 2500,
) -> AutoFitRecommendation:
    """Evaluate the fixed nine-candidate scope, then apply production policy."""

    from ezqens.fitting.core import evaluate_standard_candidates

    candidates = evaluate_standard_candidates(
        prepared_resolution,
        selection,
        group_index,
        max_lorentzians=2,
        allow_linear_background=True,
        max_nfev=max_nfev,
    )
    return recommend_standard_candidates(candidates)
