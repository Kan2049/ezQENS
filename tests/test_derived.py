"""Scientific contracts for component-resolved Multi-Q derived quantities."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from ezqens.batch import (
    MultiQBranchResult,
    MultiQExecutionStatus,
    MultiQFitOutcome,
    MultiQFitStatus,
)
from ezqens.derived import (
    ComponentAreaEstimate,
    DerivedPointStatus,
    DerivedQENSPoint,
    DerivedQENSResult,
    DerivedValueStatus,
    DerivedWarningCode,
    StatisticalUncertaintyStatus,
    derive_qens,
)
from ezqens.domain import DiagnosticSeverity, QBins
from ezqens.fitting import (
    BACKGROUND_COMPONENT,
    ELASTIC_COMPONENT,
    BackgroundModel,
    ComponentFamily,
    ComponentIdentity,
    FitDiagnostics,
    FitProvenance,
    FitResult,
    FitStatistics,
    LorentzianComponent,
    ManualFitDiagnosticCode,
    ManualFitReadinessDiagnostic,
    ModelEvaluation,
    ParameterConfiguration,
    ParameterEstimate,
    ParameterFamily,
    ParameterReference,
    ParameterTieGroup,
    SpectralModelDefinition,
)

L1 = ComponentIdentity(ComponentFamily.LORENTZIAN, "L1")
L2 = ComponentIdentity(ComponentFamily.LORENTZIAN, "L2")


def _configuration(
    value: float,
    *,
    free: bool = True,
    positive: bool = False,
) -> ParameterConfiguration:
    return ParameterConfiguration(
        value,
        np.finfo(float).tiny if positive else -np.inf,
        np.inf,
        free,
    )


def _model(
    *,
    elastic: bool = True,
    lorentzian_identities: tuple[ComponentIdentity, ...] = (L1,),
    background: BackgroundModel = BackgroundModel.NONE,
) -> SpectralModelDefinition:
    has_centered_component = elastic or bool(lorentzian_identities)
    return SpectralModelDefinition(
        energy_shift=_configuration(0.0) if has_centered_component else None,
        elastic_area=_configuration(2.0, positive=True) if elastic else None,
        lorentzians=tuple(
            LorentzianComponent(
                area=_configuration(float(index + 1), positive=True),
                fwhm=_configuration(0.2 * (index + 1), positive=True),
                identity=identity,
            )
            for index, identity in enumerate(lorentzian_identities)
        ),
        background=background,
        b0=_configuration(0.1) if background is not BackgroundModel.NONE else None,
        b1=_configuration(0.0) if background is BackgroundModel.LINEAR else None,
    )


def _estimate(
    name: str,
    value: float,
    references: tuple[ParameterReference, ...],
    *,
    free: bool = True,
    standard_error: float | None = 0.01,
) -> ParameterEstimate:
    return ParameterEstimate(
        name=name,
        value=value,
        standard_error=standard_error if free else None,
        lower_bound=-np.inf,
        upper_bound=np.inf,
        free=free,
        references=references,
    )


def _fit(
    model: SpectralModelDefinition,
    *,
    q_value: float,
    area_values: dict[ComponentIdentity, float] | None = None,
    fwhm_values: dict[ComponentIdentity, float] | None = None,
    fixed_areas: frozenset[ComponentIdentity] = frozenset(),
    fixed_fwhm: frozenset[ComponentIdentity] = frozenset(),
    covariance: np.ndarray | None | str = "identity",
    tied_area_references: tuple[ParameterReference, ...] = (),
    energy_unit: str = "meV",
) -> FitResult:
    area_values = area_values or {}
    fwhm_values = fwhm_values or {}
    parameters: list[ParameterEstimate] = []
    centered_references: list[ParameterReference] = []
    if model.elastic_area is not None:
        centered_references.append(
            ParameterReference(ELASTIC_COMPONENT, ParameterFamily.CENTER)
        )
    centered_references.extend(
        ParameterReference(component.identity, ParameterFamily.CENTER)
        for component in model.lorentzians
    )
    if centered_references:
        parameters.append(_estimate("energy_shift", 0.0, tuple(centered_references)))

    tied_components = {reference.component for reference in tied_area_references}
    if tied_area_references:
        tied_value = area_values[tied_area_references[0].component]
        parameters.append(
            _estimate("tied_area", tied_value, tied_area_references, standard_error=0.1)
        )
    if model.elastic_area is not None and ELASTIC_COMPONENT not in tied_components:
        parameters.append(
            _estimate(
                "elastic_area",
                area_values.get(ELASTIC_COMPONENT, 2.0),
                (ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA),),
                free=ELASTIC_COMPONENT not in fixed_areas,
                standard_error=0.1,
            )
        )
    for component in model.lorentzians:
        identity = component.identity
        if identity not in tied_components:
            parameters.append(
                _estimate(
                    f"{identity.component_id}_area",
                    area_values.get(identity, component.area.initial_value),
                    (ParameterReference(identity, ParameterFamily.AREA),),
                    free=identity not in fixed_areas,
                    standard_error=0.1,
                )
            )
        parameters.append(
            _estimate(
                f"{identity.component_id}_fwhm",
                fwhm_values.get(identity, component.fwhm.initial_value),
                (ParameterReference(identity, ParameterFamily.FWHM),),
                free=identity not in fixed_fwhm,
                standard_error=0.02,
            )
        )
    if model.b0 is not None:
        parameters.append(
            _estimate(
                "background_offset",
                model.b0.initial_value,
                (ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.OFFSET),),
            )
        )
    if model.b1 is not None:
        parameters.append(
            _estimate(
                "background_slope",
                model.b1.initial_value,
                (ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.SLOPE),),
            )
        )

    matrix: np.ndarray | None
    if isinstance(covariance, str):
        matrix = np.eye(len(parameters), dtype=np.float64)
    else:
        matrix = covariance
    energy = np.asarray([-0.1, 0.0, 0.1])
    evaluation = ModelEvaluation(
        energy,
        np.zeros(3),
        np.zeros(3),
        tuple(np.zeros(3) for _ in model.lorentzians),
        np.zeros(3),
    )
    return FitResult(
        configuration=model,
        parameters=tuple(parameters),
        covariance=matrix,
        correlation=None,
        evaluation=evaluation,
        raw_residuals=np.zeros(3),
        standardized_residuals=np.zeros(3),
        statistics=FitStatistics(0.0, 0.0, 3, 0, 3, 0.0, 0.0, 0.0),
        diagnostics=cast(
            FitDiagnostics,
            SimpleNamespace(optimizer_success=True),
        ),
        provenance=cast(
            FitProvenance,
            SimpleNamespace(q_value=q_value, energy_unit=energy_unit),
        ),
        fitted_model=model,
    )


def _branch(*fits: FitResult) -> tuple[MultiQBranchResult, QBins]:
    outcomes = tuple(
        MultiQFitOutcome(index, MultiQFitStatus.SUCCESS, fit)
        for index, fit in enumerate(fits)
    )
    return (
        MultiQBranchResult(0, MultiQExecutionStatus.COMPLETED, outcomes),
        QBins.from_q_values([fit.provenance.q_value for fit in fits]),
    )


def test_component_resolved_fwhm_tau_preserves_anchor_identity_order_across_q() -> None:
    anchor_model = _model(lorentzian_identities=(L1, L2))
    reversed_model = _model(lorentzian_identities=(L2, L1))
    first = _fit(
        anchor_model,
        q_value=0.5,
        fwhm_values={L1: 0.8, L2: 0.1},
    )
    second = _fit(
        reversed_model,
        q_value=1.0,
        fwhm_values={L1: 0.7, L2: 0.2},
    )
    branch, q_bins = _branch(first, second)

    result = derive_qens(branch, q_bins)

    assert result.source_branch is branch
    assert [item.component for item in result.point(0).lorentzians] == [L1, L2]
    assert [item.component for item in result.point(1).lorentzians] == [L1, L2]
    assert [item.fwhm_mev for item in result.point(0).lorentzians] == [0.8, 0.1]
    assert result.point(0).lorentzians[0].tau_ps == pytest.approx(1.3164239138 / 0.8)


def test_tau_uses_analytic_fwhm_uncertainty_and_fixed_width_is_not_zero_error() -> None:
    model = _model(lorentzian_identities=(L1, L2))
    fit = _fit(
        model,
        q_value=0.5,
        fwhm_values={L1: 0.4, L2: 0.2},
        fixed_fwhm=frozenset({L2}),
    )
    branch, q_bins = _branch(fit)

    widths = derive_qens(branch, q_bins).point(0).lorentzians

    assert widths[0].fwhm_standard_error_mev == pytest.approx(0.02)
    assert widths[0].tau_standard_error_ps == pytest.approx(
        1.3164239138 * 0.02 / 0.4**2
    )
    assert widths[1].tau_ps == pytest.approx(1.3164239138 / 0.2)
    assert widths[1].tau_standard_error_ps is None
    assert (
        widths[1].uncertainty_status is StatisticalUncertaintyStatus.FIXED_NOT_ESTIMATED
    )


@pytest.mark.parametrize("fwhm", [0.0, -0.1, np.nan, np.inf])
def test_invalid_fwhm_retains_value_but_has_no_tau(fwhm: float) -> None:
    fit = _fit(_model(), q_value=0.5, fwhm_values={L1: fwhm})
    branch, q_bins = _branch(fit)

    width = derive_qens(branch, q_bins).point(0).lorentzians[0]

    assert width.validity is DerivedValueStatus.UNAVAILABLE
    assert width.tau_ps is None
    assert width.tau_standard_error_ps is None
    assert width.warnings[0].code is DerivedWarningCode.INVALID_FWHM


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (_model(lorentzian_identities=()), 1.0),
        (_model(elastic=False), 0.0),
    ],
)
def test_structural_elastic_only_and_lorentzian_only_eisf(
    model: SpectralModelDefinition,
    expected: float,
) -> None:
    fit = _fit(model, q_value=0.5)
    branch, q_bins = _branch(fit)

    eisf = derive_qens(branch, q_bins).point(0).eisf

    assert eisf is not None
    assert eisf.value == pytest.approx(expected)
    assert eisf.standard_error is None
    assert (
        eisf.uncertainty_status
        is StatisticalUncertaintyStatus.STRUCTURAL_VALUE_NOT_ESTIMATED
    )


def test_background_only_and_zero_total_area_have_unavailable_eisf() -> None:
    background = _fit(
        _model(
            elastic=False,
            lorentzian_identities=(),
            background=BackgroundModel.CONSTANT,
        ),
        q_value=0.5,
    )
    zero_total = _fit(
        _model(),
        q_value=1.0,
        area_values={ELASTIC_COMPONENT: 0.0, L1: 0.0},
    )
    background_branch, background_q = _branch(background)
    zero_branch, zero_q = _branch(zero_total)

    background_result = derive_qens(background_branch, background_q)
    zero_result = derive_qens(zero_branch, zero_q)

    background_eisf = background_result.point(0).eisf
    zero_total_eisf = zero_result.point(0).eisf
    assert background_eisf is not None
    assert background_eisf.validity is DerivedValueStatus.UNAVAILABLE
    assert zero_total_eisf is not None
    assert zero_total_eisf.value is None


def test_multilorentzian_eisf_uses_full_covariance_including_off_diagonal() -> None:
    model = _model(lorentzian_identities=(L1, L2))
    provisional = _fit(
        model,
        q_value=0.5,
        area_values={ELASTIC_COMPONENT: 2.0, L1: 1.0, L2: 1.0},
    )
    covariance = np.zeros_like(provisional.covariance)
    assert covariance is not None
    references = (
        ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA),
        ParameterReference(L1, ParameterFamily.AREA),
        ParameterReference(L2, ParameterFamily.AREA),
    )
    indices = [
        next(
            index
            for index, parameter in enumerate(provisional.parameters)
            if reference in parameter.references
        )
        for reference in references
    ]
    area_covariance = np.asarray(
        [[0.16, 0.02, -0.01], [0.02, 0.09, 0.015], [-0.01, 0.015, 0.04]]
    )
    covariance[np.ix_(indices, indices)] = area_covariance
    fit = replace(provisional, covariance=covariance)
    branch, q_bins = _branch(fit)

    eisf = derive_qens(branch, q_bins).point(0).eisf

    gradient = np.asarray([0.125, -0.125, -0.125])
    assert eisf is not None
    assert eisf.q_value == pytest.approx(0.5)
    assert eisf.value == pytest.approx(0.5)
    assert eisf.standard_error == pytest.approx(
        np.sqrt(gradient @ area_covariance @ gradient)
    )
    assert eisf.uncertainty_status is StatisticalUncertaintyStatus.AVAILABLE
    assert [area.component for area in eisf.quasielastic_areas] == [L1, L2]


def test_tied_area_derivatives_are_aggregated_onto_one_covariance_parameter() -> None:
    tied = (
        ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA),
        ParameterReference(L1, ParameterFamily.AREA),
    )
    model = replace(
        _model(lorentzian_identities=(L1, L2)),
        parameter_ties=(
            ParameterTieGroup(
                "elastic_l1_area",
                tied,
                _configuration(2.0, positive=True),
            ),
        ),
    )
    provisional = _fit(
        model,
        q_value=0.5,
        area_values={ELASTIC_COMPONENT: 2.0, L1: 2.0, L2: 1.0},
        tied_area_references=tied,
    )
    covariance = np.zeros_like(provisional.covariance)
    assert covariance is not None
    tied_index = next(
        index
        for index, parameter in enumerate(provisional.parameters)
        if set(parameter.references) == set(tied)
    )
    l2_index = next(
        index
        for index, parameter in enumerate(provisional.parameters)
        if ParameterReference(L2, ParameterFamily.AREA) in parameter.references
    )
    covariance[tied_index, tied_index] = 4.0
    covariance[l2_index, l2_index] = 9.0
    fit = replace(provisional, covariance=covariance)
    branch, q_bins = _branch(fit)

    eisf = derive_qens(branch, q_bins).point(0).eisf

    # d/d(shared Ae=A1) combines 3/25 - 2/25; d/dA2 is -2/25.
    expected_variance = (1.0 / 25.0) ** 2 * 4.0 + (-2.0 / 25.0) ** 2 * 9.0
    assert eisf is not None
    assert eisf.value == pytest.approx(0.4)
    assert eisf.standard_error == pytest.approx(np.sqrt(expected_variance))
    assert eisf.elastic_area is not None and eisf.elastic_area.shared_references == tied


def test_mixed_fixed_free_eisf_is_conditional_and_all_fixed_is_not_zero_error() -> None:
    model = _model()
    mixed_provisional = _fit(
        model,
        q_value=0.5,
        area_values={ELASTIC_COMPONENT: 2.0, L1: 2.0},
        fixed_areas=frozenset({ELASTIC_COMPONENT}),
    )
    mixed_covariance = np.zeros_like(mixed_provisional.covariance)
    assert mixed_covariance is not None
    l1_index = next(
        index
        for index, parameter in enumerate(mixed_provisional.parameters)
        if ParameterReference(L1, ParameterFamily.AREA) in parameter.references
    )
    mixed_covariance[l1_index, l1_index] = 0.16
    mixed = replace(mixed_provisional, covariance=mixed_covariance)
    fixed = _fit(
        model,
        q_value=1.0,
        area_values={ELASTIC_COMPONENT: 2.0, L1: 2.0},
        fixed_areas=frozenset({ELASTIC_COMPONENT, L1}),
    )
    branch, q_bins = _branch(mixed, fixed)

    result = derive_qens(branch, q_bins)
    mixed_eisf = result.point(0).eisf
    fixed_eisf = result.point(1).eisf

    assert mixed_eisf is not None
    assert mixed_eisf.standard_error == pytest.approx(0.05)
    assert (
        mixed_eisf.uncertainty_status
        is StatisticalUncertaintyStatus.AVAILABLE_CONDITIONAL_ON_FIXED
    )
    assert DerivedWarningCode.EISF_FIXED_CONTRIBUTORS in {
        warning.code for warning in mixed_eisf.warnings
    }
    assert fixed_eisf is not None
    assert fixed_eisf.standard_error is None
    assert (
        fixed_eisf.uncertainty_status
        is StatisticalUncertaintyStatus.ALL_FIXED_NOT_ESTIMATED
    )


@pytest.mark.parametrize(
    ("covariance", "status", "warning_code"),
    [
        (
            None,
            StatisticalUncertaintyStatus.COVARIANCE_UNAVAILABLE,
            DerivedWarningCode.EISF_COVARIANCE_UNAVAILABLE,
        ),
        (
            "nonfinite",
            StatisticalUncertaintyStatus.COVARIANCE_UNUSABLE,
            DerivedWarningCode.EISF_COVARIANCE_UNUSABLE,
        ),
        (
            "asymmetric",
            StatisticalUncertaintyStatus.COVARIANCE_UNUSABLE,
            DerivedWarningCode.EISF_COVARIANCE_UNUSABLE,
        ),
    ],
)
def test_valid_eisf_survives_missing_or_unusable_covariance(
    covariance: None | str,
    status: StatisticalUncertaintyStatus,
    warning_code: DerivedWarningCode,
) -> None:
    fit = _fit(_model(), q_value=0.5, covariance=covariance)
    if covariance == "nonfinite":
        invalid = np.full((len(fit.parameters), len(fit.parameters)), np.nan)
        fit = replace(fit, covariance=invalid)
    elif covariance == "asymmetric":
        invalid = np.eye(len(fit.parameters))
        area_indices = [
            index
            for index, parameter in enumerate(fit.parameters)
            if any(
                reference.family is ParameterFamily.AREA
                for reference in parameter.references
            )
        ]
        invalid[area_indices[0], area_indices[1]] = 0.1
        fit = replace(fit, covariance=invalid)
    branch, q_bins = _branch(fit)

    eisf = derive_qens(branch, q_bins).point(0).eisf

    assert eisf is not None
    assert eisf.value == pytest.approx(2.0 / 3.0)
    assert eisf.standard_error is None
    assert eisf.uncertainty_status is status
    assert warning_code in {warning.code for warning in eisf.warnings}


def test_symmetric_indefinite_covariance_is_unusable_even_with_positive_variance() -> (
    None
):
    provisional = _fit(
        _model(),
        q_value=0.5,
        area_values={ELASTIC_COMPONENT: 2.0, L1: 1.0},
    )
    covariance = np.zeros_like(provisional.covariance)
    assert covariance is not None
    elastic_index = next(
        index
        for index, parameter in enumerate(provisional.parameters)
        if ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA)
        in parameter.references
    )
    l1_index = next(
        index
        for index, parameter in enumerate(provisional.parameters)
        if ParameterReference(L1, ParameterFamily.AREA) in parameter.references
    )
    covariance[elastic_index, elastic_index] = -1.0
    covariance[l1_index, l1_index] = 1.0
    gradient = np.asarray([1.0 / 9.0, -2.0 / 9.0])
    relevant = covariance[np.ix_([elastic_index, l1_index], [elastic_index, l1_index])]
    assert float(gradient @ relevant @ gradient) > 0.0
    fit = replace(provisional, covariance=covariance)
    branch, q_bins = _branch(fit)

    eisf = derive_qens(branch, q_bins).point(0).eisf

    assert eisf is not None
    assert eisf.value == pytest.approx(2.0 / 3.0)
    assert eisf.standard_error is None
    assert eisf.uncertainty_status is StatisticalUncertaintyStatus.COVARIANCE_UNUSABLE
    assert DerivedWarningCode.EISF_COVARIANCE_UNUSABLE in {
        warning.code for warning in eisf.warnings
    }


def test_roundoff_scale_negative_covariance_eigenvalue_remains_usable() -> None:
    provisional = _fit(
        _model(),
        q_value=0.5,
        area_values={ELASTIC_COMPONENT: 2.0, L1: 1.0},
    )
    covariance = np.zeros_like(provisional.covariance)
    assert covariance is not None
    elastic_index = next(
        index
        for index, parameter in enumerate(provisional.parameters)
        if ParameterReference(ELASTIC_COMPONENT, ParameterFamily.AREA)
        in parameter.references
    )
    l1_index = next(
        index
        for index, parameter in enumerate(provisional.parameters)
        if ParameterReference(L1, ParameterFamily.AREA) in parameter.references
    )
    covariance[elastic_index, elastic_index] = 1.0
    covariance[l1_index, l1_index] = -np.finfo(np.float64).eps
    fit = replace(provisional, covariance=covariance)
    branch, q_bins = _branch(fit)

    eisf = derive_qens(branch, q_bins).point(0).eisf

    assert eisf is not None
    assert eisf.standard_error is not None
    assert eisf.uncertainty_status is StatisticalUncertaintyStatus.AVAILABLE


@pytest.mark.parametrize("anchor_elastic", [True, False])
def test_elastic_presence_mismatch_blocks_derived_branch_point(
    anchor_elastic: bool,
) -> None:
    anchor = _fit(_model(elastic=anchor_elastic), q_value=0.5)
    incompatible = _fit(_model(elastic=not anchor_elastic), q_value=1.0)
    branch, q_bins = _branch(anchor, incompatible)

    result = derive_qens(branch, q_bins)

    assert result.point(0).derivation_status is DerivedPointStatus.AVAILABLE
    if not anchor_elastic:
        anchor_eisf = result.point(0).eisf
        assert anchor_eisf is not None
        assert anchor_eisf.value == pytest.approx(0.0)
    assert result.point(1).derivation_status is DerivedPointStatus.BLOCKED
    assert result.point(1).eisf is None
    assert (
        result.point(1).warnings[0].code
        is DerivedWarningCode.INCOMPATIBLE_COMPONENT_TOPOLOGY
    )
    assert result.point(1).warnings[0].message == (
        "FitResult component topology differs from the anchor branch"
    )


def test_background_difference_does_not_change_derived_branch_topology() -> None:
    anchor = _fit(_model(), q_value=0.5)
    with_background = _fit(
        _model(background=BackgroundModel.LINEAR),
        q_value=1.0,
    )
    branch, q_bins = _branch(anchor, with_background)

    result = derive_qens(branch, q_bins)

    assert all(
        point.derivation_status is DerivedPointStatus.AVAILABLE
        for point in result.points
    )


def test_unavailable_branch_points_preserve_q_and_have_no_derived_values() -> None:
    fit = _fit(_model(), q_value=0.5)
    diagnostic = ManualFitReadinessDiagnostic(
        ManualFitDiagnosticCode.INVALID_SELECTION,
        DiagnosticSeverity.ERROR,
        "blocked",
        group_index=2,
    )
    outcomes = (
        MultiQFitOutcome(0, MultiQFitStatus.SUCCESS, fit),
        MultiQFitOutcome(1, MultiQFitStatus.FAILED, error_type="failure"),
        MultiQFitOutcome(2, MultiQFitStatus.BLOCKED, diagnostics=(diagnostic,)),
        MultiQFitOutcome(3, MultiQFitStatus.EXCLUDED),
        MultiQFitOutcome(4, MultiQFitStatus.NOT_RUN),
        MultiQFitOutcome(
            5,
            MultiQFitStatus.SUCCESS,
            replace(
                fit,
                provenance=cast(
                    FitProvenance,
                    SimpleNamespace(q_value=3.0, energy_unit="meV"),
                ),
            ),
            derived_result_excluded=True,
        ),
    )
    branch = MultiQBranchResult(0, MultiQExecutionStatus.CANCELLED, outcomes)
    q_bins = QBins.from_q_values([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    result = derive_qens(branch, q_bins)

    assert [point.execution_status for point in result.points] == [
        outcome.status for outcome in outcomes
    ]
    assert [point.q_value for point in result.points] == pytest.approx(q_bins.q_values)
    assert all(
        not point.lorentzians and point.eisf is None for point in result.points[1:]
    )
    assert result.point(5).derivation_status is DerivedPointStatus.EXCLUDED


def test_noncanonical_fit_energy_unit_blocks_only_the_affected_q_point() -> None:
    canonical = _fit(_model(), q_value=0.5)
    unresolved = _fit(_model(), q_value=1.0, energy_unit="unknown")
    branch, q_bins = _branch(canonical, unresolved)

    result = derive_qens(branch, q_bins)

    assert result.point(0).derivation_status is DerivedPointStatus.AVAILABLE
    assert result.point(1).derivation_status is DerivedPointStatus.BLOCKED
    assert not result.point(1).lorentzians
    assert result.point(1).eisf is None
    assert (
        result.point(1).warnings[0].code is DerivedWarningCode.NON_CANONICAL_ENERGY_UNIT
    )


def test_separate_branches_never_rematch_component_identities() -> None:
    other = ComponentIdentity(ComponentFamily.LORENTZIAN, "other")
    first_fit = _fit(_model(lorentzian_identities=(L1,)), q_value=0.5)
    second_fit = _fit(_model(lorentzian_identities=(other,)), q_value=0.5)
    first_branch, q_bins = _branch(first_fit)
    second_branch, _ = _branch(second_fit)

    first = derive_qens(first_branch, q_bins)
    second = derive_qens(second_branch, q_bins)

    assert first.point(0).lorentzians[0].component is L1
    assert second.point(0).lorentzians[0].component is other


def test_direct_lorentzian_constructor_rejects_contradictory_states() -> None:
    fit = _fit(_model(), q_value=0.5)
    branch, q_bins = _branch(fit)
    width = derive_qens(branch, q_bins).point(0).lorentzians[0]

    with pytest.raises(ValueError, match="FWHM must be finite and positive"):
        replace(width, fwhm_mev=0.0)
    with pytest.raises(ValueError, match="relaxation time must match"):
        replace(width, tau_ps=None)
    with pytest.raises(ValueError, match="finite nonnegative errors"):
        replace(width, fwhm_standard_error_mev=None)
    with pytest.raises(ValueError, match="finite nonnegative errors"):
        replace(width, fwhm_standard_error_mev=-0.1)
    with pytest.raises(ValueError, match="must not carry errors"):
        replace(
            width,
            uncertainty_status=StatisticalUncertaintyStatus.FIXED_NOT_ESTIMATED,
        )
    with pytest.raises(ValueError, match="must not contain tau"):
        replace(
            width,
            validity=DerivedValueStatus.UNAVAILABLE,
            uncertainty_status=StatisticalUncertaintyStatus.VALUE_UNAVAILABLE,
        )


def test_direct_eisf_constructor_rejects_value_and_provenance_contradictions() -> None:
    fit = _fit(_model(), q_value=0.5)
    branch, q_bins = _branch(fit)
    eisf = derive_qens(branch, q_bins).point(0).eisf
    assert eisf is not None
    assert eisf.elastic_area is not None

    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        replace(eisf, value=1.1)
    with pytest.raises(ValueError, match="unavailable EISF"):
        replace(
            eisf,
            validity=DerivedValueStatus.UNAVAILABLE,
            uncertainty_status=StatisticalUncertaintyStatus.VALUE_UNAVAILABLE,
        )
    with pytest.raises(ValueError, match="unavailable EISF must not carry"):
        replace(
            eisf,
            validity=DerivedValueStatus.UNAVAILABLE,
            uncertainty_status=StatisticalUncertaintyStatus.VALUE_UNAVAILABLE,
            standard_error=None,
        )
    with pytest.raises(ValueError, match="available EISF uncertainty"):
        replace(eisf, standard_error=None)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        replace(eisf, standard_error=-0.1)
    with pytest.raises(ValueError, match="must match its contributing areas"):
        replace(eisf, value=0.5)
    with pytest.raises(ValueError, match="requires fixed and free contributors"):
        replace(
            eisf,
            uncertainty_status=(
                StatisticalUncertaintyStatus.AVAILABLE_CONDITIONAL_ON_FIXED
            ),
        )
    mismatched_area = replace(eisf.elastic_area, value=eisf.elastic_area.value + 1.0)
    with pytest.raises(ValueError, match="provenance must match"):
        replace(eisf, elastic_area=mismatched_area)
    foreign_reference = ParameterReference(L1, ParameterFamily.AREA)
    invalid_area = ComponentAreaEstimate(
        ELASTIC_COMPONENT,
        eisf.elastic_area.value,
        eisf.elastic_area.free,
        eisf.elastic_area.parameter_name,
        (eisf.elastic_area.reference, foreign_reference),
    )
    with pytest.raises(ValueError, match="provenance must match"):
        replace(eisf, elastic_area=invalid_area)


def test_direct_point_and_result_constructors_require_exact_source_evidence() -> None:
    fit = _fit(_model(), q_value=0.5)
    branch, q_bins = _branch(fit)
    result = derive_qens(branch, q_bins)
    point = result.point(0)

    with pytest.raises(ValueError, match="requires its source fit"):
        replace(point, eisf=None)
    inconsistent_width = replace(
        point.lorentzians[0],
        fwhm_mev=0.4,
        tau_ps=1.3164239138 / 0.4,
        tau_standard_error_ps=1.3164239138 * 0.02 / 0.4**2,
    )
    with pytest.raises(ValueError, match="must match its source FitResult"):
        replace(point, lorentzians=(inconsistent_width,))

    cloned_outcome = replace(point.source_outcome)
    cloned_point = replace(point, source_outcome=cloned_outcome)
    with pytest.raises(ValueError, match="exact source outcomes"):
        DerivedQENSResult(result.source_branch, result.q_bins, (cloned_point,))

    failed_outcome = MultiQFitOutcome(0, MultiQFitStatus.FAILED)
    with pytest.raises(ValueError, match="requires a successful fit"):
        DerivedQENSPoint(
            0,
            0.5,
            MultiQFitStatus.FAILED,
            DerivedPointStatus.AVAILABLE,
            failed_outcome,
            point.lorentzians,
            point.eisf,
        )


def test_public_derived_value_classes_remain_immutable() -> None:
    fit = _fit(_model(), q_value=0.5)
    branch, q_bins = _branch(fit)
    result = derive_qens(branch, q_bins)
    point = result.point(0)
    eisf = point.eisf
    assert eisf is not None
    values: tuple[tuple[object, str], ...] = (
        (point.lorentzians[0], "warnings"),
        (eisf, "warnings"),
        (point, "warnings"),
        (result, "points"),
    )
    for value, attribute in values:
        with pytest.raises(FrozenInstanceError):
            setattr(value, attribute, ())
