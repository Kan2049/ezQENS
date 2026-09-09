"""Fractional-coverage Q-rebin and project-lifecycle tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ezqens.domain import (
    FractionalCoverage,
    FractionalCoverageOrigin,
    QBins,
    QExclusionInterval,
    QRebinSpecification,
    ReducedDataset,
    Spectrum,
    SpectrumRole,
)
from ezqens.preprocessing import (
    QRebinDiagnosticCode,
    QRebinError,
    rebin_fractional_q,
)
from ezqens.workflow import (
    WorkflowDiagnosticCode,
    WorkflowError,
    add_project_dataset,
    applied_resolution,
    apply_resolution,
    create_project,
    project_dataset,
    rebin_project_sample_q,
)


def make_rebin_dataset(
    role: SpectrumRole = SpectrumRole.SAMPLE,
    *,
    edges: np.ndarray | None = None,
    energies: tuple[np.ndarray, ...] | None = None,
    intensities: tuple[np.ndarray, ...] | None = None,
    uncertainties: tuple[np.ndarray, ...] | None = None,
    coverage_values: tuple[np.ndarray, ...] | None = None,
) -> ReducedDataset:
    """Build a compact edge-defined dataset with explicit fractional coverage."""

    q_edges = np.asarray(
        [0.0, 1.0, 2.0, 3.0, 4.0] if edges is None else edges,
        dtype=np.float64,
    )
    group_count = q_edges.size - 1
    energy_values = energies or tuple(
        np.array([-1.0, 0.0, 1.0]) for _ in range(group_count)
    )
    intensity_values = intensities or tuple(
        np.full(energy_values[index].size, 10.0 * (index + 1))
        for index in range(group_count)
    )
    uncertainty_values = uncertainties or tuple(
        np.full(energy_values[index].size, float(index + 1))
        for index in range(group_count)
    )
    spectra = tuple(
        Spectrum(
            role=role,
            group_index=index,
            group_label=f"source-{index}",
            energy=energy_values[index],
            intensity=intensity_values[index],
            uncertainty=uncertainty_values[index],
            energy_unit="meV",
            intensity_unit="arb",
            uncertainty_unit="arb",
        )
        for index in range(group_count)
    )
    values = coverage_values or tuple(
        np.ones(spectrum.energy.size) for spectrum in spectra
    )
    coverage = FractionalCoverage.aligned_with(
        spectra,
        values=values,
        origin=(
            FractionalCoverageOrigin.CONFIRMED_UNREBINNED_SOURCE
            if all(np.all(value == 1.0) for value in values)
            else FractionalCoverageOrigin.EXPLICIT_SOURCE
        ),
    )
    return ReducedDataset(
        role=role,
        spectra=spectra,
        q_bins=QBins.from_edges(q_edges),
        fractional_coverage=coverage,
    )


def specification(
    edges: tuple[float, ...],
    exclusions: tuple[QExclusionInterval, ...] = (),
) -> QRebinSpecification:
    return QRebinSpecification(target_edges=edges, exclusions=exclusions)


def diagnostic_code(error: QRebinError) -> QRebinDiagnosticCode:
    return error.diagnostics[0].code


def test_exact_overlap_formula_propagates_signal_uncertainty_and_coverage() -> None:
    source = make_rebin_dataset()

    rebinned = rebin_fractional_q(source, specification((0.0, 2.0, 4.0)))

    assert rebinned.fractional_coverage is not None
    assert (
        rebinned.fractional_coverage.origin
        is FractionalCoverageOrigin.PROPAGATED_Q_REBIN
    )
    np.testing.assert_allclose(rebinned.spectra[0].intensity, 15.0)
    np.testing.assert_allclose(
        rebinned.spectra[0].uncertainty,
        np.sqrt(1.0**2 + 2.0**2) / 2.0,
    )
    np.testing.assert_allclose(rebinned.fractional_coverage.values[0], 2.0)
    np.testing.assert_allclose(rebinned.spectra[1].intensity, 35.0)
    np.testing.assert_allclose(
        rebinned.spectra[1].uncertainty,
        np.sqrt(3.0**2 + 4.0**2) / 2.0,
    )
    assert rebinned.q_bins is not None
    np.testing.assert_array_equal(rebinned.q_bins.edges, [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(rebinned.q_bins.q_values, [1.0, 3.0])


@pytest.mark.parametrize("invalid_value", (0.0, -0.25))
def test_positively_weighted_invalid_uncertainty_remains_invalid(
    invalid_value: float,
) -> None:
    source = make_rebin_dataset(
        edges=np.array([0.0, 1.0, 2.0]),
        uncertainties=(
            np.array([1.0, invalid_value, 1.0]),
            np.array([2.0, 2.0, 2.0]),
        ),
    )

    rebinned = rebin_fractional_q(source, specification((0.0, 2.0)))

    assert rebinned.fractional_coverage is not None
    np.testing.assert_allclose(rebinned.spectra[0].intensity, 15.0)
    np.testing.assert_allclose(rebinned.fractional_coverage.values[0], 2.0)
    assert np.isnan(rebinned.spectra[0].uncertainty[1])
    np.testing.assert_array_equal(
        rebinned.spectra[0].invalid_uncertainty_mask,
        [False, True, False],
    )


def test_mantid_style_y_e_f_and_partial_exclusion_use_requested_equations() -> None:
    source = make_rebin_dataset(
        edges=np.array([0.0, 1.0, 2.0]),
        coverage_values=(
            np.array([1.0, 0.5, 0.25]),
            np.array([0.5, 1.5, 2.0]),
        ),
    )
    spec = specification(
        (0.0, 2.0),
        (QExclusionInterval(0.0, 0.5),),
    )

    rebinned = rebin_fractional_q(source, spec)

    assert rebinned.fractional_coverage is not None
    expected_f = 0.5 * np.array([1.0, 0.5, 0.25]) + np.array([0.5, 1.5, 2.0])
    expected_y = (
        10.0 * np.array([1.0, 0.5, 0.25]) * 0.5 + 20.0 * np.array([0.5, 1.5, 2.0])
    ) / expected_f
    expected_e = (
        np.sqrt(
            np.square(1.0 * np.array([1.0, 0.5, 0.25])) * 0.5
            + np.square(2.0 * np.array([0.5, 1.5, 2.0]))
        )
        / expected_f
    )
    np.testing.assert_allclose(rebinned.fractional_coverage.values[0], expected_f)
    np.testing.assert_allclose(rebinned.spectra[0].intensity, expected_y)
    np.testing.assert_allclose(rebinned.spectra[0].uncertainty, expected_e)


def test_partial_source_target_overlap_uses_source_width_fraction() -> None:
    source = make_rebin_dataset()

    rebinned = rebin_fractional_q(source, specification((0.0, 1.5, 4.0)))

    assert rebinned.fractional_coverage is not None
    np.testing.assert_allclose(rebinned.fractional_coverage.values[0], 1.5)
    np.testing.assert_allclose(rebinned.spectra[0].intensity, (10.0 + 0.5 * 20.0) / 1.5)


def test_explicit_nonuniform_source_edges_are_used_without_inference() -> None:
    source = make_rebin_dataset(edges=np.array([0.0, 0.5, 2.0, 4.0]))

    rebinned = rebin_fractional_q(source, specification((0.0, 2.0, 4.0)))

    assert rebinned.fractional_coverage is not None
    np.testing.assert_allclose(rebinned.fractional_coverage.values[0], 2.0)
    np.testing.assert_allclose(rebinned.spectra[0].intensity, 15.0)


def test_overlapping_exclusions_are_unioned_without_double_removal() -> None:
    source = make_rebin_dataset(edges=np.array([0.0, 1.0, 2.0]))
    spec = specification(
        (0.0, 2.0),
        (
            QExclusionInterval(0.4, 0.8),
            QExclusionInterval(0.6, 1.2),
        ),
    )

    rebinned = rebin_fractional_q(source, spec)

    assert rebinned.fractional_coverage is not None
    np.testing.assert_allclose(rebinned.fractional_coverage.values[0], 1.2)


def test_exclusion_that_empties_target_is_structured_failure() -> None:
    source = make_rebin_dataset(edges=np.array([0.0, 1.0, 2.0]))

    with pytest.raises(QRebinError) as caught:
        rebin_fractional_q(
            source,
            specification(
                (0.0, 2.0),
                (QExclusionInterval(0.0, 2.0),),
            ),
        )

    assert diagnostic_code(caught.value) is QRebinDiagnosticCode.EMPTY_TARGET_OVERLAP


def test_rebin_requires_fractional_coverage_and_source_edges() -> None:
    source = make_rebin_dataset()
    without_coverage = replace(source, fractional_coverage=None)
    without_edges = replace(source, q_bins=QBins.from_q_values([0.5, 1.5, 2.5, 3.5]))

    with pytest.raises(QRebinError) as missing:
        rebin_fractional_q(without_coverage, specification((0.0, 2.0, 4.0)))
    with pytest.raises(QRebinError) as no_edges:
        rebin_fractional_q(without_edges, specification((0.0, 2.0, 4.0)))

    assert diagnostic_code(missing.value) is (
        QRebinDiagnosticCode.FRACTIONAL_COVERAGE_MISSING
    )
    assert diagnostic_code(no_edges.value) is (
        QRebinDiagnosticCode.SOURCE_Q_EDGES_REQUIRED
    )


def test_target_must_be_inside_source_and_must_not_refine() -> None:
    source = make_rebin_dataset(edges=np.array([0.0, 1.0, 2.0]))

    with pytest.raises(QRebinError) as outside:
        rebin_fractional_q(source, specification((-0.1, 2.0)))
    with pytest.raises(QRebinError) as refined:
        rebin_fractional_q(source, specification((0.0, 0.5, 2.0)))

    assert diagnostic_code(outside.value) is (
        QRebinDiagnosticCode.TARGET_OUTSIDE_SOURCE_COVERAGE
    )
    assert diagnostic_code(refined.value) is QRebinDiagnosticCode.TARGET_REFINES_SOURCE


def test_100_to_20_and_100_to_30_are_allowed_but_20_to_30_is_blocked() -> None:
    source = make_rebin_dataset(edges=np.linspace(0.0, 1.0, 101))

    twenty = rebin_fractional_q(
        source,
        specification(tuple(np.linspace(0.0, 1.0, 21))),
    )
    thirty = rebin_fractional_q(
        source,
        specification(tuple(np.linspace(0.0, 1.0, 31))),
    )
    with pytest.raises(QRebinError) as refined:
        rebin_fractional_q(
            twenty,
            specification(tuple(np.linspace(0.0, 1.0, 31))),
        )

    assert len(twenty.spectra) == 20
    assert len(thirty.spectra) == 30
    assert diagnostic_code(refined.value) is QRebinDiagnosticCode.TARGET_REFINES_SOURCE


def test_contributors_require_exact_energy_correspondence() -> None:
    source = make_rebin_dataset(
        edges=np.array([0.0, 1.0, 2.0]),
        energies=(
            np.array([-1.0, 0.0, 1.0]),
            np.array([-1.0, 0.1, 1.0]),
        ),
    )

    with pytest.raises(QRebinError) as caught:
        rebin_fractional_q(source, specification((0.0, 2.0)))

    assert diagnostic_code(caught.value) is (
        QRebinDiagnosticCode.INCOMPATIBLE_ENERGY_GRIDS
    )


def test_unequal_energy_grids_are_allowed_across_independent_targets() -> None:
    short = np.array([-1.0, 0.0, 1.0])
    long = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    source = make_rebin_dataset(
        energies=(short, short, long, long),
    )

    rebinned = rebin_fractional_q(source, specification((0.0, 2.0, 4.0)))

    np.testing.assert_array_equal(rebinned.spectra[0].energy, short)
    np.testing.assert_array_equal(rebinned.spectra[1].energy, long)


def test_output_point_without_usable_fractional_weight_is_blocked() -> None:
    source = make_rebin_dataset(
        edges=np.array([0.0, 1.0, 2.0]),
        coverage_values=(
            np.array([1.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 1.0]),
        ),
    )

    with pytest.raises(QRebinError) as caught:
        rebin_fractional_q(source, specification((0.0, 2.0)))

    assert diagnostic_code(caught.value) is (
        QRebinDiagnosticCode.OUTPUT_COVERAGE_MISSING
    )


def test_successive_coarsening_matches_direct_coarsening_and_preserves_history() -> (
    None
):
    source = make_rebin_dataset()

    intermediate = rebin_fractional_q(source, specification((0.0, 2.0, 4.0)))
    successive = rebin_fractional_q(intermediate, specification((0.0, 4.0)))
    direct = rebin_fractional_q(source, specification((0.0, 4.0)))

    np.testing.assert_allclose(
        successive.spectra[0].intensity,
        direct.spectra[0].intensity,
    )
    np.testing.assert_allclose(
        successive.spectra[0].uncertainty,
        direct.spectra[0].uncertainty,
    )
    assert successive.fractional_coverage is not None
    assert direct.fractional_coverage is not None
    np.testing.assert_allclose(
        successive.fractional_coverage.values[0],
        direct.fractional_coverage.values[0],
    )
    assert len(successive.fractional_coverage.q_rebin_history) == 2
    assert len(direct.fractional_coverage.q_rebin_history) == 1


def test_rebin_does_not_mutate_source_dataset_or_arrays() -> None:
    source = make_rebin_dataset()
    original_energy = tuple(item.energy.copy() for item in source.spectra)
    original_intensity = tuple(item.intensity.copy() for item in source.spectra)
    original_uncertainty = tuple(item.uncertainty.copy() for item in source.spectra)

    rebinned = rebin_fractional_q(source, specification((0.0, 2.0, 4.0)))

    assert rebinned is not source
    assert len(source.spectra) == 4
    for index, spectrum in enumerate(source.spectra):
        np.testing.assert_array_equal(spectrum.energy, original_energy[index])
        np.testing.assert_array_equal(spectrum.intensity, original_intensity[index])
        np.testing.assert_array_equal(spectrum.uncertainty, original_uncertainty[index])


def test_project_rebins_associated_sample_and_resolution_transactionally() -> None:
    sample_data = make_rebin_dataset(SpectrumRole.SAMPLE)
    resolution_data = make_rebin_dataset(
        SpectrumRole.RESOLUTION,
        energies=tuple(np.array([-0.5, 0.0, 0.5]) for _ in range(4)),
    )
    project = create_project("paired", project_id="paired-project")
    project, sample = add_project_dataset(project, sample_data, dataset_id="sample")
    project, resolution = add_project_dataset(
        project,
        resolution_data,
        dataset_id="resolution",
    )
    project = apply_resolution(project, sample, resolution)

    project, rebinned_sample, rebinned_resolution = rebin_project_sample_q(
        project,
        sample,
        specification((0.0, 2.0, 4.0)),
    )

    assert rebinned_resolution is not None
    assert rebinned_sample.dataset.q_bins is not None
    assert rebinned_resolution.dataset.q_bins is not None
    np.testing.assert_array_equal(rebinned_sample.dataset.q_bins.edges, [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(
        rebinned_resolution.dataset.q_bins.edges,
        [0.0, 2.0, 4.0],
    )
    assert applied_resolution(project, rebinned_sample) is rebinned_resolution
    assert len(sample_data.spectra) == 4
    assert len(resolution_data.spectra) == 4


def test_paired_rebin_failure_leaves_original_project_unchanged() -> None:
    sample_data = make_rebin_dataset(SpectrumRole.SAMPLE)
    resolution_data = make_rebin_dataset(
        SpectrumRole.RESOLUTION,
        energies=(
            np.array([-1.0, 0.0, 1.0]),
            np.array([-1.0, 0.1, 1.0]),
            np.array([-1.0, 0.0, 1.0]),
            np.array([-1.0, 0.0, 1.0]),
        ),
    )
    project = create_project("rollback", project_id="rollback-project")
    project, sample = add_project_dataset(project, sample_data, dataset_id="sample")
    project, resolution = add_project_dataset(
        project,
        resolution_data,
        dataset_id="resolution",
    )
    project = apply_resolution(project, sample, resolution)

    with pytest.raises(WorkflowError) as caught:
        rebin_project_sample_q(
            project,
            sample,
            specification((0.0, 2.0, 4.0)),
        )

    assert caught.value.diagnostics[0].code is WorkflowDiagnosticCode.Q_REBIN_FAILED
    assert project_dataset(project, sample).dataset is sample_data
    assert project_dataset(project, resolution).dataset is resolution_data


def test_paired_rebin_branches_resolution_shared_with_another_sample() -> None:
    first_sample_data = make_rebin_dataset(SpectrumRole.SAMPLE)
    second_sample_data = make_rebin_dataset(SpectrumRole.SAMPLE)
    resolution_data = make_rebin_dataset(SpectrumRole.RESOLUTION)
    project = create_project("shared", project_id="shared-project")
    project, first_sample = add_project_dataset(
        project,
        first_sample_data,
        dataset_id="first-sample",
    )
    project, second_sample = add_project_dataset(
        project,
        second_sample_data,
        dataset_id="second-sample",
    )
    project, resolution = add_project_dataset(
        project,
        resolution_data,
        dataset_id="resolution",
    )
    project = apply_resolution(project, first_sample, resolution)
    project = apply_resolution(project, second_sample, resolution)

    project, first_sample, rebinned_resolution = rebin_project_sample_q(
        project,
        first_sample,
        specification((0.0, 2.0, 4.0)),
    )

    assert rebinned_resolution is not None
    assert rebinned_resolution.dataset_id != resolution.dataset_id
    assert applied_resolution(project, first_sample) is rebinned_resolution
    assert applied_resolution(project, second_sample) is resolution
    assert project_dataset(project, resolution).dataset is resolution_data


def test_later_resolution_association_replays_sample_rebin_history() -> None:
    sample_data = make_rebin_dataset(SpectrumRole.SAMPLE)
    resolution_data = make_rebin_dataset(SpectrumRole.RESOLUTION)
    project = create_project("replay", project_id="replay-project")
    project, sample = add_project_dataset(project, sample_data, dataset_id="sample")
    project, sample, _ = rebin_project_sample_q(
        project,
        sample,
        specification((0.0, 2.0, 4.0)),
    )
    project, resolution = add_project_dataset(
        project,
        resolution_data,
        dataset_id="resolution",
    )

    project = apply_resolution(project, sample, resolution)

    replayed = applied_resolution(project, sample)
    assert replayed is not None
    assert replayed.dataset.fractional_coverage is not None
    assert replayed.dataset.q_bins is not None
    np.testing.assert_array_equal(replayed.dataset.q_bins.edges, [0.0, 2.0, 4.0])
    assert len(replayed.dataset.fractional_coverage.q_rebin_history) == 1
    assert len(resolution_data.spectra) == 4


def test_two_step_resolution_replay_failure_does_not_escape_partial_state() -> None:
    sample_data = make_rebin_dataset(SpectrumRole.SAMPLE)
    first = specification((0.0, 2.0, 4.0))
    second = specification((0.0, 4.0))
    project = create_project("two-step-replay", project_id="two-step-replay-project")
    project, sample = add_project_dataset(project, sample_data, dataset_id="sample")
    project, sample, _ = rebin_project_sample_q(project, sample, first)
    project, sample, _ = rebin_project_sample_q(project, sample, second)

    first_grid = np.array([-1.0, 0.0, 1.0])
    second_grid = np.array([-1.0, 0.25, 1.0])
    resolution_data = make_rebin_dataset(
        SpectrumRole.RESOLUTION,
        energies=(first_grid, first_grid, second_grid, second_grid),
    )
    first_replay = rebin_fractional_q(resolution_data, first)
    assert first_replay.fractional_coverage is not None
    assert first_replay.fractional_coverage.q_rebin_history == (first,)
    with pytest.raises(QRebinError) as second_failure:
        rebin_fractional_q(first_replay, second)
    assert diagnostic_code(second_failure.value) is (
        QRebinDiagnosticCode.INCOMPATIBLE_ENERGY_GRIDS
    )
    project, resolution = add_project_dataset(
        project,
        resolution_data,
        dataset_id="resolution",
    )

    with pytest.raises(WorkflowError) as caught:
        apply_resolution(project, sample, resolution)

    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.code is WorkflowDiagnosticCode.RESOLUTION_Q_REBIN_REPLAY_FAILED
    assert diagnostic.q_rebin_diagnostics[0].code is (
        QRebinDiagnosticCode.INCOMPATIBLE_ENERGY_GRIDS
    )
    assert applied_resolution(project, sample) is None
    stored_resolution = project_dataset(project, resolution)
    assert stored_resolution.dataset is resolution_data
    assert stored_resolution.dataset.fractional_coverage is not None
    assert stored_resolution.dataset.fractional_coverage.q_rebin_history == ()
    assert len(stored_resolution.dataset.spectra) == 4


def test_incompatible_later_resolution_replay_is_structured_and_not_attached() -> None:
    sample_data = make_rebin_dataset(SpectrumRole.SAMPLE)
    resolution_data = replace(
        make_rebin_dataset(SpectrumRole.RESOLUTION),
        fractional_coverage=None,
    )
    project = create_project("bad-replay", project_id="bad-replay-project")
    project, sample = add_project_dataset(project, sample_data, dataset_id="sample")
    project, sample, _ = rebin_project_sample_q(
        project,
        sample,
        specification((0.0, 2.0, 4.0)),
    )
    project, resolution = add_project_dataset(
        project,
        resolution_data,
        dataset_id="resolution",
    )

    with pytest.raises(WorkflowError) as caught:
        apply_resolution(project, sample, resolution)

    assert caught.value.diagnostics[0].code is (
        WorkflowDiagnosticCode.RESOLUTION_Q_REBIN_REPLAY_FAILED
    )
    assert caught.value.diagnostics[0].q_rebin_diagnostics[0].code is (
        QRebinDiagnosticCode.FRACTIONAL_COVERAGE_MISSING
    )
    assert applied_resolution(project, sample) is None
    assert project_dataset(project, resolution).dataset is resolution_data
