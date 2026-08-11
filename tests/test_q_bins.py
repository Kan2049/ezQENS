"""Behavior tests for dataset-level Q bins and DAVE parameters."""

from pathlib import Path

import numpy as np
import pytest

from qensfit.domain import (
    ImportValidationError,
    QBins,
    ReducedDataset,
    Spectrum,
    SpectrumRole,
    uniform_q_bins,
)
from qensfit.io import parse_dave_q_bins

FIXTURES = Path(__file__).parent / "fixtures" / "q_bins"


def make_dataset(group_count: int = 2) -> ReducedDataset:
    spectra = tuple(
        Spectrum(
            role=SpectrumRole.SAMPLE,
            group_index=index,
            group_label=f"group-{index}",
            energy=np.array([-1.0, 0.0, 1.0]),
            intensity=np.array([1.0, 2.0, 1.0]),
            uncertainty=np.array([0.1, 0.1, 0.1]),
            energy_unit="meV",
            intensity_unit="counts",
            uncertainty_unit="counts",
        )
        for index in range(group_count)
    )
    return ReducedDataset(role=SpectrumRole.SAMPLE, spectra=spectra)


def test_nonuniform_edges_are_accepted_and_midpoints_are_derived() -> None:
    q_bins = QBins.from_edges([0.2, 0.5, 1.1, 2.0])

    np.testing.assert_array_equal(q_bins.edges, [0.2, 0.5, 1.1, 2.0])
    np.testing.assert_allclose(q_bins.q_values, [0.35, 0.8, 1.55])
    assert q_bins.group_count == 3
    assert q_bins.unit == "Å^-1"


@pytest.mark.parametrize(
    ("edges", "message"),
    [
        ([0.2], "at least two"),
        ([0.2, np.nan], "finite"),
        ([0.2, np.inf], "finite"),
        ([0.2, 0.2, 0.5], "strictly increasing"),
        ([0.5, 0.2], "strictly increasing"),
    ],
)
def test_invalid_edges_fail_clearly(edges: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        QBins.from_edges(edges)


def test_direct_edge_and_representative_lengths_must_match() -> None:
    with pytest.raises(ValueError, match="one more"):
        QBins(q_values=np.array([0.4, 0.8]), edges=np.array([0.2, 0.6]))


def test_uniform_edges_use_outer_boundaries_and_authoritative_group_count() -> None:
    q_bins = uniform_q_bins(
        lower_q_edge=0.4,
        upper_q_edge=2.5,
        group_count=14,
    )

    assert q_bins.edges is not None
    assert q_bins.edges.size == 15
    assert q_bins.edges[0] == 0.4
    assert q_bins.edges[-1] == 2.5
    np.testing.assert_allclose(np.diff(q_bins.edges), 0.15)
    assert q_bins.q_values[0] == pytest.approx(0.475)
    assert q_bins.q_values[0] != q_bins.edges[0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lower_q_edge": np.nan, "upper_q_edge": 2.0, "group_count": 2},
        {"lower_q_edge": 2.0, "upper_q_edge": 1.0, "group_count": 2},
        {"lower_q_edge": 0.0, "upper_q_edge": 1.0, "group_count": 0},
        {"lower_q_edge": 0.0, "upper_q_edge": 1.0, "group_count": 2.5},
    ],
)
def test_uniform_q_bin_definition_is_validated(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        uniform_q_bins(**kwargs)  # type: ignore[arg-type]


def test_explicit_q_values_preserve_nonlinear_order_without_edges() -> None:
    supplied = [0.8, 0.3, 0.3, 1.7]

    q_bins = QBins.from_q_values(supplied)

    np.testing.assert_array_equal(q_bins.q_values, supplied)
    assert q_bins.edges is None


def test_known_edges_do_not_permanently_require_midpoint_representatives() -> None:
    q_bins = QBins(
        q_values=np.array([0.36, 0.92]),
        edges=np.array([0.2, 0.5, 1.2]),
    )

    np.testing.assert_array_equal(q_bins.q_values, [0.36, 0.92])


@pytest.mark.parametrize("values", [[], [0.2, np.nan], [0.2, np.inf]])
def test_explicit_q_values_must_be_nonempty_and_finite(values: list[float]) -> None:
    with pytest.raises(ValueError):
        QBins.from_q_values(values)


def test_q_assignment_is_dataset_level_and_count_checked() -> None:
    dataset = make_dataset(2)
    assigned = dataset.assign_q_bins(QBins.from_q_values([0.4, 0.9]))

    assert assigned.q_bins is not None
    np.testing.assert_array_equal(assigned.q_bins.q_values, [0.4, 0.9])
    assert dataset.q_bins is None
    assert all(not hasattr(spectrum, "q") for spectrum in assigned.spectra)
    assert all(not hasattr(spectrum, "q_value") for spectrum in assigned.spectra)
    with pytest.raises(ValueError, match="Q-bin count"):
        dataset.assign_q_bins(QBins.from_q_values([0.4]))


def test_q_bin_arrays_are_read_only() -> None:
    q_bins = QBins.from_edges([0.2, 0.4, 0.8])

    assert q_bins.edges is not None
    for array in (q_bins.edges, q_bins.q_values):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = 99.0


def test_dave_step_reconstructs_complete_bins_with_unused_upper_remainder() -> None:
    result = parse_dave_q_bins(FIXTURES / "dave_valid.txt")

    assert result.q_bins.edges is not None
    assert result.q_bins.group_count == 14
    np.testing.assert_allclose(result.q_bins.edges, 1.0 + 0.25 * np.arange(15))
    np.testing.assert_allclose(
        result.q_bins.q_values,
        1.125 + 0.25 * np.arange(14),
    )
    assert result.q_bins.edges[-1] == 4.5
    assert result.lower_limit == 1.0
    assert result.upper_limit == 4.6
    assert result.step == 0.25
    assert result.reported_group_count == 14
    assert result.diagnostics == ()


def test_dave_exact_fit_retains_last_complete_bin() -> None:
    result = parse_dave_q_bins(FIXTURES / "dave_exact_fit.txt")

    assert result.q_bins.edges is not None
    np.testing.assert_allclose(result.q_bins.edges, 1.0 + 0.25 * np.arange(15))
    assert result.q_bins.group_count == 14
    assert result.q_bins.edges[-1] == result.upper_limit == 4.5
    assert result.diagnostics == ()


def test_dave_reported_count_mismatch_warns_and_uses_reconstructed_bins() -> None:
    result = parse_dave_q_bins(FIXTURES / "dave_count_mismatch.txt")

    assert result.q_bins.edges is not None
    assert result.q_bins.group_count == 14
    assert result.reported_group_count == 13
    assert result.q_bins.edges[-1] == 4.5
    assert [item.code for item in result.diagnostics] == [
        "dave_q_bins_group_count_mismatch"
    ]


def test_count_driven_uniform_bins_and_step_driven_dave_bins_remain_distinct() -> None:
    uniform = uniform_q_bins(
        lower_q_edge=1.0,
        upper_q_edge=4.6,
        group_count=14,
    )
    dave = parse_dave_q_bins(FIXTURES / "dave_valid.txt").q_bins

    assert uniform.edges is not None
    assert dave.edges is not None
    assert uniform.group_count == dave.group_count == 14
    assert uniform.edges[-1] == 4.6
    assert dave.edges[-1] == 4.5
    np.testing.assert_allclose(np.diff(uniform.edges), 3.6 / 14.0)
    np.testing.assert_allclose(np.diff(dave.edges), 0.25)


def test_dave_floating_point_exact_fit_keeps_final_bin() -> None:
    result = parse_dave_q_bins(FIXTURES / "dave_float_exact.txt")

    assert result.q_bins.edges is not None
    assert result.q_bins.group_count == 3
    np.testing.assert_allclose(result.q_bins.edges, [0.0, 0.1, 0.2, 0.3])
    assert result.q_bins.edges[-1] == pytest.approx(result.upper_limit)
    assert result.diagnostics == ()


def test_dave_incomplete_final_bin_is_excluded() -> None:
    result = parse_dave_q_bins(FIXTURES / "dave_incomplete.txt")

    assert result.q_bins.edges is not None
    assert result.q_bins.group_count == 3
    assert result.q_bins.edges[-1] == pytest.approx(0.3)
    assert result.q_bins.edges[-1] < result.upper_limit
    assert result.diagnostics == ()


def test_coordinate_scale_cannot_make_tolerance_admit_an_incomplete_bin() -> None:
    result = parse_dave_q_bins(FIXTURES / "dave_large_offset_incomplete.txt")

    assert result.q_bins.edges is not None
    assert result.q_bins.group_count == 14
    assert result.q_bins.edges[-1] == 1_000_000_000_000_014.0
    assert result.q_bins.edges[-1] < result.upper_limit
    assert result.diagnostics == ()


@pytest.mark.parametrize(
    ("filename", "code"),
    [
        ("dave_malformed.txt", "dave_q_bins_value_not_numeric"),
        ("dave_nonfinite.txt", "dave_q_bins_value_not_finite"),
        ("dave_invalid_count.txt", "dave_q_bins_group_count_invalid"),
        ("dave_wrong_value_count.txt", "dave_q_bins_expected_four_values"),
        ("dave_reversed_edges.txt", "dave_q_bins_edge_order_invalid"),
        ("dave_invalid_step.txt", "dave_q_bins_step_invalid"),
    ],
)
def test_invalid_dave_parameters_fail_with_diagnostics(
    filename: str, code: str
) -> None:
    with pytest.raises(ImportValidationError) as caught:
        parse_dave_q_bins(FIXTURES / filename)

    assert caught.value.diagnostics[0].code == code
