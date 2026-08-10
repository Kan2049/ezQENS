"""Tests for DAVE, wide-table, and single-spectrum import."""

from pathlib import Path

import numpy as np
import pytest

from qensfit.domain import (
    ImportValidationError,
    ReducedDataFormat,
    SpectrumRole,
)
from qensfit.io.importers import import_reduced_data

FIXTURES = Path(__file__).parent / "fixtures" / "reduced_data"


def diagnostic_codes(error: ImportValidationError) -> set[str]:
    """Return diagnostic codes without inspecting source data."""

    return {diagnostic.code for diagnostic in error.diagnostics}


def test_dave_one_group_preserves_values_and_order() -> None:
    dataset = import_reduced_data(
        FIXTURES / "dave_one_group.dat",
        role=SpectrumRole.SAMPLE,
        energy_unit="meV",
        intensity_unit="counts",
        uncertainty_unit="counts",
    )

    assert dataset.source_layout is ReducedDataFormat.DAVE_GROUP_BLOCKS
    assert dataset.source_reference == "dave_one_group.dat"
    assert len(dataset.spectra) == 1
    spectrum = dataset.spectra[0]
    assert spectrum.group_index == 0
    assert spectrum.group_label == "first"
    assert dataset.source_columns[0].source_row_numbers == (4, 5, 6)
    np.testing.assert_array_equal(spectrum.energy, [-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(spectrum.intensity, [2.0, 3.0, 2.5])
    np.testing.assert_array_equal(spectrum.uncertainty, [0.1, 0.15, 0.2])
    assert dataset.shared_energy_grid


def test_dave_multiple_groups_keep_unequal_grids_and_extra_columns() -> None:
    dataset = import_reduced_data(
        FIXTURES / "dave_multiple_groups.dat",
        role="sample",
    )

    assert tuple(spectrum.group_label for spectrum in dataset.spectra) == (
        "alpha",
        "beta",
    )
    assert tuple(len(spectrum.energy) for spectrum in dataset.spectra) == (2, 3)
    np.testing.assert_array_equal(dataset.spectra[0].energy, [-1.0, 0.0])
    np.testing.assert_array_equal(dataset.spectra[1].energy, [-2.0, 0.0, 2.0])
    assert not dataset.shared_energy_grid
    assert dataset.detected_extra_columns == ("ModelFit", "Func1", "Func2")
    assert dataset.source_columns[0].extra_columns == (
        "ModelFit",
        "Func1",
    )
    assert any(
        diagnostic.code == "extra_columns_ignored" for diagnostic in dataset.diagnostics
    )


@pytest.mark.parametrize(
    ("filename", "expected_code"),
    [
        ("dave_missing_yerr.dat", "dave_required_columns_missing"),
        ("dave_malformed_row.dat", "malformed_numeric_row"),
    ],
)
def test_dave_malformed_input_fails_with_diagnostics(
    filename: str,
    expected_code: str,
) -> None:
    with pytest.raises(ImportValidationError) as caught:
        import_reduced_data(FIXTURES / filename, role="sample")

    assert expected_code in diagnostic_codes(caught.value)
    assert "array(" not in str(caught.value)


def test_dave_invalid_values_are_preserved_and_flagged() -> None:
    dataset = import_reduced_data(
        FIXTURES / "dave_invalid_values.dat",
        role=SpectrumRole.RESOLUTION,
    )
    spectrum = dataset.spectra[0]

    assert spectrum.role is SpectrumRole.RESOLUTION
    assert np.isnan(spectrum.energy[0])
    assert np.isinf(spectrum.intensity[1])
    np.testing.assert_array_equal(
        spectrum.invalid_uncertainty_mask,
        [False, True, True, True],
    )
    assert "invalid_uncertainty_values" in {
        diagnostic.code for diagnostic in dataset.diagnostics
    }


def test_wide_one_pair_imports_one_independent_spectrum() -> None:
    dataset = import_reduced_data(FIXTURES / "wide_one_pair.txt", role="sample")

    assert dataset.source_layout is ReducedDataFormat.WIDE_QENS_TABLE
    assert len(dataset.spectra) == 1
    assert dataset.spectra[0].group_label == "1"
    assert dataset.shared_energy_grid
    assert not hasattr(dataset, "shared_energy_axis")
    assert not hasattr(dataset.spectra[0], "q")


def test_wide_multiple_pairs_share_energy_but_expose_spectra() -> None:
    dataset = import_reduced_data(
        FIXTURES / "wide_multiple_pairs.txt",
        role="sample",
    )

    assert len(dataset.spectra) == 2
    assert tuple(spectrum.group_label for spectrum in dataset.spectra) == ("1", "2")
    np.testing.assert_array_equal(
        dataset.spectra[0].energy,
        dataset.spectra[1].energy,
    )
    np.testing.assert_array_equal(dataset.spectra[0].intensity, [2.0, 3.0, 2.5])
    np.testing.assert_array_equal(dataset.spectra[1].intensity, [4.0, 5.0, 4.5])


def test_wide_reordered_nonsequential_pairs_use_suffix_order() -> None:
    dataset = import_reduced_data(
        FIXTURES / "wide_reordered_nonsequential.txt",
        role="sample",
    )

    assert tuple(spectrum.group_label for spectrum in dataset.spectra) == ("2", "05")
    np.testing.assert_array_equal(dataset.spectra[0].intensity, [2.0, 3.0, 2.5])
    np.testing.assert_array_equal(dataset.spectra[1].intensity, [5.0, 6.0, 5.5])
    assert dataset.source_columns[1].uncertainty == "yerr05"


@pytest.mark.parametrize(
    ("filename", "expected_code"),
    [
        ("wide_missing_yerr.txt", "wide_uncertainty_column_missing"),
        ("wide_missing_y.txt", "wide_intensity_column_missing"),
        ("wide_duplicate_suffix.txt", "wide_duplicate_intensity_suffix"),
        ("wide_inconsistent_width.txt", "inconsistent_row_width"),
    ],
)
def test_wide_malformed_input_fails_with_diagnostics(
    filename: str,
    expected_code: str,
) -> None:
    with pytest.raises(ImportValidationError) as caught:
        import_reduced_data(FIXTURES / filename, role="sample")

    assert expected_code in diagnostic_codes(caught.value)


def test_wide_invalid_uncertainties_are_independent_by_spectrum() -> None:
    dataset = import_reduced_data(
        FIXTURES / "wide_invalid_uncertainties.txt",
        role="sample",
    )

    np.testing.assert_array_equal(
        dataset.spectra[0].invalid_uncertainty_mask,
        [True, True, True],
    )
    np.testing.assert_array_equal(
        dataset.spectra[1].invalid_uncertainty_mask,
        [False, True, False],
    )
    assert np.isnan(dataset.spectra[0].uncertainty[0])
    assert np.isinf(dataset.spectra[1].uncertainty[1])


def test_single_spectrum_import_preserves_layout_metadata() -> None:
    dataset = import_reduced_data(
        FIXTURES / "single_valid.csv",
        role="resolution",
        energy_unit="meV",
    )

    assert dataset.role is SpectrumRole.RESOLUTION
    assert dataset.source_layout is ReducedDataFormat.SINGLE_SPECTRUM_TABLE
    assert len(dataset.spectra) == 1
    spectrum = dataset.spectra[0]
    assert dataset.source_columns[0].energy == "x"
    assert spectrum.energy_unit == "meV"
    assert dataset.source_columns[0].source_row_numbers == (2, 3, 4)
    assert not hasattr(spectrum, "q_value")


def test_single_invalid_uncertainty_is_preserved() -> None:
    dataset = import_reduced_data(
        FIXTURES / "single_invalid_uncertainty.txt",
        role="sample",
    )
    spectrum = dataset.spectra[0]

    assert np.isneginf(spectrum.uncertainty[1])
    assert spectrum.uncertainty[2] == 0.0
    np.testing.assert_array_equal(
        spectrum.invalid_uncertainty_mask,
        [False, True, True],
    )


def test_single_missing_required_column_fails() -> None:
    with pytest.raises(ImportValidationError) as caught:
        import_reduced_data(
            FIXTURES / "single_missing_yerr.txt",
            role="sample",
        )

    assert "single_required_columns_missing" in diagnostic_codes(caught.value)


def test_explicit_format_override_is_used_for_ambiguous_table() -> None:
    dataset = import_reduced_data(
        FIXTURES / "ambiguous_table.txt",
        role="sample",
        explicit_format=ReducedDataFormat.SINGLE_SPECTRUM_TABLE,
    )

    assert dataset.source_layout is ReducedDataFormat.SINGLE_SPECTRUM_TABLE
    assert len(dataset.spectra) == 1
    assert dataset.detected_extra_columns == ("y1", "yerr1")


def test_inconsistent_explicit_override_fails_without_fallback() -> None:
    with pytest.raises(ImportValidationError) as caught:
        import_reduced_data(
            FIXTURES / "wide_one_pair.txt",
            role="sample",
            explicit_format=ReducedDataFormat.SINGLE_SPECTRUM_TABLE,
        )

    assert "single_required_columns_missing" in diagnostic_codes(caught.value)


def test_privacy_safe_summary_contains_only_structural_information() -> None:
    dataset = import_reduced_data(
        FIXTURES / "dave_multiple_groups.dat",
        role="sample",
    )

    summary = dataset.structural_summary()
    representation = repr(summary)
    assert summary.spectrum_count == 2
    assert summary.row_counts == (2, 3)
    assert summary.finite_energy_ranges == ((-1.0, 0.0), (-2.0, 2.0))
    assert summary.shared_energy_grid is False
    assert "array(" not in representation
    assert "[4.0" not in representation
    assert "dave_multiple_groups.dat" not in representation
