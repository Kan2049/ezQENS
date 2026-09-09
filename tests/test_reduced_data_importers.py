"""Tests for DAVE, wide-table, and single-spectrum import."""

from pathlib import Path

import numpy as np
import pytest

from ezqens.domain import (
    FractionalCoverageAvailability,
    FractionalCoverageOrigin,
    ImportValidationError,
    QBins,
    ReducedDataFormat,
    SpectrumRole,
)
from ezqens.io.importers import import_reduced_data

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


def test_mantid_xye_blocks_preserve_order_grids_values_and_missing_metadata() -> None:
    dataset = import_reduced_data(
        FIXTURES / "mantid_xye_blocks.txt",
        role=SpectrumRole.SAMPLE,
    )

    assert dataset.source_layout is ReducedDataFormat.MANTID_XYE_BLOCKS
    assert tuple(spectrum.group_label for spectrum in dataset.spectra) == (
        "1",
        "2",
        "3",
    )
    assert tuple(spectrum.energy.size for spectrum in dataset.spectra) == (2, 3, 2)
    assert not dataset.shared_energy_grid
    np.testing.assert_array_equal(dataset.spectra[0].energy, [-1.0, 0.0])
    np.testing.assert_array_equal(dataset.spectra[0].intensity, [2.0, 0.0])
    np.testing.assert_array_equal(dataset.spectra[0].uncertainty, [0.1, 0.0])
    np.testing.assert_array_equal(dataset.spectra[1].energy, [-0.8, 0.1, 0.9])
    np.testing.assert_array_equal(dataset.spectra[2].energy, [-0.5, 0.5])
    assert dataset.source_columns[0].source_row_numbers == (3, 4)
    assert dataset.source_columns[1].source_row_numbers == (7, 8, 9)
    assert dataset.source_columns[2].source_row_numbers == (12, 13)
    assert all(spectrum.energy_unit == "unknown" for spectrum in dataset.spectra)
    assert all(spectrum.intensity_unit == "unknown" for spectrum in dataset.spectra)
    assert all(spectrum.uncertainty_unit == "unknown" for spectrum in dataset.spectra)
    assert dataset.q_bins is None
    assert dataset.fractional_coverage is None
    assert (
        dataset.fractional_coverage_availability
        is FractionalCoverageAvailability.MISSING
    )
    assert dataset.spectra[0].intensity[1] == 0.0
    assert dataset.spectra[0].uncertainty[1] == 0.0
    assert dataset.spectra[0].invalid_uncertainty_mask[1]
    codes = {diagnostic.code for diagnostic in dataset.diagnostics}
    assert "mantid_energy_unit_missing" in codes
    assert "mantid_q_assignment_missing" in codes


def test_mantid_xye_import_accepts_manual_q_assignment() -> None:
    imported = import_reduced_data(
        FIXTURES / "mantid_xye_blocks.txt",
        role="resolution",
        energy_unit="meV",
    )
    q_bins = QBins.from_q_values([0.45, 0.65, 0.85])

    assigned = imported.assign_q_bins(q_bins)

    assert imported.q_bins is None
    assert assigned.q_bins is q_bins
    assert "mantid_energy_unit_missing" not in {
        diagnostic.code for diagnostic in assigned.diagnostics
    }


def test_headerless_single_xye_block_imports_with_explicit_format(
    tmp_path: Path,
) -> None:
    source = tmp_path / "one-block.dat"
    source.write_text("-1.0 2.0 0.1\n0.0 3.0 0.2\n", encoding="utf-8")

    dataset = import_reduced_data(
        source,
        role="sample",
        explicit_format=ReducedDataFormat.MANTID_XYE_BLOCKS,
    )

    assert len(dataset.spectra) == 1
    np.testing.assert_array_equal(dataset.spectra[0].energy, [-1.0, 0.0])


def test_unrebinned_confirmation_initializes_aligned_read_only_coverage() -> None:
    imported = import_reduced_data(
        FIXTURES / "mantid_xye_blocks.txt",
        role="sample",
    )
    original_arrays = tuple(
        (spectrum.energy, spectrum.intensity, spectrum.uncertainty)
        for spectrum in imported.spectra
    )

    confirmed = imported.confirm_unrebinned_source()

    assert imported.fractional_coverage is None
    assert confirmed.fractional_coverage is not None
    assert (
        confirmed.fractional_coverage.origin
        is FractionalCoverageOrigin.CONFIRMED_UNREBINNED_SOURCE
    )
    assert (
        confirmed.fractional_coverage_availability
        is FractionalCoverageAvailability.AVAILABLE
    )
    for coverage, spectrum in zip(
        confirmed.fractional_coverage.values,
        confirmed.spectra,
        strict=True,
    ):
        np.testing.assert_array_equal(coverage, np.ones(spectrum.energy.size))
        assert not coverage.flags.writeable
    for spectrum, arrays in zip(imported.spectra, original_arrays, strict=True):
        assert spectrum.energy is arrays[0]
        assert spectrum.intensity is arrays[1]
        assert spectrum.uncertainty is arrays[2]


def test_unrebinned_confirmation_does_not_reset_existing_coverage() -> None:
    imported = import_reduced_data(
        FIXTURES / "mantid_xye_blocks.txt",
        role="sample",
    ).confirm_unrebinned_source()

    with pytest.raises(ValueError, match="must not be reset"):
        imported.confirm_unrebinned_source()


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


def test_rich_dave_metadata_units_q_and_unequal_groups_are_preserved() -> None:
    source = FIXTURES / "dave_rich_metadata.dat"
    dataset = import_reduced_data(source, role=SpectrumRole.SAMPLE)

    assert dataset.source_layout is ReducedDataFormat.DAVE_GROUP_BLOCKS
    assert tuple(spectrum.group_label for spectrum in dataset.spectra) == ("1", "2")
    assert tuple(spectrum.energy.size for spectrum in dataset.spectra) == (2, 3)
    assert not dataset.shared_energy_grid
    assert all(spectrum.energy_unit == "meV" for spectrum in dataset.spectra)
    assert all(
        spectrum.intensity_unit == "arbitrary units"
        and spectrum.uncertainty_unit == "arbitrary units"
        for spectrum in dataset.spectra
    )
    assert dataset.q_bins is not None
    np.testing.assert_array_equal(dataset.q_bins.q_values, [0.575, 0.825])
    assert dataset.q_bins.edges is None

    metadata = dataset.source_metadata
    assert metadata is not None
    assert metadata.source_filename == source.name
    assert metadata.instrument == "SYNTHETIC-SPECTROMETER"
    assert metadata.sample == "public synthetic sample"
    assert metadata.title == "Rich DAVE metadata fixture"
    assert metadata.temperature_kelvin == pytest.approx(301.193)
    assert metadata.wavelength_angstrom == pytest.approx(4.060545)
    expected_header = tuple(
        source.read_text(encoding="utf-8").split("#Begin", maxsplit=1)[0].splitlines()
    )
    assert metadata.raw_header_lines == expected_header
    assert "#Group Type: Points" in metadata.raw_header_lines
    assert metadata.raw_header_lines[-3:] == (
        "#Comment: first repeated-key value",
        "#Comment: second repeated-key value",
        "# free-form header text retained exactly",
    )
    assert dataset.source_columns[0].energy == "X Value"
    assert dataset.source_columns[0].intensity == "Intensity"
    assert dataset.source_columns[0].uncertainty == "dIntensity"
    assert "dave_number_channels_reported" in {
        diagnostic.code for diagnostic in dataset.diagnostics
    }


def test_conflicting_explicit_dave_energy_units_fail(tmp_path: Path) -> None:
    source = tmp_path / "conflicting-units.dat"
    source.write_text(
        "#X Units: Energy / meV\n"
        "#Energy Units = eV\n"
        "#Begin\n"
        "#Group Number: 1\n"
        "#X Value Intensity dIntensity\n"
        "0.0 1.0 0.1\n",
        encoding="utf-8",
    )

    with pytest.raises(ImportValidationError) as caught:
        import_reduced_data(source, role="sample")

    assert "dave_energy_unit_conflict" in diagnostic_codes(caught.value)


def test_incomplete_dave_q_metadata_warns_without_inventing_q(tmp_path: Path) -> None:
    source = tmp_path / "incomplete-q.dat"
    source.write_text(
        "#Group Label: Q\n"
        "#Group Units: wavevector:A-1\n"
        "#Number of Groups: 2\n"
        "#Begin\n"
        "#Group Number: 1\n"
        "#Group Value: 0.5\n"
        "#X Value Intensity dIntensity\n"
        "0.0 1.0 0.1\n"
        "#Group Number: 2\n"
        "#Group Value: not-a-number\n"
        "#X Value Intensity dIntensity\n"
        "0.0 2.0 0.2\n",
        encoding="utf-8",
    )

    dataset = import_reduced_data(source, role="sample")

    assert len(dataset.spectra) == 2
    assert dataset.q_bins is None
    codes = {diagnostic.code for diagnostic in dataset.diagnostics}
    assert "dave_q_group_value_invalid" in codes
    assert "dave_q_values_incomplete" in codes


def test_dave_reported_group_count_mismatch_is_warning_only(tmp_path: Path) -> None:
    source = tmp_path / "group-count.dat"
    source.write_text(
        "#Number of Groups: 3\n"
        "#Begin\n"
        "#Group Number: 1\n"
        "#X Value Intensity dIntensity\n"
        "0.0 1.0 0.1\n"
        "#Group Number: 2\n"
        "#X Value Intensity dIntensity\n"
        "0.0 2.0 0.2\n",
        encoding="utf-8",
    )

    dataset = import_reduced_data(source, role="resolution")

    assert len(dataset.spectra) == 2
    assert "dave_group_count_mismatch" in {
        diagnostic.code for diagnostic in dataset.diagnostics
    }


def test_dave_y_units_are_authoritative_for_intensity_and_uncertainty(
    tmp_path: Path,
) -> None:
    source = tmp_path / "authoritative-y-units.dat"
    source.write_text(
        "#X Units: Energy / meV\n"
        "#Y Units: counts\n"
        "#Begin\n"
        "#Group Number: 1\n"
        "#X Value Intensity dIntensity\n"
        "0.0 3.0 0.2\n",
        encoding="utf-8",
    )

    dataset = import_reduced_data(
        source,
        role="sample",
        intensity_unit="arb. unit",
        uncertainty_unit="arb. unit",
    )

    spectrum = dataset.spectra[0]
    assert spectrum.intensity_unit == "counts"
    assert spectrum.uncertainty_unit == "counts"


@pytest.mark.parametrize(
    ("declarations", "expected_code"),
    [
        (
            "#X Units: Energy / meV\n#Energy Units = unsupported\n",
            "dave_energy_unit_unrecognized",
        ),
        (
            "#X Units: Energy / meV\n#Energy Units = eV\n",
            "dave_energy_unit_conflict",
        ),
        (
            "#Energy Units = unsupported\n",
            "dave_energy_unit_unrecognized",
        ),
    ],
)
def test_invalid_explicit_dave_energy_declarations_fail_without_fallback(
    tmp_path: Path,
    declarations: str,
    expected_code: str,
) -> None:
    source = tmp_path / "invalid-energy-unit.dat"
    source.write_text(
        declarations + "#Begin\n"
        "#Group Number: 1\n"
        "#X Value Intensity dIntensity\n"
        "0.0 1.0 0.1\n",
        encoding="utf-8",
    )

    with pytest.raises(ImportValidationError) as caught:
        import_reduced_data(source, role="sample", energy_unit="meV")

    assert expected_code in diagnostic_codes(caught.value)


def test_authoritative_dave_energy_unit_must_agree_with_explicit_caller(
    tmp_path: Path,
) -> None:
    source = tmp_path / "caller-unit-conflict.dat"
    source.write_text(
        "#X Units: Energy / meV\n"
        "#Begin\n"
        "#Group Number: 1\n"
        "#X Value Intensity dIntensity\n"
        "0.0 1.0 0.1\n",
        encoding="utf-8",
    )

    with pytest.raises(ImportValidationError) as caught:
        import_reduced_data(source, role="sample", energy_unit="eV")

    assert "dave_energy_unit_caller_conflict" in diagnostic_codes(caught.value)


@pytest.mark.parametrize(
    ("extra_header", "extra_values", "expected_extras"),
    [
        ("ModelFit", "9.0", ("ModelFit",)),
        ("ModelFit Func", "9.0 8.0", ("ModelFit", "Func")),
    ],
)
def test_formal_dave_header_preserves_extra_fit_columns_without_importing_them(
    tmp_path: Path,
    extra_header: str,
    extra_values: str,
    expected_extras: tuple[str, ...],
) -> None:
    source = tmp_path / "formal-extra-columns.dat"
    source.write_text(
        "#Begin\n"
        "#Group Number: 1\n"
        f"#X Value Intensity dIntensity {extra_header}\n"
        f"-1.0 2.0 0.1 {extra_values}\n"
        f"0.0 3.0 0.2 {extra_values}\n",
        encoding="utf-8",
    )

    dataset = import_reduced_data(source, role="sample")

    spectrum = dataset.spectra[0]
    np.testing.assert_array_equal(spectrum.energy, [-1.0, 0.0])
    np.testing.assert_array_equal(spectrum.intensity, [2.0, 3.0])
    np.testing.assert_array_equal(spectrum.uncertainty, [0.1, 0.2])
    assert dataset.source_columns[0].extra_columns == expected_extras
    assert dataset.detected_extra_columns == expected_extras


def test_invalid_optional_dave_metadata_warns_and_remains_raw(
    tmp_path: Path,
) -> None:
    source = tmp_path / "invalid-optional-metadata.dat"
    source.write_text(
        "#Temperature / K: -1\n"
        "#Wavelength / A: 0\n"
        "#Begin\n"
        "#Group Number: 1\n"
        "#X Value Intensity dIntensity\n"
        "0.0 1.0 0.1\n",
        encoding="utf-8",
    )

    dataset = import_reduced_data(source, role="sample")

    metadata = dataset.source_metadata
    assert metadata is not None
    assert metadata.temperature_kelvin is None
    assert metadata.wavelength_angstrom is None
    assert "#Temperature / K: -1" in metadata.raw_header_lines
    assert "#Wavelength / A: 0" in metadata.raw_header_lines
    codes = {diagnostic.code for diagnostic in dataset.diagnostics}
    assert "dave_temperature_invalid" in codes
    assert "dave_wavelength_invalid" in codes
