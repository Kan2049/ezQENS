"""Tests for content-based reduced-data format detection."""

from pathlib import Path

import pytest

from ezqens.domain import DiagnosticSeverity, ReducedDataFormat
from ezqens.io.importers import detect_reduced_data_format

FIXTURES = Path(__file__).parent / "fixtures" / "reduced_data"


@pytest.mark.parametrize(
    ("filename", "expected_format", "expected_count"),
    [
        ("dave_multiple_groups.dat", ReducedDataFormat.DAVE_GROUP_BLOCKS, 2),
        ("mantid_xye_blocks.txt", ReducedDataFormat.MANTID_XYE_BLOCKS, 3),
        ("wide_multiple_pairs.txt", ReducedDataFormat.WIDE_QENS_TABLE, 2),
        ("single_valid.csv", ReducedDataFormat.SINGLE_SPECTRUM_TABLE, 1),
    ],
)
def test_supported_formats_are_detected_from_content(
    filename: str,
    expected_format: ReducedDataFormat,
    expected_count: int,
) -> None:
    result = detect_reduced_data_format(FIXTURES / filename)

    assert result.proposed_format is expected_format
    assert result.detected_count == expected_count
    assert result.evidence
    assert not result.has_errors


def test_dave_detection_records_extra_columns() -> None:
    result = detect_reduced_data_format(FIXTURES / "dave_multiple_groups.dat")

    assert result.detected_extra_columns == ("ModelFit", "Func1", "Func2")
    assert result.detected_required_columns == ("x", "y", "yerr")


def test_mantid_xye_detection_reports_block_structure() -> None:
    result = detect_reduced_data_format(FIXTURES / "mantid_xye_blocks.txt")

    assert result.proposed_format is ReducedDataFormat.MANTID_XYE_BLOCKS
    assert result.detected_required_columns == ("X", "Y", "E")
    assert result.detected_extra_columns == ()
    assert result.detected_count == 3
    assert not result.has_errors


def test_headerless_blank_separated_xye_blocks_are_detected(tmp_path: Path) -> None:
    source = tmp_path / "headerless-blocks.dat"
    source.write_text(
        "-1.0 2.0 0.1\n0.0 3.0 0.2\n\n-0.5 4.0 0.3\n0.5 4.5 0.4\n",
        encoding="utf-8",
    )

    result = detect_reduced_data_format(source)

    assert result.proposed_format is ReducedDataFormat.MANTID_XYE_BLOCKS
    assert result.detected_count == 2
    assert not result.has_errors


@pytest.mark.parametrize(
    ("header", "expected_format"),
    [
        ("# x y yerr", ReducedDataFormat.SINGLE_SPECTRUM_TABLE),
        ("# x y1 yerr1", ReducedDataFormat.WIDE_QENS_TABLE),
    ],
)
def test_recognized_table_headers_precede_headerless_mantid_heuristic(
    tmp_path: Path,
    header: str,
    expected_format: ReducedDataFormat,
) -> None:
    source = tmp_path / "recognized-with-blank-lines.txt"
    source.write_text(
        f"{header}\n-1.0 2.0 0.1\n\n0.0 3.0 0.2\n",
        encoding="utf-8",
    )

    result = detect_reduced_data_format(source)

    assert result.proposed_format is expected_format
    assert not result.has_errors


def test_single_and_one_pair_wide_tables_are_distinct() -> None:
    single = detect_reduced_data_format(FIXTURES / "single_valid.csv")
    wide = detect_reduced_data_format(FIXTURES / "wide_one_pair.txt")

    assert single.proposed_format is ReducedDataFormat.SINGLE_SPECTRUM_TABLE
    assert wide.proposed_format is ReducedDataFormat.WIDE_QENS_TABLE
    assert single.evidence != wide.evidence


def test_explicit_override_resolves_ambiguous_content() -> None:
    automatic = detect_reduced_data_format(FIXTURES / "ambiguous_table.txt")
    overridden = detect_reduced_data_format(
        FIXTURES / "ambiguous_table.txt",
        explicit_format=ReducedDataFormat.SINGLE_SPECTRUM_TABLE,
    )

    assert automatic.proposed_format is ReducedDataFormat.AMBIGUOUS
    assert automatic.alternative_formats == (
        ReducedDataFormat.WIDE_QENS_TABLE,
        ReducedDataFormat.SINGLE_SPECTRUM_TABLE,
    )
    assert automatic.has_errors
    assert overridden.proposed_format is ReducedDataFormat.SINGLE_SPECTRUM_TABLE
    assert not overridden.has_errors


def test_inconsistent_explicit_override_does_not_fall_back() -> None:
    result = detect_reduced_data_format(
        FIXTURES / "wide_multiple_pairs.txt",
        explicit_format=ReducedDataFormat.SINGLE_SPECTRUM_TABLE,
    )

    assert result.proposed_format is ReducedDataFormat.SINGLE_SPECTRUM_TABLE
    assert result.has_errors
    assert "single_required_columns_missing" in {
        diagnostic.code for diagnostic in result.diagnostics
    }


def test_explicit_dave_override_requires_group_markers() -> None:
    result = detect_reduced_data_format(
        FIXTURES / "unknown_table.txt",
        explicit_format=ReducedDataFormat.DAVE_GROUP_BLOCKS,
    )

    assert result.proposed_format is ReducedDataFormat.DAVE_GROUP_BLOCKS
    assert result.has_errors
    assert result.diagnostics[0].code == "dave_group_markers_missing"


def test_extension_does_not_determine_classification(tmp_path: Path) -> None:
    content = (FIXTURES / "wide_one_pair.txt").read_text(encoding="utf-8")
    misleading = tmp_path / "looks_like_single.csv"
    extensionless = tmp_path / "no_extension"
    misleading.write_text(content, encoding="utf-8")
    extensionless.write_text(content, encoding="utf-8")

    with_extension = detect_reduced_data_format(misleading)
    without_extension = detect_reduced_data_format(extensionless)

    assert with_extension == without_extension
    assert with_extension.proposed_format is ReducedDataFormat.WIDE_QENS_TABLE
    assert not hasattr(with_extension, "extension_hint")


@pytest.mark.parametrize(
    "filename",
    ["unknown_table.txt", "single_missing_yerr.txt"],
)
def test_unknown_or_malformed_input_returns_diagnostics(filename: str) -> None:
    result = detect_reduced_data_format(FIXTURES / filename)

    assert result.proposed_format is ReducedDataFormat.UNKNOWN
    assert result.has_errors
    assert all(
        diagnostic.severity is DiagnosticSeverity.ERROR
        for diagnostic in result.diagnostics
    )


def test_malformed_wide_header_keeps_layout_and_reports_pair_error() -> None:
    result = detect_reduced_data_format(FIXTURES / "wide_missing_yerr.txt")

    assert result.proposed_format is ReducedDataFormat.WIDE_QENS_TABLE
    assert result.has_errors
    assert "wide_uncertainty_column_missing" in {
        diagnostic.code for diagnostic in result.diagnostics
    }


def test_unsupported_explicit_format_is_diagnostic() -> None:
    result = detect_reduced_data_format(
        FIXTURES / "single_valid.csv",
        explicit_format="generic_custom",
    )

    assert result.proposed_format is ReducedDataFormat.UNKNOWN
    assert result.has_errors
    assert result.diagnostics[0].code == "explicit_format_unsupported"


def test_detection_result_has_no_gui_confirmation_or_confidence_state() -> None:
    result = detect_reduced_data_format(FIXTURES / "single_valid.csv")

    assert not hasattr(result, "confidence")
    assert not hasattr(result, "requires_confirmation")
    assert not hasattr(result, "explicit_override")


def test_formal_rich_dave_group_syntax_is_detected() -> None:
    result = detect_reduced_data_format(FIXTURES / "dave_rich_metadata.dat")

    assert result.proposed_format is ReducedDataFormat.DAVE_GROUP_BLOCKS
    assert result.detected_count == 2
    assert result.detected_required_columns == ("x", "y", "yerr")
    assert not result.has_errors
