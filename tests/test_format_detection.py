"""Tests for content-based reduced-data format detection."""

from pathlib import Path

import pytest

from qensfit.domain import (
    DetectionConfidence,
    DiagnosticSeverity,
    ReducedDataFormat,
)
from qensfit.io.importers import detect_reduced_data_format

FIXTURES = Path(__file__).parent / "fixtures" / "reduced_data"


@pytest.mark.parametrize(
    ("filename", "expected_format", "expected_count"),
    [
        ("dave_multiple_groups.dat", ReducedDataFormat.DAVE_GROUP_BLOCKS, 2),
        ("wide_multiple_pairs.txt", ReducedDataFormat.WIDE_QENS_TABLE, 2),
        ("single_valid.csv", ReducedDataFormat.SINGLE_SPECTRUM_TABLE, 1),
    ],
)
def test_supported_formats_are_detected_with_high_confidence(
    filename: str,
    expected_format: ReducedDataFormat,
    expected_count: int,
) -> None:
    result = detect_reduced_data_format(FIXTURES / filename)

    assert result.proposed_format is expected_format
    assert result.confidence is DetectionConfidence.HIGH
    assert result.detected_count == expected_count
    assert result.requires_confirmation
    assert not result.has_errors


def test_dave_detection_records_extra_columns() -> None:
    result = detect_reduced_data_format(FIXTURES / "dave_multiple_groups.dat")

    assert result.detected_extra_columns == ("ModelFit", "Func1", "Func2")
    assert result.detected_required_columns == ("x", "y", "yerr")


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
    assert overridden.explicit_override
    assert not overridden.requires_confirmation
    assert not overridden.has_errors


def test_inconsistent_explicit_override_does_not_fall_back() -> None:
    result = detect_reduced_data_format(
        FIXTURES / "wide_multiple_pairs.txt",
        explicit_format=ReducedDataFormat.SINGLE_SPECTRUM_TABLE,
    )

    assert result.proposed_format is ReducedDataFormat.SINGLE_SPECTRUM_TABLE
    assert result.has_errors
    assert ReducedDataFormat.WIDE_QENS_TABLE in result.alternative_formats
    assert "explicit_format_inconsistent" in {
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

    assert with_extension.proposed_format is ReducedDataFormat.WIDE_QENS_TABLE
    assert without_extension.proposed_format is ReducedDataFormat.WIDE_QENS_TABLE
    assert with_extension.extension_hint == ".csv"
    assert without_extension.extension_hint is None


@pytest.mark.parametrize(
    "filename",
    ["unknown_table.txt", "single_missing_yerr.txt"],
)
def test_unknown_or_malformed_input_returns_diagnostics(filename: str) -> None:
    result = detect_reduced_data_format(FIXTURES / filename)

    assert result.proposed_format is ReducedDataFormat.UNKNOWN
    assert result.confidence is DetectionConfidence.NONE
    assert result.has_errors
    assert all(
        diagnostic.severity is DiagnosticSeverity.ERROR
        for diagnostic in result.diagnostics
    )


def test_malformed_wide_header_keeps_layout_and_reports_pair_error() -> None:
    result = detect_reduced_data_format(FIXTURES / "wide_missing_yerr.txt")

    assert result.proposed_format is ReducedDataFormat.WIDE_QENS_TABLE
    assert result.confidence is DetectionConfidence.LOW
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
