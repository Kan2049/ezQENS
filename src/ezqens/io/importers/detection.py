"""Content-based detection for reduced QENS text layouts."""

from __future__ import annotations

from pathlib import Path

from ezqens.domain import (
    DiagnosticSeverity,
    FormatDetectionResult,
    ImportDiagnostic,
    ReducedDataFormat,
)
from ezqens.io.importers._text import (
    TextHeader,
    analyze_wide_columns,
    find_group_markers,
    find_table_header,
    normalized_columns,
    read_text_lines,
)

_IMPORTABLE_FORMATS = (
    ReducedDataFormat.DAVE_GROUP_BLOCKS,
    ReducedDataFormat.WIDE_QENS_TABLE,
    ReducedDataFormat.SINGLE_SPECTRUM_TABLE,
)
_REQUIRED_COLUMNS = ("x", "y", "yerr")


def _coerce_explicit_format(
    explicit_format: ReducedDataFormat | str | None,
) -> ReducedDataFormat | None:
    if explicit_format is None:
        return None
    try:
        selected = ReducedDataFormat(explicit_format)
    except ValueError:
        return ReducedDataFormat.UNKNOWN
    return selected if selected in _IMPORTABLE_FORMATS else ReducedDataFormat.UNKNOWN


def _dave_detection(lines: tuple[str, ...]) -> FormatDetectionResult:
    markers = find_group_markers(lines)
    diagnostics: list[ImportDiagnostic] = []
    extra_columns: list[str] = []
    compatible_headers = 0
    if not markers:
        diagnostics.append(
            ImportDiagnostic(
                code="dave_group_markers_missing",
                severity=DiagnosticSeverity.ERROR,
                message="DAVE format requires at least one group marker",
            )
        )
    for marker_index, marker in enumerate(markers):
        end_index = (
            markers[marker_index + 1].line_index
            if marker_index + 1 < len(markers)
            else len(lines)
        )
        header = find_table_header(
            lines,
            start_index=marker.line_index + 1,
            end_index=end_index,
        )
        if header is None:
            diagnostics.append(
                ImportDiagnostic(
                    code="dave_group_header_missing",
                    severity=DiagnosticSeverity.ERROR,
                    message="DAVE group has no detectable column header",
                    group=marker.label,
                    row=marker.line_number,
                )
            )
            continue
        normalized = normalized_columns(header.columns)
        missing = tuple(
            column for column in _REQUIRED_COLUMNS if column not in normalized
        )
        if missing:
            diagnostics.append(
                ImportDiagnostic(
                    code="dave_required_columns_missing",
                    severity=DiagnosticSeverity.ERROR,
                    message="DAVE group is missing required columns: "
                    + ", ".join(missing),
                    group=marker.label,
                    row=header.line_number,
                )
            )
            continue
        compatible_headers += 1
        for column, normalized_column in zip(header.columns, normalized, strict=True):
            if normalized_column not in _REQUIRED_COLUMNS:
                if column not in extra_columns:
                    extra_columns.append(column)

    return FormatDetectionResult(
        proposed_format=ReducedDataFormat.DAVE_GROUP_BLOCKS,
        evidence=(f"Found {len(markers)} DAVE group marker(s)",),
        detected_required_columns=_REQUIRED_COLUMNS,
        detected_extra_columns=tuple(extra_columns),
        detected_count=len(markers),
        diagnostics=tuple(diagnostics),
    )


def _single_detection(header: TextHeader | None) -> FormatDetectionResult:
    diagnostics: list[ImportDiagnostic] = []
    normalized: tuple[str, ...] = ()
    extra_columns: tuple[str, ...] = ()
    if header is None:
        diagnostics.append(
            ImportDiagnostic(
                code="table_header_missing",
                severity=DiagnosticSeverity.ERROR,
                message="No table header was detected",
            )
        )
    else:
        normalized = normalized_columns(header.columns)
        missing = tuple(
            column for column in _REQUIRED_COLUMNS if column not in normalized
        )
        if missing:
            diagnostics.append(
                ImportDiagnostic(
                    code="single_required_columns_missing",
                    severity=DiagnosticSeverity.ERROR,
                    message="Single-spectrum table is missing required columns: "
                    + ", ".join(missing),
                    row=header.line_number,
                )
            )
        for required in _REQUIRED_COLUMNS:
            if normalized.count(required) > 1:
                diagnostics.append(
                    ImportDiagnostic(
                        code="single_duplicate_required_column",
                        severity=DiagnosticSeverity.ERROR,
                        message=f"Single-spectrum column {required!r} is duplicated",
                        row=header.line_number,
                        column=required,
                    )
                )
        extra_columns = tuple(
            column
            for column, normalized_column in zip(
                header.columns, normalized, strict=True
            )
            if normalized_column not in _REQUIRED_COLUMNS
        )

    has_errors = any(
        diagnostic.severity is DiagnosticSeverity.ERROR for diagnostic in diagnostics
    )
    return FormatDetectionResult(
        proposed_format=ReducedDataFormat.SINGLE_SPECTRUM_TABLE,
        evidence=("Found an x/y/yerr table header",),
        detected_required_columns=tuple(
            column for column in _REQUIRED_COLUMNS if column in normalized
        ),
        detected_extra_columns=extra_columns,
        detected_count=0 if has_errors else 1,
        diagnostics=tuple(diagnostics),
    )


def _wide_detection(header: TextHeader | None) -> FormatDetectionResult:
    diagnostics: list[ImportDiagnostic] = []
    columns = () if header is None else header.columns
    if header is None:
        diagnostics.append(
            ImportDiagnostic(
                code="table_header_missing",
                severity=DiagnosticSeverity.ERROR,
                message="No table header was detected",
            )
        )
    analysis = analyze_wide_columns(columns)

    if len(analysis.energy_positions) != 1:
        diagnostics.append(
            ImportDiagnostic(
                code="wide_energy_column_count",
                severity=DiagnosticSeverity.ERROR,
                message="Wide table requires exactly one x energy column",
                row=None if header is None else header.line_number,
                column="x",
            )
        )
    for suffix in analysis.all_suffixes:
        intensity_count = len(analysis.intensity_positions.get(suffix, ()))
        uncertainty_count = len(analysis.uncertainty_positions.get(suffix, ()))
        if intensity_count == 0:
            diagnostics.append(
                ImportDiagnostic(
                    code="wide_intensity_column_missing",
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Wide table is missing y{suffix}",
                    row=None if header is None else header.line_number,
                    column=f"y{suffix}",
                )
            )
        elif intensity_count > 1:
            diagnostics.append(
                ImportDiagnostic(
                    code="wide_duplicate_intensity_suffix",
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Wide table has duplicate y{suffix} columns",
                    row=None if header is None else header.line_number,
                    column=f"y{suffix}",
                )
            )
        if uncertainty_count == 0:
            diagnostics.append(
                ImportDiagnostic(
                    code="wide_uncertainty_column_missing",
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Wide table is missing yerr{suffix}",
                    row=None if header is None else header.line_number,
                    column=f"yerr{suffix}",
                )
            )
        elif uncertainty_count > 1:
            diagnostics.append(
                ImportDiagnostic(
                    code="wide_duplicate_uncertainty_suffix",
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Wide table has duplicate yerr{suffix} columns",
                    row=None if header is None else header.line_number,
                    column=f"yerr{suffix}",
                )
            )
    if not analysis.all_suffixes:
        diagnostics.append(
            ImportDiagnostic(
                code="wide_pairs_missing",
                severity=DiagnosticSeverity.ERROR,
                message="Wide table has no yN/yerrN column pair",
                row=None if header is None else header.line_number,
            )
        )

    required_columns: list[str] = []
    if len(analysis.energy_positions) == 1:
        required_columns.append(columns[analysis.energy_positions[0]])
    for suffix in analysis.complete_suffixes:
        required_columns.append(columns[analysis.intensity_positions[suffix][0]])
        required_columns.append(columns[analysis.uncertainty_positions[suffix][0]])
    return FormatDetectionResult(
        proposed_format=ReducedDataFormat.WIDE_QENS_TABLE,
        evidence=(
            f"Found {len(analysis.complete_suffixes)} complete yN/yerrN pair(s)",
        ),
        detected_required_columns=tuple(required_columns),
        detected_extra_columns=tuple(
            columns[index] for index in analysis.extra_positions
        ),
        detected_count=len(analysis.complete_suffixes),
        diagnostics=tuple(diagnostics),
    )


def _auto_detect(lines: tuple[str, ...]) -> FormatDetectionResult:
    if find_group_markers(lines):
        return _dave_detection(lines)

    header = find_table_header(lines)
    if header is None:
        return FormatDetectionResult(
            proposed_format=ReducedDataFormat.UNKNOWN,
            evidence=("No recognizable reduced-data header was found",),
            detected_required_columns=(),
            detected_extra_columns=(),
            detected_count=0,
            diagnostics=(
                ImportDiagnostic(
                    code="format_unknown",
                    severity=DiagnosticSeverity.ERROR,
                    message="Reduced-data format could not be determined safely",
                ),
            ),
        )

    normalized = normalized_columns(header.columns)
    wide_analysis = analyze_wide_columns(header.columns)
    has_single = all(column in normalized for column in _REQUIRED_COLUMNS)
    has_wide = bool(wide_analysis.all_suffixes)
    if has_single and has_wide:
        return FormatDetectionResult(
            proposed_format=ReducedDataFormat.AMBIGUOUS,
            evidence=("Header matches both single and wide table layouts",),
            detected_required_columns=_REQUIRED_COLUMNS,
            detected_extra_columns=tuple(
                column
                for column in header.columns
                if column.casefold() not in _REQUIRED_COLUMNS
            ),
            detected_count=len(wide_analysis.complete_suffixes) + 1,
            diagnostics=(
                ImportDiagnostic(
                    code="format_ambiguous",
                    severity=DiagnosticSeverity.ERROR,
                    message="Multiple reduced-data layouts are plausible",
                    row=header.line_number,
                ),
            ),
            alternative_formats=(
                ReducedDataFormat.WIDE_QENS_TABLE,
                ReducedDataFormat.SINGLE_SPECTRUM_TABLE,
            ),
        )
    if has_wide:
        return _wide_detection(header)
    if has_single:
        return _single_detection(header)

    detected_partial = tuple(
        column for column in _REQUIRED_COLUMNS if column in normalized
    )
    missing = tuple(column for column in _REQUIRED_COLUMNS if column not in normalized)
    diagnostic = (
        ImportDiagnostic(
            code="single_required_columns_missing",
            severity=DiagnosticSeverity.ERROR,
            message="Single-spectrum table is missing required columns: "
            + ", ".join(missing),
            row=header.line_number,
        )
        if detected_partial
        else ImportDiagnostic(
            code="format_unknown",
            severity=DiagnosticSeverity.ERROR,
            message="Reduced-data format could not be determined safely",
            row=header.line_number,
        )
    )
    return FormatDetectionResult(
        proposed_format=ReducedDataFormat.UNKNOWN,
        evidence=("Header does not satisfy a supported reduced-data layout",),
        detected_required_columns=detected_partial,
        detected_extra_columns=tuple(header.columns),
        detected_count=0,
        diagnostics=(diagnostic,),
    )


def _explicit_detection(
    lines: tuple[str, ...], selected: ReducedDataFormat
) -> FormatDetectionResult:
    if selected is ReducedDataFormat.DAVE_GROUP_BLOCKS:
        return _dave_detection(lines)

    header = find_table_header(lines)
    result = (
        _wide_detection(header)
        if selected is ReducedDataFormat.WIDE_QENS_TABLE
        else _single_detection(header)
    )
    if find_group_markers(lines):
        mismatch = ImportDiagnostic(
            code="explicit_format_inconsistent",
            severity=DiagnosticSeverity.ERROR,
            message=(
                "DAVE group markers are incompatible with the selected table layout"
            ),
        )
        return FormatDetectionResult(
            proposed_format=result.proposed_format,
            evidence=result.evidence,
            detected_required_columns=result.detected_required_columns,
            detected_extra_columns=result.detected_extra_columns,
            detected_count=result.detected_count,
            diagnostics=result.diagnostics + (mismatch,),
            alternative_formats=(ReducedDataFormat.DAVE_GROUP_BLOCKS,),
        )
    return result


def detect_reduced_data_format(
    path: str | Path,
    explicit_format: ReducedDataFormat | str | None = None,
) -> FormatDetectionResult:
    """Detect or validate a supported reduced-data layout from file contents."""

    lines = read_text_lines(Path(path))
    selected = _coerce_explicit_format(explicit_format)
    if explicit_format is not None and selected is ReducedDataFormat.UNKNOWN:
        return FormatDetectionResult(
            proposed_format=ReducedDataFormat.UNKNOWN,
            evidence=("Explicit format override is unsupported",),
            detected_required_columns=(),
            detected_extra_columns=(),
            detected_count=0,
            diagnostics=(
                ImportDiagnostic(
                    code="explicit_format_unsupported",
                    severity=DiagnosticSeverity.ERROR,
                    message="Explicit format must select a supported reduced layout",
                ),
            ),
        )
    return (
        _auto_detect(lines)
        if selected is None
        else _explicit_detection(lines, selected)
    )
