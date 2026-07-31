"""Import supported Milestone 1 reduced-data text layouts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qensfit.domain import (
    DiagnosticSeverity,
    FormatDetectionResult,
    ImportDiagnostic,
    ImportedDataset,
    ImportValidationError,
    ReducedDataFormat,
    SourceColumnMetadata,
    Spectrum,
    SpectrumRole,
)
from qensfit.io.importers._text import (
    TextHeader,
    analyze_wide_columns,
    find_group_markers,
    find_table_header,
    normalized_columns,
    read_text_lines,
    split_columns,
)
from qensfit.io.importers.detection import detect_reduced_data_format

_REQUIRED_COLUMNS = ("x", "y", "yerr")


def _coerce_role(role: SpectrumRole | str) -> SpectrumRole:
    try:
        return SpectrumRole(role)
    except ValueError as error:
        raise ValueError("role must be 'sample' or 'resolution'") from error


def _raise_on_errors(diagnostics: tuple[ImportDiagnostic, ...]) -> None:
    if any(
        diagnostic.severity is DiagnosticSeverity.ERROR
        for diagnostic in diagnostics
    ):
        raise ImportValidationError(diagnostics)


def _required_positions(
    header: TextHeader,
    *,
    group: str | None = None,
) -> tuple[dict[str, int], tuple[ImportDiagnostic, ...]]:
    normalized = normalized_columns(header.columns)
    positions: dict[str, int] = {}
    diagnostics: list[ImportDiagnostic] = []
    for required in _REQUIRED_COLUMNS:
        matches = tuple(
            index for index, column in enumerate(normalized) if column == required
        )
        if not matches:
            diagnostics.append(
                ImportDiagnostic(
                    code="required_column_missing",
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Required column {required!r} is missing",
                    group=group,
                    row=header.line_number,
                    column=required,
                )
            )
        elif len(matches) > 1:
            diagnostics.append(
                ImportDiagnostic(
                    code="required_column_duplicated",
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Required column {required!r} is duplicated",
                    group=group,
                    row=header.line_number,
                    column=required,
                )
            )
        else:
            positions[required] = matches[0]
    return positions, tuple(diagnostics)


def _parse_numeric_rows(
    lines: tuple[str, ...],
    *,
    header: TextHeader,
    end_index: int,
    group: str | None,
) -> tuple[np.ndarray, tuple[int, ...], tuple[ImportDiagnostic, ...]]:
    rows: list[tuple[float, ...]] = []
    row_numbers: list[int] = []
    diagnostics: list[ImportDiagnostic] = []
    expected_width = len(header.columns)
    for line_index in range(header.line_index + 1, end_index):
        stripped = lines[line_index].strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = split_columns(lines[line_index])
        line_number = line_index + 1
        if len(tokens) != expected_width:
            diagnostics.append(
                ImportDiagnostic(
                    code="inconsistent_row_width",
                    severity=DiagnosticSeverity.ERROR,
                    message=(
                        f"Numerical row has {len(tokens)} columns; "
                        f"expected {expected_width}"
                    ),
                    group=group,
                    row=line_number,
                )
            )
            continue
        try:
            values = tuple(float(token) for token in tokens)
        except ValueError:
            diagnostics.append(
                ImportDiagnostic(
                    code="malformed_numeric_row",
                    severity=DiagnosticSeverity.ERROR,
                    message="Numerical row contains a non-numeric value",
                    group=group,
                    row=line_number,
                )
            )
            continue
        rows.append(values)
        row_numbers.append(line_number)

    if not rows:
        diagnostics.append(
            ImportDiagnostic(
                code="data_rows_missing",
                severity=DiagnosticSeverity.ERROR,
                message="No valid numerical rows were found",
                group=group,
                row=header.line_number,
            )
        )
        matrix = np.empty((0, expected_width), dtype=np.float64)
    else:
        matrix = np.asarray(rows, dtype=np.float64)
    return matrix, tuple(row_numbers), tuple(diagnostics)


def _invalid_value_diagnostics(spectrum: Spectrum) -> tuple[ImportDiagnostic, ...]:
    diagnostics: list[ImportDiagnostic] = []
    invalid_fields = (
        (
            "invalid_energy_values",
            spectrum.invalid_energy_mask,
            spectrum.source_columns.energy,
        ),
        (
            "invalid_intensity_values",
            spectrum.invalid_intensity_mask,
            spectrum.source_columns.intensity,
        ),
        (
            "invalid_uncertainty_values",
            spectrum.invalid_uncertainty_mask,
            spectrum.source_columns.uncertainty,
        ),
    )
    for code, mask, column in invalid_fields:
        count = int(np.count_nonzero(mask))
        if count:
            diagnostics.append(
                ImportDiagnostic(
                    code=code,
                    severity=DiagnosticSeverity.WARNING,
                    message=f"Detected {count} invalid value(s)",
                    group=spectrum.group_label,
                    column=column,
                )
            )
    return tuple(diagnostics)


def _shared_axis(
    spectra: tuple[Spectrum, ...],
) -> tuple[bool, np.ndarray | None]:
    first = spectra[0].energy
    shared = all(
        np.array_equal(spectrum.energy, first, equal_nan=True)
        for spectrum in spectra[1:]
    )
    return (True, first) if shared else (False, None)


def _extra_columns_diagnostic(
    extra_columns: tuple[str, ...],
) -> tuple[ImportDiagnostic, ...]:
    if not extra_columns:
        return ()
    return (
        ImportDiagnostic(
            code="extra_columns_ignored",
            severity=DiagnosticSeverity.INFO,
            message=(
                f"Recorded {len(extra_columns)} additional source column(s) "
                "outside the primary x/y/yerr mapping"
            ),
        ),
    )


def _import_dave_groups(
    *,
    lines: tuple[str, ...],
    role: SpectrumRole,
    detection: FormatDetectionResult,
    source: Path,
    energy_unit: str,
    intensity_unit: str,
    uncertainty_unit: str,
) -> ImportedDataset:
    markers = find_group_markers(lines)
    if not markers:
        raise ImportValidationError(
            (
                ImportDiagnostic(
                    code="dave_group_markers_missing",
                    severity=DiagnosticSeverity.ERROR,
                    message="No DAVE group markers were found",
                ),
            )
        )

    spectra: list[Spectrum] = []
    diagnostics: list[ImportDiagnostic] = list(detection.diagnostics)
    extra_columns: list[str] = []
    for group_index, marker in enumerate(markers):
        end_index = (
            markers[group_index + 1].line_index
            if group_index + 1 < len(markers)
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
        positions, header_diagnostics = _required_positions(
            header,
            group=marker.label,
        )
        diagnostics.extend(header_diagnostics)
        matrix, row_numbers, row_diagnostics = _parse_numeric_rows(
            lines,
            header=header,
            end_index=end_index,
            group=marker.label,
        )
        diagnostics.extend(row_diagnostics)
        if header_diagnostics or row_diagnostics:
            continue

        normalized = normalized_columns(header.columns)
        group_extra_columns = tuple(
            column
            for column, normalized_column in zip(
                header.columns,
                normalized,
                strict=True,
            )
            if normalized_column not in _REQUIRED_COLUMNS
        )
        for column in group_extra_columns:
            if column not in extra_columns:
                extra_columns.append(column)
        spectrum = Spectrum.from_imported_arrays(
            role=role,
            group_index=group_index,
            group_label=marker.label,
            energy=matrix[:, positions["x"]],
            intensity=matrix[:, positions["y"]],
            uncertainty=matrix[:, positions["yerr"]],
            energy_unit=energy_unit,
            intensity_unit=intensity_unit,
            uncertainty_unit=uncertainty_unit,
            source_row_numbers=row_numbers,
            source_columns=SourceColumnMetadata(
                energy=header.columns[positions["x"]],
                intensity=header.columns[positions["y"]],
                uncertainty=header.columns[positions["yerr"]],
                extra_columns=group_extra_columns,
            ),
            source_layout=ReducedDataFormat.DAVE_GROUP_BLOCKS,
            source_layout_metadata=(
                ("group_marker_line", str(marker.line_number)),
                ("header_line", str(header.line_number)),
            ),
        )
        spectra.append(spectrum)
        diagnostics.extend(_invalid_value_diagnostics(spectrum))

    _raise_on_errors(tuple(diagnostics))
    ordered_spectra = tuple(spectra)
    shared, shared_axis = _shared_axis(ordered_spectra)
    extras = tuple(extra_columns)
    diagnostics.extend(_extra_columns_diagnostic(extras))
    return ImportedDataset(
        role=role,
        source_layout=ReducedDataFormat.DAVE_GROUP_BLOCKS,
        spectra=ordered_spectra,
        source_reference=source.name,
        diagnostics=tuple(diagnostics),
        detected_extra_columns=extras,
        shared_energy_grid=shared,
        shared_energy_axis=shared_axis,
        format_detection=detection,
    )


def _import_wide_table(
    *,
    lines: tuple[str, ...],
    role: SpectrumRole,
    detection: FormatDetectionResult,
    source: Path,
    energy_unit: str,
    intensity_unit: str,
    uncertainty_unit: str,
) -> ImportedDataset:
    header = find_table_header(lines)
    if header is None:
        raise ImportValidationError(detection.diagnostics)
    analysis = analyze_wide_columns(header.columns)
    matrix, row_numbers, row_diagnostics = _parse_numeric_rows(
        lines,
        header=header,
        end_index=len(lines),
        group=None,
    )
    diagnostics = list(detection.diagnostics) + list(row_diagnostics)
    _raise_on_errors(tuple(diagnostics))

    energy_position = analysis.energy_positions[0]
    extras = tuple(header.columns[index] for index in analysis.extra_positions)
    spectra: list[Spectrum] = []
    for group_index, suffix in enumerate(analysis.complete_suffixes):
        intensity_position = analysis.intensity_positions[suffix][0]
        uncertainty_position = analysis.uncertainty_positions[suffix][0]
        source_suffix = header.columns[intensity_position][1:]
        spectrum = Spectrum.from_imported_arrays(
            role=role,
            group_index=group_index,
            group_label=source_suffix,
            energy=matrix[:, energy_position],
            intensity=matrix[:, intensity_position],
            uncertainty=matrix[:, uncertainty_position],
            energy_unit=energy_unit,
            intensity_unit=intensity_unit,
            uncertainty_unit=uncertainty_unit,
            source_row_numbers=row_numbers,
            source_columns=SourceColumnMetadata(
                energy=header.columns[energy_position],
                intensity=header.columns[intensity_position],
                uncertainty=header.columns[uncertainty_position],
                extra_columns=extras,
            ),
            source_layout=ReducedDataFormat.WIDE_QENS_TABLE,
            source_layout_metadata=(
                ("header_line", str(header.line_number)),
                ("column_suffix", source_suffix),
            ),
        )
        spectra.append(spectrum)
        diagnostics.extend(_invalid_value_diagnostics(spectrum))
    diagnostics.extend(_extra_columns_diagnostic(extras))

    ordered_spectra = tuple(spectra)
    return ImportedDataset(
        role=role,
        source_layout=ReducedDataFormat.WIDE_QENS_TABLE,
        spectra=ordered_spectra,
        source_reference=source.name,
        diagnostics=tuple(diagnostics),
        detected_extra_columns=extras,
        shared_energy_grid=True,
        shared_energy_axis=matrix[:, energy_position],
        format_detection=detection,
    )


def _import_single_table(
    *,
    lines: tuple[str, ...],
    role: SpectrumRole,
    detection: FormatDetectionResult,
    source: Path,
    energy_unit: str,
    intensity_unit: str,
    uncertainty_unit: str,
) -> ImportedDataset:
    header = find_table_header(lines)
    if header is None:
        raise ImportValidationError(detection.diagnostics)
    positions, header_diagnostics = _required_positions(header)
    matrix, row_numbers, row_diagnostics = _parse_numeric_rows(
        lines,
        header=header,
        end_index=len(lines),
        group=None,
    )
    diagnostics = (
        list(detection.diagnostics)
        + list(header_diagnostics)
        + list(row_diagnostics)
    )
    _raise_on_errors(tuple(diagnostics))

    normalized = normalized_columns(header.columns)
    extras = tuple(
        column
        for column, normalized_column in zip(
            header.columns,
            normalized,
            strict=True,
        )
        if normalized_column not in _REQUIRED_COLUMNS
    )
    spectrum = Spectrum.from_imported_arrays(
        role=role,
        group_index=0,
        group_label="spectrum",
        energy=matrix[:, positions["x"]],
        intensity=matrix[:, positions["y"]],
        uncertainty=matrix[:, positions["yerr"]],
        energy_unit=energy_unit,
        intensity_unit=intensity_unit,
        uncertainty_unit=uncertainty_unit,
        source_row_numbers=row_numbers,
        source_columns=SourceColumnMetadata(
            energy=header.columns[positions["x"]],
            intensity=header.columns[positions["y"]],
            uncertainty=header.columns[positions["yerr"]],
            extra_columns=extras,
        ),
        source_layout=ReducedDataFormat.SINGLE_SPECTRUM_TABLE,
        source_layout_metadata=(("header_line", str(header.line_number)),),
    )
    diagnostics.extend(_invalid_value_diagnostics(spectrum))
    diagnostics.extend(_extra_columns_diagnostic(extras))
    return ImportedDataset(
        role=role,
        source_layout=ReducedDataFormat.SINGLE_SPECTRUM_TABLE,
        spectra=(spectrum,),
        source_reference=source.name,
        diagnostics=tuple(diagnostics),
        detected_extra_columns=extras,
        shared_energy_grid=True,
        shared_energy_axis=spectrum.energy,
        format_detection=detection,
    )


def import_reduced_data(
    path: str | Path,
    *,
    role: SpectrumRole | str,
    explicit_format: ReducedDataFormat | str | None = None,
    energy_unit: str = "unknown",
    intensity_unit: str = "unknown",
    uncertainty_unit: str = "unknown",
) -> ImportedDataset:
    """Detect and import one supported reduced-data text source.

    The importer preserves source order and numerical values, classifies invalid
    values, and never creates physical Q values.
    """

    source = Path(path)
    selected_role = _coerce_role(role)
    detection = detect_reduced_data_format(source, explicit_format)
    if detection.proposed_format in {
        ReducedDataFormat.UNKNOWN,
        ReducedDataFormat.AMBIGUOUS,
    }:
        raise ImportValidationError(detection.diagnostics)
    _raise_on_errors(detection.diagnostics)
    lines = read_text_lines(source)

    if detection.proposed_format is ReducedDataFormat.DAVE_GROUP_BLOCKS:
        return _import_dave_groups(
            lines=lines,
            role=selected_role,
            detection=detection,
            source=source,
            energy_unit=energy_unit,
            intensity_unit=intensity_unit,
            uncertainty_unit=uncertainty_unit,
        )
    if detection.proposed_format is ReducedDataFormat.WIDE_QENS_TABLE:
        return _import_wide_table(
            lines=lines,
            role=selected_role,
            detection=detection,
            source=source,
            energy_unit=energy_unit,
            intensity_unit=intensity_unit,
            uncertainty_unit=uncertainty_unit,
        )
    return _import_single_table(
        lines=lines,
        role=selected_role,
        detection=detection,
        source=source,
        energy_unit=energy_unit,
        intensity_unit=intensity_unit,
        uncertainty_unit=uncertainty_unit,
    )
