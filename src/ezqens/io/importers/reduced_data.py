"""Import supported reduced QENS text layouts."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ezqens.domain import (
    DiagnosticSeverity,
    FormatDetectionResult,
    ImportDiagnostic,
    ImportValidationError,
    QBins,
    ReducedDataFormat,
    ReducedDataset,
    SourceMetadata,
    Spectrum,
    SpectrumRole,
)
from ezqens.domain.models import SourceColumnMetadata
from ezqens.io.importers._text import (
    TextHeader,
    analyze_wide_columns,
    find_group_markers,
    find_table_header,
    mantid_xye_blocks,
    normalized_columns,
    normalized_dave_columns,
    read_text_lines,
    split_columns,
)
from ezqens.io.importers.detection import detect_reduced_data_format

_REQUIRED_COLUMNS = ("x", "y", "yerr")


def _coerce_role(role: SpectrumRole | str) -> SpectrumRole:
    try:
        return SpectrumRole(role)
    except ValueError as error:
        raise ValueError("role must be 'sample' or 'resolution'") from error


def _raise_on_errors(diagnostics: tuple[ImportDiagnostic, ...]) -> None:
    if any(
        diagnostic.severity is DiagnosticSeverity.ERROR for diagnostic in diagnostics
    ):
        raise ImportValidationError(diagnostics)


def _required_positions(
    header: TextHeader,
    *,
    group: str | None = None,
    dave: bool = False,
) -> tuple[dict[str, int], tuple[ImportDiagnostic, ...]]:
    normalized = (
        normalized_dave_columns(header.columns)
        if dave
        else normalized_columns(header.columns)
    )
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


def _invalid_value_diagnostics(
    spectrum: Spectrum,
    columns: SourceColumnMetadata,
) -> tuple[ImportDiagnostic, ...]:
    diagnostics: list[ImportDiagnostic] = []
    invalid_fields = (
        ("invalid_energy_values", spectrum.invalid_energy_mask, columns.energy),
        (
            "invalid_intensity_values",
            spectrum.invalid_intensity_mask,
            columns.intensity,
        ),
        (
            "invalid_uncertainty_values",
            spectrum.invalid_uncertainty_mask,
            columns.uncertainty,
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
                "outside the primary energy/intensity/uncertainty mapping"
            ),
        ),
    )


def _make_spectrum(
    *,
    role: SpectrumRole,
    group_index: int,
    group_label: str,
    energy: np.ndarray,
    intensity: np.ndarray,
    uncertainty: np.ndarray,
    energy_unit: str,
    intensity_unit: str,
    uncertainty_unit: str,
) -> Spectrum:
    return Spectrum(
        role=role,
        group_index=group_index,
        group_label=group_label,
        energy=energy,
        intensity=intensity,
        uncertainty=uncertainty,
        energy_unit=energy_unit,
        intensity_unit=intensity_unit,
        uncertainty_unit=uncertainty_unit,
    )


_HEADER_ENTRY_PATTERN = re.compile(r"^\s*#\s*([^:=]+?)\s*[:=]\s*(.*?)\s*$")
_GROUP_VALUE_PATTERN = re.compile(
    r"^\s*#\s*group\s+value\s*:\s*(.*?)\s*$",
    re.IGNORECASE,
)


def _dave_header_lines(
    lines: tuple[str, ...],
    first_group_index: int,
) -> tuple[str, ...]:
    begin_index = next(
        (
            index
            for index, line in enumerate(lines[:first_group_index])
            if line.strip().casefold() == "#begin"
        ),
        first_group_index,
    )
    return lines[:begin_index]


def _header_entries(
    raw_header_lines: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    entries: list[tuple[str, str]] = []
    for line in raw_header_lines:
        match = _HEADER_ENTRY_PATTERN.match(line)
        if match is not None:
            key = " ".join(match.group(1).casefold().split())
            entries.append((key, match.group(2)))
    return tuple(entries)


def _unique_header_value(
    entries: tuple[tuple[str, str], ...],
    key: str,
) -> str | None:
    values = tuple(value for entry_key, value in entries if entry_key == key and value)
    return values[0] if values and all(value == values[0] for value in values) else None


def _optional_header_number(
    entries: tuple[tuple[str, str], ...],
    *,
    key: str,
    diagnostic_code: str,
    field_name: str,
    minimum: float,
    inclusive: bool,
) -> tuple[float | None, tuple[ImportDiagnostic, ...]]:
    declarations = tuple(value for entry_key, value in entries if entry_key == key)
    if not declarations:
        return None, ()
    raw = (
        declarations[0]
        if all(value == declarations[0] for value in declarations)
        else None
    )
    value: float | None = None
    if raw is not None:
        try:
            candidate = float(raw)
        except ValueError:
            candidate = np.nan
        if np.isfinite(candidate) and (
            candidate >= minimum if inclusive else candidate > minimum
        ):
            value = candidate
    if value is not None:
        return value, ()
    return None, (
        ImportDiagnostic(
            code=diagnostic_code,
            severity=DiagnosticSeverity.WARNING,
            message=(
                f"DAVE {field_name} metadata is invalid and was not promoted; "
                "the original header remains preserved"
            ),
        ),
    )


def _canonical_energy_unit(value: str) -> str | None:
    candidate = value.strip()
    match = re.search(r"energy\s*/\s*([^\s]+)", candidate, re.IGNORECASE)
    if match is not None:
        candidate = match.group(1)
    normalized = candidate.casefold().replace("μ", "µ")
    aliases = {"mev": "meV", "ev": "eV", "uev": "µeV", "µev": "µeV"}
    return aliases.get(normalized)


def _dave_source_metadata(
    *,
    source: Path,
    raw_header_lines: tuple[str, ...],
) -> tuple[SourceMetadata, tuple[ImportDiagnostic, ...]]:
    entries = _header_entries(raw_header_lines)
    temperature, temperature_diagnostics = _optional_header_number(
        entries,
        key="temperature / k",
        diagnostic_code="dave_temperature_invalid",
        field_name="temperature",
        minimum=0.0,
        inclusive=True,
    )
    wavelength, wavelength_diagnostics = _optional_header_number(
        entries,
        key="wavelength / a",
        diagnostic_code="dave_wavelength_invalid",
        field_name="wavelength",
        minimum=0.0,
        inclusive=False,
    )
    metadata = SourceMetadata(
        source_filename=source.name,
        raw_header_lines=raw_header_lines,
        instrument=_unique_header_value(entries, "instrument"),
        sample=_unique_header_value(entries, "sample"),
        title=_unique_header_value(entries, "title"),
        temperature_kelvin=temperature,
        wavelength_angstrom=wavelength,
    )
    return metadata, (*temperature_diagnostics, *wavelength_diagnostics)


def _dave_energy_unit(
    entries: tuple[tuple[str, str], ...],
    fallback: str,
) -> tuple[str, tuple[ImportDiagnostic, ...]]:
    declarations = tuple(
        value.strip()
        for key, value in entries
        if key in {"x units", "energy units"} and value.strip()
    )
    if not declarations:
        return fallback, ()

    canonical = tuple(_canonical_energy_unit(value) for value in declarations)
    recognized = tuple(unit for unit in canonical if unit is not None)
    diagnostics: list[ImportDiagnostic] = []
    if len(recognized) != len(declarations):
        diagnostics.append(
            ImportDiagnostic(
                code="dave_energy_unit_unrecognized",
                severity=DiagnosticSeverity.ERROR,
                message=(
                    "Every explicit DAVE energy-unit declaration must be recognized"
                ),
            )
        )
    if len(set(recognized)) > 1:
        diagnostics.append(
            ImportDiagnostic(
                code="dave_energy_unit_conflict",
                severity=DiagnosticSeverity.ERROR,
                message="DAVE header contains conflicting energy-unit declarations",
            )
        )
    if diagnostics:
        return fallback, tuple(diagnostics)

    header_unit = recognized[0]
    caller_is_missing = not fallback.strip() or fallback.strip().casefold() == "unknown"
    if not caller_is_missing and _canonical_energy_unit(fallback) != header_unit:
        return fallback, (
            ImportDiagnostic(
                code="dave_energy_unit_caller_conflict",
                severity=DiagnosticSeverity.ERROR,
                message=(
                    "Explicit caller energy unit conflicts with the authoritative "
                    "DAVE header unit"
                ),
            ),
        )
    return header_unit, ()


def _canonical_y_unit(value: str) -> str:
    declaration = value.strip()
    if "arbitrary units" in declaration.casefold():
        return "arbitrary units"
    return declaration


def _dave_y_unit(
    entries: tuple[tuple[str, str], ...],
) -> tuple[str | None, tuple[ImportDiagnostic, ...]]:
    declarations = tuple(
        _canonical_y_unit(value)
        for key, value in entries
        if key == "y units" and value.strip()
    )
    if not declarations:
        return None, ()
    if len({value.casefold() for value in declarations}) > 1:
        return None, (
            ImportDiagnostic(
                code="dave_y_unit_conflict",
                severity=DiagnosticSeverity.ERROR,
                message="DAVE header contains conflicting Y-unit declarations",
            ),
        )
    return declarations[0], ()


def _dave_group_values(
    lines: tuple[str, ...],
    marker_indices: tuple[int, ...],
) -> tuple[tuple[float, ...] | None, tuple[ImportDiagnostic, ...]]:
    diagnostics: list[ImportDiagnostic] = []
    values: list[float] = []
    found_any = False
    for group_index, start_index in enumerate(marker_indices):
        end_index = (
            marker_indices[group_index + 1]
            if group_index + 1 < len(marker_indices)
            else len(lines)
        )
        raw_values = tuple(
            match.group(1)
            for line in lines[start_index + 1 : end_index]
            if (match := _GROUP_VALUE_PATTERN.match(line)) is not None
        )
        found_any = found_any or bool(raw_values)
        if len(raw_values) != 1:
            diagnostics.append(
                ImportDiagnostic(
                    code="dave_q_group_value_incomplete",
                    severity=DiagnosticSeverity.WARNING,
                    message="DAVE Q metadata requires one Group Value per group",
                    group=str(group_index + 1),
                )
            )
            continue
        try:
            value = float(raw_values[0])
        except ValueError:
            value = np.nan
        if not np.isfinite(value):
            diagnostics.append(
                ImportDiagnostic(
                    code="dave_q_group_value_invalid",
                    severity=DiagnosticSeverity.WARNING,
                    message="DAVE Group Value must be finite for Q assignment",
                    group=str(group_index + 1),
                )
            )
            continue
        values.append(value)
    if not found_any:
        return None, ()
    if diagnostics or len(values) != len(marker_indices):
        return None, tuple(diagnostics)
    return tuple(values), ()


def _dave_q_bins(
    *,
    entries: tuple[tuple[str, str], ...],
    group_values: tuple[float, ...] | None,
    group_count: int,
) -> tuple[QBins | None, tuple[ImportDiagnostic, ...]]:
    label = _unique_header_value(entries, "group label")
    unit = _unique_header_value(entries, "group units")
    q_metadata_present = (
        label is not None or unit is not None or group_values is not None
    )
    if not q_metadata_present:
        return None, ()
    diagnostics: list[ImportDiagnostic] = []
    if label is None or label.strip().casefold() != "q":
        diagnostics.append(
            ImportDiagnostic(
                code="dave_q_label_invalid",
                severity=DiagnosticSeverity.WARNING,
                message="DAVE Group Label must explicitly identify Q",
            )
        )
    normalized_unit = "" if unit is None else unit.casefold().replace("å", "a")
    if normalized_unit not in {"wavevector:a-1", "a-1", "1/a"}:
        diagnostics.append(
            ImportDiagnostic(
                code="dave_q_unit_invalid",
                severity=DiagnosticSeverity.WARNING,
                message="DAVE Group Units must identify inverse angstrom",
            )
        )
    if group_values is None or len(group_values) != group_count:
        diagnostics.append(
            ImportDiagnostic(
                code="dave_q_values_incomplete",
                severity=DiagnosticSeverity.WARNING,
                message="DAVE Group Values are incomplete; Q was not assigned",
            )
        )
    if diagnostics:
        return None, tuple(diagnostics)
    assert group_values is not None
    return QBins.from_q_values(group_values), ()


def _dave_number_channels_diagnostic(
    entries: tuple[tuple[str, str], ...],
) -> tuple[ImportDiagnostic, ...]:
    reported = _unique_header_value(entries, "number of channels")
    if reported is None:
        return ()
    return (
        ImportDiagnostic(
            code="dave_number_channels_reported",
            severity=DiagnosticSeverity.INFO,
            message=(
                "DAVE Number of Channels was retained as source metadata; "
                "per-group row counts remain authoritative"
            ),
        ),
    )


def _dave_group_count_diagnostic(
    entries: tuple[tuple[str, str], ...],
    actual_count: int,
) -> tuple[ImportDiagnostic, ...]:
    raw = _unique_header_value(entries, "number of groups")
    if raw is None:
        return ()
    try:
        reported = int(raw)
    except ValueError:
        reported = -1
    if reported == actual_count:
        return ()
    return (
        ImportDiagnostic(
            code="dave_group_count_mismatch",
            severity=DiagnosticSeverity.WARNING,
            message=(
                f"DAVE header reports {raw!r} groups but {actual_count} were parsed"
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
) -> ReducedDataset:
    markers = find_group_markers(lines)
    if not markers:
        raise ImportValidationError(detection.diagnostics)

    raw_header_lines = _dave_header_lines(lines, markers[0].line_index)
    header_entries = _header_entries(raw_header_lines)
    source_metadata, metadata_diagnostics = _dave_source_metadata(
        source=source,
        raw_header_lines=raw_header_lines,
    )
    resolved_energy_unit, energy_diagnostics = _dave_energy_unit(
        header_entries,
        energy_unit,
    )
    header_y_unit, y_unit_diagnostics = _dave_y_unit(header_entries)
    if header_y_unit is None:
        resolved_intensity_unit = intensity_unit
        resolved_uncertainty_unit = uncertainty_unit
    else:
        resolved_intensity_unit = header_y_unit
        resolved_uncertainty_unit = header_y_unit

    spectra: list[Spectrum] = []
    source_columns: list[SourceColumnMetadata] = []
    diagnostics: list[ImportDiagnostic] = [
        *detection.diagnostics,
        *energy_diagnostics,
        *y_unit_diagnostics,
        *metadata_diagnostics,
        *_dave_group_count_diagnostic(header_entries, len(markers)),
        *_dave_number_channels_diagnostic(header_entries),
    ]
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
            dave=True,
        )
        matrix, row_numbers, row_diagnostics = _parse_numeric_rows(
            lines,
            header=header,
            end_index=end_index,
            group=marker.label,
        )
        diagnostics.extend(header_diagnostics)
        diagnostics.extend(row_diagnostics)
        if header_diagnostics or row_diagnostics:
            continue

        normalized = normalized_dave_columns(header.columns)
        extras = tuple(
            column
            for column, normalized_column in zip(
                header.columns, normalized, strict=True
            )
            if normalized_column not in _REQUIRED_COLUMNS
        )
        columns = SourceColumnMetadata(
            group_identity=marker.label,
            energy=header.columns[positions["x"]],
            intensity=header.columns[positions["y"]],
            uncertainty=header.columns[positions["yerr"]],
            extra_columns=extras,
            source_row_numbers=row_numbers,
        )
        spectrum = _make_spectrum(
            role=role,
            group_index=group_index,
            group_label=marker.label,
            energy=matrix[:, positions["x"]],
            intensity=matrix[:, positions["y"]],
            uncertainty=matrix[:, positions["yerr"]],
            energy_unit=resolved_energy_unit,
            intensity_unit=resolved_intensity_unit,
            uncertainty_unit=resolved_uncertainty_unit,
        )
        spectra.append(spectrum)
        source_columns.append(columns)
        diagnostics.extend(_invalid_value_diagnostics(spectrum, columns))

    _raise_on_errors(tuple(diagnostics))
    group_values, group_value_diagnostics = _dave_group_values(
        lines,
        tuple(marker.line_index for marker in markers),
    )
    q_bins, q_diagnostics = _dave_q_bins(
        entries=header_entries,
        group_values=group_values,
        group_count=len(spectra),
    )
    diagnostics.extend(group_value_diagnostics)
    diagnostics.extend(q_diagnostics)
    extras = tuple(
        dict.fromkeys(
            column for metadata in source_columns for column in metadata.extra_columns
        )
    )
    diagnostics.extend(_extra_columns_diagnostic(extras))
    return ReducedDataset(
        role=role,
        spectra=tuple(spectra),
        source_reference=source.name,
        source_layout=ReducedDataFormat.DAVE_GROUP_BLOCKS,
        diagnostics=tuple(diagnostics),
        source_columns=tuple(source_columns),
        source_metadata=source_metadata,
        q_bins=q_bins,
    )


def _import_mantid_xye_blocks(
    *,
    lines: tuple[str, ...],
    role: SpectrumRole,
    detection: FormatDetectionResult,
    source: Path,
    energy_unit: str,
    intensity_unit: str,
    uncertainty_unit: str,
) -> ReducedDataset:
    """Import blank-line-separated Mantid-style X,Y,E spectrum blocks."""

    _raise_on_errors(detection.diagnostics)
    blocks = mantid_xye_blocks(lines)
    spectra: list[Spectrum] = []
    source_columns: list[SourceColumnMetadata] = []
    diagnostics = list(detection.diagnostics)
    for group_index, block in enumerate(blocks):
        matrix = np.asarray(
            [tuple(float(token) for token in row.tokens) for row in block],
            dtype=np.float64,
        )
        group_label = str(group_index + 1)
        columns = SourceColumnMetadata(
            group_identity=group_label,
            energy="X",
            intensity="Y",
            uncertainty="E",
            source_row_numbers=tuple(row.line_number for row in block),
        )
        spectrum = _make_spectrum(
            role=role,
            group_index=group_index,
            group_label=group_label,
            energy=matrix[:, 0],
            intensity=matrix[:, 1],
            uncertainty=matrix[:, 2],
            energy_unit=energy_unit,
            intensity_unit=intensity_unit,
            uncertainty_unit=uncertainty_unit,
        )
        spectra.append(spectrum)
        source_columns.append(columns)
        diagnostics.extend(_invalid_value_diagnostics(spectrum, columns))

    if not energy_unit.strip() or energy_unit.strip().casefold() == "unknown":
        diagnostics.append(
            ImportDiagnostic(
                code="mantid_energy_unit_missing",
                severity=DiagnosticSeverity.WARNING,
                message=(
                    "Mantid X,Y,E source does not declare an energy unit; "
                    "manual unit assignment remains required before physical analysis"
                ),
            )
        )
    diagnostics.append(
        ImportDiagnostic(
            code="mantid_q_assignment_missing",
            severity=DiagnosticSeverity.INFO,
            message=(
                "Mantid X,Y,E block order does not define physical Q values; "
                "manual Q assignment remains available"
            ),
        )
    )
    return ReducedDataset(
        role=role,
        spectra=tuple(spectra),
        source_reference=source.name,
        source_layout=ReducedDataFormat.MANTID_XYE_BLOCKS,
        diagnostics=tuple(diagnostics),
        source_columns=tuple(source_columns),
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
) -> ReducedDataset:
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
    source_columns: list[SourceColumnMetadata] = []
    for group_index, suffix in enumerate(analysis.complete_suffixes):
        intensity_position = analysis.intensity_positions[suffix][0]
        uncertainty_position = analysis.uncertainty_positions[suffix][0]
        source_suffix = header.columns[intensity_position][1:]
        columns = SourceColumnMetadata(
            group_identity=source_suffix,
            energy=header.columns[energy_position],
            intensity=header.columns[intensity_position],
            uncertainty=header.columns[uncertainty_position],
            extra_columns=extras,
            source_row_numbers=row_numbers,
        )
        spectrum = _make_spectrum(
            role=role,
            group_index=group_index,
            group_label=source_suffix,
            energy=matrix[:, energy_position],
            intensity=matrix[:, intensity_position],
            uncertainty=matrix[:, uncertainty_position],
            energy_unit=energy_unit,
            intensity_unit=intensity_unit,
            uncertainty_unit=uncertainty_unit,
        )
        spectra.append(spectrum)
        source_columns.append(columns)
        diagnostics.extend(_invalid_value_diagnostics(spectrum, columns))
    diagnostics.extend(_extra_columns_diagnostic(extras))

    return ReducedDataset(
        role=role,
        spectra=tuple(spectra),
        source_reference=source.name,
        source_layout=ReducedDataFormat.WIDE_QENS_TABLE,
        diagnostics=tuple(diagnostics),
        source_columns=tuple(source_columns),
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
) -> ReducedDataset:
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
        list(detection.diagnostics) + list(header_diagnostics) + list(row_diagnostics)
    )
    _raise_on_errors(tuple(diagnostics))

    normalized = normalized_columns(header.columns)
    extras = tuple(
        column
        for column, normalized_column in zip(header.columns, normalized, strict=True)
        if normalized_column not in _REQUIRED_COLUMNS
    )
    columns = SourceColumnMetadata(
        group_identity="spectrum",
        energy=header.columns[positions["x"]],
        intensity=header.columns[positions["y"]],
        uncertainty=header.columns[positions["yerr"]],
        extra_columns=extras,
        source_row_numbers=row_numbers,
    )
    spectrum = _make_spectrum(
        role=role,
        group_index=0,
        group_label="spectrum",
        energy=matrix[:, positions["x"]],
        intensity=matrix[:, positions["y"]],
        uncertainty=matrix[:, positions["yerr"]],
        energy_unit=energy_unit,
        intensity_unit=intensity_unit,
        uncertainty_unit=uncertainty_unit,
    )
    diagnostics.extend(_invalid_value_diagnostics(spectrum, columns))
    diagnostics.extend(_extra_columns_diagnostic(extras))
    return ReducedDataset(
        role=role,
        spectra=(spectrum,),
        source_reference=source.name,
        source_layout=ReducedDataFormat.SINGLE_SPECTRUM_TABLE,
        diagnostics=tuple(diagnostics),
        source_columns=(columns,),
    )


def import_reduced_data(
    path: str | Path,
    *,
    role: SpectrumRole | str,
    explicit_format: ReducedDataFormat | str | None = None,
    energy_unit: str = "unknown",
    intensity_unit: str = "unknown",
    uncertainty_unit: str = "unknown",
) -> ReducedDataset:
    """Import supported reduced text data without changing values or inferring Q."""

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

    importers = {
        ReducedDataFormat.DAVE_GROUP_BLOCKS: _import_dave_groups,
        ReducedDataFormat.MANTID_XYE_BLOCKS: _import_mantid_xye_blocks,
        ReducedDataFormat.WIDE_QENS_TABLE: _import_wide_table,
        ReducedDataFormat.SINGLE_SPECTRUM_TABLE: _import_single_table,
    }
    return importers[detection.proposed_format](
        lines=lines,
        role=selected_role,
        detection=detection,
        source=source,
        energy_unit=energy_unit,
        intensity_unit=intensity_unit,
        uncertainty_unit=uncertainty_unit,
    )
