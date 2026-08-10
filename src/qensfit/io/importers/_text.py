"""Shared line-oriented helpers for reduced-data text import."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from qensfit.domain import (
    DiagnosticSeverity,
    ImportDiagnostic,
    ImportValidationError,
)

_GROUP_PATTERN = re.compile(r"^\s*#\s*group\s+(.+?)\s*$", re.IGNORECASE)
_WIDE_INTENSITY_PATTERN = re.compile(r"^y(\d+)$", re.IGNORECASE)
_WIDE_UNCERTAINTY_PATTERN = re.compile(r"^yerr(\d+)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class TextHeader:
    """A detected source header and its one-based line number."""

    line_index: int
    line_number: int
    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GroupMarker:
    """A DAVE-style group marker."""

    line_index: int
    line_number: int
    label: str


@dataclass(frozen=True, slots=True)
class WideColumnAnalysis:
    """Positions and suffixes found in a candidate wide-table header."""

    energy_positions: tuple[int, ...]
    intensity_positions: dict[int, tuple[int, ...]]
    uncertainty_positions: dict[int, tuple[int, ...]]
    extra_positions: tuple[int, ...]

    @property
    def all_suffixes(self) -> tuple[int, ...]:
        """Return all suffixes in ascending numerical order."""

        return tuple(
            sorted(set(self.intensity_positions) | set(self.uncertainty_positions))
        )

    @property
    def complete_suffixes(self) -> tuple[int, ...]:
        """Return suffixes with exactly one y and one yerr column."""

        return tuple(
            suffix
            for suffix in self.all_suffixes
            if len(self.intensity_positions.get(suffix, ())) == 1
            and len(self.uncertainty_positions.get(suffix, ())) == 1
        )


def read_text_lines(path: Path) -> tuple[str, ...]:
    """Read UTF-8 text without including source contents in failures."""

    try:
        return tuple(path.read_text(encoding="utf-8-sig").splitlines())
    except (OSError, UnicodeError) as error:
        raise ImportValidationError(
            (
                ImportDiagnostic(
                    code="source_read_failed",
                    severity=DiagnosticSeverity.ERROR,
                    message=(
                        f"Could not read reduced-data source: {type(error).__name__}"
                    ),
                ),
            )
        ) from error


def split_columns(line: str) -> tuple[str, ...]:
    """Split a whitespace- or comma-delimited line into source tokens."""

    stripped = line.strip()
    if stripped.startswith("#"):
        stripped = stripped[1:].strip()
    if not stripped:
        return ()
    return tuple(token for token in re.split(r"[\s,]+", stripped) if token)


def normalized_columns(columns: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize column names for matching while retaining originals elsewhere."""

    return tuple(column.casefold() for column in columns)


def find_group_markers(lines: tuple[str, ...]) -> tuple[GroupMarker, ...]:
    """Return DAVE-style group markers in source order."""

    markers: list[GroupMarker] = []
    for line_index, line in enumerate(lines):
        match = _GROUP_PATTERN.match(line)
        if match is not None:
            markers.append(
                GroupMarker(
                    line_index=line_index,
                    line_number=line_index + 1,
                    label=match.group(1).strip(),
                )
            )
    return tuple(markers)


def _looks_like_header(columns: tuple[str, ...]) -> bool:
    normalized = normalized_columns(columns)
    if any(column in {"x", "y", "yerr"} for column in normalized):
        return True
    return any(
        _WIDE_INTENSITY_PATTERN.fullmatch(column) is not None
        or _WIDE_UNCERTAINTY_PATTERN.fullmatch(column) is not None
        for column in normalized
    )


def find_table_header(
    lines: tuple[str, ...],
    *,
    start_index: int = 0,
    end_index: int | None = None,
) -> TextHeader | None:
    """Find the first plausible table header in a line range."""

    stop = len(lines) if end_index is None else end_index
    for line_index in range(start_index, stop):
        line = lines[line_index]
        stripped = line.strip()
        if not stripped or _GROUP_PATTERN.match(line):
            continue
        columns = split_columns(line)
        if not columns:
            continue
        if stripped.startswith("#") and not _looks_like_header(columns):
            continue
        if _looks_like_header(columns):
            return TextHeader(
                line_index=line_index,
                line_number=line_index + 1,
                columns=columns,
            )
        if not stripped.startswith("#"):
            return TextHeader(
                line_index=line_index,
                line_number=line_index + 1,
                columns=columns,
            )
    return None


def analyze_wide_columns(columns: tuple[str, ...]) -> WideColumnAnalysis:
    """Analyze shared-energy and paired wide-table columns."""

    normalized = normalized_columns(columns)
    energy_positions = tuple(
        index for index, column in enumerate(normalized) if column == "x"
    )
    intensity: dict[int, list[int]] = {}
    uncertainty: dict[int, list[int]] = {}
    recognized: set[int] = set(energy_positions)
    for index, column in enumerate(normalized):
        intensity_match = _WIDE_INTENSITY_PATTERN.fullmatch(column)
        uncertainty_match = _WIDE_UNCERTAINTY_PATTERN.fullmatch(column)
        if intensity_match is not None:
            suffix = int(intensity_match.group(1))
            intensity.setdefault(suffix, []).append(index)
            recognized.add(index)
        elif uncertainty_match is not None:
            suffix = int(uncertainty_match.group(1))
            uncertainty.setdefault(suffix, []).append(index)
            recognized.add(index)
    return WideColumnAnalysis(
        energy_positions=energy_positions,
        intensity_positions={
            suffix: tuple(positions) for suffix, positions in intensity.items()
        },
        uncertainty_positions={
            suffix: tuple(positions) for suffix, positions in uncertainty.items()
        },
        extra_positions=tuple(
            index for index in range(len(columns)) if index not in recognized
        ),
    )
