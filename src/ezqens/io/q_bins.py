"""Import the owner-approved four-value DAVE Q-bin parameter format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import numpy as np

from ezqens.domain import (
    DiagnosticSeverity,
    ImportDiagnostic,
    ImportValidationError,
    QBins,
    fixed_width_q_bins,
)


@dataclass(frozen=True, slots=True)
class DAVEQBinsResult:
    """Reconstructed Q bins plus the original DAVE source parameters."""

    q_bins: QBins
    lower_limit: float
    upper_limit: float
    step: float
    reported_group_count: int
    diagnostics: tuple[ImportDiagnostic, ...] = ()


def _fail(code: str, message: str, *, row: int | None = None) -> NoReturn:
    raise ImportValidationError(
        (
            ImportDiagnostic(
                code=code,
                severity=DiagnosticSeverity.ERROR,
                message=message,
                row=row,
            ),
        )
    )


def _read_four_values(path: Path) -> tuple[float, ...]:
    lines = tuple(
        (row, line.strip())
        for row, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if line.strip()
    )
    if len(lines) != 4 or any(len(line.split()) != 1 for _, line in lines):
        _fail(
            "dave_q_bins_expected_four_values",
            "DAVE Q-bin parameters require exactly four nonblank numeric lines",
        )

    values: list[float] = []
    for row, token in lines:
        try:
            value = float(token)
        except ValueError:
            _fail(
                "dave_q_bins_value_not_numeric",
                "DAVE Q-bin parameter must be numeric",
                row=row,
            )
        if not np.isfinite(value):
            _fail(
                "dave_q_bins_value_not_finite",
                "DAVE Q-bin parameter must be finite",
                row=row,
            )
        values.append(value)
    return tuple(values)


def parse_dave_q_bins(path: str | Path) -> DAVEQBinsResult:
    """Parse the supported four-line DAVE Q-bin parameter file.

    Complete fixed-width bins are reconstructed from the lower limit, upper
    limit, and step. The stored group count is retained and checked afterward.
    """

    lower_limit, upper_limit, raw_group_count, step = _read_four_values(Path(path))
    if not raw_group_count.is_integer() or raw_group_count < 1.0:
        _fail(
            "dave_q_bins_group_count_invalid",
            "DAVE Q-bin group count must be a positive integer",
            row=3,
        )
    if lower_limit >= upper_limit:
        _fail(
            "dave_q_bins_edge_order_invalid",
            "DAVE lower Q limit must be less than the upper Q limit",
        )
    if step <= 0.0:
        _fail(
            "dave_q_bins_step_invalid",
            "DAVE Q-bin step must be positive",
            row=4,
        )

    reported_group_count = int(raw_group_count)
    try:
        q_bins = fixed_width_q_bins(
            lower_q_edge=lower_limit,
            upper_q_limit=upper_limit,
            step=step,
        )
    except ValueError as error:
        _fail(
            "dave_q_bins_no_complete_bins",
            f"DAVE Q limits and step do not define valid complete Q bins: {error}",
        )
    actual_group_count = q_bins.group_count

    diagnostics: tuple[ImportDiagnostic, ...] = ()
    if reported_group_count != actual_group_count:
        diagnostics = (
            ImportDiagnostic(
                code="dave_q_bins_group_count_mismatch",
                severity=DiagnosticSeverity.WARNING,
                message=(
                    "Reported DAVE group count differs from the complete bins "
                    "reconstructed from the source limits and step"
                ),
                row=3,
            ),
        )
    return DAVEQBinsResult(
        q_bins=q_bins,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        step=step,
        reported_group_count=reported_group_count,
        diagnostics=diagnostics,
    )
