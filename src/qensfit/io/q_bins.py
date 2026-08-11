"""Import the owner-approved four-value DAVE Q-bin parameter format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qensfit.domain import (
    DiagnosticSeverity,
    ImportDiagnostic,
    ImportValidationError,
    QBins,
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


def _fail(code: str, message: str, *, row: int | None = None) -> None:
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


def _complete_bin_count(lower_limit: float, upper_limit: float, step: float) -> int:
    """Return the number of complete fixed-width bins within the DAVE limits."""

    quotient = (upper_limit - lower_limit) / step
    group_count = int(np.floor(quotient))

    def edge_tolerance(candidate_count: int) -> float:
        candidate_edge = lower_limit + candidate_count * step
        roundoff = (
            np.finfo(np.float64).eps
            * max(1.0, abs(lower_limit), abs(upper_limit), abs(candidate_edge))
            * 16.0
        )
        # Never let coordinate-scale roundoff become a material fraction of a
        # Q bin. Normal binary noise is far below this local one-step cap.
        one_step_cap = float(np.sqrt(np.finfo(np.float64).eps) * abs(step))
        return min(roundoff, one_step_cap)

    # Confirm the complete-edge condition directly. At most the bin adjacent
    # to floor(span / step) can be affected by ordinary floating-point noise.
    if (
        group_count > 0
        and lower_limit + group_count * step > upper_limit + edge_tolerance(group_count)
    ):
        group_count -= 1
    next_count = group_count + 1
    if lower_limit + next_count * step <= upper_limit + edge_tolerance(next_count):
        group_count += 1
    return group_count


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
    actual_group_count = _complete_bin_count(lower_limit, upper_limit, step)
    if actual_group_count < 1:
        _fail(
            "dave_q_bins_no_complete_bins",
            "DAVE Q limits and step do not define a complete Q bin",
        )
    edges = lower_limit + step * np.arange(actual_group_count + 1)
    q_bins = QBins.from_edges(edges)

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
