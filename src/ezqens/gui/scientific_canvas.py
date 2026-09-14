"""Matplotlib-backed scientific canvas shell."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

SCIENTIFIC_BACKGROUND = "#ffffff"
SCIENTIFIC_MEASURED_COLOR = "#292928"
SCIENTIFIC_UNCERTAINTY_COLOR = "#727272"
SCIENTIFIC_TOTAL_FIT_COLOR = "#314b63"
SCIENTIFIC_ELASTIC_COLOR = "#1f77b4"
SCIENTIFIC_LORENTZIAN_COLORS = ("#ff7f0e", "#2ca02c")
SCIENTIFIC_BACKGROUND_COLOR = "#8a7564"
SCIENTIFIC_RESIDUAL_COLOR = "#496d91"


def log_display_values(values: np.ndarray) -> np.ndarray:
    """Mask nonpositive values only in the Matplotlib Log presentation layer."""

    return np.ma.masked_where(~np.isfinite(values) | (values <= 0.0), values)


def symlog_linthresh(values: np.ndarray) -> float:
    """Choose the shared display-only central linear band for signed plot data."""

    finite = np.abs(values[np.isfinite(values)])
    if not finite.size:
        return 1.0
    return max(float(np.max(finite)) * 0.01, np.finfo(np.float64).tiny)


def zoom_limits(
    limits: tuple[float, float],
    cursor: float,
    scale: float,
) -> tuple[float, float]:
    """Scale display limits around a pointer without changing scientific state."""

    lower, upper = limits
    return (
        cursor - (cursor - lower) * scale,
        cursor + (upper - cursor) * scale,
    )


def create_empty_figure() -> Figure:
    """Create an empty scientific figure without placeholder analysis data."""
    return Figure(facecolor=SCIENTIFIC_BACKGROUND, layout="constrained")


class ScientificCanvas(FigureCanvasQTAgg):
    """Real embedded Matplotlib canvas reserved for scientific content."""

    def __init__(self) -> None:
        super().__init__(create_empty_figure())  # type: ignore[no-untyped-call]
        self.setObjectName("scientificCanvas")
        self.setMinimumSize(320, 240)
        self.figure.set_facecolor(SCIENTIFIC_BACKGROUND)
