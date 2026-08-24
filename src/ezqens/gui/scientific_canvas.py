"""Matplotlib-backed scientific canvas shell."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

SCIENTIFIC_BACKGROUND = "#ffffff"


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
