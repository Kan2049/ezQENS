"""GUI-independent preprocessing services."""

from ezqens.preprocessing.edge_padding import (
    BoundaryPaddingResult,
    BoundarySide,
    EdgePaddingDetectionResult,
    PaddingStatus,
    SpectrumPaddingResult,
    detect_edge_padding,
)
from ezqens.preprocessing.fitting_selection import FittingRange, FittingSelection

__all__ = [
    "BoundaryPaddingResult",
    "BoundarySide",
    "EdgePaddingDetectionResult",
    "FittingRange",
    "FittingSelection",
    "PaddingStatus",
    "SpectrumPaddingResult",
    "detect_edge_padding",
]
