"""Minimal scientific domain API for reduced-data import."""

from qensfit.domain.diagnostics import (
    DiagnosticSeverity,
    ImportDiagnostic,
    ImportValidationError,
)
from qensfit.domain.models import (
    FormatDetectionResult,
    ReducedDataFormat,
    ReducedDataset,
    Spectrum,
    SpectrumRole,
)

__all__ = [
    "DiagnosticSeverity",
    "FormatDetectionResult",
    "ImportDiagnostic",
    "ImportValidationError",
    "ReducedDataset",
    "ReducedDataFormat",
    "Spectrum",
    "SpectrumRole",
]
