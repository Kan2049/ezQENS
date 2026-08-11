"""Minimal scientific domain API for reduced-data import."""

from qensfit.domain.diagnostics import (
    DiagnosticSeverity,
    ImportDiagnostic,
    ImportValidationError,
)
from qensfit.domain.models import (
    FormatDetectionResult,
    QBins,
    ReducedDataFormat,
    ReducedDataset,
    Spectrum,
    SpectrumRole,
    uniform_q_bins,
)

__all__ = [
    "DiagnosticSeverity",
    "FormatDetectionResult",
    "ImportDiagnostic",
    "ImportValidationError",
    "QBins",
    "ReducedDataset",
    "ReducedDataFormat",
    "Spectrum",
    "SpectrumRole",
    "uniform_q_bins",
]
