"""Minimal scientific domain API for reduced-data import."""

from ezqens.domain.diagnostics import (
    DiagnosticSeverity,
    ImportDiagnostic,
    ImportValidationError,
)
from ezqens.domain.models import (
    FormatDetectionResult,
    QBins,
    ReducedDataFormat,
    ReducedDataset,
    SourceMetadata,
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
    "SourceMetadata",
    "uniform_q_bins",
]
