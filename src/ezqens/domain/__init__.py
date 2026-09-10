"""Minimal scientific domain API for reduced-data import."""

from ezqens.domain.diagnostics import (
    DiagnosticSeverity,
    ImportDiagnostic,
    ImportValidationError,
)
from ezqens.domain.models import (
    FormatDetectionResult,
    FractionalCoverage,
    FractionalCoverageAvailability,
    FractionalCoverageOrigin,
    QBins,
    QExclusionInterval,
    QRebinSpecification,
    ReducedDataFormat,
    ReducedDataset,
    SourceMetadata,
    Spectrum,
    SpectrumRole,
    fixed_width_q_bins,
    uniform_q_bins,
)

__all__ = [
    "DiagnosticSeverity",
    "FractionalCoverage",
    "FractionalCoverageAvailability",
    "FractionalCoverageOrigin",
    "FormatDetectionResult",
    "ImportDiagnostic",
    "ImportValidationError",
    "QExclusionInterval",
    "QBins",
    "QRebinSpecification",
    "ReducedDataset",
    "ReducedDataFormat",
    "Spectrum",
    "SpectrumRole",
    "SourceMetadata",
    "fixed_width_q_bins",
    "uniform_q_bins",
]
