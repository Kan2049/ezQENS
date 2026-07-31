"""Minimal scientific domain API for reduced-data import."""

from qensfit.domain.diagnostics import (
    DiagnosticSeverity,
    ImportDiagnostic,
    ImportValidationError,
)
from qensfit.domain.models import (
    DetectionConfidence,
    FormatDetectionResult,
    ImportedDataset,
    ImportSummary,
    InvalidValueCounts,
    ReducedDataFormat,
    SourceColumnMetadata,
    Spectrum,
    SpectrumRole,
)

__all__ = [
    "DetectionConfidence",
    "DiagnosticSeverity",
    "FormatDetectionResult",
    "ImportDiagnostic",
    "ImportedDataset",
    "ImportSummary",
    "ImportValidationError",
    "InvalidValueCounts",
    "ReducedDataFormat",
    "SourceColumnMetadata",
    "Spectrum",
    "SpectrumRole",
]
