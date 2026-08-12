"""Measured-resolution preparation for later numerical convolution."""

from ezqens.resolution.preparation import (
    NORMALIZATION_METHOD,
    PADDING_BOUNDARY_ATOL,
    PADDING_BOUNDARY_RTOL,
    Q_MATCH_ATOL,
    Q_MATCH_RTOL,
    PreparedResolution,
    PreparedResolutionSpectrum,
    ResolutionDiagnostic,
    ResolutionPaddingComparison,
    ResolutionPreparationError,
    ResolutionSupport,
    ResolutionSupportSource,
    prepare_measured_resolution,
)

__all__ = [
    "NORMALIZATION_METHOD",
    "PADDING_BOUNDARY_ATOL",
    "PADDING_BOUNDARY_RTOL",
    "Q_MATCH_ATOL",
    "Q_MATCH_RTOL",
    "PreparedResolution",
    "PreparedResolutionSpectrum",
    "ResolutionDiagnostic",
    "ResolutionPaddingComparison",
    "ResolutionPreparationError",
    "ResolutionSupport",
    "ResolutionSupportSource",
    "prepare_measured_resolution",
]
