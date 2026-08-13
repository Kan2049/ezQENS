"""GUI-independent single-Q spectral fitting and candidate evidence."""

from ezqens.fitting.core import (
    FittingError,
    evaluate_spectral_model,
    evaluate_standard_candidates,
    fit_single_q,
    fit_standard_candidate,
    generate_standard_candidates,
)
from ezqens.fitting.models import (
    AlternativeStartResult,
    BackgroundModel,
    CandidateFitResult,
    FitDiagnostics,
    FitProvenance,
    FitResult,
    FitStatistics,
    LorentzianComponent,
    ModelEvaluation,
    ParameterConfiguration,
    ParameterEstimate,
    ResidualDiagnostics,
    SpectralModelDefinition,
    StandardModelCandidate,
)

__all__ = [
    "AlternativeStartResult",
    "BackgroundModel",
    "CandidateFitResult",
    "FitDiagnostics",
    "FitProvenance",
    "FitResult",
    "FitStatistics",
    "FittingError",
    "LorentzianComponent",
    "ModelEvaluation",
    "ParameterConfiguration",
    "ParameterEstimate",
    "ResidualDiagnostics",
    "SpectralModelDefinition",
    "StandardModelCandidate",
    "evaluate_spectral_model",
    "evaluate_standard_candidates",
    "fit_single_q",
    "fit_standard_candidate",
    "generate_standard_candidates",
]
