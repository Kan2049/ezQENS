"""Structured, privacy-safe diagnostics for reduced-data import."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class DiagnosticSeverity(StrEnum):
    """Severity of an import or format-detection diagnostic."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ImportDiagnostic:
    """One structured diagnostic without embedded scientific arrays."""

    code: str
    severity: DiagnosticSeverity
    message: str
    group: str | None = None
    row: int | None = None
    column: str | None = None


class ImportValidationError(ValueError):
    """Raised when reduced scientific data cannot be imported safely."""

    diagnostics: Final[tuple[ImportDiagnostic, ...]]

    def __init__(self, diagnostics: tuple[ImportDiagnostic, ...]) -> None:
        if not diagnostics:
            raise ValueError("ImportValidationError requires at least one diagnostic")
        self.diagnostics = diagnostics
        summary = "; ".join(
            f"{diagnostic.code}: {diagnostic.message}"
            for diagnostic in diagnostics
            if diagnostic.severity is DiagnosticSeverity.ERROR
        )
        if not summary:
            summary = "; ".join(
                f"{diagnostic.code}: {diagnostic.message}" for diagnostic in diagnostics
            )
        super().__init__(summary)
