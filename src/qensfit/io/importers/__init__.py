"""Minimal public entry points for Milestone 1 reduced-data import."""

from qensfit.io.importers.detection import detect_reduced_data_format
from qensfit.io.importers.reduced_data import import_reduced_data

__all__ = ["detect_reduced_data_format", "import_reduced_data"]

