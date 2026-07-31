"""Print privacy-safe structural summaries of local reduced-data files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from qensfit.domain import ImportValidationError
from qensfit.io.importers import (
    detect_reduced_data_format,
    import_reduced_data,
)
from qensfit.preprocessing import detect_edge_padding

_SEVERITIES = ("info", "warning", "error")


def _diagnostic_counts(diagnostics: tuple[Any, ...]) -> dict[str, int]:
    counts = Counter(str(diagnostic.severity) for diagnostic in diagnostics)
    return {severity: counts.get(severity, 0) for severity in _SEVERITIES}


def _empty_dataset_fields() -> dict[str, object]:
    return {
        "number_of_spectra": None,
        "group_identities": [],
        "row_counts": [],
        "shared_energy_grid": None,
        "detected_ignored_extra_columns": [],
        "finite_energy_ranges": [],
        "invalid_value_counts": [],
        "edge_padding_algorithm_version": None,
        "edge_padding_by_spectrum": [],
        "total_auto_padding_mask_count": None,
    }


def _inspect_file(path: Path, *, role: str) -> tuple[dict[str, object], bool]:
    report: dict[str, object] = {
        "source_basename": path.name,
        "dataset_role": role,
    }

    try:
        detection = detect_reduced_data_format(path)
    except ImportValidationError as error:
        report.update(
            {
                "proposed_detected_format": "unavailable",
                "detection_confidence": "none",
                "detection_evidence": [],
                "detected_required_columns": [],
                **_empty_dataset_fields(),
                "diagnostic_counts_by_severity": _diagnostic_counts(
                    error.diagnostics
                ),
                "inspection_status": "failed",
            }
        )
        return report, False

    report.update(
        {
            "proposed_detected_format": detection.proposed_format.value,
            "detection_confidence": detection.confidence.value,
            "detection_evidence": list(detection.evidence),
            "detected_required_columns": list(
                detection.detected_required_columns
            ),
        }
    )

    try:
        dataset = import_reduced_data(path, role=role)
    except ImportValidationError as error:
        report.update(
            {
                **_empty_dataset_fields(),
                "detected_ignored_extra_columns": list(
                    detection.detected_extra_columns
                ),
                "diagnostic_counts_by_severity": _diagnostic_counts(
                    error.diagnostics
                ),
                "inspection_status": "failed",
            }
        )
        return report, False
    except Exception as error:
        report.update(
            {
                **_empty_dataset_fields(),
                "detected_ignored_extra_columns": list(
                    detection.detected_extra_columns
                ),
                "diagnostic_counts_by_severity": {
                    "info": 0,
                    "warning": 0,
                    "error": 1,
                },
                "inspection_status": f"failed:{type(error).__name__}",
            }
        )
        return report, False

    summary = dataset.structural_summary()
    padding_detection = detect_edge_padding(dataset)
    padding_summary = padding_detection.structural_summary(dataset)
    report.update(
        {
            "number_of_spectra": summary.spectrum_count,
            "group_identities": [
                spectrum.group_label for spectrum in dataset.spectra
            ],
            "row_counts": list(summary.row_counts),
            "shared_energy_grid": summary.shared_energy_grid,
            "detected_ignored_extra_columns": list(
                summary.detected_extra_columns
            ),
            "finite_energy_ranges": [
                {
                    "group_identity": spectrum.group_label,
                    "minimum": energy_range[0],
                    "maximum": energy_range[1],
                }
                for spectrum, energy_range in zip(
                    dataset.spectra,
                    summary.finite_energy_ranges,
                    strict=True,
                )
            ],
            "invalid_value_counts": [
                {
                    "group_identity": spectrum.group_label,
                    "energy": counts.energy,
                    "intensity": counts.intensity,
                    "uncertainty": counts.uncertainty,
                }
                for spectrum, counts in zip(
                    dataset.spectra,
                    summary.invalid_value_counts,
                    strict=True,
                )
            ],
            "diagnostic_counts_by_severity": _diagnostic_counts(
                dataset.diagnostics
            ),
            "edge_padding_algorithm_version": (
                padding_summary.algorithm_version
            ),
            "edge_padding_by_spectrum": [
                {
                    "group_identity": item.group_identity,
                    "left_auto_masked_point_count": (
                        item.left_auto_masked_point_count
                    ),
                    "right_auto_masked_point_count": (
                        item.right_auto_masked_point_count
                    ),
                    "derived_valid_energy_range": {
                        "minimum": item.derived_valid_energy_range[0],
                        "maximum": item.derived_valid_energy_range[1],
                    },
                    "confidence": item.confidence.value,
                    "evidence_codes": list(item.evidence_codes),
                    "total_auto_padding_mask_count": (
                        item.total_auto_padding_mask_count
                    ),
                }
                for item in padding_summary.spectra
            ],
            "total_auto_padding_mask_count": (
                padding_summary.total_auto_padding_mask_count
            ),
            "inspection_status": "ok",
        }
    )
    return report, True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect local reduced-data files without printing scientific arrays "
            "or absolute source paths."
        )
    )
    parser.add_argument(
        "--role",
        required=True,
        choices=("sample", "resolution"),
        help="Scientific role assigned to every input dataset.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more local reduced-data files.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    succeeded = True
    for path in args.paths:
        report, file_succeeded = _inspect_file(path, role=args.role)
        print(json.dumps(report, indent=2, allow_nan=False))
        succeeded = succeeded and file_succeeded
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
