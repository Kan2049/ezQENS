"""Tests for minimal Milestone 1 scientific domain models."""

from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from qensfit.domain import (
    DetectionConfidence,
    FormatDetectionResult,
    ImportedDataset,
    ReducedDataFormat,
    SourceColumnMetadata,
    Spectrum,
    SpectrumRole,
)


def make_spectrum(
    *,
    role: SpectrumRole = SpectrumRole.SAMPLE,
    energy: np.ndarray | None = None,
    intensity: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
) -> Spectrum:
    """Build a compact spectrum for validation tests."""

    energy_values = np.array([-1.0, 0.0, 1.0]) if energy is None else energy
    intensity_values = (
        np.array([2.0, 3.0, 2.5]) if intensity is None else intensity
    )
    uncertainty_values = (
        np.array([0.1, 0.2, 0.1]) if uncertainty is None else uncertainty
    )
    return Spectrum.from_imported_arrays(
        role=role,
        group_index=0,
        group_label="test",
        energy=energy_values,
        intensity=intensity_values,
        uncertainty=uncertainty_values,
        energy_unit="meV",
        intensity_unit="counts",
        uncertainty_unit="counts",
        source_row_numbers=tuple(range(2, 2 + len(energy_values))),
        source_columns=SourceColumnMetadata("x", "y", "yerr"),
        source_layout=ReducedDataFormat.SINGLE_SPECTRUM_TABLE,
    )


def test_equal_length_arrays_are_required() -> None:
    with pytest.raises(ValueError, match="equal length"):
        make_spectrum(intensity=np.array([1.0, 2.0]))


def test_nonempty_arrays_are_required() -> None:
    empty = np.array([], dtype=np.float64)

    with pytest.raises(ValueError, match="must not be empty"):
        make_spectrum(energy=empty, intensity=empty, uncertainty=empty)


def test_invalid_mask_shape_is_validated() -> None:
    spectrum = make_spectrum()

    with pytest.raises(ValueError, match="one-dimensional"):
        replace(spectrum, invalid_energy_mask=np.zeros((1, 3), dtype=np.bool_))


def test_role_is_validated_at_runtime() -> None:
    invalid_role = cast(SpectrumRole, "invalid-role")

    with pytest.raises(ValueError, match="SpectrumRole"):
        make_spectrum(role=invalid_role)


def test_invalid_values_are_retained_and_classified() -> None:
    spectrum = make_spectrum(
        energy=np.array([np.nan, 0.0, 1.0, 2.0]),
        intensity=np.array([1.0, np.inf, 3.0, 4.0]),
        uncertainty=np.array([np.nan, 0.0, -1.0, np.inf]),
    )

    assert np.isnan(spectrum.energy[0])
    assert np.isinf(spectrum.intensity[1])
    assert np.isnan(spectrum.uncertainty[0])
    assert spectrum.uncertainty[1] == 0.0
    assert spectrum.uncertainty[2] == -1.0
    np.testing.assert_array_equal(
        spectrum.invalid_energy_mask,
        [True, False, False, False],
    )
    np.testing.assert_array_equal(
        spectrum.invalid_intensity_mask,
        [False, True, False, False],
    )
    np.testing.assert_array_equal(
        spectrum.invalid_uncertainty_mask,
        [True, True, True, True],
    )


def test_imported_arrays_are_read_only() -> None:
    spectrum = make_spectrum()

    with pytest.raises(ValueError):
        spectrum.energy[0] = 99.0


def test_dataset_role_must_match_spectrum_role() -> None:
    spectrum = make_spectrum(role=SpectrumRole.SAMPLE)
    detection = FormatDetectionResult(
        proposed_format=ReducedDataFormat.SINGLE_SPECTRUM_TABLE,
        confidence=DetectionConfidence.HIGH,
        evidence=("test",),
        detected_required_columns=("x", "y", "yerr"),
        detected_extra_columns=(),
        detected_count=1,
    )

    with pytest.raises(ValueError, match="roles must match"):
        ImportedDataset(
            role=SpectrumRole.RESOLUTION,
            source_layout=ReducedDataFormat.SINGLE_SPECTRUM_TABLE,
            spectra=(spectrum,),
            source_reference="synthetic.txt",
            diagnostics=(),
            detected_extra_columns=(),
            shared_energy_grid=True,
            shared_energy_axis=spectrum.energy,
            format_detection=detection,
        )


def test_spectrum_repr_does_not_include_arrays() -> None:
    representation = repr(make_spectrum())

    assert "array(" not in representation
    assert "2.5" not in representation
    assert "source_row_numbers" not in representation

