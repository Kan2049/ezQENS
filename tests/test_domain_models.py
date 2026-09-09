"""Tests for the reduced scientific domain boundary."""

from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from ezqens.domain import (
    FractionalCoverage,
    FractionalCoverageAvailability,
    FractionalCoverageOrigin,
    QBins,
    ReducedDataset,
    SourceMetadata,
    Spectrum,
    SpectrumRole,
)


def make_spectrum(
    *,
    role: SpectrumRole = SpectrumRole.SAMPLE,
    group_index: int = 0,
    energy: np.ndarray | None = None,
    intensity: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
) -> Spectrum:
    """Build a compact source-independent spectrum."""

    energy_values = np.array([-1.0, 0.0, 1.0]) if energy is None else energy
    intensity_values = np.array([2.0, 3.0, 2.5]) if intensity is None else intensity
    uncertainty_values = (
        np.array([0.1, 0.2, 0.1]) if uncertainty is None else uncertainty
    )
    return Spectrum(
        role=role,
        group_index=group_index,
        group_label=f"group-{group_index}",
        energy=energy_values,
        intensity=intensity_values,
        uncertainty=uncertainty_values,
        energy_unit="meV",
        intensity_unit="counts",
        uncertainty_unit="counts",
    )


def test_equal_length_arrays_are_required() -> None:
    with pytest.raises(ValueError, match="equal length"):
        make_spectrum(intensity=np.array([1.0, 2.0]))


def test_nonempty_arrays_are_required() -> None:
    empty = np.array([], dtype=np.float64)

    with pytest.raises(ValueError, match="must not be empty"):
        make_spectrum(energy=empty, intensity=empty, uncertainty=empty)


def test_arrays_must_be_one_dimensional() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        make_spectrum(energy=np.zeros((1, 3)))


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


def test_values_and_masks_are_read_only() -> None:
    spectrum = make_spectrum()

    for array in (
        spectrum.energy,
        spectrum.intensity,
        spectrum.uncertainty,
        spectrum.invalid_energy_mask,
        spectrum.invalid_intensity_mask,
        spectrum.invalid_uncertainty_mask,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = 0


def test_dataset_role_must_match_spectrum_role() -> None:
    spectrum = make_spectrum(role=SpectrumRole.SAMPLE)

    with pytest.raises(ValueError, match="roles must match"):
        ReducedDataset(
            role=SpectrumRole.RESOLUTION,
            spectra=(spectrum,),
        )


def test_dataset_grid_identity_is_derived() -> None:
    first = make_spectrum(group_index=0)
    matching = make_spectrum(group_index=1)
    unequal = make_spectrum(group_index=1, energy=np.array([-2.0, 0.0, 2.0]))

    assert ReducedDataset(
        role=SpectrumRole.SAMPLE, spectra=(first, matching)
    ).shared_energy_grid
    assert not ReducedDataset(
        role=SpectrumRole.SAMPLE, spectra=(first, unequal)
    ).shared_energy_grid


def test_spectrum_is_independent_of_text_import_metadata() -> None:
    spectrum = make_spectrum()

    assert not hasattr(spectrum, "source_columns")
    assert not hasattr(spectrum, "source_row_numbers")
    assert not hasattr(spectrum, "source_layout")


def test_spectrum_repr_does_not_include_arrays() -> None:
    representation = repr(make_spectrum())

    assert "array(" not in representation
    assert "2.5" not in representation


def test_source_metadata_rejects_invalid_promoted_physical_values() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        SourceMetadata(source_filename="synthetic.dat", temperature_kelvin=-1.0)
    with pytest.raises(ValueError, match="positive"):
        SourceMetadata(source_filename="synthetic.dat", wavelength_angstrom=0.0)


def test_fractional_coverage_is_validated_against_spectrum_alignment() -> None:
    spectra = (make_spectrum(group_index=0), make_spectrum(group_index=1))
    mismatched = FractionalCoverage.aligned_with(
        spectra,
        values=(np.ones(3), np.ones(2)),
        origin=FractionalCoverageOrigin.PROPAGATED_Q_REBIN,
    )

    with pytest.raises(ValueError, match="align with every spectrum point"):
        ReducedDataset(
            role=SpectrumRole.SAMPLE,
            spectra=spectra,
            fractional_coverage=mismatched,
        )


def test_propagated_fractional_coverage_remains_nonunit_and_available() -> None:
    spectra = (make_spectrum(group_index=0), make_spectrum(group_index=1))
    values = (np.array([1.0, 0.75, 0.25]), np.array([0.5, 0.5, 0.0]))
    coverage = FractionalCoverage.aligned_with(
        spectra,
        values=values,
        origin=FractionalCoverageOrigin.PROPAGATED_Q_REBIN,
    )

    dataset = ReducedDataset(
        role=SpectrumRole.SAMPLE,
        spectra=spectra,
        fractional_coverage=coverage,
    )

    assert (
        dataset.fractional_coverage_availability
        is FractionalCoverageAvailability.AVAILABLE
    )
    assert dataset.fractional_coverage is not None
    np.testing.assert_array_equal(dataset.fractional_coverage.values[0], values[0])
    assert not dataset.fractional_coverage.values[0].flags.writeable


def test_propagated_fractional_coverage_may_exceed_one_but_not_be_negative() -> None:
    spectrum = make_spectrum()
    coverage = FractionalCoverage.aligned_with(
        (spectrum,),
        values=(np.array([2.0, 1.25, 0.0]),),
        origin=FractionalCoverageOrigin.PROPAGATED_Q_REBIN,
    )

    np.testing.assert_array_equal(coverage.values[0], [2.0, 1.25, 0.0])
    with pytest.raises(ValueError, match="nonnegative"):
        FractionalCoverage.aligned_with(
            (spectrum,),
            values=(np.array([1.0, -0.01, 1.0]),),
            origin=FractionalCoverageOrigin.EXPLICIT_SOURCE,
        )


def test_confirmed_unrebinned_origin_requires_exactly_unit_coverage() -> None:
    with pytest.raises(ValueError, match="must all equal one"):
        FractionalCoverage.aligned_with(
            (make_spectrum(),),
            values=(np.array([1.0, 0.75, 1.0]),),
            origin=FractionalCoverageOrigin.CONFIRMED_UNREBINNED_SOURCE,
        )


def test_coverage_rejects_same_length_changed_point_correspondence() -> None:
    dataset = ReducedDataset(
        role=SpectrumRole.SAMPLE,
        spectra=(make_spectrum(),),
    ).confirm_unrebinned_source()
    changed = replace(
        dataset.spectra[0],
        intensity=dataset.spectra[0].intensity + 0.01,
    )

    with pytest.raises(ValueError, match="different ordered X/Y/E points"):
        replace(dataset, spectra=(changed,))


def test_q_assignment_preserves_fractional_coverage_availability() -> None:
    dataset = ReducedDataset(
        role=SpectrumRole.SAMPLE,
        spectra=(make_spectrum(),),
    ).confirm_unrebinned_source()

    assigned = dataset.assign_q_bins(QBins.from_q_values([0.75]))

    assert assigned.fractional_coverage is dataset.fractional_coverage
    assert (
        assigned.fractional_coverage_availability
        is FractionalCoverageAvailability.AVAILABLE
    )
