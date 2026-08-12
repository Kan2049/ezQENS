"""Meaningful noninteractive tests for scientific inspection plots."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ezqens.domain import QBins, ReducedDataset, Spectrum, SpectrumRole
from ezqens.preprocessing import FittingSelection, detect_edge_padding
from ezqens.resolution import prepare_measured_resolution
from ezqens.visualization import (
    plot_q_bins,
    plot_resolution_inspection,
    plot_spectrum_inspection,
)


def make_selection() -> FittingSelection:
    spectrum = Spectrum(
        role=SpectrumRole.SAMPLE,
        group_index=0,
        group_label="sample-group",
        energy=np.arange(-5.0, 5.0),
        intensity=np.array([-2.0] * 5 + [3.0, 4.0, 2.0] + [-1.0] * 2),
        uncertainty=np.array([0.2] * 6 + [0.0] + [0.2] * 3),
        energy_unit="meV",
        intensity_unit="counts",
        uncertainty_unit="counts",
    )
    dataset = ReducedDataset(
        role=SpectrumRole.SAMPLE,
        spectra=(spectrum,),
    ).assign_q_bins(QBins.from_edges([0.4, 0.55]))
    return FittingSelection.uniform(
        dataset,
        detect_edge_padding(dataset),
        lower_energy=-3.0,
        upper_energy=3.0,
    )


def test_q_bin_plot_uses_scientific_edges_and_representatives() -> None:
    q_bins = QBins.from_edges([0.4, 0.55, 0.8])

    figure, axes = plot_q_bins(q_bins)

    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)
    interval_lines = [
        line for line in axes.lines if line.get_label() == "Q-bin interval"
    ]
    representative = next(
        line for line in axes.lines if line.get_label() == "Representative Q"
    )
    np.testing.assert_allclose(
        np.asarray(interval_lines[0].get_xdata(), dtype=np.float64),
        np.asarray([0.4, 0.55]),
    )
    np.testing.assert_allclose(
        np.asarray(representative.get_xdata(), dtype=np.float64),
        np.asarray([0.475, 0.675]),
    )
    plt.close(figure)


def test_q_value_only_plot_does_not_invent_edges() -> None:
    q_bins = QBins.from_q_values([0.8, 0.35, 1.4])

    figure, axes = plot_q_bins(q_bins)

    assert "edges unknown" in axes.get_title()
    assert all(line.get_label() != "Q-bin interval" for line in axes.lines)
    plt.close(figure)


def test_spectrum_plot_reflects_group_range_q_and_selection_without_mutation() -> None:
    selection = make_selection()
    spectrum = selection.dataset.spectra[0]
    originals = tuple(
        array.copy()
        for array in (spectrum.energy, spectrum.intensity, spectrum.uncertainty)
    )

    figure, axes = plot_spectrum_inspection(selection, 0)

    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)
    range_patch = axes.patches[0]
    assert isinstance(range_patch, Rectangle)
    assert range_patch.get_x() == -3.0
    assert range_patch.get_width() == 6.0
    assert "Q=0.475" in axes.get_title()
    labels = axes.get_legend_handles_labels()[1]
    assert "AUTO padding" in labels
    assert "REVIEW padding (retained unless otherwise excluded)" in labels
    assert "Invalid measurement" in labels
    assert "Outside fitting range" in labels
    assert "Retained for later fitting" in labels
    for current, original in zip(
        (spectrum.energy, spectrum.intensity, spectrum.uncertainty),
        originals,
        strict=True,
    ):
        np.testing.assert_array_equal(current, original)
    plt.close(figure)


def test_resolution_plot_consumes_prepared_state_without_mutation() -> None:
    energy = np.arange(-5.0, 5.0)
    q_bins = QBins.from_q_values([0.5])
    sample_spectrum = Spectrum(
        role=SpectrumRole.SAMPLE,
        group_index=0,
        group_label="sample-group",
        energy=energy,
        intensity=np.linspace(1.0, 2.0, energy.size),
        uncertainty=np.full(energy.size, 0.2),
        energy_unit="meV",
        intensity_unit="counts",
        uncertainty_unit="counts",
    )
    resolution_spectrum = Spectrum(
        role=SpectrumRole.RESOLUTION,
        group_index=0,
        group_label="resolution-group",
        energy=energy,
        intensity=np.array([10.0] * 5 + [1.0, 3.0, 4.0, 2.0, 1.0]),
        uncertainty=np.full(energy.size, 0.2),
        energy_unit="meV",
        intensity_unit="counts",
        uncertainty_unit="counts",
    )
    sample = ReducedDataset(
        role=SpectrumRole.SAMPLE,
        spectra=(sample_spectrum,),
    ).assign_q_bins(q_bins)
    resolution = ReducedDataset(
        role=SpectrumRole.RESOLUTION,
        spectra=(resolution_spectrum,),
    ).assign_q_bins(q_bins)
    originals = tuple(
        value.copy()
        for value in (
            resolution_spectrum.energy,
            resolution_spectrum.intensity,
            resolution_spectrum.uncertainty,
        )
    )
    prepared = prepare_measured_resolution(sample, resolution)

    figure, axes = plot_resolution_inspection(prepared, 0)

    assert isinstance(figure, Figure)
    assert len(axes) == 2
    source_labels = axes[0].get_legend_handles_labels()[1]
    normalized_labels = axes[1].get_legend_handles_labels()[1]
    assert "Original measured resolution" in source_labels
    assert "Accepted kernel points" in source_labels
    assert "AUTO padding (excluded)" in source_labels
    assert "Corresponding sample (alignment only)" in source_labels
    assert "E = 0" in source_labels
    assert "Unit-area measured resolution" in normalized_labels
    normalized_line = next(
        line
        for line in axes[1].lines
        if line.get_label() == "Unit-area measured resolution"
    )
    np.testing.assert_array_equal(
        np.asarray(normalized_line.get_xdata(), dtype=np.float64),
        prepared.spectra[0].energy,
    )
    for current, original in zip(
        (
            resolution_spectrum.energy,
            resolution_spectrum.intensity,
            resolution_spectrum.uncertainty,
        ),
        originals,
        strict=True,
    ):
        np.testing.assert_array_equal(current, original)
    plt.close(figure)
