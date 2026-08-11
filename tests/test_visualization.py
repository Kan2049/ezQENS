"""Meaningful noninteractive tests for scientific inspection plots."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from qensfit.domain import QBins, ReducedDataset, Spectrum, SpectrumRole
from qensfit.preprocessing import FittingSelection, detect_edge_padding
from qensfit.visualization import plot_q_bins, plot_spectrum_inspection


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
