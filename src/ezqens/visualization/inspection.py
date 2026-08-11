"""Scientific inspection plots for Q bins and fitting-point selection."""

from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ezqens.domain import QBins
from ezqens.preprocessing import FittingSelection


def _figure_and_axes(
    axes: Axes | None, *, figsize: tuple[float, float]
) -> tuple[Figure, Axes]:
    if axes is not None:
        return cast(Figure, axes.figure), axes
    figure, created_axes = plt.subplots(figsize=figsize)
    return figure, created_axes


def plot_q_bins(q_bins: QBins, *, axes: Axes | None = None) -> tuple[Figure, Axes]:
    """Plot ordered representative Q values and known bin intervals."""

    figure, axes = _figure_and_axes(axes, figsize=(8.0, 4.5))
    group_indices = np.arange(q_bins.group_count)
    if q_bins.edges is not None:
        for group_index, (lower_edge, upper_edge) in enumerate(
            zip(q_bins.edges[:-1], q_bins.edges[1:], strict=True)
        ):
            axes.plot(
                [lower_edge, upper_edge],
                [group_index, group_index],
                linewidth=7.0,
                alpha=0.35,
                color="tab:blue",
                solid_capstyle="butt",
                label="Q-bin interval" if group_index == 0 else None,
            )
        axes.vlines(
            q_bins.edges,
            -0.4,
            max(q_bins.group_count - 0.6, 0.4),
            color="0.35",
            linewidth=0.8,
            label="Q-bin edges",
        )
    axes.plot(
        q_bins.q_values,
        group_indices,
        linestyle="none",
        marker="o",
        color="tab:red",
        label="Representative Q",
    )
    axes.set_xlabel(f"Q ({q_bins.unit})")
    axes.set_ylabel("Spectrum group index")
    axes.set_yticks(group_indices)
    axes.set_title(
        "Q-bin edges and representatives"
        if q_bins.edges is not None
        else "Ordered representative Q values (bin edges unknown)"
    )
    axes.legend()
    figure.tight_layout()
    return figure, axes


def plot_spectrum_inspection(
    selection: FittingSelection,
    group_index: int,
    *,
    axes: Axes | None = None,
    show_uncertainties: bool = True,
) -> tuple[Figure, Axes]:
    """Plot one measured group and every Milestone-2 point-selection state."""

    spectrum = selection.dataset.spectra[group_index]
    padding = selection.padding.spectra[group_index]
    fitting_range = selection.ranges[group_index]
    invalid = selection.invalid_mask(group_index)
    in_range = selection.in_range_mask(group_index)
    retained = selection.retained_mask(group_index)
    finite_coordinates = np.isfinite(spectrum.energy) & np.isfinite(spectrum.intensity)

    figure, axes = _figure_and_axes(axes, figsize=(9.0, 5.0))
    axes.axvspan(
        fitting_range.lower_energy,
        fitting_range.upper_energy,
        color="tab:green",
        alpha=0.10,
        label="Selected fitting range",
    )
    axes.plot(
        spectrum.energy[finite_coordinates],
        spectrum.intensity[finite_coordinates],
        linestyle="none",
        marker=".",
        color="0.65",
        label="Original measured points",
    )
    outside_range = ~in_range & finite_coordinates
    axes.scatter(
        spectrum.energy[outside_range],
        spectrum.intensity[outside_range],
        marker=".",
        color="tab:cyan",
        label="Outside fitting range",
    )
    axes.scatter(
        spectrum.energy[padding.auto_mask & finite_coordinates],
        spectrum.intensity[padding.auto_mask & finite_coordinates],
        marker="x",
        color="tab:red",
        label="AUTO padding",
    )
    axes.scatter(
        spectrum.energy[padding.review_mask & finite_coordinates],
        spectrum.intensity[padding.review_mask & finite_coordinates],
        marker="s",
        facecolors="none",
        edgecolors="tab:orange",
        label="REVIEW padding (retained unless otherwise excluded)",
    )
    axes.scatter(
        spectrum.energy[invalid & finite_coordinates],
        spectrum.intensity[invalid & finite_coordinates],
        marker="+",
        color="tab:purple",
        label="Invalid measurement",
    )
    axes.plot(
        spectrum.energy[retained & finite_coordinates],
        spectrum.intensity[retained & finite_coordinates],
        linestyle="none",
        marker="o",
        markerfacecolor="none",
        color="tab:blue",
        label="Retained for later fitting",
    )
    if show_uncertainties:
        uncertainty_points = (
            finite_coordinates
            & np.isfinite(spectrum.uncertainty)
            & (spectrum.uncertainty > 0.0)
        )
        axes.errorbar(
            spectrum.energy[uncertainty_points],
            spectrum.intensity[uncertainty_points],
            yerr=spectrum.uncertainty[uncertainty_points],
            fmt="none",
            color="0.45",
            alpha=0.35,
            label="Measurement uncertainty",
        )

    title_parts = [f"Group {spectrum.group_label}"]
    q_bins = selection.dataset.q_bins
    if q_bins is not None:
        title_parts.append(f"Q={q_bins.q_values[group_index]:.6g} {q_bins.unit}")
        if q_bins.edges is not None:
            title_parts.append(
                "bin "
                f"[{q_bins.edges[group_index]:.6g}, "
                f"{q_bins.edges[group_index + 1]:.6g}] {q_bins.unit}"
            )
    axes.set_title("; ".join(title_parts))
    axes.set_xlabel(f"Energy ({spectrum.energy_unit})")
    axes.set_ylabel(f"Intensity ({spectrum.intensity_unit})")
    axes.legend(fontsize=8)
    figure.tight_layout()
    return figure, axes
