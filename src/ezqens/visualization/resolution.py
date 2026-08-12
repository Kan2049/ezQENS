"""Inspection plots for authoritative measured-resolution preparation state."""

from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ezqens.resolution import PreparedResolution


def _resolution_axes(
    axes: tuple[Axes, Axes] | None,
) -> tuple[Figure, Axes, Axes]:
    if axes is not None:
        return cast(Figure, axes[0].figure), axes[0], axes[1]
    figure, created = plt.subplots(1, 2, figsize=(13.0, 5.0))
    return figure, cast(Axes, created[0]), cast(Axes, created[1])


def plot_resolution_inspection(
    prepared: PreparedResolution,
    group_index: int,
    *,
    axes: tuple[Axes, Axes] | None = None,
    show_uncertainties: bool = True,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot source, mask/support, alignment, and normalized resolution state."""

    if not 0 <= group_index < len(prepared.spectra):
        raise IndexError("group_index is outside the prepared resolution")
    prepared_spectrum = prepared.spectra[group_index]
    resolution = prepared_spectrum.source_spectrum
    sample = prepared.sample_dataset.spectra[group_index]
    padding = prepared_spectrum.padding
    comparison = prepared.padding_comparisons[group_index]
    accepted = prepared_spectrum.accepted_mask
    invalid = prepared_spectrum.invalid_mask
    finite_resolution = np.isfinite(resolution.energy) & np.isfinite(
        resolution.intensity
    )
    finite_sample = np.isfinite(sample.energy) & np.isfinite(sample.intensity)

    figure, source_axes, normalized_axes = _resolution_axes(axes)
    source_axes.axvspan(
        prepared_spectrum.support.lower_energy,
        prepared_spectrum.support.upper_energy,
        color="tab:green",
        alpha=0.10,
        label="Accepted support range",
    )
    source_axes.axvline(0.0, color="0.25", linestyle="--", label="E = 0")
    source_axes.plot(
        sample.energy[finite_sample],
        sample.intensity[finite_sample],
        color="0.65",
        linewidth=1.0,
        label="Corresponding sample (alignment only)",
    )
    source_axes.plot(
        resolution.energy[finite_resolution],
        resolution.intensity[finite_resolution],
        linestyle="none",
        marker=".",
        color="0.4",
        label="Original measured resolution",
    )
    source_axes.plot(
        resolution.energy[accepted],
        resolution.intensity[accepted],
        linestyle="none",
        marker="o",
        markerfacecolor="none",
        color="tab:blue",
        label="Accepted kernel points",
    )
    source_axes.scatter(
        resolution.energy[padding.auto_mask & finite_resolution],
        resolution.intensity[padding.auto_mask & finite_resolution],
        marker="x",
        color="tab:red",
        label=(
            "AUTO padding (excluded)"
            if prepared_spectrum.auto_padding_applied
            else "AUTO padding (restored)"
        ),
    )
    source_axes.scatter(
        resolution.energy[padding.review_mask & finite_resolution],
        resolution.intensity[padding.review_mask & finite_resolution],
        marker="s",
        facecolors="none",
        edgecolors="tab:orange",
        label="REVIEW padding (retained by default)",
    )
    source_axes.scatter(
        resolution.energy[invalid & finite_resolution],
        resolution.intensity[invalid & finite_resolution],
        marker="+",
        color="tab:purple",
        label="Invalid source value",
    )
    if show_uncertainties:
        valid_uncertainty = finite_resolution & ~resolution.invalid_uncertainty_mask
        source_axes.errorbar(
            resolution.energy[valid_uncertainty],
            resolution.intensity[valid_uncertainty],
            yerr=resolution.uncertainty[valid_uncertainty],
            fmt="none",
            color="0.45",
            alpha=0.35,
            label="Resolution uncertainty",
        )

    normalized_axes.axvline(0.0, color="0.25", linestyle="--", label="E = 0")
    normalized_axes.plot(
        prepared_spectrum.energy,
        prepared_spectrum.normalized_intensity,
        marker="o",
        markersize=3.0,
        color="tab:blue",
        label="Unit-area measured resolution",
    )
    if show_uncertainties:
        accepted_valid_uncertainty = accepted & ~resolution.invalid_uncertainty_mask
        normalized_axes.errorbar(
            resolution.energy[accepted_valid_uncertainty],
            resolution.intensity[accepted_valid_uncertainty]
            * prepared_spectrum.normalization_factor,
            yerr=resolution.uncertainty[accepted_valid_uncertainty]
            * prepared_spectrum.normalization_factor,
            fmt="none",
            color="0.45",
            alpha=0.35,
            label="Scaled uncertainty",
        )

    q_value = prepared.q_value(group_index)
    consistency = (
        "matching retained boundaries"
        if comparison.is_consistent
        else ("padding-boundary warning")
    )
    source_axes.set_title(
        f"Sample/resolution group {group_index}; Q={q_value:.6g} Å^-1; {consistency}"
    )
    source_axes.set_xlabel(f"Energy ({resolution.energy_unit})")
    source_axes.set_ylabel(f"Source intensity ({resolution.intensity_unit})")
    source_axes.legend(fontsize=7)

    normalized_axes.set_title(
        "Prepared measured resolution; "
        f"area={prepared_spectrum.normalized_integral:.6g}"
    )
    normalized_axes.set_xlabel(f"Energy ({resolution.energy_unit})")
    normalized_axes.set_ylabel(f"Normalized resolution ({resolution.energy_unit}^-1)")
    normalized_axes.legend(fontsize=8)
    figure.tight_layout()
    return figure, (source_axes, normalized_axes)
