"""Numerical linear convolution on fixed physical energy grids."""

from ezqens.convolution.core import (
    CANONICAL_ENERGY_UNIT,
    ConvolutionError,
    ConvolutionPlan,
    ConvolvedProfile,
    automatic_grid_spacing,
    build_convolution_plan,
    cell_integrated_lorentzian,
)

__all__ = [
    "CANONICAL_ENERGY_UNIT",
    "ConvolvedProfile",
    "ConvolutionError",
    "ConvolutionPlan",
    "automatic_grid_spacing",
    "build_convolution_plan",
    "cell_integrated_lorentzian",
]
