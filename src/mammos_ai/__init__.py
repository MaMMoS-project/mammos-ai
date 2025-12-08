"""Pre-trained AI models."""

import importlib.metadata

from .beyond_stoner_wohlfarth_fixed_angle.bsw import (
    Hc_Mr_BHmax_from_Ms_A_K,
    classify_magnetic_from_Ms_A_K,
)

__version__ = importlib.metadata.version(__package__)

__all__ = ["Hc_Mr_BHmax_from_Ms_A_K", "classify_magnetic_from_Ms_A_K"]
