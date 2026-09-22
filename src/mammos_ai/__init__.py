"""Pre-trained AI models."""

import importlib.metadata

from ._beyond_stoner_wohlfarth_fixed_angle import (
    Hc_Mr_BHmax_from_Ms_A_K1,
    Hc_Mr_BHmax_from_Ms_A_K1_metadata,
    is_hard_magnet_from_Ms_A_K1,
    is_hard_magnet_from_Ms_A_K1_metadata,
)

__version__ = importlib.metadata.version(__package__)

__all__ = [
    "Hc_Mr_BHmax_from_Ms_A_K1",
    "Hc_Mr_BHmax_from_Ms_A_K1_metadata",
    "is_hard_magnet_from_Ms_A_K1",
    "is_hard_magnet_from_Ms_A_K1_metadata",
]
