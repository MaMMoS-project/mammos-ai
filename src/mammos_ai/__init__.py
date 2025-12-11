"""Pre-trained AI models."""

import importlib.metadata

from ._beyond_stoner_wohlfarth_fixed_angle import (
    Hc_Mr_BHmax_from_Ms_A_K,
    Hc_Mr_BHmax_from_Ms_A_K_metadata,
    is_hard_magnet_from_Ms_A_K,
    is_hard_magnet_from_Ms_A_K_metadata,
)

__version__ = importlib.metadata.version(__package__)

__all__ = [
    "Hc_Mr_BHmax_from_Ms_A_K",
    "Hc_Mr_BHmax_from_Ms_A_K_metadata",
    "is_hard_magnet_from_Ms_A_K",
    "is_hard_magnet_from_Ms_A_K_metadata",
]
