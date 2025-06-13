"""Pre-trained AI models."""

import importlib.metadata

from ._hysteresis import Hc_Mr_BHmax_from_Ms_A_K

__version__ = importlib.metadata.version(__package__)

__all__ = ["Hc_Mr_BHmax_from_Ms_A_K"]
