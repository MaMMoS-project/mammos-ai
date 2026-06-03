"""Shared helpers used by all models in this subpackage."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mammos_entity
    import mammos_units
    import numpy

import mammos_entity as me
import numpy as np
import onnxruntime as ort

SESSION_OPTIONS = ort.SessionOptions()
SESSION_OPTIONS.log_severity_level = 3


def prepare_Ms_A_K1(
    Ms: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    A: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    K1: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn (Ms, A, K1) inputs into numpy value arrays in SI units.

    Args:
        Ms: :entity:`SpontaneousMagnetization`.
            If no unit is provided, values are interpreted as 'A/m'.
        A: :entity:`ExchangeStiffnessConstant`.
            If no unit is provided, values are interpreted as 'J/m'.
        K1: :entity:`UniaxialAnisotropyConstant`.
            If no unit is provided, values are interpreted as 'J/m^3'.

    Returns:
        ``(Ms_arr, A_arr, K1_arr)``. The arrays use SI units.

    Raises:
        ValueError: if the three inputs do not have the same shape.
    """
    Ms = me._entity.from_compatible(
        "SpontaneousMagnetization", "A/m", Ms=Ms, enforce_unit=True
    )
    A = me._entity.from_compatible(
        "ExchangeStiffnessConstant", "J/m", A=A, enforce_unit=True
    )
    K1 = me._entity.from_compatible(
        "UniaxialAnisotropyConstant", "J/m^3", K1=K1, enforce_unit=True
    )

    Ms_arr = Ms.value
    A_arr = A.value
    K1_arr = K1.value

    if not (Ms_arr.shape == A_arr.shape == K1_arr.shape):
        raise ValueError(
            f"Input arrays must have the same shape. Shapes are Ms: {Ms_arr.shape}, "
            f"A: {A_arr.shape}, Ku: {K1_arr.shape}"
        )

    return Ms_arr, A_arr, K1_arr
