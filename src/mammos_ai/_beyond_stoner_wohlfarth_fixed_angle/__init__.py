"""Functions to predict properties related to hysteresis loops.

Each model lives in its own module in this subpackage. The public functions
below look up the requested model in ``_REGISTRY`` and call it. To add a
new model, write a new module that provides ``NAME``, ``CLASSIFY_METADATA``,
``PREDICT_METADATA``, ``is_hard_magnet`` and ``predict_extrinsic``, then add
it to ``_REGISTRY``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mammos_entity
    import mammos_units
    import numpy

import mammos_analysis
import mammos_entity as me

from . import cube50_singlegrain_random_forest_v0_1 as _cube50_v0_1
from ._common import prepare_Ms_A_K1

_REGISTRY = {
    "cube50_singlegrain_random_forest_v0.1": _cube50_v0_1,
}


def _choose_model(model: str):
    """Find the model module registered under the given name.

    Args:
        model: Name of a registered model.

    Returns:
        The matching model module.

    Raises:
        ValueError: if ``model`` is not registered.
    """
    try:
        return _REGISTRY[model]
    except KeyError:
        raise ValueError(f"Unknown model {model}") from None


def is_hard_magnet_from_Ms_A_K(
    Ms: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    A: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    K1: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    model: str = "cube50_singlegrain_random_forest_v0.1",
) -> bool | numpy.ndarray:
    """Classify material as soft or hard magnetic from micromagnetic parameters.

    This function classifies a magnetic material as either soft or hard magnetic
    based on its micromagnetic parameters spontaneous magnetization Ms, exchange
    stiffness constant A and uniaxial anisotropy constant K1.
    The shape of the input parameters needs to be the same. If single values are
    provided, a single classification is returned. If arrays are provided, a
    numpy array with the same shape is returned.

    The following models are available for the prediction:

    - ``cube50_singlegrain_random_forest_v0.1``: Random forest model trained on
      simulated data for single grain cubic particles with 50 nm edge length with
      the external field applied parallel to the anisotropy axis. These are both
      aligned along an edge of the cube. Further details on the training data
      and model can be found in the
      `model repository <https://github.com/MaMMoS-project/ML-models/tree/main/beyond-stoner-wohlfarth/single-grain-easy-axis-model>`_.

    Args:
        Ms: :entity:`SpontaneousMagnetization`.
            If no unit is provided, values are interpreted as 'A/m'.
        A: :entity:`ExchangeStiffnessConstant`.
            If no unit is provided, values are interpreted as 'J/m'.
        K1: :entity:`UniaxialAnisotropyConstant`.
            If no unit is provided, values are interpreted as 'J/m^3'.
        model: AI model used for the classification

    Returns:
        Classification as False (soft) or True (hard).
        Returns a boolean for scalar inputs, or a numpy array
        with the same shape as the input for array inputs.

    Examples:
        >>> import mammos_ai
        >>> import mammos_entity as me
        >>> mammos_ai.is_hard_magnet_from_Ms_A_K(me.Ms(1e6), me.A(1e-12), me.Ku(1e6))
        True

    """
    m = _choose_model(model)
    Ms_arr, A_arr, K1_arr, original_shape, is_scalar = prepare_Ms_A_K1(Ms, A, K1)
    labels = m.is_hard_magnet(Ms_arr, A_arr, K1_arr)
    if is_scalar:
        return labels.item()
    return labels.reshape(original_shape)


def is_hard_magnet_from_Ms_A_K_metadata(
    model: str = "cube50_singlegrain_random_forest_v0.1",
) -> dict:
    """Get metadata for the specified classification model.

    Args:
       model: AI model used for the classification

    """
    return _choose_model(model).CLASSIFY_METADATA


def Hc_Mr_BHmax_from_Ms_A_K(
    Ms: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    A: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    K1: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    model: str = "cube50_singlegrain_random_forest_v0.1",
) -> mammos_analysis.hysteresis.ExtrinsicProperties:
    """Predict Hc, Mr and BHmax from micromagnetic properties Ms, A and K1.

    This function predicts extrinsic properties coercive field Hc, remanent
    magnetization Mr and maximum energy product BHmax given a set of micromagnetic
    material parameters.

    The following models are available for the prediction:

    - ``cube50_singlegrain_random_forest_v0.1``: Random forest model trained on
      simulated data for single grain cubic particles with 50 nm edge length with
      the external field applied parallel to the anisotropy axis. These are both
      aligned along an edge of the cube. Further details on the training data
      and model can be found in the
      `model repository <https://github.com/MaMMoS-project/ML-models/tree/main/beyond-stoner-wohlfarth/single-grain-easy-axis-model>`_.

    Args:
        Ms: :entity:`SpontaneousMagnetization`.
            If no unit is provided, values are interpreted as 'A/m'.
        A: :entity:`ExchangeStiffnessConstant`.
            If no unit is provided, values are interpreted as 'J/m'.
        K1: :entity:`UniaxialAnisotropyConstant`.
            If no unit is provided, values are interpreted as 'J/m^3'.
        model: AI model used for the prediction

    Returns:
        An object containing extrinsic properties Hc, Mr, BHmax

    Examples:
        >>> import mammos_ai
        >>> import mammos_entity as me
        >>> mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(me.Ms(1e6), me.A(1e-12), me.Ku(1e6))
        ExtrinsicProperties(Hc=..., Mr=..., BHmax=...)
    """
    m = _choose_model(model)
    Ms_arr, A_arr, K1_arr, original_shape, is_scalar = prepare_Ms_A_K1(Ms, A, K1)
    y = m.predict_extrinsic(Ms_arr, A_arr, K1_arr)
    if is_scalar:
        Hc_val, Mr_val, BHmax_val = y[0]
    else:
        y = y.reshape(original_shape + (3,))
        Hc_val = y[..., 0]
        Mr_val = y[..., 1]
        BHmax_val = y[..., 2]
    return mammos_analysis.hysteresis.ExtrinsicProperties(
        Hc=me.Hc(Hc_val, "A/m"),
        Mr=me.Mr(Mr_val, "A/m"),
        BHmax=me.BHmax(BHmax_val, "J/m3"),
    )


def Hc_Mr_BHmax_from_Ms_A_K_metadata(
    model: str = "cube50_singlegrain_random_forest_v0.1",
) -> dict:
    """Get metadata for the specified Hc, Mr, BHmax prediction model.

    Args:
       model: AI model used for the prediction

    """
    return _choose_model(model).PREDICT_METADATA
