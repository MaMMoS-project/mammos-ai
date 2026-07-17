"""Functions to predict properties related to hysteresis loops.

Each model lives in its own module in this subpackage. The public functions
below look up the requested model in ``_REGISTRY`` and call it. To add a
new model, write a new module that provides ``NAME``, ``CLASSIFY_METADATA``,
``PREDICT_METADATA``, ``is_hard_magnet`` and ``predict_extrinsic``, then add
it to ``_REGISTRY``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mammos_entity
    import mammos_units
    import numpy

import mammos_analysis
import mammos_entity as me
import numpy as np

from . import cube50_singlegrain_random_forest_v0_1, cube50_singlegrain_random_forest_v1_0
from ._common import prepare_Ms_A_K1

_REGISTRY = {
    "cube50_singlegrain_random_forest_v0.1": cube50_singlegrain_random_forest_v0_1,
    "cube50_singlegrain_random_forest_v1.0": cube50_singlegrain_random_forest_v1_0,
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
    model: str = "cube50_singlegrain_random_forest_v1.0",
) -> bool | numpy.ndarray:
    """Classify material as soft or hard magnetic from micromagnetic parameters.

    This function classifies a magnetic material as either soft or hard magnetic
    based on its micromagnetic parameters spontaneous magnetization Ms, exchange
    stiffness constant A and uniaxial anisotropy constant K1.
    The shape of the input parameters needs to be the same. If single values are
    provided, a single classification is returned. If arrays are provided, a
    numpy array with the same shape is returned.

    The following models are available for the prediction:

    - ``cube50_singlegrain_random_forest_v1.0``: Random forest model trained on extended
      simulated data for single grain cubic particles with 50 nm edge length with
      the external field applied parallel to the anisotropy axis. These are both
      aligned along an edge of the cube. Further details on the training data
      can be found in the
      `training repository <https://github.com/MaMMoS-project/ML-models/tree/main/beyond-stoner-wohlfarth/single-grain-easy-axis-model>`_.
      Model files are downloaded from the
      `Hugging Face model repository <https://huggingface.co/mammos-project/mammos-ai-models>`_.

    - ``cube50_singlegrain_random_forest_v0.1``: Random forest model trained on
      simulated data for single grain cubic particles with 50 nm edge length with
      the external field applied parallel to the anisotropy axis. These are both
      aligned along an edge of the cube. This version uses separate
      classifiers to determine whether a sample is valid and, if valid, whether
      it is soft or hard magnetic. If the sample is invalid, the prediction will
      return NaN values. Further details on the training data can be found in the
      `training repository <https://github.com/MaMMoS-project/ML-models/tree/main/beyond-stoner-wohlfarth/single-grain-easy-axis-model>`_.
      Model files are downloaded from the
      `Hugging Face model repository <https://huggingface.co/mammos-project/mammos-ai-models>`_.

    Args:
        Ms: :entity:`SpontaneousMagnetization` or
            :entity:`SaturationMagnetization`.
            If no unit is provided, values are interpreted as 'A/m'.
        A: :entity:`ExchangeStiffnessConstant`.
            If no unit is provided, values are interpreted as 'J/m'.
        K1: :entity:`UniaxialAnisotropyConstant`.
            If no unit is provided, values are interpreted as 'J/m^3'.
        model: AI model used for the classification

    Returns:
        Classification as False (soft), True (hard), or NaN (invalid).
        Returns a boolean for scalar inputs, or a numpy array
        with the same shape as the input for array inputs.

    Examples:
        >>> import mammos_ai
        >>> import mammos_entity as me
        >>> mammos_ai.is_hard_magnet_from_Ms_A_K(
        ...     me.Entity("SpontaneousMagnetization", 1e6),
        ...     me.Entity("ExchangeStiffnessConstant", 1e-12),
        ...     me.Entity("UniaxialAnisotropyConstant", 1e6),
        ... )
        array(True, dtype=object)
    """
    m = _choose_model(model)
    if not hasattr(m, "is_hard_magnet"):
        raise NotImplementedError(f"Model {model} cannot classify materials as hard or soft.")
    Ms_arr, A_arr, K1_arr = prepare_Ms_A_K1(Ms, A, K1)
    labels = m.is_hard_magnet(Ms_arr, A_arr, K1_arr)
    return labels


def is_hard_magnet_from_Ms_A_K_metadata(
    model: str = "cube50_singlegrain_random_forest_v1.0",
) -> dict:
    """Get metadata for the specified classification model.

    Args:
       model: AI model used for the classification

    """
    m = _choose_model(model)
    if not hasattr(m, "CLASSIFY_METADATA"):
        raise NotImplementedError(f"Model {model} does not provide classification metadata.")
    return m.CLASSIFY_METADATA


def Hc_Mr_BHmax_from_Ms_A_K(
    Ms: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    A: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    K1: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    model: str = "cube50_singlegrain_random_forest_v1.0",
) -> mammos_analysis.hysteresis.ExtrinsicProperties:
    """Predict Hc, Mr and BHmax from micromagnetic properties Ms, A and K1.

    This function predicts extrinsic properties coercive field Hc, remanent
    magnetization Mr and maximum energy product BHmax given a set of micromagnetic
    material parameters.

    The following models are available for the prediction:

    - ``cube50_singlegrain_random_forest_v1.0``: Random forest model trained on extended
      simulated data for single grain cubic particles with 50 nm edge length with
      the external field applied parallel to the anisotropy axis. These are both
      aligned along an edge of the cube. Further details on the training data
      can be found in the
      `training repository <https://github.com/MaMMoS-project/ML-models/tree/main/beyond-stoner-wohlfarth/single-grain-easy-axis-model>`_.
      Model files are downloaded from the
      `Hugging Face model repository <https://huggingface.co/mammos-project/mammos-ai-models>`_.

    - ``cube50_singlegrain_random_forest_v0.1``: Random forest model trained on
      simulated data for single grain cubic particles with 50 nm edge length with
      the external field applied parallel to the anisotropy axis. These are both
      aligned along an edge of the cube. This version uses separate
      classifiers to determine whether a sample is valid and, if valid, whether
      it is soft or hard magnetic. If the sample is invalid, the prediction will
      return NaN values. Further details on the training data can be found in the
      `training repository <https://github.com/MaMMoS-project/ML-models/tree/main/beyond-stoner-wohlfarth/single-grain-easy-axis-model>`_.
      Model files are downloaded from the
      `Hugging Face model repository <https://huggingface.co/mammos-project/mammos-ai-models>`_.

    Args:
        Ms: :entity:`SpontaneousMagnetization` or
            :entity:`SaturationMagnetization`.
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
        >>> mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(
        ...     me.Entity("SpontaneousMagnetization", 1e6),
        ...     me.Entity("ExchangeStiffnessConstant", 1e-12),
        ...     me.Entity("UniaxialAnisotropyConstant", 1e6),
        ... )
        ExtrinsicProperties(Hc=..., Mr=..., BHmax=...)
    """
    m = _choose_model(model)
    if not hasattr(m, "predict_extrinsic"):
        raise NotImplementedError(f"Model {model} cannot predict Hc, Mr or BHmax.")
    Ms_arr, A_arr, K1_arr = prepare_Ms_A_K1(Ms, A, K1)
    Hc_val, Mr_val, BHmax_val = m.predict_extrinsic(Ms_arr, A_arr, K1_arr)
    return mammos_analysis.hysteresis.ExtrinsicProperties(
        Hc=me.Entity("CoercivityHcExternal", Hc_val, "A/m"),
        Mr=me.Entity("Remanence", Mr_val, "A/m"),
        BHmax=me.Entity("MaximumEnergyProduct", BHmax_val, "J/m3"),
    )


def Hc_Mr_BHmax_from_Ms_A_K_metadata(
    model: str = "cube50_singlegrain_random_forest_v1.0",
) -> dict:
    """Get metadata for the specified Hc, Mr, BHmax prediction model.

    Args:
       model: AI model used for the prediction

    """
    m = _choose_model(model)
    if not hasattr(m, "PREDICT_METADATA"):
        raise NotImplementedError(f"Model {model} does not provide Hc, Mr or BHmax prediction metadata.")
    return m.PREDICT_METADATA


def mc_error_propagation_Hc_Mr_BHmax_from_Ms_A_K(
    Ms_mean: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    A_mean: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    K1_mean: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    Ms_std: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    A_std: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    K1_std: mammos_entity.Entity | mammos_units.Quantity | numpy.typing.ArrayLike,
    n_samples: int = 10000,
    model: str = "cube50_singlegrain_random_forest_v1.0",
    random_seed: int | None = 1,
) -> mammos_entity.EntityCollection:
    """Estimate prediction uncertainty by Monte Carlo error propagation.

    The function samples normally distributed input parameters around the
    provided mean values and standard deviations, evaluates
    :func:`Hc_Mr_BHmax_from_Ms_A_K` for all samples, and returns the mean and
    standard deviation of the finite predictions. Samples for which the model
    returns ``NaN`` are ignored in the mean and standard deviation.

    Args:
        Ms_mean: Mean :entity:`SpontaneousMagnetization`.
            If no unit is provided, values are interpreted as 'A/m'.
        A_mean: Mean :entity:`ExchangeStiffnessConstant`.
            If no unit is provided, values are interpreted as 'J/m'.
        K1_mean: Mean :entity:`UniaxialAnisotropyConstant`.
            If no unit is provided, values are interpreted as 'J/m^3'.
        Ms_std: Standard deviation of :entity:`SpontaneousMagnetization`.
            If no unit is provided, values are interpreted as 'A/m'.
        A_std: Standard deviation of :entity:`ExchangeStiffnessConstant`.
            If no unit is provided, values are interpreted as 'J/m'.
        K1_std: Standard deviation of :entity:`UniaxialAnisotropyConstant`.
            If no unit is provided, values are interpreted as 'J/m^3'.
        n_samples: Number of Monte Carlo samples.
        model: AI model used for the prediction.
        random_seed: Seed for the random number generator. Use ``None`` for
            non-deterministic sampling.

    Returns:
        An entity collection containing ``Hc_mean``, ``Mr_mean``,
        ``BHmax_mean``, ``Hc_std``, ``Mr_std``, ``BHmax_std``, and the
        diagnostic fields ``valid_count``, and ``valid_fraction``.

    Raises:
        ValueError: if input means or standard deviations have incompatible
            shapes, if a standard deviation is negative, or if ``n_samples`` is
            smaller than one.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    Ms_mean_arr, A_mean_arr, K1_mean_arr = prepare_Ms_A_K1(Ms_mean, A_mean, K1_mean)
    Ms_std_arr, A_std_arr, K1_std_arr = prepare_Ms_A_K1(Ms_std, A_std, K1_std)

    input_shape = Ms_mean_arr.shape
    if not all(arr.shape == input_shape for arr in [A_mean_arr, K1_mean_arr, Ms_std_arr, A_std_arr, K1_std_arr]):
        raise ValueError(
            f"Mean and standard deviation arrays must have the same shape. Shapes are "
            f"Ms_mean: {Ms_mean_arr.shape}, A_mean: {A_mean_arr.shape}, K1_mean: {K1_mean_arr.shape}, "
            f"Ms_std: {Ms_std_arr.shape}, A_std: {A_std_arr.shape}, K1_std: {K1_std_arr.shape}"
        )

    if np.any(Ms_std_arr < 0) or np.any(A_std_arr < 0) or np.any(K1_std_arr < 0):
        raise ValueError("Standard deviations must be non-negative.")

    rng = np.random.default_rng(random_seed)
    sample_shape = (n_samples, *Ms_mean_arr.shape)
    Ms_samples = np.clip(rng.normal(Ms_mean_arr, Ms_std_arr, sample_shape), 1.0, None)
    A_samples = np.clip(rng.normal(A_mean_arr, A_std_arr, sample_shape), 1e-15, None)
    K1_samples = np.clip(rng.normal(K1_mean_arr, K1_std_arr, sample_shape), 1.0, None)

    predictions = Hc_Mr_BHmax_from_Ms_A_K(
        me.Ms(Ms_samples),
        me.A(A_samples),
        me.Ku(K1_samples),
        model=model,
    )

    Hc = predictions.Hc.q.value
    Mr = predictions.Mr.q.value
    BHmax = predictions.BHmax.q.value
    valid = np.isfinite(Hc) & np.isfinite(Mr) & np.isfinite(BHmax)
    valid_count = np.sum(valid, axis=0)
    Hc_valid = np.where(valid, Hc, np.nan)
    Mr_valid = np.where(valid, Mr, np.nan)
    BHmax_valid = np.where(valid, BHmax, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        Hc_mean = np.nanmean(Hc_valid, axis=0)
        Mr_mean = np.nanmean(Mr_valid, axis=0)
        BHmax_mean = np.nanmean(BHmax_valid, axis=0)
        Hc_std = np.nanstd(Hc_valid, axis=0)
        Mr_std = np.nanstd(Mr_valid, axis=0)
        BHmax_std = np.nanstd(BHmax_valid, axis=0)

    return me.EntityCollection(
        Hc_mean=me.Hc(Hc_mean),
        Mr_mean=me.Mr(Mr_mean),
        BHmax_mean=me.BHmax(BHmax_mean),
        Hc_std=me.Hc(Hc_std),
        Mr_std=me.Mr(Mr_std),
        BHmax_std=me.BHmax(BHmax_std),
        valid_count=valid_count,
        valid_fraction=valid_count / n_samples,
    )
