"""Functions to predict properties related to hysteresis loops."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numbers

    import astropy.units
    import mammos_entity

import mammos_analysis
import mammos_entity as me
import mammos_units as u
import numpy as np
import onnxruntime as ort

MODEL_DIR = Path(__file__).parent

MODELS = {
    "classifier": MODEL_DIR / "classifier_cube50_singlegrain_random_forest_v0.1.onnx",
    False: MODEL_DIR / "soft_cube50_singlegrain_random_forest_v0.1.onnx",
    True: MODEL_DIR / "hard_cube50_singlegrain_random_forest_v0.1.onnx",
}

_SESSION_OPTIONS = ort.SessionOptions()
_SESSION_OPTIONS.log_severity_level = 3


def is_hard_magnet_from_Ms_A_K(
    Ms: mammos_entity.Entity | astropy.units.Quantity | numbers.Number | np.ndarray,
    A: mammos_entity.Entity | astropy.units.Quantity | numbers.Number | np.ndarray,
    K1: mammos_entity.Entity | astropy.units.Quantity | numbers.Number | np.ndarray,
    model: str = "cube50_singlegrain_random_forest_v0.1",
) -> bool | np.ndarray:
    """Classify material as soft or hard magnetic from micromagnetic parameters.

    This function classifies a magnetic material as either soft or hard magnetic
    based on its micromagnetic parameters spontaneous magnetization Ms, exchange
    stiffness constant A and uniaxial anisotropy constant K1.
    The shape of the input parameters needs to be the same. If single values are
    provided, a single classification is returned. If arrays are provided, a
    numpy array with the same shape is returned.

    Different models are available for the prediction. The following list provides an
    overview:

    - ``cube50_singlegrain_random_forest_v0.1``: Random forest model trained on
      simulated data for single grain cubic particles with 50 nm edge length with
      the external field applied parallel to the anisotropy axis. These are both
      aligned along an edge of the cube. Further details on the training data
      and model can be found in the
      `model repository <https://github.com/MaMMoS-project/ML-models/tree/main/beyond-stoner-wohlfarth/single-grain-easy-axis-model>`_.

    Args:
       Ms: Spontaneous magnetization.
       A: Exchange stiffness constant.
       K1: Uniaxial anisotropy constant.
       model: AI model used for the classification

    Returns:
       Classification as False (soft) or True (hard).
       Returns a boolean for scalar inputs, or a numpy array
       with the same shape as the input for array inputs.

    """
    Ms = me.Ms(Ms, unit=u.A / u.m)
    A = me.A(A, unit=u.J / u.m)
    K1 = me.Ku(K1, unit=u.J / u.m**3)

    Ms_arr = np.atleast_1d(Ms.value)
    A_arr = np.atleast_1d(A.value)
    K1_arr = np.atleast_1d(K1.value)

    if not (Ms_arr.shape == A_arr.shape == K1_arr.shape):
        raise ValueError(
            f"Input arrays must have the same shape. Shapes are Ms: {Ms_arr.shape}, "
            f"A: {A_arr.shape}, Ku: {K1_arr.shape}"
        )

    original_shape = Ms_arr.shape
    is_scalar = Ms_arr.ndim == 0 or (Ms_arr.ndim == 1 and Ms_arr.size == 1)

    match model:
        case "cube50_singlegrain_random_forest_v0.1":
            classifier_path = MODELS["classifier"]
        case _:
            raise ValueError(f"Unknown model {model}")

    session = ort.InferenceSession(classifier_path, _SESSION_OPTIONS)
    X = np.column_stack([Ms_arr.ravel(), A_arr.ravel(), K1_arr.ravel()]).astype(
        np.float32
    )

    # input name obtained from model expects a shape
    # (n_samples, 3) containing [Ms, A, K1], returns 1D
    # array with shape (n_samples,) containing class labels
    # (0=soft magnetic, 1=hard magnetic)
    results = session.run(None, {session.get_inputs()[0].name: X})[0]
    labels = np.where(results == 0, False, True)

    if is_scalar:
        return labels.item()
    else:
        return labels.reshape(original_shape)


def Hc_Mr_BHmax_from_Ms_A_K(
    Ms: mammos_entity.Entity | astropy.units.Quantity | numbers.Number | np.ndarray,
    A: mammos_entity.Entity | astropy.units.Quantity | numbers.Number | np.ndarray,
    K1: mammos_entity.Entity | astropy.units.Quantity | numbers.Number | np.ndarray,
    model: str = "cube50_singlegrain_random_forest_v0.1",
) -> mammos_analysis.hysteresis.ExtrinsicProperties:
    """Predict Hc, Mr and BHmax from micromagnetic properties Ms, A and K1.

    This function predicts extrinsic properties coercive field Hc, remanent
    magnetization Mr and maximum energy product BHmax given a set of micromagnetic
    material parameters.

    Different models are available for the prediction. The following list provides an
    overview:

    - ``cube50_singlegrain_random_forest_v0.1``: Random forest model trained on
      simulated data for single grain cubic particles with 50 nm edge length with
      the external field applied parallel to the anisotropy axis. These are both
      aligned along an edge of the cube. Further details on the training data
      and model can be found in the
      `model repository <https://github.com/MaMMoS-project/ML-models/tree/main/beyond-stoner-wohlfarth/single-grain-easy-axis-model>`_.



    Args:
       Ms: Spontaneous magnetization.
       A: Exchange stiffness constant.
       K1: Uniaxial anisotropy constant.
       model: AI model used for the prediction

    Returns:
       An object containing extrinsic properties Hc, Mr, BHmax

    Examples:
    >>> import mammos_ai
    >>> import mammos_entity as me
    >>> mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(me.Ms(1e5), me.A(1e-12), me.Ku(1e6))
    ExtrinsicProperties(Hc=..., Mr=..., BHmax=...)
    """
    Ms = me.Ms(Ms, unit=u.A / u.m)
    A = me.A(A, unit=u.J / u.m)
    K1 = me.Ku(K1, unit=u.J / u.m**3)

    Ms_arr = np.atleast_1d(Ms.value)
    A_arr = np.atleast_1d(A.value)
    K1_arr = np.atleast_1d(K1.value)

    if not (Ms_arr.shape == A_arr.shape == K1_arr.shape):
        raise ValueError(
            f"Input arrays must have the same shape. Shapes are Ms: {Ms_arr.shape}, "
            f"A: {A_arr.shape}, Ku: {K1_arr.shape}"
        )

    original_shape = Ms_arr.shape
    is_scalar = Ms_arr.ndim == 0 or (Ms_arr.ndim == 1 and Ms_arr.size == 1)

    match model:
        case "cube50_singlegrain_random_forest_v0.1":
            # 1. Determine class
            mat_class = is_hard_magnet_from_Ms_A_K(Ms, A, K1, model=model)

            # 2. Preprocess
            X_log = np.log1p(
                np.column_stack([Ms_arr.ravel(), A_arr.ravel(), K1_arr.ravel()]).astype(
                    np.float32
                )
            )

            y_log = np.empty((X_log.shape[0], 3), dtype=np.float32)
            classes = np.atleast_1d(mat_class).ravel()

            for cls in [False, True]:
                mask = classes == cls
                if np.any(mask):
                    # 3. Load regression model
                    session = ort.InferenceSession(MODELS[cls], _SESSION_OPTIONS)
                    X_subset = X_log[mask]

                    # 4. Predict: input name obtained from model expects a shape
                    # (n_samples_in_class, 3) containing [Ms, A, K1], returns 2D
                    # array with shape (n_samples_in_class, 3) containing
                    # [Hc, Mr, BHmax] predictions
                    res = session.run(None, {session.get_inputs()[0].name: X_subset})[0]
                    y_log[mask] = res

            # 5. Postprocess
            y = np.expm1(y_log)

            y = y[0] if is_scalar else y.reshape(original_shape + (3,))

        case _:
            raise ValueError(f"Unknown model {model}")

    if is_scalar:
        Hc_val = y[0]
        Mr_val = y[1]
        BHmax_val = y[2]
    else:
        Hc_val = y[..., 0]
        Mr_val = y[..., 1]
        BHmax_val = y[..., 2]

    Hc = me.Hc(Hc_val, "A/m")
    Mr = me.Mr(Mr_val, "A/m")
    BHmax = me.BHmax(BHmax_val, "J/m3")
    return mammos_analysis.hysteresis.ExtrinsicProperties(Hc=Hc, Mr=Mr, BHmax=BHmax)
