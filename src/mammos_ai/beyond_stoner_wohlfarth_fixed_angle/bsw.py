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

_MODEL_DIR = Path(__file__).parent

MODELS = {
    "soft": _MODEL_DIR / "random_forest_soft_v1.onnx",
    "hard": _MODEL_DIR / "random_forest_hard_v1.onnx",
}

_SESSION_OPTIONS = ort.SessionOptions()
_SESSION_OPTIONS.log_severity_level = 3


def classify_magnetic_from_Ms_A_K(
    Ms: mammos_entity.Entity | astropy.units.Quantity | numbers.Number,
    A: mammos_entity.Entity | astropy.units.Quantity | numbers.Number,
    Ku: mammos_entity.Entity | astropy.units.Quantity | numbers.Number,
    model: str = "random-forest-v1",
) -> str | list[str]:
    """Classify material as soft or hard magnetic from micromagnetic properties.

    This function classifies a magnetic material as either soft or hard magnetic
    based on its micromagnetic properties spontaneous magnetization Ms, exchange
    stiffness constant A and uniaxial anisotropy constant Ku.

    Args:
       Ms: Spontaneous magnetization.
       A: Exchange stiffness constant.
       Ku: Uniaxial anisotropy constant.
       model: AI model used for the classification

    Returns:
       Classification as "soft" or "hard".

    """
    Ms = me.Ms(Ms, unit=u.A / u.m)
    A = me.A(A, unit=u.J / u.m)
    Ku = me.Ku(Ku, unit=u.J / u.m**3)

    Ms_arr = np.atleast_1d(Ms.value)
    A_arr = np.atleast_1d(A.value)
    Ku_arr = np.atleast_1d(Ku.value)

    if not (Ms_arr.shape == A_arr.shape == Ku_arr.shape):
        raise ValueError(
            f"Input arrays must have the same length. Shapes are Ms: {Ms_arr.shape}, "
            f"A: {A_arr.shape}, Ku: {Ku_arr.shape}"
        )

    match model:
        case "random-forest-v1":
            classifier_path = _MODEL_DIR / "classifier_random_forest_v1.onnx"
        case _:
            raise ValueError(f"Unknown model {model}")

    session = ort.InferenceSession(str(classifier_path), _SESSION_OPTIONS)
    X = np.column_stack([Ms_arr, A_arr, Ku_arr]).astype(np.float32)

    results = session.run(None, {session.get_inputs()[0].name: X})[0]
    labels = np.where(results == 0, "soft", "hard")

    if X.shape[0] == 1:
        return labels.item()
    return labels.flatten().tolist()


def Hc_Mr_BHmax_from_Ms_A_K(
    Ms: mammos_entity.Entity | astropy.units.Quantity | numbers.Number,
    A: mammos_entity.Entity | astropy.units.Quantity | numbers.Number,
    Ku: mammos_entity.Entity | astropy.units.Quantity | numbers.Number,
    model: str = "random-forest-v1",
) -> mammos_analysis.hysteresis.ExtrinsicProperties:
    """Predict Hc, Mr and BHmax from micromagnetic properties Ms, A and Ku.

    This function predicts extrinsic properties coercive field Hc, remanent
    magnetization Mr and maximum energy product BHmax given a set of micromagnetic
    material parameters.

    Different models are available for the prediction. The following list provides an
    overview:

    - ``random-forest-v1``: TODO explain details of model, training data,
      Zenodo DOI to dataset and simulation script


    Args:
       Ms: Spontaneous magnetization.
       A: Exchange stiffness constant.
       Ku: Uniaxial anisotropy constant.
       model: AI model used for the prediction

    Returns:
       Extrinsic properties Hc, Mr, BHmax

    Examples:
    >>> import mammos_ai
    >>> import mammos_entity as me
    >>> mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(me.Ms(1e5), me.A(1e-12), me.Ku(1e6))
    ExtrinsicProperties(Hc=..., Mr=..., BHmax=...)
    """
    Ms = me.Ms(Ms, unit=u.A / u.m)
    A = me.A(A, unit=u.J / u.m)
    Ku = me.Ku(Ku, unit=u.J / u.m**3)

    Ms_arr = np.atleast_1d(Ms.value)
    A_arr = np.atleast_1d(A.value)
    Ku_arr = np.atleast_1d(Ku.value)

    if not (Ms_arr.shape == A_arr.shape == Ku_arr.shape):
        raise ValueError(
            f"Input arrays must have the same length. Shapes are Ms: {Ms_arr.shape}, "
            f"A: {A_arr.shape}, Ku: {Ku_arr.shape}"
        )

    match model:
        case "random-forest-v1":
            # 1. Determine class
            mat_class = classify_magnetic_from_Ms_A_K(Ms, A, Ku, model=model)

            # 2. Preprocess
            X_log = np.log1p(
                np.column_stack([Ms_arr, A_arr, Ku_arr]).astype(np.float32)
            )

            y_log = np.empty((X_log.shape[0], 3), dtype=np.float32)
            classes = np.atleast_1d(mat_class)

            for cls in ["soft", "hard"]:
                mask = classes == cls
                if np.any(mask):
                    # 3. Load regression model
                    session = ort.InferenceSession(str(MODELS[cls]), _SESSION_OPTIONS)
                    X_subset = X_log[mask]
                    # 4. Predict
                    res = session.run(None, {session.get_inputs()[0].name: X_subset})[0]
                    y_log[mask] = res

            # 5. Postprocess
            y = np.expm1(y_log)

            if X_log.shape[0] == 1:
                y = y[0]

        case _:
            raise ValueError(f"Unknown model {model}")

    if y.ndim == 1:
        Hc_val = y[0]
        Mr_val = y[1]
        BHmax_val = y[2]
    else:
        Hc_val = y[:, 0]
        Mr_val = y[:, 1]
        BHmax_val = y[:, 2]

    Hc = me.Hc(Hc_val, "A/m")
    Mr = me.Mr(Mr_val, "A/m")
    BHmax = me.BHmax(BHmax_val, "J/m3")
    return mammos_analysis.hysteresis.ExtrinsicProperties(Hc=Hc, Mr=Mr, BHmax=BHmax)
