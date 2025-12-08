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
    "soft": _MODEL_DIR / "random_forest_soft.onnx",
    "hard": _MODEL_DIR / "random_forest_hard.onnx",
}

_SESSION_OPTIONS = ort.SessionOptions()
_SESSION_OPTIONS.log_severity_level = 3


def classify_magnetic_from_Ms_A_K(
    Ms: mammos_entity.Entity | astropy.units.Quantity | numbers.Number,
    A: mammos_entity.Entity | astropy.units.Quantity | numbers.Number,
    Ku: mammos_entity.Entity | astropy.units.Quantity | numbers.Number,
    model: str = "random-forest-v1",
) -> str:
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

    match model:
        case "random-forest-v1":
            classifier_path = _MODEL_DIR / "classifier_random_forest_v1.onnx"
        case _:
            raise ValueError(f"Unknown model {model}")

    session = ort.InferenceSession(str(classifier_path), _SESSION_OPTIONS)
    X = np.array([[Ms.value, A.value, Ku.value]], dtype=np.float32)
    return (
        "soft"
        if session.run(None, {session.get_inputs()[0].name: X})[0][0] == 0
        else "hard"
    )


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
    # TODO for MPSD validate input arguments (required in multiple packages)
    # @MPCDF: assume the arguments are of type mammos_entity.Entity

    match model:
        case "random-forest-v1":
            # TODO call function for the correct model
            pass
        case _:
            raise ValueError(f"Unknown model {model}")

    Hc = me.Hc(0, "A/m")  # TODO: replace with model output
    Mr = me.Mr(0, "A/m")  # TODO: replace with model output
    BHmax = me.BHmax(0, "J/m3")  # TODO: replace with model output
    return mammos_analysis.hysteresis.ExtrinsicProperties(Hc=Hc, Mr=Mr, BHmax=BHmax)
