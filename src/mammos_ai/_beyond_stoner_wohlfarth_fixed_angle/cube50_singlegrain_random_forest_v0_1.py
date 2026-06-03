"""Beyond Stoner-Wohlfarth fixed-angle, cube50 single-grain random forest, v0.1.

Random forest model trained on simulated data for single grain cubic particles
with 50 nm edge length, external field parallel to the anisotropy axis.
See the
`Hugging Face model repository <https://huggingface.co/mammos-project/mammos-ai-models>`_
for model files and the
`training repository <https://github.com/MaMMoS-project/ML-models/tree/main/beyond-stoner-wohlfarth/single-grain-easy-axis-model>`_
for full training-data details.
"""

from __future__ import annotations

import mammos_entity as me
import numpy as np
import onnxruntime as ort

from ._common import SESSION_OPTIONS, download_model_file

NAME = "cube50_singlegrain_random_forest_v0.1"
MODEL_SUBFOLDER = "beyond-stoner-wohlfarth"

FILENAMES = {
    "classifier": "classifier_cube50_singlegrain_random_forest_v0.1.onnx",
    "soft": "soft_cube50_singlegrain_random_forest_v0.1.onnx",
    "hard": "hard_cube50_singlegrain_random_forest_v0.1.onnx",
}

_DESCRIPTION = (
    "Random forest model trained on simulated data for single grain "
    "cubic particles with 50 nm edge length with the external field "
    "applied parallel to the anisotropy axis."
)

_MODEL_SOURCE = "https://huggingface.co/mammos-project/mammos-ai-models"
_TRAINING_SOURCE = (
    "https://github.com/MaMMoS-project/ML-models/tree/main/"
    "beyond-stoner-wohlfarth/single-grain-easy-axis-model"
)

_TRAINING_DATA_RANGE = {
    "Ms": (me.Ms(79.58e3), me.Ms(3.98e6)),
    "A": (me.A(1e-13), me.A(1e-11)),
    "K": (me.Ku(1e4), me.Ku(1e7)),
}

CLASSIFY_METADATA = {
    "model_name": NAME,
    "description": _DESCRIPTION,
    "training_data_range": _TRAINING_DATA_RANGE,
    "input_parameters": ["Ms (A/m)", "A (J/m)", "K1 (J/m^3)"],
    "output_classes": {0: "soft magnetic", 1: "hard magnetic"},
    "source": _MODEL_SOURCE,
    "training_source": _TRAINING_SOURCE,
}

PREDICT_METADATA = {
    "model_name": NAME,
    "description": _DESCRIPTION,
    "training_data_range": _TRAINING_DATA_RANGE,
    "input_parameters": ["Ms (A/m)", "A (J/m)", "K1 (J/m^3)"],
    "output_parameters": ["Hc (A/m)", "Mr (A/m)", "BHmax (J/m^3)"],
    "source": _MODEL_SOURCE,
    "training_source": _TRAINING_SOURCE,
}


def _model_path(model_key: str) -> str:
    """Return a local cached path for one ONNX file."""
    return download_model_file(FILENAMES[model_key], MODEL_SUBFOLDER)


def is_hard_magnet(
    Ms_arr: np.ndarray, A_arr: np.ndarray, K1_arr: np.ndarray
) -> np.ndarray:
    """Classify each sample as soft (False) or hard (True) magnetic.

    Args:
        Ms_arr: Spontaneous magnetization values in A/m.
        A_arr: Exchange stiffness values in J/m.
        K1_arr: Uniaxial anisotropy values in J/m^3.

    Returns:
        Flat boolean array of shape ``(N,)`` where N is the number of input
        samples after flattening.
    """
    session = ort.InferenceSession(_model_path("classifier"), SESSION_OPTIONS)
    X = np.column_stack([Ms_arr.ravel(), A_arr.ravel(), K1_arr.ravel()]).astype(
        np.float32
    )
    # The classifier expects input shape (n_samples, 3) containing [Ms, A, K1]
    # and returns a 1-D array of class labels (0=soft, 1=hard).
    results = session.run(None, {session.get_inputs()[0].name: X})[0]
    return np.where(results == 0, False, True).reshape(Ms_arr.shape)


def predict_extrinsic(
    Ms_arr: np.ndarray, A_arr: np.ndarray, K1_arr: np.ndarray
) -> np.ndarray:
    """Predict Hc, Mr and BHmax for each sample.

    Args:
        Ms_arr: Spontaneous magnetization values in A/m.
        A_arr: Exchange stiffness values in J/m.
        K1_arr: Uniaxial anisotropy values in J/m^3.

    Returns:
        Array of shape ``(N, 3)`` containing ``[Hc, Mr, BHmax]`` predictions in
        SI units.
    """
    mat_class = is_hard_magnet(Ms_arr, A_arr, K1_arr)

    X_log = np.log1p(
        np.column_stack([Ms_arr.ravel(), A_arr.ravel(), K1_arr.ravel()]).astype(
            np.float32
        )
    )

    y_log = np.empty((X_log.shape[0], 3), dtype=np.float32)
    classes = np.atleast_1d(mat_class).ravel()
    for is_hard in (False, True):
        mask = classes == is_hard
        if np.any(mask):
            path = _model_path("hard" if is_hard else "soft")
            session = ort.InferenceSession(path, SESSION_OPTIONS)
            X_subset = X_log[mask]
            # Regressor expects (n, 3) [Ms, A, K1]; returns (n, 3)
            # [Hc, Mr, BHmax] predictions in log1p space.
            res = session.run(None, {session.get_inputs()[0].name: X_subset})[0]
            y_log[mask] = res

    out = np.expm1(y_log).reshape(Ms_arr.shape + (3,))
    Hc_val = out[..., 0]
    Mr_val = out[..., 1]
    BHmax_val = out[..., 2]
    return Hc_val, Mr_val, BHmax_val
