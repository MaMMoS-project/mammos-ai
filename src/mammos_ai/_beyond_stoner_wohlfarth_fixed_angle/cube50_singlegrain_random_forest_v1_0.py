"""Beyond Stoner-Wohlfarth fixed-angle, cube50 single-grain random forest, v1.0.

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
import mammos_units as u
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from ._common import MODEL_REPO_ID, SESSION_OPTIONS

NAME = "cube50_singlegrain_random_forest_v1.0"
MODEL_SUBFOLDER = "beyond-stoner-wohlfarth"

FILENAMES = {
    "classifier_valid_invalid": "classifier_valid_invalid_cube50_singlegrain_random_forest_v1.0.onnx",
    "classifier_hard_soft": "classifier_hard_soft_cube50_singlegrain_random_forest_v1.0.onnx",
    "soft": "soft_cube50_singlegrain_random_forest_v1.0.onnx",
    "hard": "hard_cube50_singlegrain_random_forest_v1.0.onnx",
}

_DESCRIPTION = (
    "Random forest model trained on extended simulated data for single grain "  # TODO: double check description.
    "cubic particles with 50 nm edge length with the external field "
    "applied parallel to the anisotropy axis."
)

_MODEL_SOURCE = "https://huggingface.co/mammos-project/mammos-ai-models"
_TRAINING_SOURCE = (
    "https://github.com/MaMMoS-project/ML-models/tree/main/beyond-stoner-wohlfarth/single-grain-easy-axis-model"
)

_TRAINING_DATA_RANGE = {
    "Ms": (
        me.Ms((0.1 * u.T).to(u.A / u.m, equivalencies=u.magnetic_flux_field())),
        me.Ms((5.0 * u.T).to(u.A / u.m, equivalencies=u.magnetic_flux_field())),
    ),
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
    """Return a local cached path for one ONNX file.

    If the file is not already cached, it will be downloaded from Hugging Face.
    """
    return hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=FILENAMES[model_key],
        subfolder=MODEL_SUBFOLDER,
    )


def _in_training_range(Ms_arr, A_arr, K1_arr) -> np.ndarray:
    """Check if each sample is within the training data range for all parameters."""
    Ms_min, Ms_max = (value.q.to_value("A/m") for value in _TRAINING_DATA_RANGE["Ms"])
    A_min, A_max = (value.q.to_value("J/m") for value in _TRAINING_DATA_RANGE["A"])
    K_min, K_max = (value.q.to_value("J/m3") for value in _TRAINING_DATA_RANGE["K"])

    in_range = (
        (Ms_arr >= Ms_min)
        & (Ms_arr <= Ms_max)
        & (A_arr >= A_min)
        & (A_arr <= A_max)
        & (K1_arr >= K_min)
        & (K1_arr <= K_max)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        l_A = np.sqrt(2 * A_arr / (u.constants.mu0.value * Ms_arr**2))

        l_K = np.sqrt(A_arr / K1_arr)

    threshold = 1e-9  # 1 nm

    in_range &= (l_A >= threshold) & (l_K >= threshold)
    return in_range


def is_hard_magnet(Ms_arr: np.ndarray, A_arr: np.ndarray, K1_arr: np.ndarray) -> np.ndarray:
    """Classify each sample as soft (False) or hard (True) magnetic.

    Args:
        Ms_arr: Spontaneous magnetization values in A/m.
        A_arr: Exchange stiffness values in J/m.
        K1_arr: Uniaxial anisotropy values in J/m^3.

    Returns:
        Flat boolean array of shape ``(N,)`` where N is the number of input
        samples after flattening.
    """
    in_range = _in_training_range(Ms_arr, A_arr, K1_arr)
    session = ort.InferenceSession(_model_path("classifier_hard_soft"), SESSION_OPTIONS)
    X = np.column_stack([Ms_arr.ravel(), A_arr.ravel(), K1_arr.ravel()]).astype(np.float32)
    # The classifier expects input shape (n_samples, 3) containing [Ms, A, K1]
    # and returns a 1-D array of class labels (0=soft, 1=hard).
    results = session.run(None, {session.get_inputs()[0].name: X})[0]
    labels = results.astype(bool).astype(object)
    # TODO: add valid-invalid classification here
    labels[~in_range.ravel()] = np.nan
    return labels.reshape(Ms_arr.shape)


def predict_extrinsic(Ms_arr: np.ndarray, A_arr: np.ndarray, K1_arr: np.ndarray) -> np.ndarray:
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

    X_log = np.log1p(np.column_stack([Ms_arr.ravel(), A_arr.ravel(), K1_arr.ravel()]).astype(np.float32))

    y_log = np.full((X_log.shape[0], 3), np.nan, dtype=np.float32)
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
