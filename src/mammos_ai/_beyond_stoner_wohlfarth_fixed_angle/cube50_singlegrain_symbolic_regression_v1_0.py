"""Beyond Stoner-Wohlfarth fixed-angle, cube50 single-grain symbolic regression, v1.0.

The symbolic regression model generates data-driven mathematical equations
by learning algorithmically the relationships between inputs and outputs.
The model was trained on simulated data for single grain cubic particles
with 50 nm edge length, and external field parallel to the anisotropy axis.

More information on the equations and the use of this model is available at:
https://mammos-project.github.io/mammos/examples/mammos-ai/symbolic_regression.html.

TODO: add citation / DOI from paper.
"""

from __future__ import annotations

import mammos_entity as me
import mammos_units as u
import numpy as np

NAME = "cube50_singlegrain_symbolic_regression_v1.0"

_DESCRIPTION = (
    "Symbolic regression model trained on extended simulated data for single "
    "grain cubic particles with 50 nm edge length with the external field "
    "applied parallel to the anisotropy axis."
)

_TRAINING_DATA_RANGE = {
    "Ms": (
        me.Ms((0.1 * u.T).to(u.A / u.m, equivalencies=u.magnetic_flux_field())),
        me.Ms((5.0 * u.T).to(u.A / u.m, equivalencies=u.magnetic_flux_field())),
    ),
    "A": (me.A(1e-13), me.A(1e-11)),
    "K": (me.Ku(1e4), me.Ku(1e7)),
}

_MODEL_SOURCE = "https://doi.org/10.48550/arXiv.2607.29249"
# TODO: add paper DOI when published

_TRAINING_SOURCE = "TODO"  # TODO: add training repo

PREDICT_METADATA = {
    "model_name": NAME,
    "description": _DESCRIPTION,
    "training_data_range": _TRAINING_DATA_RANGE,
    "input_parameters": ["Ms (A/m)", "A (J/m)", "K1 (J/m^3)"],
    "output_parameters": ["Hc (A/m)", "Mr (A/m)", "BHmax (J/m^3)"],
    "source": _MODEL_SOURCE,
    "training_source": _TRAINING_SOURCE,
}


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
    mu0_Ms = u.constants.mu0.value * Ms_arr
    mu0_Ms2 = mu0_Ms * Ms_arr
    H_A = 2 * K1_arr / mu0_Ms
    BH_s = mu0_Ms2 / 4
    kappa = np.sqrt(K1_arr / mu0_Ms2)
    l_ex = np.sqrt(2 * A_arr / mu0_Ms2)
    L_tilde = 50e-9 / l_ex
    fact = L_tilde / (kappa**4)

    alpha = 0.942
    n = 0.0921
    eps_m = 5.18e-5
    eps_b = 7.81e-5

    Hc = (alpha - n * np.log(L_tilde) / kappa) * H_A
    Mr = (1 - eps_m * fact) * Ms_arr
    BH_max = (1 - eps_b * fact) ** 2 * BH_s

    return Hc, Mr, BH_max
