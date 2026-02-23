import mammos_analysis
import mammos_entity as me
import numpy as np
import pytest

import mammos_ai
from mammos_ai._beyond_stoner_wohlfarth_fixed_angle import MODELS

requires_v0_2 = pytest.mark.skipif(
    not all(
        p.exists() for p in MODELS["cube50_singlegrain_random_forest_v0.2"].values()
    ),
    reason="v0.2 model files not available",
)


@pytest.mark.parametrize("Ms", [me.Ms(1e6), me.Ms(1e6).q, me.Ms(1e6).value])
@pytest.mark.parametrize("A", [me.A(1e-12), me.A(1e-12).q, me.A(1e-12).value])
@pytest.mark.parametrize("Ku", [me.Ku(1e6), me.Ku(1e6).q, me.Ku(1e6).value])
def test_classify_magnetic_from_Ms_A_K_single_input(Ms, A, Ku):
    """Test classification of magnetic materials from Ms, A, Ku."""
    classification = mammos_ai.is_hard_magnet_from_Ms_A_K(Ms, A, Ku)
    assert classification in [True, False]


@pytest.mark.parametrize(
    "Ms", [me.Ms([5e5, 1e6]), me.Ms([5e5, 1e6]).q, me.Ms([5e5, 1e6]).value]
)
@pytest.mark.parametrize(
    "A", [me.A([5e-12, 1e-11]), me.A([5e-12, 1e-11]).q, me.A([5e-12, 1e-11]).value]
)
@pytest.mark.parametrize(
    "Ku", [me.Ku([5e4, 1e5]), me.Ku([5e4, 1e5]).q, me.Ku([5e4, 1e5]).value]
)
def test_classify_magnetic_from_Ms_A_K_1d_array(Ms, A, Ku):
    """Test classification of magnetic materials from Ms, A, Ku."""
    classification = mammos_ai.is_hard_magnet_from_Ms_A_K(Ms, A, Ku)
    assert isinstance(classification, np.ndarray)
    assert classification.shape == (2,)
    assert all(c in [True, False] for c in classification)


@pytest.mark.parametrize(
    "Ms",
    [
        me.Ms([[1e5, 5e5], [2e5, 3e5]]),
        me.Ms([[1e5, 5e5], [2e5, 3e5]]).q,
        me.Ms([[1e5, 5e5], [2e5, 3e5]]).value,
    ],
)
@pytest.mark.parametrize(
    "A",
    [
        me.A([[5e-12, 1e-11], [7e-12, 8e-12]]),
        me.A([[5e-12, 1e-11], [7e-12, 8e-12]]).q,
        me.A([[5e-12, 1e-11], [7e-12, 8e-12]]).value,
    ],
)
@pytest.mark.parametrize(
    "Ku",
    [
        me.Ku([[5e4, 1e5], [2e4, 3e4]]),
        me.Ku([[5e4, 1e5], [2e4, 3e4]]).q,
        me.Ku([[5e4, 1e5], [2e4, 3e4]]).value,
    ],
)
def test_classify_magnetic_from_Ms_A_K_nd_array(Ms, A, Ku):
    """Test classification of magnetic materials from Ms, A, Ku."""
    classification = mammos_ai.is_hard_magnet_from_Ms_A_K(Ms, A, Ku)
    assert isinstance(classification, np.ndarray)
    assert classification.shape == (2, 2)
    assert all(c in [True, False] for c in classification.flatten())


def test_classify_magnetic_from_Ms_A_K_out_of_range():
    """Test that out-of-range inputs return NaN."""
    classification = mammos_ai.is_hard_magnet_from_Ms_A_K(0, 0, 0)
    assert np.isnan(classification)


def test_classify_magnetic_from_Ms_A_K_soft():
    """Test classification of a soft magnetic material."""
    Ms = me.Ms(2e6)
    A = me.A(1e-11)
    Ku = me.Ku(1e4)
    classification = mammos_ai.is_hard_magnet_from_Ms_A_K(Ms, A, Ku)
    assert not classification


def test_classify_magnetic_from_Ms_A_K_hard():
    """Test classification of a hard magnetic material."""
    Ms = me.Ms(1e6)
    A = me.A(1e-11)
    Ku = me.Ku(1e5)
    classification = mammos_ai.is_hard_magnet_from_Ms_A_K(Ms, A, Ku)
    assert classification


def test_classify_magnetic_from_Ms_A_K_specify_model():
    """Test specifying different models for classification."""
    Ms = me.Ms(1e6)
    A = me.A(1e-12)
    Ku = me.Ku(1e6)

    classification = mammos_ai.is_hard_magnet_from_Ms_A_K(
        Ms, A, Ku, model="cube50_singlegrain_random_forest_v0.1"
    )
    assert classification in [True, False]

    with pytest.raises(ValueError):
        mammos_ai.is_hard_magnet_from_Ms_A_K(Ms, A, Ku, model="non-existent-model")


def test_classify_magnetic_array_inputs():
    """Test that array inputs are processed correctly for classification."""
    Ms = me.Ms([2e6, 1e6])
    A = me.A([1e-11, 1e-11])
    Ku = me.Ku([1e4, 1e5])

    classification = mammos_ai.is_hard_magnet_from_Ms_A_K(Ms, A, Ku)

    assert isinstance(classification, np.ndarray)
    assert len(classification) == 2
    assert not classification[0]
    assert classification[1]


def test_classify_magnetic_array_inputs_mixed_lengths():
    """Test that array inputs of mixed lengths raise an error."""
    Ms = me.Ms(1e6)
    A = me.A([5e-12, 1e-11])
    Ku = me.Ku([5e4, 1e5])

    with pytest.raises(ValueError, match="Input arrays must have the same shape"):
        mammos_ai.is_hard_magnet_from_Ms_A_K(Ms, A, Ku)

    Ms = me.Ms([1e6])
    A = me.A([5e-12, 1e-11])
    Ku = me.Ku([5e4, 1e5])

    with pytest.raises(ValueError, match="Input arrays must have the same shape"):
        mammos_ai.is_hard_magnet_from_Ms_A_K(Ms, A, Ku)


@pytest.mark.parametrize("Ms", [me.Ms(1e6), me.Ms(1e6).q, me.Ms(1e6).value])
@pytest.mark.parametrize("A", [me.A(1e-12), me.A(1e-12).q, me.A(1e-12).value])
@pytest.mark.parametrize("Ku", [me.Ku(1e6), me.Ku(1e6).q, me.Ku(1e6).value])
def test_Hc_Mr_BHmax_from_Ms_A_K_single_input(Ms, A, Ku):
    """Test Hc, Mr, BHmax prediction from Ms, A, Ku."""
    extrinsic_properties = mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(Ms, A, Ku)

    assert isinstance(
        extrinsic_properties, mammos_analysis.hysteresis.ExtrinsicProperties
    )
    assert isinstance(extrinsic_properties.Hc, me.Entity)
    assert isinstance(extrinsic_properties.Mr, me.Entity)
    assert isinstance(extrinsic_properties.BHmax, me.Entity)

    assert np.all(extrinsic_properties.Hc.q > 0)
    assert np.all(extrinsic_properties.Mr.q > 0)
    assert np.all(extrinsic_properties.BHmax.q > 0)


@pytest.mark.parametrize(
    "Ms", [me.Ms([5e5, 1e6]), me.Ms([5e5, 1e6]).q, me.Ms([5e5, 1e6]).value]
)
@pytest.mark.parametrize(
    "A", [me.A([5e-12, 1e-11]), me.A([5e-12, 1e-11]).q, me.A([5e-12, 1e-11]).value]
)
@pytest.mark.parametrize(
    "Ku", [me.Ku([5e4, 1e5]), me.Ku([5e4, 1e5]).q, me.Ku([5e4, 1e5]).value]
)
def test_Hc_Mr_BHmax_from_Ms_A_K_1d_array(Ms, A, Ku):
    """Test Hc, Mr, BHmax prediction from Ms, A, Ku."""
    extrinsic_properties = mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(Ms, A, Ku)

    assert isinstance(
        extrinsic_properties, mammos_analysis.hysteresis.ExtrinsicProperties
    )
    assert isinstance(extrinsic_properties.Hc, me.Entity)
    assert isinstance(extrinsic_properties.Mr, me.Entity)
    assert isinstance(extrinsic_properties.BHmax, me.Entity)

    assert np.all(extrinsic_properties.Hc.q > 0)
    assert np.all(extrinsic_properties.Mr.q > 0)
    assert np.all(extrinsic_properties.BHmax.q > 0)


@pytest.mark.parametrize("model", ["cube50_singlegrain_random_forest_v0.1"])
def test_Hc_Mr_BHmax_from_Ms_A_K_specify_model(model):
    """Test specifying different models for Hc, Mr, BHmax prediction."""
    Ms = me.Ms(1e6)
    A = me.A(1e-12)
    Ku = me.Ku(1e6)

    extrinsic_properties = mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(Ms, A, Ku, model=model)

    assert isinstance(
        extrinsic_properties, mammos_analysis.hysteresis.ExtrinsicProperties
    )

    with pytest.raises(ValueError):
        mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(Ms, A, Ku, model="non-existent-model")


def test_Hc_Mr_BHmax_2d_array_inputs():
    """Test that array inputs produce correct shape outputs for predictions."""
    Ms = me.Ms([[1e5, 5e5], [2e5, 3e5]])
    A = me.A([[5e-12, 1e-11], [7e-12, 8e-12]])
    Ku = me.Ku([[5e4, 1e5], [2e4, 3e4]])
    extrinsic_properties = mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(Ms, A, Ku)

    assert isinstance(
        extrinsic_properties, mammos_analysis.hysteresis.ExtrinsicProperties
    )

    assert np.shape(extrinsic_properties.Hc.value) == (2, 2)
    assert np.shape(extrinsic_properties.Mr.value) == (2, 2)
    assert np.shape(extrinsic_properties.BHmax.value) == (2, 2)

    assert np.all(extrinsic_properties.Hc.value > 0)
    assert np.all(extrinsic_properties.Mr.value > 0)
    assert np.all(extrinsic_properties.BHmax.value > 0)


def test_Hc_Mr_BHmax_array_inputs_mixed_lengths():
    """Test that array inputs of mixed lengths raise an error."""
    Ms = me.Ms(1e6)
    A = me.A([1e-12, 2e-12])
    Ku = me.Ku([1e6, 2e6])

    with pytest.raises(ValueError, match="Input arrays must have the same shape"):
        mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(Ms, A, Ku)

    Ms = me.Ms([1e6])
    A = me.A([1e-12, 2e-12])
    Ku = me.Ku([1e6, 2e6])

    with pytest.raises(ValueError, match="Input arrays must have the same shape"):
        mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(Ms, A, Ku)


def test_is_hard_magnet_metadata_default_model():
    """Test metadata function returns dict with model_name."""
    metadata = mammos_ai.is_hard_magnet_from_Ms_A_K_metadata()

    assert isinstance(metadata, dict)
    assert "model_name" in metadata
    assert metadata["model_name"] == "cube50_singlegrain_random_forest_v0.1"


def test_is_hard_magnet_metadata_specified_model():
    """Test metadata function with explicitly specified model."""
    metadata = mammos_ai.is_hard_magnet_from_Ms_A_K_metadata(
        model="cube50_singlegrain_random_forest_v0.1"
    )

    assert isinstance(metadata, dict)
    assert "model_name" in metadata


def test_is_hard_magnet_metadata_unknown_model():
    """Test metadata function raises error for unknown model."""
    with pytest.raises(ValueError, match="Unknown model"):
        mammos_ai.is_hard_magnet_from_Ms_A_K_metadata(model="non-existent-model")


def test_Hc_Mr_BHmax_metadata_default_model():
    """Test metadata function returns dict with model_name."""
    metadata = mammos_ai.Hc_Mr_BHmax_from_Ms_A_K_metadata()

    assert isinstance(metadata, dict)
    assert "model_name" in metadata
    assert metadata["model_name"] == "cube50_singlegrain_random_forest_v0.1"


def test_Hc_Mr_BHmax_metadata_specified_model():
    """Test metadata function with explicitly specified model."""
    metadata = mammos_ai.Hc_Mr_BHmax_from_Ms_A_K_metadata(
        model="cube50_singlegrain_random_forest_v0.1"
    )

    assert isinstance(metadata, dict)
    assert "model_name" in metadata


def test_Hc_Mr_BHmax_metadata_unknown_model():
    """Test metadata function raises error for unknown model."""
    with pytest.raises(ValueError, match="Unknown model"):
        mammos_ai.Hc_Mr_BHmax_from_Ms_A_K_metadata(model="non-existent-model")


def test_Hc_Mr_BHmax_out_of_range_returns_nan():
    """Test that out-of-range scalar inputs return NaN predictions."""
    result = mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(me.Ms(1e8), me.A(1e-12), me.Ku(1e6))
    assert np.isnan(result.Hc.value)
    assert np.isnan(result.Mr.value)
    assert np.isnan(result.BHmax.value)


# --- v0.2 tests ---
#
# Exchange-length constraint case: Ms=3.9e6 A/m, A=1e-13 J/m, K1=1e4 J/m^3
# gives l_A = sqrt(2*A / (mu0*Ms^2)) ≈ 0.32 nm < 1 nm threshold.
# All three parameters are within the Ms/A/K1 box.
# v0.1 has no length constraint  → valid result.
# v0.2 enforces the constraint   → NaN.
_MS_SHORT_EXCH = me.Ms(3.9e6)
_A_SHORT_EXCH = me.A(1e-13)
_K1_SHORT_EXCH = me.Ku(1e4)


def test_length_constraint_v0_1_not_enforced():
    """v0.1 does not enforce exchange length constraint.

    Short-exchange-length inputs are accepted as valid.
    """
    result = mammos_ai.is_hard_magnet_from_Ms_A_K(
        _MS_SHORT_EXCH,
        _A_SHORT_EXCH,
        _K1_SHORT_EXCH,
        model="cube50_singlegrain_random_forest_v0.1",
    )
    assert result in [True, False]


@requires_v0_2
def test_length_constraint_v0_2_enforced_classify():
    """v0.2 enforces exchange length >= 1 nm.

    Short-exchange-length inputs return NaN.
    """
    result = mammos_ai.is_hard_magnet_from_Ms_A_K(
        _MS_SHORT_EXCH,
        _A_SHORT_EXCH,
        _K1_SHORT_EXCH,
        model="cube50_singlegrain_random_forest_v0.2",
    )
    assert np.isnan(result)


@requires_v0_2
def test_length_constraint_v0_2_enforced_predict():
    """v0.2 enforces exchange length >= 1 nm.

    Short-exchange-length inputs return NaN.
    """
    result = mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(
        _MS_SHORT_EXCH,
        _A_SHORT_EXCH,
        _K1_SHORT_EXCH,
        model="cube50_singlegrain_random_forest_v0.2",
    )
    assert np.isnan(result.Hc.value)
    assert np.isnan(result.Mr.value)
    assert np.isnan(result.BHmax.value)


@requires_v0_2
def test_classify_magnetic_from_Ms_A_K_v0_2():
    """Test classification works with v0.2 model for an in-range input."""
    result = mammos_ai.is_hard_magnet_from_Ms_A_K(
        me.Ms(1e6),
        me.A(1e-12),
        me.Ku(1e6),
        model="cube50_singlegrain_random_forest_v0.2",
    )
    assert result in [True, False]


@requires_v0_2
def test_Hc_Mr_BHmax_from_Ms_A_K_v0_2():
    """Test Hc, Mr, BHmax prediction works with v0.2 model for an in-range input."""
    result = mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(
        me.Ms(1e6),
        me.A(1e-12),
        me.Ku(1e6),
        model="cube50_singlegrain_random_forest_v0.2",
    )
    assert isinstance(result, mammos_analysis.hysteresis.ExtrinsicProperties)
    assert np.all(result.Hc.q > 0)
    assert np.all(result.Mr.q > 0)
    assert np.all(result.BHmax.q > 0)


def test_is_hard_magnet_metadata_v0_2():
    """Test metadata for v0.2 classifier has correct model_name."""
    metadata = mammos_ai.is_hard_magnet_from_Ms_A_K_metadata(
        model="cube50_singlegrain_random_forest_v0.2"
    )
    assert isinstance(metadata, dict)
    assert metadata["model_name"] == "cube50_singlegrain_random_forest_v0.2"


def test_Hc_Mr_BHmax_metadata_v0_2():
    """Test metadata for v0.2 predictor has correct model_name."""
    metadata = mammos_ai.Hc_Mr_BHmax_from_Ms_A_K_metadata(
        model="cube50_singlegrain_random_forest_v0.2"
    )
    assert isinstance(metadata, dict)
    assert metadata["model_name"] == "cube50_singlegrain_random_forest_v0.2"
