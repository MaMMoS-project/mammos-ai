import mammos_analysis
import mammos_entity as me
import numpy as np
import pytest

import mammos_ai


@pytest.mark.parametrize(
    "Ms,A,Ku",
    [
        (me.Ms(1e6), me.A(1e-12), me.Ku(1e6)),
        (me.Ms(1e6).q, me.A(1e-12).q, me.Ku(1e6).q),
        (me.Ms(1e6).value, me.A(1e-12).value, me.Ku(1e6).value),
        (me.Ms([1e6, 2e6]), me.A([1e-12, 2e-12]), me.Ku([1e6, 2e6])),
        (me.Ms([1e6, 2e6]).q, me.A([1e-12, 2e-12]).q, me.Ku([1e6, 2e6]).q),
        (me.Ms([1e6, 2e6]).value, me.A([1e-12, 2e-12]).value, me.Ku([1e6, 2e6]).value),
    ],
)
def test_classify_magnetic_from_Ms_A_K(Ms, A, Ku):
    classification = mammos_ai.classify_magnetic_from_Ms_A_K(Ms, A, Ku)

    if isinstance(classification, str):
        assert classification in ["soft", "hard"]
    else:
        assert isinstance(classification, list)
        assert np.all(np.isin(classification, ["soft", "hard"]))


def test_classify_magnetic_from_Ms_A_K_zeros():
    classification = mammos_ai.classify_magnetic_from_Ms_A_K(0, 0, 0)
    assert classification == "soft"


def test_classify_magnetic_from_Ms_A_K_soft():
    Ms = me.Ms(1e6)
    A = me.A(1e-12)
    Ku = me.Ku(1e3)
    classification = mammos_ai.classify_magnetic_from_Ms_A_K(Ms, A, Ku)
    assert classification == "soft"


def test_classify_magnetic_from_Ms_A_K_hard():
    Ms = me.Ms(1e6)
    A = me.A(1e-12)
    Ku = me.Ku(1e8)
    classification = mammos_ai.classify_magnetic_from_Ms_A_K(Ms, A, Ku)
    assert classification == "hard"


def test_classify_magnetic_from_Ms_A_K_specify_model():
    Ms = me.Ms(1e6)
    A = me.A(1e-12)
    Ku = me.Ku(1e6)

    classification = mammos_ai.classify_magnetic_from_Ms_A_K(
        Ms, A, Ku, model="random-forest-v1"
    )
    assert classification in ["soft", "hard"]

    with pytest.raises(ValueError):
        mammos_ai.classify_magnetic_from_Ms_A_K(Ms, A, Ku, model="non-existent-model")


@pytest.mark.parametrize(
    "Ms,A,Ku",
    [
        (me.Ms(1e6), me.A(1e-12), me.Ku(1e6)),
        (me.Ms(1e6).q, me.A(1e-12).q, me.Ku(1e6).q),
        (me.Ms(1e6).value, me.A(1e-12).value, me.Ku(1e6).value),
        (me.Ms([1e6, 2e6]), me.A([1e-12, 2e-12]), me.Ku([1e6, 2e6])),
        (me.Ms([1e6, 2e6]).q, me.A([1e-12, 2e-12]).q, me.Ku([1e6, 2e6]).q),
        (me.Ms([1e6, 2e6]).value, me.A([1e-12, 2e-12]).value, me.Ku([1e6, 2e6]).value),
    ],
)
def test_Hc_Mr_BHmax_from_Ms_A_K(Ms, A, Ku):
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


def test_Hc_Mr_BHmax_from_Ms_A_K_specify_model():
    Ms = me.Ms(1e6)
    A = me.A(1e-12)
    Ku = me.Ku(1e6)

    extrinsic_properties = mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(
        Ms, A, Ku, model="random-forest-v1"
    )

    assert isinstance(
        extrinsic_properties, mammos_analysis.hysteresis.ExtrinsicProperties
    )

    with pytest.raises(ValueError):
        mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(Ms, A, Ku, model="non-existent-model")


def test_classify_magnetic_array_inputs():
    """Test that array inputs are processed correctly for classification."""
    # Test with array inputs - soft and hard materials
    Ms = me.Ms([1e6, 1e6]).value
    A = me.A([1e-12, 1e-12]).value
    Ku = me.Ku([1e3, 1e8]).value  # First soft, second hard

    classification = mammos_ai.classify_magnetic_from_Ms_A_K(Ms, A, Ku)

    # Check that we get an array-like output with correct length and values
    assert isinstance(classification, list)
    assert len(classification) == 2
    assert classification[0] == "soft"
    assert classification[1] == "hard"


def test_Hc_Mr_BHmax_array_inputs():
    """Test that array inputs produce correct shape outputs for predictions."""
    # Test with multiple samples
    Ms = me.Ms([1e6, 2e6]).value
    A = me.A([1e-12, 2e-12]).value
    Ku = me.Ku([1e6, 2e6]).value

    extrinsic_properties = mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(Ms, A, Ku)

    # Verify output types
    assert isinstance(
        extrinsic_properties, mammos_analysis.hysteresis.ExtrinsicProperties
    )

    # Check shapes - should match input length of 2
    assert np.shape(extrinsic_properties.Hc.value) == (2,)
    assert np.shape(extrinsic_properties.Mr.value) == (2,)
    assert np.shape(extrinsic_properties.BHmax.value) == (2,)

    # All values should be positive
    assert np.all(extrinsic_properties.Hc.value > 0)
    assert np.all(extrinsic_properties.Mr.value > 0)
    assert np.all(extrinsic_properties.BHmax.value > 0)
