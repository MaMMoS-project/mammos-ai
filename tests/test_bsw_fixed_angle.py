import mammos_analysis
import mammos_entity as me
import numpy as np
import pytest

import mammos_ai
from mammos_ai._beyond_stoner_wohlfarth_fixed_angle import (
    cube50_singlegrain_random_forest_v0_1 as cube50_model,
)


def test_in_training_range_2d_array_inputs():
    """Test that training range checks preserve input shape."""
    Ms = np.array([[79.58e3, 3.98e6], [79.58e3 - 1, 79.58e3]])
    A = np.array([[1e-13, 1e-11], [1e-13, 1e-14]])
    Ku = np.array([[1e4, 1e7], [1e4, 1e4]])

    in_range = cube50_model._in_training_range(Ms, A, Ku)

    assert isinstance(in_range, np.ndarray)
    assert in_range.shape == (2, 2)
    assert np.all(in_range == np.array([[True, True], [False, False]]))


def test_classify_magnetic_from_Ms_A_K_out_of_range_2d_array():
    """Test that out-of-training-range array inputs preserve input shape."""
    Ms = me.Ms([[1e6, 1e6], [1e6, 1e6]])
    A = me.A([[1e-12, 1e-12], [1e-12, 1e-12]])
    Ku = me.Ku([[1e4, 1e6], [1e3, 1e8]])

    classification = mammos_ai.is_hard_magnet_from_Ms_A_K(Ms, A, Ku)

    assert isinstance(classification, np.ndarray)
    assert classification.shape == (2, 2)
    assert not classification[0, 0]
    assert classification[0, 1]
    assert np.isnan(classification[1, 0])
    assert np.isnan(classification[1, 1])


@pytest.mark.parametrize("Ms", [me.Ms(1e6), me.Ms(1e6).q, me.Ms(1e6).value])
@pytest.mark.parametrize("A", [me.A(1e-12), me.A(1e-12).q, me.A(1e-12).value])
@pytest.mark.parametrize("Ku", [me.Ku(1e6), me.Ku(1e6).q, me.Ku(1e6).value])
def test_classify_magnetic_from_Ms_A_K_single_input(Ms, A, Ku):
    """Test classification of magnetic materials from Ms, A, Ku."""
    classification = mammos_ai.is_hard_magnet_from_Ms_A_K(Ms, A, Ku)
    assert classification in [True, False]


@pytest.mark.parametrize("Ms", [me.Ms([1e6, 0.5e6]), me.Ms([1e6, 0.5e6]).q, me.Ms([1e6, 0.5e6]).value])
@pytest.mark.parametrize("A", [me.A([1e-12, 2e-12]), me.A([1e-12, 2e-12]).q, me.A([1e-12, 2e-12]).value])
@pytest.mark.parametrize("Ku", [me.Ku([1e6, 2e6]), me.Ku([1e6, 2e6]).q, me.Ku([1e6, 2e6]).value])
def test_classify_magnetic_from_Ms_A_K_1d_array(Ms, A, Ku):
    """Test classification of magnetic materials from Ms, A, Ku."""
    classification = mammos_ai.is_hard_magnet_from_Ms_A_K(Ms, A, Ku)
    assert isinstance(classification, np.ndarray)
    assert classification.shape == (2,)
    assert all(c in [True, False] for c in classification)


@pytest.mark.parametrize(
    "Ms",
    [
        me.Ms([[1e5, 2e5], [3e5, 3.9e5]]),
        me.Ms([[1e5, 2e5], [3e5, 3.9e5]]).q,
        me.Ms([[1e5, 2e5], [3e5, 3.9e5]]).value,
    ],
)
@pytest.mark.parametrize(
    "A",
    [
        me.A([[1e-12, 2e-12], [3e-12, 4e-12]]),
        me.A([[1e-12, 2e-12], [3e-12, 4e-12]]).q,
        me.A([[1e-12, 2e-12], [3e-12, 4e-12]]).value,
    ],
)
@pytest.mark.parametrize(
    "Ku",
    [
        me.Ku([[1e5, 2e5], [3e5, 4e5]]),
        me.Ku([[1e5, 2e5], [3e5, 4e5]]).q,
        me.Ku([[1e5, 2e5], [3e5, 4e5]]).value,
    ],
)
def test_classify_magnetic_from_Ms_A_K_nd_array(Ms, A, Ku):
    """Test classification of magnetic materials from Ms, A, Ku."""
    classification = mammos_ai.is_hard_magnet_from_Ms_A_K(Ms, A, Ku)
    assert isinstance(classification, np.ndarray)
    assert classification.shape == (2, 2)
    assert all(c in [True, False] for c in classification.flatten())


def test_classify_magnetic_from_Ms_A_K_zeros():
    classification = mammos_ai.is_hard_magnet_from_Ms_A_K(0, 0, 0)
    assert np.isnan(classification.item())


def test_classify_magnetic_from_Ms_A_K_soft():
    """Test classification of a soft magnetic material."""
    Ms = me.Ms(1e6)
    A = me.A(1e-12)
    Ku = me.Ku(1e4)
    classification = mammos_ai.is_hard_magnet_from_Ms_A_K(Ms, A, Ku)
    assert not classification


def test_classify_magnetic_from_Ms_A_K_hard():
    """Test classification of a hard magnetic material."""
    Ms = me.Ms(1e6)
    A = me.A(1e-12)
    Ku = me.Ku(1e6)
    classification = mammos_ai.is_hard_magnet_from_Ms_A_K(Ms, A, Ku)
    assert classification


def test_classify_magnetic_from_Ms_A_K_specify_model():
    """Test specifying different models for classification."""
    Ms = me.Ms(1e6)
    A = me.A(1e-12)
    Ku = me.Ku(1e6)

    classification = mammos_ai.is_hard_magnet_from_Ms_A_K(Ms, A, Ku, model="cube50_singlegrain_random_forest_v0.1")
    assert classification in [True, False]

    with pytest.raises(ValueError):
        mammos_ai.is_hard_magnet_from_Ms_A_K(Ms, A, Ku, model="non-existent-model")


def test_classify_magnetic_array_inputs():
    """Test that array inputs are processed correctly for classification."""
    Ms = me.Ms([1e6, 1e6])
    A = me.A([1e-12, 1e-12])
    Ku = me.Ku([1e4, 1e6])

    classification = mammos_ai.is_hard_magnet_from_Ms_A_K(Ms, A, Ku)

    assert isinstance(classification, np.ndarray)
    assert len(classification) == 2
    assert not classification[0]
    assert classification[1]


def test_classify_magnetic_array_inputs_mixed_lengths():
    """Test that array inputs of mixed lengths raise an error."""
    Ms = me.Ms(1e6)
    A = me.A([1e-12, 1e-12])
    Ku = me.Ku([1e3, 1e8])

    with pytest.raises(ValueError, match="Input arrays must have the same shape"):
        mammos_ai.is_hard_magnet_from_Ms_A_K(Ms, A, Ku)

    Ms = me.Ms([1e6])
    A = me.A([1e-12, 2e-12])
    Ku = me.Ku([1e3, 1e8])

    with pytest.raises(ValueError, match="Input arrays must have the same shape"):
        mammos_ai.is_hard_magnet_from_Ms_A_K(Ms, A, Ku)


@pytest.mark.parametrize("Ms", [me.Ms(1e6), me.Ms(1e6).q, me.Ms(1e6).value])
@pytest.mark.parametrize("A", [me.A(1e-12), me.A(1e-12).q, me.A(1e-12).value])
@pytest.mark.parametrize("Ku", [me.Ku(1e6), me.Ku(1e6).q, me.Ku(1e6).value])
def test_Hc_Mr_BHmax_from_Ms_A_K_single_input(Ms, A, Ku):
    """Test Hc, Mr, BHmax prediction from Ms, A, Ku."""
    extrinsic_properties = mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(Ms, A, Ku)

    assert isinstance(extrinsic_properties, mammos_analysis.hysteresis.ExtrinsicProperties)
    assert isinstance(extrinsic_properties.Hc, me.Entity)
    assert isinstance(extrinsic_properties.Mr, me.Entity)
    assert isinstance(extrinsic_properties.BHmax, me.Entity)

    assert np.all(extrinsic_properties.Hc.q > 0)
    assert np.all(extrinsic_properties.Mr.q > 0)
    assert np.all(extrinsic_properties.BHmax.q > 0)


@pytest.mark.parametrize("Ms", [me.Ms([1e5, 2e5]), me.Ms([1e5, 2e5]).q, me.Ms([1e5, 2e5]).value])
@pytest.mark.parametrize("A", [me.A([1e-12, 2e-12]), me.A([1e-12, 2e-12]).q, me.A([1e-12, 2e-12]).value])
@pytest.mark.parametrize("Ku", [me.Ku([1e5, 2e5]), me.Ku([1e5, 2e5]).q, me.Ku([1e5, 2e5]).value])
def test_Hc_Mr_BHmax_from_Ms_A_K_1d_array(Ms, A, Ku):
    """Test Hc, Mr, BHmax prediction from Ms, A, Ku."""
    extrinsic_properties = mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(Ms, A, Ku)

    assert isinstance(extrinsic_properties, mammos_analysis.hysteresis.ExtrinsicProperties)
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

    assert isinstance(extrinsic_properties, mammos_analysis.hysteresis.ExtrinsicProperties)

    with pytest.raises(ValueError):
        mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(Ms, A, Ku, model="non-existent-model")


def test_Hc_Mr_BHmax_2d_array_inputs():
    """Test that array inputs produce correct shape outputs for predictions."""
    Ms = me.Ms([[1e5, 2e5], [3e5, 3.9e5]])
    A = me.A([[1e-12, 2e-12], [3e-12, 4e-12]])
    Ku = me.Ku([[1e5, 2e5], [3e5, 4e5]])
    extrinsic_properties = mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(Ms, A, Ku)

    assert isinstance(extrinsic_properties, mammos_analysis.hysteresis.ExtrinsicProperties)

    assert np.shape(extrinsic_properties.Hc.value) == (2, 2)
    assert np.shape(extrinsic_properties.Mr.value) == (2, 2)
    assert np.shape(extrinsic_properties.BHmax.value) == (2, 2)

    assert np.all(extrinsic_properties.Hc.value > 0)
    assert np.all(extrinsic_properties.Mr.value > 0)
    assert np.all(extrinsic_properties.BHmax.value > 0)


def test_Hc_Mr_BHmax_out_of_range_2d_array_inputs():
    """Test that out-of-training-range array inputs produce nan predictions."""
    Ms = me.Ms([[1e6, 1e6], [1e6, 1e6]])
    A = me.A([[1e-12, 1e-12], [1e-12, 1e-12]])
    Ku = me.Ku([[1e4, 1e6], [1e3, 1e8]])

    extrinsic_properties = mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(Ms, A, Ku)

    assert np.shape(extrinsic_properties.Hc.value) == (2, 2)
    assert np.shape(extrinsic_properties.Mr.value) == (2, 2)
    assert np.shape(extrinsic_properties.BHmax.value) == (2, 2)

    assert np.all(extrinsic_properties.Hc.value[0] > 0)
    assert np.all(extrinsic_properties.Mr.value[0] > 0)
    assert np.all(extrinsic_properties.BHmax.value[0] > 0)

    assert np.all(np.isnan(extrinsic_properties.Hc.value[1]))
    assert np.all(np.isnan(extrinsic_properties.Mr.value[1]))
    assert np.all(np.isnan(extrinsic_properties.BHmax.value[1]))


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
    assert metadata["model_name"] == "cube50_singlegrain_random_forest_v1.0"


def test_is_hard_magnet_metadata_specified_model():
    """Test metadata function with explicitly specified model."""
    metadata = mammos_ai.is_hard_magnet_from_Ms_A_K_metadata(model="cube50_singlegrain_random_forest_v0.1")

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
    assert metadata["model_name"] == "cube50_singlegrain_random_forest_v1.0"


def test_Hc_Mr_BHmax_metadata_specified_model():
    """Test metadata function with explicitly specified model."""
    metadata = mammos_ai.Hc_Mr_BHmax_from_Ms_A_K_metadata(model="cube50_singlegrain_random_forest_v0.1")

    assert isinstance(metadata, dict)
    assert "model_name" in metadata


def test_Hc_Mr_BHmax_metadata_unknown_model():
    """Test metadata function raises error for unknown model."""
    with pytest.raises(ValueError, match="Unknown model"):
        mammos_ai.Hc_Mr_BHmax_from_Ms_A_K_metadata(model="non-existent-model")
