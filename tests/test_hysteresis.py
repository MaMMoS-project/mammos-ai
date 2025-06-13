import mammos_entity as me
import pytest

import mammos_ai


def test_Hc_Mr_BHmax_from_Ms_A_K():
    Ms = me.Ms(1e5)
    A = me.A(1e-12)
    Ku = me.Ku(1e6)

    extrinsic_properties = mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(Ms, A, Ku)
    assert extrinsic_properties.Hc > 0
    assert extrinsic_properties.Mr > 0
    assert extrinsic_properties.BHmax > 0


def test_Hc_Mr_BHmax_from_Ms_A_K_wrong_model():
    Ms = me.Ms(1e5)
    A = me.A(1e-12)
    Ku = me.Ku(1e6)

    with pytest.raises(ValueError):
        mammos_ai.Hc_Mr_BHmax_from_Ms_A_K(Ms, A, Ku, model="non-existent-model")
