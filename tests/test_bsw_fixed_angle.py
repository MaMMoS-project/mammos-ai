import mammos_analysis
import mammos_entity as me
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
    assert classification in ["soft", "hard"]


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

    assert extrinsic_properties.Hc.q > 0
    assert extrinsic_properties.Mr.q > 0
    assert extrinsic_properties.BHmax.q > 0


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
