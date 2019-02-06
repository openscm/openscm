import numpy as np
import pytest

from openscm.units import unit_registry
from openscm.units import DimensionalityError, UndefinedUnitError, UnitConverter


def test_unit_registry():
    CO2 = unit_registry("CO2")
    np.testing.assert_allclose(CO2.to("C").magnitude, 12 / 44)


def test_alias():
    CO2 = unit_registry("carbon_dioxide")
    np.testing.assert_allclose(CO2.to("C").magnitude, 12 / 44)


def test_base_unit():
    assert unit_registry("carbon") == unit_registry("C")


def test_nitrogen():
    N = unit_registry("N")
    np.testing.assert_allclose(N.to("N2ON").magnitude, 28 / 14)


def test_nox():
    NOx = unit_registry("NOx")
    with pytest.raises(DimensionalityError):
        NOx.to("N")

    N = unit_registry("N")
    with unit_registry.context("NOx_conversions"):
        np.testing.assert_allclose(NOx.to("N").magnitude, 14 / 46)
        np.testing.assert_allclose(N.to("NOx").magnitude, 46 / 14)
        # this also becomes allowed, unfortunately...
        np.testing.assert_allclose(NOx.to("N2O").magnitude, 44 / 46)


def test_methane():
    CH4 = unit_registry("CH4")
    with pytest.raises(DimensionalityError):
        CH4.to("C")

    C = unit_registry("C")
    with unit_registry.context("CH4_conversions"):
        np.testing.assert_allclose(CH4.to("C").magnitude, 12 / 16)
        np.testing.assert_allclose(C.to("CH4").magnitude, 16 / 12)
        # this also becomes allowed, unfortunately...
        np.testing.assert_allclose(CH4.to("CO2").magnitude, 44 / 16)


def test_ppm():
    ppm = unit_registry("ppm")
    np.testing.assert_allclose(ppm.to("ppb").magnitude, 1000)


def test_ppt():
    ppt = unit_registry("ppt")
    np.testing.assert_allclose(ppt.to("ppb").magnitude, 1 / 1000)


def test_short_definition():
    tC = unit_registry("tC")
    np.testing.assert_allclose(tC.to("tCO2").magnitude, 44 / 12)
    np.testing.assert_allclose(tC.to("gC").magnitude, 10 ** 6)


def test_uppercase():
    tC = unit_registry("HFC4310MEE")
    np.testing.assert_allclose(tC.to("HFC4310mee").magnitude, 1)


def test_emissions_flux():
    tOC = unit_registry("tOC/day")
    np.testing.assert_allclose(tOC.to("tOC/hr").magnitude, 1 / 24)


def test_kt():
    kt = unit_registry("kt")
    np.testing.assert_allclose(kt.to("t").magnitude, 1000)


def test_h():
    h = unit_registry("h")
    np.testing.assert_allclose(h.to("min").magnitude, 60)


def test_a():
    a = unit_registry("a")
    np.testing.assert_allclose(a.to("day").magnitude, 365.24219878)


def test_forcing():
    uc = UnitConverter("kW/m^2", "W/m^2")
    np.testing.assert_allclose(uc.convert_from(1), 10 ** 3)
    np.testing.assert_allclose(uc.convert_to(-600), -0.6)


def test_conversion_without_offset():
    uc = UnitConverter("kg", "t")
    assert uc._source == "kg"
    assert uc._target == "t"
    np.testing.assert_allclose(uc.convert_from(1000), 1)
    np.testing.assert_allclose(uc.convert_to(1), 1000)


def test_conversion_with_offset():
    uc = UnitConverter("degC", "degF")
    np.testing.assert_allclose(uc.convert_from(1), 33.8)
    np.testing.assert_allclose(uc.convert_to(1), -17.22222, rtol=1e-5)


def test_conversion_unknown_unit():
    with pytest.raises(UndefinedUnitError):
        UnitConverter("UNKOWN", "degF")


def test_conversion_incompatible_units():
    with pytest.raises(DimensionalityError):
        UnitConverter("kg", "degF")


def test_context():
    CO2 = unit_registry("CO2")
    with pytest.raises(DimensionalityError):
        CO2.to("N")

    N = unit_registry("N")
    with unit_registry.context("AR4GWP12"):
        np.testing.assert_allclose(CO2.to("N").magnitude, 12 / 44 * 20)
        np.testing.assert_allclose(N.to("CO2").magnitude, 44 / 12 / 20)
