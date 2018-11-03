import numpy as np
from openscm.units import unit_registry
from openscm.units import DimensionalityError, UndefinedUnitError, UnitConverter
import pytest


def test_unit_registry():
    CO2 = unit_registry("CO2")
    assert CO2.to("C").magnitude == 12.0 / 44.0


def test_alias():
    CO2 = unit_registry("carbon_dioxide")
    assert CO2.to("C").magnitude == 12.0 / 44.0


def test_base_unit():
    assert unit_registry("carbon") == unit_registry("C")


def test_nitrogen():
    N = unit_registry("N")
    assert N.to("N2ON").magnitude == 28 / 14


def test_nox():
    # do we want to allow this?
    NOx = unit_registry("NOx")
    with pytest.raises(DimensionalityError):
        NOx.to("N")


def test_ppm():
    # do we want to allow this?
    ppm = unit_registry("ppm")
    assert ppm.to("ppb").magnitude == 1000


def test_ppt():
    # do we want to allow this?
    ppt = unit_registry("ppt")
    assert ppt.to("ppb").magnitude == 1 / 1000


def test_methane():
    # do we want to allow this?
    CH4 = unit_registry("CH4")
    with pytest.raises(DimensionalityError):
        CH4.to("C")


def test_short_definition():
    tC = unit_registry("tC")
    np.testing.assert_allclose(tC.to("tCO2").magnitude, 44.0 / 12.0)
    np.testing.assert_allclose(tC.to("gC").magnitude, 10 ** 6)


def test_uppercase():
    tC = unit_registry("HFC4310MEE")
    np.testing.assert_allclose(tC.to("HFC4310mee").magnitude, 1)


def test_emissions_flux():
    tOC = unit_registry("tOC/day")
    np.testing.assert_allclose(tOC.to("tOC/hr").magnitude, 1 / 24)


def test_kt():
    kt = unit_registry("kt")
    assert kt.to("t").magnitude == 1000


def test_h():
    h = unit_registry("h")
    assert h.to("min").magnitude == 60


def test_a():
    a = unit_registry("a")
    np.testing.assert_allclose(a.to("day").magnitude, 365.24219878)


def test_conversion_without_offset():
    uc = UnitConverter("kg", "t")
    assert uc.convert_from(1000) == 1
    assert uc.convert_to(1) == 1000


def test_conversion_with_offset():
    uc = UnitConverter("degC", "degF")
    assert round(uc.convert_from(1), 5) == 33.8
    assert round(uc.convert_to(1), 5) == -17.22222


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
