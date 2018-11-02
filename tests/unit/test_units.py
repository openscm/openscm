import numpy as np


from openscm.units import unit_registry
from openscm.units import DimensionalityError, UndefinedUnitError, UnitConverter
import pytest


def test_unit_registry():
    CO2 = unit_registry("CO2")
    assert CO2.to("C").magnitude == 12. / 44.


def test_alias():
    CO2 = unit_registry("carbon_dioxide")
    assert CO2.to("C").magnitude == 12. / 44.


def test_base_unit():
    # I don't know why this is this way, anyway
    with pytest.raises(UndefinedUnitError):
        C = unit_registry("carbon")


def test_nitrogen():
    N = unit_registry("N")
    assert N.to("N2ON").magnitude == 28 / 14


def test_short_definition():
    tC = unit_registry("tC")
    np.testing.assert_allclose(tC.to("tCO2").magnitude, 44. / 12.)
    np.testing.assert_allclose(tC.to("gC").magnitude, 10**6)


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
        uc = UnitConverter("UNKOWN", "degF")


def test_conversion_incompatible_units():
    with pytest.raises(DimensionalityError):
        uc = UnitConverter("kg", "degF")

def test_context():
    CO2 = unit_registry("CO2")
    with pytest.raises(DimensionalityError):
        CO2.to("N")

    with unit_registry.context('AR4GWP12'):
        np.testing.assert_allclose(CO2.to("N").magnitude, 12 / 44 * 20)
