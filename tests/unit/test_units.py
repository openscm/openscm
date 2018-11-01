from openscm.units import unit_registry
from openscm.units import DimensionalityError, UndefinedUnitError, UnitConverter
import pytest


def test_unit_registry():
    tq = unit_registry("CO2")
    assert tq.to("C").magnitude == 12. / 44.


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
