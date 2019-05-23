import warnings

import numpy as np
import pytest

from openscm.core.units import UnitConverter, _unit_registry
from openscm.errors import DimensionalityError, UndefinedUnitError


def test_conversion_without_offset():
    uc = UnitConverter("kg", "t")
    assert uc.source == "kg"
    assert uc.target == "t"
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


def test_metric_conversion_unit_converter_with_context():
    uc = UnitConverter("kg SF5CF3 / yr", "kg CO2 / yr", context="AR4GWP100")
    assert uc.convert_from(1) == 17700
    assert uc.convert_to(1) == 1 / 17700


def test_metric_conversion_unit_converter_error():
    with pytest.raises(DimensionalityError):
        UnitConverter("kg SF5CF3 / yr", "kg CO2 / yr")


def test_metric_conversion_unit_converter_nan():
    src_species = "CHCl3"
    target_species = "CO2"
    expected_warning = (
        "No conversion from {} to {} available, nan will be returned "
        "upon conversion".format(src_species, target_species)
    )
    with _unit_registry.context("AR4GWP100"):
        with warnings.catch_warnings(record=True) as recorded_warnings:
            UnitConverter(src_species, target_species)

    assert len(recorded_warnings) == 1
    assert str(recorded_warnings[0].message) == expected_warning


def test_properties():
    assert UnitConverter("CO2", "C").contexts
    assert UnitConverter("CO2", "C").unit_registry
