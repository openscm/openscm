import numpy as np
import pytest

from openscm.units import DimensionalityError, unit_registry


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


def test_context():
    CO2 = unit_registry("CO2")
    N = unit_registry("N")
    with unit_registry.context("AR4GWP100"):
        np.testing.assert_allclose(CO2.to("N").magnitude, 14 / (44 * 298))
        np.testing.assert_allclose(N.to("CO2").magnitude, 44 * 298 / 14)


def test_context_with_magnitude():
    CO2 = 1 * unit_registry("CO2")
    N = 1 * unit_registry("N")
    with unit_registry.context("AR4GWP100"):
        np.testing.assert_allclose(CO2.to("N").magnitude, 14 / (44 * 298))
        np.testing.assert_allclose(N.to("CO2").magnitude, 44 * 298 / 14)


def test_context_compound_unit():
    CO2 = 1 * unit_registry("kg CO2 / yr")
    N = 1 * unit_registry("kg N / yr")
    with unit_registry.context("AR4GWP100"):
        np.testing.assert_allclose(CO2.to("kg N / yr").magnitude, 14 / (44 * 298))
        np.testing.assert_allclose(N.to("kg CO2 / yr").magnitude, 44 * 298 / 14)


def test_context_dimensionality_error():
    CO2 = unit_registry("CO2")
    with pytest.raises(DimensionalityError):
        CO2.to("N")


@pytest.mark.parametrize(
    "metric_name,species,conversion",
    (
        ["AR4GWP100", "CH4", 25],
        ["AR4GWP100", "N2O", 298],
        ["AR4GWP100", "CCl4", 1400],
        ["AR4GWP100", "HFC32", 675],
        ["AR4GWP100", "SF6", 22800],
        ["AR4GWP100", "C2F6", 12200],
        ["AR4GWP100", "HCFC142b", 2310],
        ["AR4GWP100", "HFC32", 675],
        ["AR4GWP100", "cC4F8", 10300],
        ["AR4GWP100", "cC4F8", 10300],
        ["AR4GWP100", "HFE356pcc3", 413],
        ["AR4GWP100", "CH2Cl2", 8.7],
        ["SARGWP100", "CH4", 21],
        ["SARGWP100", "N2O", 310],
        ["SARGWP100", "HFC32", 650],
        ["SARGWP100", "SF6", 23900],
        ["SARGWP100", "CF4", 6500],
        ["SARGWP100", "C2F6", 9200],
    ),
)
def test_metric_conversion(metric_name, species, conversion):
    base_str_formats = ["{}", "kg {} / yr", "kg {}", "{} / yr"]
    for base_str_format in base_str_formats:
        base = unit_registry(base_str_format.format(species))
        dest = unit_registry(base_str_format.format("CO2"))
        with unit_registry.context(metric_name):
            np.testing.assert_allclose(base.to(dest).magnitude, conversion)
            np.testing.assert_allclose(dest.to(base).magnitude, 1 / conversion)
