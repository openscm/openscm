import numpy as np
from math import isnan
from openscm.core import Core, ParameterType
from openscm.errors import (
    ParameterReadError,
    ParameterReadonlyError,
    ParameterTypeError,
    ParameterWrittenError,
    RegionAggregatedError,
)
from openscm.units import DimensionalityError
import pytest


@pytest.fixture
def model():
    return "DICE"


@pytest.fixture
def start_time():
    return 30 * 365 * 24 * 3600


@pytest.fixture
def end_time():
    return 130 * 365 * 24 * 3600


@pytest.fixture
def core(model, start_time, end_time):
    core = Core(model, start_time, end_time)
    core.parameters._get_or_create_region(("DEU", "BER"))
    return core


def test_core(core, model, start_time, end_time):
    assert core.start_time == start_time
    assert core.end_time == end_time
    assert core.model == model


def test_region(core):
    parameterset = core.parameters
    region_deu = parameterset._get_or_create_region(("DEU",))
    assert region_deu.full_name == ("DEU",)
    assert region_deu.name == "DEU"

    region_ber = parameterset._get_or_create_region(("DEU", "BER"))
    assert region_ber.full_name == ("DEU", "BER")
    assert region_ber.name == "BER"
    assert region_ber.parent == region_deu

    region_deu.attempt_aggregate()
    with pytest.raises(RegionAggregatedError):
        parameterset._get_or_create_region(("DEU", "BRB"))


def test_parameter(core):
    parameterset = core.parameters
    region_ber = parameterset._get_or_create_region(("DEU", "BER"))

    with pytest.raises(ValueError, match="No parameter name given"):
        parameterset._get_or_create_parameter((), region_ber)

    param_co2 = parameterset._get_or_create_parameter(("Emissions", "CO2"), region_ber)
    assert param_co2.get_subparameter(()) == param_co2
    assert param_co2.parent.get_subparameter(("CO2",)) == param_co2
    assert region_ber.get_parameter(("Emissions", "CO2")) == param_co2
    assert param_co2.full_name == ("Emissions", "CO2")
    assert param_co2.info.region == ("DEU", "BER")
    assert param_co2.info.name == "CO2"
    assert (
        parameterset.get_parameter_info(("Emissions", "CO2"), ("DEU", "BER"))
        == param_co2.info
    )
    assert (
        parameterset.get_parameter_info(("Emissions",), ("DEU", "BER"))
        == param_co2.parent.info
    )
    assert parameterset.get_parameter_info(("Emissions", "NOx"), ("DEU", "BER")) is None
    assert parameterset.get_parameter_info(("Emissions",), ("DEU", "BRB")) is None

    with pytest.raises(ValueError, match="No parameter name given"):
        parameterset.get_parameter_info(None, ("DEU", "BER"))
    with pytest.raises(ValueError, match="No parameter name given"):
        parameterset.get_parameter_info((), ("DEU", "BER"))

    param_emissions = param_co2.parent
    assert param_emissions.full_name == ("Emissions",)
    assert param_emissions.info.name == "Emissions"
    # Before any read/write attempt these should be None:
    assert param_emissions.info.parameter_type is None
    assert param_emissions.info.unit is None

    param_industry = parameterset._get_or_create_parameter(
        ("Emissions", "CO2", "Industry"), region_ber
    )
    assert param_industry.full_name == ("Emissions", "CO2", "Industry")
    assert param_industry.info.name == "Industry"

    param_industry.attempt_read("GtCO2/a", ParameterType.TIMESERIES)
    assert param_industry.info.parameter_type == ParameterType.TIMESERIES
    assert param_industry.info.unit == "GtCO2/a"

    with pytest.raises(ParameterReadonlyError):
        param_co2.attempt_write("GtCO2/a", ParameterType.TIMESERIES)

    param_co2.attempt_read("GtCO2/a", ParameterType.TIMESERIES)
    with pytest.raises(ParameterTypeError):
        param_co2.attempt_read("GtCO2/a", ParameterType.SCALAR)

    with pytest.raises(ParameterReadError):
        parameterset._get_or_create_parameter(
            ("Emissions", "CO2", "Landuse"), region_ber
        )

    with pytest.raises(ParameterTypeError):
        param_industry.attempt_write("GtCO2/a", ParameterType.SCALAR)

    param_industry.attempt_write("GtCO2/a", ParameterType.TIMESERIES)
    with pytest.raises(ParameterWrittenError):
        parameterset._get_or_create_parameter(
            ("Emissions", "CO2", "Industry", "Other"), region_ber
        )


def test_scalar_parameter_view(core):
    parameterset = core.parameters
    cs = parameterset.get_scalar_view(("Climate Sensitivity"), (), "degC")
    assert isnan(cs.get())
    assert cs.is_empty
    cs_writable = parameterset.get_writable_scalar_view(
        ("Climate Sensitivity"), (), "degF"
    )
    cs_writable.set(68)
    assert cs_writable.get() == 68
    assert not cs.is_empty
    np.testing.assert_allclose(cs.get(), 20)
    with pytest.raises(ParameterTypeError):
        parameterset.get_timeseries_view(("Climate Sensitivity"), (), "degC", 0, 1)
    with pytest.raises(DimensionalityError):
        parameterset.get_scalar_view(("Climate Sensitivity"), (), "kg")


@pytest.fixture(
    params=[
        (range(5 * 365), [0.24373829, 0.7325541, 1.22136991, 1.71018572, 2.19900153]),
        ([1] * 5 * 365, [365 * 44 / 12 / 1e6] * 5),
    ]
)
def series(request):
    return np.array(request.param[0]), np.array(request.param[1])


def test_timeseries_parameter_view(core, start_time, series):
    parameterset = core.parameters
    carbon = parameterset.get_timeseries_view(
        ("Emissions", "CO2"), (), "GtCO2/a", start_time, 365 * 24 * 3600
    )
    carbon_writable = parameterset.get_writable_timeseries_view(
        ("Emissions", "CO2"), (), "ktC/d", start_time, 24 * 3600
    )
    inseries = series[0]
    outseries = series[1]
    carbon_writable.set_series(inseries)
    assert carbon_writable.length == len(inseries)
    np.testing.assert_allclose(carbon_writable.get_series(), inseries)
    assert carbon.length == 5
    np.testing.assert_allclose(carbon.get_series(), outseries, rtol=1e-3)
    with pytest.raises(ParameterTypeError):
        parameterset.get_scalar_view(("Emissions", "CO2"), (), "GtCO2/a")
    with pytest.raises(DimensionalityError):
        parameterset.get_timeseries_view(("Emissions", "CO2"), (), "kg", 0, 1)