from openscm.errors import (
    ParameterAggregatedError,
    ParameterReadonlyError,
    ParameterTypeError,
    ParameterWrittenError,
    RegionAggregatedError,
)
from openscm.core import ParameterSet, ParameterType
import pytest


@pytest.fixture
def parameterset():
    res = ParameterSet()
    res._get_or_create_region(("DEU", "BER"))
    return res


def test_region(parameterset):
    region_deu = parameterset._get_or_create_region(("DEU",))
    assert region_deu.name == "DEU"

    region_ber = parameterset._get_or_create_region(("DEU", "BER"))
    assert region_ber.name == "BER"
    assert region_ber.parent == region_deu

    region_deu.attempt_aggregate()
    with pytest.raises(RegionAggregatedError):
        parameterset._get_or_create_region(("DEU", "BRB"))


def test_parameter(parameterset):
    region_ber = parameterset._get_or_create_region(("DEU", "BER"))

    with pytest.raises(ValueError, match="No parameter name given"):
        parameterset._get_or_create_parameter(
            (), region_ber, "GtCO2/a", ParameterType.TIMESERIES
        )

    param_co2 = parameterset._get_or_create_parameter(
        ("Emissions", "CO2"), region_ber, "GtCO2/a", ParameterType.TIMESERIES
    )
    assert param_co2.full_name == ("Emissions", "CO2")
    assert param_co2.name == "CO2"

    param_emissions = param_co2.parent
    assert param_emissions.full_name == ("Emissions",)
    assert param_emissions.name == "Emissions"
    # Before any read/write attempt these should be None:
    assert param_emissions.parameter_type is None
    assert param_emissions.unit is None

    param_industry = parameterset._get_or_create_parameter(
        ("Emissions", "CO2", "Industry"),
        region_ber,
        "GtCO2/a",
        ParameterType.TIMESERIES,
    )
    assert param_industry.full_name == ("Emissions", "CO2", "Industry")
    assert param_industry.name == "Industry"
    assert param_industry.parameter_type == ParameterType.TIMESERIES
    assert param_industry.unit == "GtCO2/a"

    with pytest.raises(ParameterReadonlyError):
        param_co2.attempt_write(ParameterType.TIMESERIES)

    with pytest.raises(ParameterTypeError):
        param_co2.attempt_aggregate(ParameterType.SCALAR)

    param_co2.attempt_aggregate(ParameterType.TIMESERIES)
    with pytest.raises(ParameterAggregatedError):
        parameterset._get_or_create_parameter(
            ("Emissions", "CO2", "Landuse"),
            region_ber,
            "GtCO2/a",
            ParameterType.TIMESERIES,
        )

    with pytest.raises(ParameterTypeError):
        param_industry.attempt_write(ParameterType.SCALAR)

    param_industry.attempt_write(ParameterType.TIMESERIES)
    with pytest.raises(ParameterWrittenError):
        parameterset._get_or_create_parameter(
            ("Emissions", "CO2", "Industry", "Other"),
            region_ber,
            "GtCO2/a",
            ParameterType.TIMESERIES,
        )
