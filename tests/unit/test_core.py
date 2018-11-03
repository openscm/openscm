from openscm.core import ParameterSet, ParameterType


def test_parameterset():
    parameterset = ParameterSet()
    r = parameterset._get_or_create_region(("DEU", "BER"))
    p = parameterset._get_or_create_parameter(
        ("Emissions", "CO2", "Industry"), r, "CO2/a", ParameterType.TIMESERIES
    )
    assert "|".join(p.full_name) == "Emissions|CO2|Industry"
