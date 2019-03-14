import warnings

import numpy as np
import pytest

from openscm.core import Core, ParameterSet
from openscm.errors import (
    ParameterAggregationError,
    ParameterEmptyError,
    ParameterReadError,
    ParameterReadonlyError,
    ParameterTypeError,
    ParameterWrittenError,
    RegionAggregatedError,
    TimeseriesPointsValuesMismatchError,
)
from openscm.parameters import ParameterType
from openscm.timeseries_converter import (
    ExtrapolationType,
    InterpolationType,
    create_time_points,
)
from openscm.units import DimensionalityError


@pytest.fixture
def model():
    return "DICE"


@pytest.fixture
def start_time():
    return 30 * 365 * 24 * 3600


@pytest.fixture
def stop_time():
    return 130 * 365 * 24 * 3600


@pytest.fixture
def core(model, start_time, stop_time):
    core = Core(model, start_time, stop_time)
    core.parameters._get_or_create_region(("World", "DEU", "BER"))
    return core


def test_core(core, model, start_time, stop_time):
    assert core.start_time == start_time
    assert core.stop_time == stop_time
    assert core.model == model


def test_region(core):
    parameterset = core.parameters

    with pytest.raises(ValueError, match="No region name given"):
        parameterset._get_or_create_region(())

    for accessor in ["World", ("World"), ("World",), ["World"]]:
        with warnings.catch_warnings():
            # silence warning about conversion, that's tested elsewhere
            warnings.simplefilter("ignore")
            region_world = parameterset._get_or_create_region(accessor)
        assert region_world.full_name == ("World",)
        assert region_world.name == "World"

    for accessor in [("World", "DEU"), ["World", "DEU"]]:
        region_deu = parameterset._get_or_create_region(accessor)
        assert region_deu.full_name == ("World", "DEU")
        assert region_deu.name == "DEU"

    region_ber = parameterset._get_or_create_region(("World", "DEU", "BER"))
    assert region_ber.full_name == ("World", "DEU", "BER")
    assert region_ber.name == "BER"
    assert region_ber.parent == region_deu

    region_deu.attempt_aggregate()
    with pytest.raises(RegionAggregatedError):
        parameterset._get_or_create_region(("World", "DEU", "BRB"))

    assert parameterset._get_region(("INVALID", "DEU", "BER")) is None


def test_parameter(core):
    parameterset = core.parameters
    region_ber = parameterset._get_or_create_region(("World", "DEU", "BER"))

    with pytest.raises(ValueError, match="No parameter name given"):
        parameterset._get_or_create_parameter((), region_ber)

    param_co2 = parameterset._get_or_create_parameter(("Emissions", "CO2"), region_ber)
    # not clear what the value of being able to access with `()` is...
    assert param_co2.get_subparameter(()) == param_co2
    for accessor in ["CO2", ("CO2"), ("CO2",), ["CO2"]]:
        assert param_co2.parent.get_subparameter(accessor) == param_co2
    assert region_ber.get_parameter(("Emissions", "CO2")) == param_co2
    assert param_co2.full_name == ("Emissions", "CO2")
    assert param_co2.info.region == ("World", "DEU", "BER")
    assert param_co2.info.name == "CO2"
    assert (
        parameterset.get_parameter_info(("Emissions", "CO2"), ("World", "DEU", "BER"))
        == param_co2.info
    )
    for accessor in ["Emissions", ("Emissions"), ("Emissions",), ["Emissions"]]:
        assert (
            parameterset.get_parameter_info(accessor, ("World", "DEU", "BER"))
            == param_co2.parent.info
        )
    assert (
        parameterset.get_parameter_info(("Emissions", "NOx"), ("World", "DEU", "BER"))
        is None
    )
    assert (
        parameterset.get_parameter_info(("Emissions",), ("World", "DEU", "BRB")) is None
    )

    with pytest.raises(ValueError, match="No parameter name given"):
        parameterset.get_parameter_info(None, ("World", "DEU", "BER"))
    with pytest.raises(ValueError, match="No parameter name given"):
        parameterset.get_parameter_info((), ("World", "DEU", "BER"))

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

    param_industry.attempt_read(
        ParameterType.AVERAGE_TIMESERIES, "GtCO2/a", np.array([0])
    )
    assert param_industry.info.parameter_type == ParameterType.AVERAGE_TIMESERIES
    assert param_industry.info.unit == "GtCO2/a"

    with pytest.raises(ParameterReadonlyError):
        param_co2.attempt_write(
            ParameterType.AVERAGE_TIMESERIES, "GtCO2/a", np.array([0])
        )

    param_co2.attempt_read(ParameterType.AVERAGE_TIMESERIES, "GtCO2/a", np.array([0]))
    with pytest.raises(ParameterTypeError):
        param_co2.attempt_read(ParameterType.SCALAR, "GtCO2/a")

    with pytest.raises(ParameterReadError):
        parameterset._get_or_create_parameter(
            ("Emissions", "CO2", "Landuse"), region_ber
        )

    with pytest.raises(ParameterTypeError):
        param_industry.attempt_write(ParameterType.SCALAR, "GtCO2/a")

    param_industry.attempt_write(
        ParameterType.AVERAGE_TIMESERIES, "GtCO2/a", np.array([0])
    )
    with pytest.raises(ParameterWrittenError):
        parameterset._get_or_create_parameter(
            ("Emissions", "CO2", "Industry", "Other"), region_ber
        )


def test_parameterset_default_initialization():
    paraset = ParameterSet()

    assert paraset._get_or_create_region(("World",)) == paraset._root
    error_msg = (
        "Cannot access region Earth, root region for this parameter set is World"
    )
    with pytest.raises(ValueError, match=error_msg):
        paraset._get_or_create_region("Earth")


def test_parameterset_named_initialization():
    paraset_named = ParameterSet("Earth")
    assert paraset_named._get_or_create_region(("Earth",)) == paraset_named._root


def test_scalar_parameter_view(core):
    parameterset = core.parameters
    cs = parameterset.get_scalar_view(("Climate Sensitivity"), ("World",), "degC")
    with pytest.raises(ParameterEmptyError):
        cs.get()
    assert cs.is_empty
    cs_writable = parameterset.get_writable_scalar_view(
        ("Climate Sensitivity"), ("World",), "degF"
    )
    cs_writable.set(68)
    assert cs_writable.get() == 68
    assert not cs.is_empty
    np.testing.assert_allclose(cs.get(), 20)
    with pytest.raises(ParameterTypeError):
        parameterset.get_timeseries_view(
            ("Climate Sensitivity"),
            ("World",),
            "degC",
            (0,),
            ParameterType.AVERAGE_TIMESERIES,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
        )
    with pytest.raises(DimensionalityError):
        parameterset.get_scalar_view(("Climate Sensitivity"), ("World",), "kg")


def test_scalar_parameter_view_aggregation(core, start_time):
    ta_1 = 0.6
    ta_2 = 0.3
    tb = 0.1

    parameterset = core.parameters

    a_1_writable = parameterset.get_writable_scalar_view(
        ("Top", "a", "1"), ("World",), "dimensionless"
    )
    a_1_writable.set(ta_1)

    a_2_writable = parameterset.get_writable_scalar_view(
        ("Top", "a", "2"), ("World",), "dimensionless"
    )
    a_2_writable.set(ta_2)

    b_writable = parameterset.get_writable_scalar_view(
        ("Top", "b"), ("World",), "dimensionless"
    )
    b_writable.set(tb)

    a_1 = parameterset.get_scalar_view(("Top", "a", "1"), ("World",), "dimensionless")
    np.testing.assert_allclose(a_1.get(), ta_1)

    a_2 = parameterset.get_scalar_view(("Top", "a", "2"), ("World",), "dimensionless")
    np.testing.assert_allclose(a_2.get(), ta_2)

    a = parameterset.get_scalar_view(("Top", "a"), ("World",), "dimensionless")
    np.testing.assert_allclose(a.get(), ta_1 + ta_2)

    b = parameterset.get_scalar_view(("Top", "b"), ("World",), "dimensionless")
    np.testing.assert_allclose(b.get(), tb)

    with pytest.raises(ParameterReadonlyError):
        parameterset.get_writable_scalar_view(("Top", "a"), ("World",), "dimensionless")

    total = parameterset.get_scalar_view(("Top"), ("World",), "dimensionless")
    np.testing.assert_allclose(total.get(), ta_1 + ta_2 + tb)


@pytest.fixture(
    params=[
        (range(5 * 365), [0.24373829, 0.7325541, 1.22136991, 1.71018572, 2.19900153]),
        ([1] * 5 * 365, [365 * 44 / 12 / 1e6] * 5),
    ]
)
def series(request):
    return np.array(request.param[0]), np.array(request.param[1])


def test_timeseries_parameter_view(core, start_time, series):
    inseries = series[0]
    outseries = series[1]

    parameterset = core.parameters
    carbon = parameterset.get_timeseries_view(
        ("Emissions", "CO2"),
        ("World",),
        "GtCO2/a",
        create_time_points(
            start_time,
            365 * 24 * 3600,
            len(outseries),
            ParameterType.AVERAGE_TIMESERIES,
        ),
        ParameterType.AVERAGE_TIMESERIES,
        InterpolationType.LINEAR,
        ExtrapolationType.LINEAR,
    )
    assert carbon.is_empty
    with pytest.raises(ParameterEmptyError):
        carbon.get()

    carbon_writable = parameterset.get_writable_timeseries_view(
        ("Emissions", "CO2"),
        ("World",),
        "ktC/d",
        create_time_points(
            start_time, 24 * 3600, len(inseries), ParameterType.AVERAGE_TIMESERIES
        ),
        ParameterType.AVERAGE_TIMESERIES,
        InterpolationType.LINEAR,
        ExtrapolationType.LINEAR,
    )
    with pytest.raises(TimeseriesPointsValuesMismatchError):
        carbon_writable.set(inseries[::2])
    carbon_writable.set(inseries)
    assert carbon_writable.length == len(inseries)
    np.testing.assert_allclose(
        carbon_writable.get(), inseries, atol=inseries.max() * 1e-10
    )
    assert carbon.length == 5
    np.testing.assert_allclose(carbon.get(), outseries, rtol=1e-3)
    with pytest.raises(ParameterTypeError):
        parameterset.get_scalar_view(("Emissions", "CO2"), ("World",), "GtCO2/a")
    with pytest.raises(DimensionalityError):
        parameterset.get_timeseries_view(
            ("Emissions", "CO2"),
            ("World",),
            "kg",
            (0,),
            ParameterType.AVERAGE_TIMESERIES,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
        )


def test_timeseries_parameter_view_aggregation(core, start_time):
    fossil_industry_emms = np.array([0, 1, 2])
    fossil_energy_emms = np.array([2, 1, 4])
    land_emms = np.array([0.05, 0.1, 0.2])

    parameterset = core.parameters

    fossil_industry_writable = parameterset.get_writable_timeseries_view(
        ("Emissions", "CO2", "Fossil", "Industry"),
        ("World",),
        "GtC/yr",
        create_time_points(
            start_time,
            24 * 3600,
            len(fossil_industry_emms),
            ParameterType.AVERAGE_TIMESERIES,
        ),
        ParameterType.AVERAGE_TIMESERIES,
        InterpolationType.LINEAR,
        ExtrapolationType.LINEAR,
    )
    fossil_industry_writable.set(fossil_industry_emms)

    fossil_energy_writable = parameterset.get_writable_timeseries_view(
        ("Emissions", "CO2", "Fossil", "Energy"),
        ("World",),
        "GtC/yr",
        create_time_points(
            start_time,
            24 * 3600,
            len(fossil_energy_emms),
            ParameterType.AVERAGE_TIMESERIES,
        ),
        ParameterType.AVERAGE_TIMESERIES,
        InterpolationType.LINEAR,
        ExtrapolationType.LINEAR,
    )
    fossil_energy_writable.set(fossil_energy_emms)

    land_writable = parameterset.get_writable_timeseries_view(
        ("Emissions", "CO2", "Land"),
        ("World",),
        "MtC/yr",
        create_time_points(
            start_time,
            24 * 3600,
            len(fossil_energy_emms),
            ParameterType.AVERAGE_TIMESERIES,
        ),
        ParameterType.AVERAGE_TIMESERIES,
        InterpolationType.LINEAR,
        ExtrapolationType.LINEAR,
    )
    land_writable.set(land_emms * 1000)

    fossil_industry = parameterset.get_timeseries_view(
        ("Emissions", "CO2", "Fossil", "Industry"),
        ("World",),
        "GtC/yr",
        create_time_points(
            start_time,
            24 * 3600,
            len(fossil_industry_emms),
            ParameterType.AVERAGE_TIMESERIES,
        ),
        ParameterType.AVERAGE_TIMESERIES,
        InterpolationType.LINEAR,
        ExtrapolationType.LINEAR,
    )
    np.testing.assert_allclose(
        fossil_industry.get(),
        fossil_industry_emms,
        atol=fossil_industry_emms.max() * 1e-10,
    )

    fossil_energy = parameterset.get_timeseries_view(
        ("Emissions", "CO2", "Fossil", "Energy"),
        ("World",),
        "GtC/yr",
        create_time_points(
            start_time,
            24 * 3600,
            len(fossil_energy_emms),
            ParameterType.AVERAGE_TIMESERIES,
        ),
        ParameterType.AVERAGE_TIMESERIES,
        InterpolationType.LINEAR,
        ExtrapolationType.LINEAR,
    )
    np.testing.assert_allclose(fossil_energy.get(), fossil_energy_emms)

    fossil = parameterset.get_timeseries_view(
        ("Emissions", "CO2", "Fossil"),
        ("World",),
        "GtC/yr",
        create_time_points(
            start_time,
            24 * 3600,
            len(fossil_energy_emms),
            ParameterType.AVERAGE_TIMESERIES,
        ),
        ParameterType.AVERAGE_TIMESERIES,
        InterpolationType.LINEAR,
        ExtrapolationType.LINEAR,
    )
    np.testing.assert_allclose(fossil.get(), fossil_industry_emms + fossil_energy_emms)

    # ensure that you can't write extra children once you've got a parent view, this
    # avoids ever having the child views become out of date
    with pytest.raises(ParameterReadError):
        parameterset.get_writable_timeseries_view(
            ("Emissions", "CO2", "Fossil", "Transport"),
            ("World",),
            "GtC/yr",
            create_time_points(
                start_time,
                24 * 3600,
                len(fossil_industry_emms),
                ParameterType.AVERAGE_TIMESERIES,
            ),
            ParameterType.AVERAGE_TIMESERIES,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
        )

    land = parameterset.get_timeseries_view(
        ("Emissions", "CO2", "Land"),
        ("World",),
        "GtC/yr",
        create_time_points(
            start_time, 24 * 3600, len(land_emms), ParameterType.AVERAGE_TIMESERIES
        ),
        ParameterType.AVERAGE_TIMESERIES,
        InterpolationType.LINEAR,
        ExtrapolationType.LINEAR,
    )
    np.testing.assert_allclose(land.get(), land_emms)

    with pytest.raises(ParameterReadonlyError):
        parameterset.get_writable_timeseries_view(
            ("Emissions", "CO2"),
            ("World",),
            "GtC/yr",
            create_time_points(
                start_time,
                24 * 3600,
                len(fossil_energy_emms),
                ParameterType.AVERAGE_TIMESERIES,
            ),
            ParameterType.AVERAGE_TIMESERIES,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
        )

    total = parameterset.get_timeseries_view(
        ("Emissions", "CO2"),
        ("World",),
        "GtC/yr",
        create_time_points(
            start_time,
            24 * 3600,
            len(fossil_energy_emms),
            ParameterType.AVERAGE_TIMESERIES,
        ),
        ParameterType.AVERAGE_TIMESERIES,
        InterpolationType.LINEAR,
        ExtrapolationType.LINEAR,
    )
    np.testing.assert_allclose(
        total.get(), land_emms + fossil_energy_emms + fossil_industry_emms
    )


def test_boolean_parameter_view(core):
    parameterset = core.parameters
    cs = parameterset.get_boolean_view(("Model Options", "Boolean Option"), ("World",))
    with pytest.raises(ParameterAggregationError):
        parameterset.get_boolean_view(("Model Options"), ("World",))
    with pytest.raises(ParameterTypeError):
        parameterset.get_scalar_view(("Model Options"), ("World",), "dimensionless")
    with pytest.raises(ParameterEmptyError):
        cs.get()
    assert cs.is_empty
    cs_writable = parameterset.get_writable_boolean_view(
        ("Model Options", "Boolean Option"), ("World",)
    )
    cs_writable.set(True)
    assert cs_writable.get()
    assert not cs.is_empty
    assert cs.get()


def test_string_parameter_view(core):
    parameterset = core.parameters
    cs = parameterset.get_string_view(("Model Options", "String Option"), ("World",))
    with pytest.raises(ParameterAggregationError):
        parameterset.get_string_view(("Model Options"), ("World",))
    with pytest.raises(ParameterTypeError):
        parameterset.get_scalar_view(("Model Options"), ("World",), "dimensionless")
    with pytest.raises(ParameterEmptyError):
        cs.get()
    assert cs.is_empty
    cs_writable = parameterset.get_writable_string_view(
        ("Model Options", "String Option"), ("World",)
    )
    cs_writable.set("enabled")
    assert cs_writable.get() == "enabled"
    assert not cs.is_empty
    assert cs.get() == "enabled"
