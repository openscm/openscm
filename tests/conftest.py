from datetime import datetime


import pytest
import pandas as pd


from openscm.core import ParameterSet


TEST_DF = pd.DataFrame(
    [
        ["a_model", "a_iam", "a_scenario", "World", "Primary Energy", "EJ/y", 1, 6.0],
        [
            "a_model",
            "a_iam",
            "a_scenario",
            "World",
            "Primary Energy|Coal",
            "EJ/y",
            0.5,
            3,
        ],
        ["a_model", "a_iam", "a_scenario2", "World", "Primary Energy", "EJ/y", 2, 7],
    ],
    columns=[
        "climate_model",
        "model",
        "scenario",
        "region",
        "variable",
        "unit",
        datetime(1005, 1, 1),
        datetime(3010, 12, 31),
    ],
)


@pytest.fixture(scope="function")
def test_pd_df():
    yield TEST_DF


@pytest.fixture(scope="function")
def test_adapter(request):
    """
    Get an initialised instance of an the requesting classes ``tadapter`` property.
    """
    try:
        yield request.cls.tadapter(ParameterSet(), ParameterSet())
    except TypeError:
        pytest.skip("{} cannot be instantiated".format(str(request.cls.tadapter)))


@pytest.fixture(scope="function")
def test_config_paraset():
    parameters = ParameterSet()
    parameters.get_writable_scalar_view("ecs", ("World",), "K").set(3)
    parameters.get_writable_scalar_view("rf2xco2", ("World",), "W / m^2").set(4.0)

    yield parameters


@pytest.fixture(scope="function")
def test_drivers_core():
    # doesn't exist yet but we'd want something like this for testing
    core = rcp_26_core

    yield core
