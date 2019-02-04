from datetime import datetime


import pytest
import pandas as pd


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
