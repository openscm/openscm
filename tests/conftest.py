from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pyam import IamDataFrame

from openscm.highlevel import ScmDataFrame


TEST_DF_LONG_TIMES = pd.DataFrame(
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

TEST_DF = pd.DataFrame([
    ['a_model', 'a_iam', 'a_scenario', 'World', 'Primary Energy', 'EJ/y', 1, 6.],
    ['a_model', 'a_iam', 'a_scenario', 'World', 'Primary Energy|Coal', 'EJ/y', 0.5, 3],
    ['a_model', 'a_iam', 'a_scenario2', 'World', 'Primary Energy', 'EJ/y', 2, 7],
],
    columns=['climate_model', 'model', 'scenario', 'region', 'variable', 'unit', 2005, 2010],
)

TEST_TS = np.array([
    [1, 6.],
    [0.5, 3],
    [2, 7],
]).T


@pytest.fixture(scope="function")
def test_pd_df():
    yield TEST_DF


@pytest.fixture(scope="function")
def test_pd_longtime_df():
    yield TEST_DF_LONG_TIMES


@pytest.fixture(scope="function")
def test_ts():
    yield TEST_TS


@pytest.fixture(scope="function")
def test_iam_df():
    yield IamDataFrame(TEST_DF)


@pytest.fixture(scope="function",
                params=[
                    {'data': TEST_DF},
                    {'data': IamDataFrame(TEST_DF).data},
                    {'data': IamDataFrame(TEST_DF).timeseries()},
                    # {'data': TEST_TS, 'columns': {
                    #     'index': [2005, 2010],
                    #     'model': ['a_iam'],
                    #     'climate_model': ['a_model'],
                    #     'scenario': ['a_scenario', 'a_scenario', 'a_scenario2'],
                    #     'region': ['World'],
                    #     'variable': ['Primary Energy', 'Primary Energy|Coal', 'Primary Energy'],
                    #     'unit': ['EJ/y']
                    # }
                    #  }
                ]
                )
def test_scm_df(request):
    yield ScmDataFrame(**request.param)
