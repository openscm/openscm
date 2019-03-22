from datetime import datetime

import numpy as np
import pandas as pd
from base import _AdapterTester

from openscm.adapters.dice import DICE, YEAR
from openscm.parameters import ParameterType
from openscm.timeseries_converter import create_time_points


def _run_and_compare(test_adapter, filename):
    original_data = pd.read_csv(filename)
    start_time = int(datetime(2010, 1, 1).timestamp())
    timestep_count = 100
    stop_time = start_time + timestep_count * YEAR
    time_points = create_time_points(
        start_time, YEAR, timestep_count, ParameterType.AVERAGE_TIMESERIES
    )
    test_adapter._parameters.get_writable_timeseries_view(
        ("Emissions", "CO2"),
        ("World",),
        "GtCO2/a",
        time_points,
        ParameterType.AVERAGE_TIMESERIES,
    ).set(original_data.E.values[:timestep_count])
    test_adapter.initialize_model_input()
    test_adapter.initialize_run_parameters(start_time, stop_time)
    test_adapter.reset()
    test_adapter.run()
    output_parameters = [
        (("Concentration", "Atmosphere"), "GtC", "MAT"),
        (("Concentration", "Ocean", "lower"), "GtC", "ML"),
        (("Concentration", "Ocean", "shallow"), "GtC", "MU"),
        (("Radiative forcing",), "W/m^2", "FORC"),
        (("Temperature Increase", "Atmosphere"), "degC", "TATM"),
        (("Temperature Increase", "Ocean", "lower"), "degC", "TOCEAN"),
    ]
    for name, unit, original_name in output_parameters:
        np.testing.assert_allclose(
            test_adapter._output.get_timeseries_view(
                name,
                ("World",),
                unit,
                time_points[:-1],  # these are point timeseries
                ParameterType.POINT_TIMESERIES,
            ).get(),
            original_data[original_name][:timestep_count],
        )


class TestMyAdapter(_AdapterTester):
    tadapter = DICE

    def test_match_original(self, test_adapter):
        _run_and_compare(test_adapter, "tests/data/dice/original_results.csv")

    def test_match_original_bau(self, test_adapter):
        _run_and_compare(test_adapter, "tests/data/dice/original_results_bau.csv")
