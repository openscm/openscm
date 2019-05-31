from datetime import datetime

import numpy as np
import pandas as pd
from base import _AdapterTester

from openscm.adapters.dice import DICE, YEAR
from openscm.core.parameters import ParameterType
from openscm.core.time import ExtrapolationType, InterpolationType, create_time_points


def _run_and_compare(test_adapter, filename):
    original_data = pd.read_csv(filename)
    start_time = int(datetime(2010, 1, 1).timestamp())
    timestep_count = len(original_data)
    stop_time = start_time + (timestep_count - 1) * YEAR

    test_adapter.initialize_model_input()
    test_adapter.initialize_run_parameters(start_time, stop_time)

    test_adapter._parameters.generic(
        ("DICE", "forcoth_saturation_time"), writable=True
    ).value = (start_time + 90 * YEAR)
    time_points = create_time_points(
        start_time, YEAR, timestep_count, ParameterType.AVERAGE_TIMESERIES
    )
    test_adapter._parameters.timeseries(
        ("Emissions", "CO2"),
        "GtCO2/a",
        time_points,
        timeseries_type="average",
        writable=True,
    ).values = original_data.E.values[:timestep_count]

    test_adapter.reset()
    test_adapter.run()

    output_parameters = [
        (("Pool", "CO2", "Atmosphere"), "GtC", "MAT"),
        (("Pool", "CO2", "Ocean", "lower"), "GtC", "ML"),
        (("Pool", "CO2", "Ocean", "shallow"), "GtC", "MU"),
        (("Radiative Forcing", "CO2"), "W/m^2", "FORC"),
        (("Surface Temperature", "Increase"), "degC", "TATM"),
        (("Ocean Temperature", "Increase"), "degC", "TOCEAN"),
    ]
    for name, unit, original_name in output_parameters:
        np.testing.assert_allclose(
            test_adapter._output.timeseries(
                name,
                unit,
                time_points[:-1],  # these are point timeseries
                timeseries_type="point",
            ).values,
            original_data[original_name][:timestep_count],
            err_msg="Not matching original results for variable '{}'".format(
                "|".join(name)
            ),
        )


class TestMyAdapter(_AdapterTester):
    tadapter = DICE

    def test_match_original(self, test_adapter):
        _run_and_compare(test_adapter, "tests/data/dice/original_results.csv")

    def test_match_original_bau(self, test_adapter):
        _run_and_compare(test_adapter, "tests/data/dice/original_results_bau.csv")

    def prepare_run_input(self, test_adapter, start_time, stop_time):
        """
        Overload this in your adapter test if you need to set required input parameters.
        This method is called directly after ``test_adapter.initialize_run_parameters``
        during tests.
        """
        time_points_for_averages = create_time_points(
            start_time,
            YEAR,
            test_adapter._timestep_count,
            ParameterType.AVERAGE_TIMESERIES,
        )
        test_adapter._parameters.timeseries(
            ("Emissions", "CO2"),
            "GtCO2/a",
            time_points_for_averages,
            timeseries_type="average",
            writable=True,
        ).values = np.zeros(test_adapter._timestep_count)
