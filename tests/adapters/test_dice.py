from datetime import datetime

import numpy as np
import pandas as pd
from base import _AdapterTester

from openscm.adapters.dice import DICE, YEAR
from openscm.parameters import ParameterType
from openscm.timeseries_converter import (
    ExtrapolationType,
    InterpolationType,
    create_time_points,
)


def _run_and_compare(test_adapter, filename):
    original_data = pd.read_csv(filename)
    start_time = int(datetime(2010, 1, 1).timestamp())
    timestep_count = len(original_data)
    stop_time = start_time + (timestep_count - 1) * YEAR

    test_adapter.initialize_model_input()
    test_adapter.initialize_run_parameters(start_time, stop_time)

    test_adapter._parameters.get_writable_generic_view(
        ("DICE", "forcoth_saturation_time"), ("World",)  # , ""
    ).set(start_time + 90 * YEAR)
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
        test_adapter._parameters.get_writable_timeseries_view(
            ("Emissions", "CO2"),
            ("World",),
            "GtCO2/a",
            time_points_for_averages,
            ParameterType.AVERAGE_TIMESERIES,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
        ).set(np.zeros(test_adapter._timestep_count))
