import numpy as np
import pandas as pd
from base import _AdapterTester

from openscm.adapters.dice import DICE
from openscm.core.parameters import ParameterType
from openscm.core.time import create_time_points


def _run_and_compare(test_adapter, filename):
    original_data = pd.read_csv(filename)
    start_time = np.datetime64("2010-01-01")
    timestep_count = len(original_data)
    stop_time = start_time + (timestep_count - 1) * np.timedelta64(365, "D")

    test_adapter._parameters.generic(("Start Time")).value = start_time
    test_adapter._parameters.generic(("Stop Time")).value = stop_time
    test_adapter._parameters.generic(
        ("DICE", "forcoth_saturation_time")
    ).value = start_time + np.timedelta64(90 * 365, "D")
    time_points = create_time_points(
        start_time,
        np.timedelta64(365, "D"),
        timestep_count,
        ParameterType.AVERAGE_TIMESERIES,
    )
    test_adapter._parameters.timeseries(
        ("Emissions", "CO2"), "GtCO2/a", time_points, timeseries_type="average"
    ).values = original_data.E.values[:timestep_count]

    test_adapter.initialize_model_input()
    test_adapter.initialize_run_parameters()
    test_adapter.reset()
    test_adapter.run()

    output_parameters = [
        (("Pool", "CO2", "Atmosphere"), "GtC", "MAT"),
        (("Pool", "CO2", "Ocean", "lower"), "GtC", "ML"),
        (("Pool", "CO2", "Ocean", "shallow"), "GtC", "MU"),
        (("Radiative Forcing", "CO2"), "W/m^2", "FORC"),
        (("Surface Temperature", "Increase"), "delta_degC", "TATM"),
        (("Ocean Temperature", "Increase"), "delta_degC", "TOCEAN"),
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
            rtol=1e-4,
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
        This method is called directly before ``test_adapter.initialize_model_input``
        during tests.
        """
        test_adapter._parameters.generic("Start Time").value = start_time
        test_adapter._parameters.generic("Stop Time").value = stop_time

        npoints = 10  # setting to zero so doesn't matter
        time_points_for_averages = create_time_points(
            start_time,
            stop_time - start_time,
            npoints,
            ParameterType.AVERAGE_TIMESERIES,
        )
        test_adapter._parameters.timeseries(
            ("Emissions", "CO2"),
            "GtCO2/a",
            time_points_for_averages,
            timeseries_type="average",
        ).values = np.zeros(npoints)
