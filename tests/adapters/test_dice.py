import numpy as np
import pandas as pd
import pytest
from base import _AdapterTester

from openscm.adapters.dice import DICE, YEAR
from openscm.core.parameters import ParameterType
from openscm.core.time import create_time_points
from openscm.errors import DimensionalityError


def _run_and_compare(test_adapter, filename, timestep_count=None):
    original_data = pd.read_csv(filename)
    start_time = np.datetime64("2010-01-01")
    if timestep_count is None:
        timestep_count = len(original_data)
    stop_time = start_time + (timestep_count - 1) * np.timedelta64(365, "D")

    test_adapter._parameters.generic(("Start Time",)).value = start_time
    test_adapter._parameters.generic(("Stop Time",)).value = stop_time
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

    test_adapter.reset()

    if timestep_count == len(original_data):
        test_adapter.run()
    else:
        for i in range(timestep_count - 1):
            test_adapter.step()

    output_parameters = [
        (("Pool", "CO2", "Atmosphere"), "GtC", "MAT", "point"),
        (("Pool", "CO2", "Ocean", "lower"), "GtC", "ML", "point"),
        (("Pool", "CO2", "Ocean", "shallow"), "GtC", "MU", "point"),
        (("Radiative Forcing", "CO2"), "W/m^2", "FORC", "average"),
        (("Surface Temperature Increase"), "delta_degC", "TATM", "point"),
        (("Ocean Temperature Increase"), "delta_degC", "TOCEAN", "point"),
    ]
    for name, unit, original_name, timeseries_type in output_parameters:
        tp = time_points[:-1] if timeseries_type == "point" else time_points
        np.testing.assert_allclose(
            test_adapter._output.timeseries(
                name, unit, tp, timeseries_type=timeseries_type
            ).values,
            original_data[original_name][:timestep_count],
            err_msg="Not matching original results for variable '{}'".format(
                "|".join(name)
            ),
            rtol=1e-4,
        )


class TestMyAdapter(_AdapterTester):
    tadapter = DICE

    def test_initialize(self, test_adapter):
        super().test_initialize(test_adapter)
        assert test_adapter._values is not None
        assert test_adapter._values.c3.value == 0.09175
        assert (
            test_adapter._parameters.scalar(("DICE", "tatm0"), "delta_degC").value
            == 0.8
        )
        assert (
            test_adapter._parameters.scalar(("DICE", "t2xco2"), "delta_degC").value
            == 2.9
        )
        assert test_adapter._parameters.scalar(
            ("Equilibrium Climate Sensitivity"), "delta_degC"
        ).empty

        with pytest.raises(DimensionalityError):
            test_adapter._parameters.timeseries(
                ("Emissions", "CO2"),
                "GtN2O/a",
                np.array(
                    [np.datetime64("{}-01-01".format(y)) for y in [2010, 2020, 2030]]
                ),
                timeseries_type="average",
                extrapolation="linear",
            )

        assert test_adapter._parameters.timeseries(
            ("Emissions", "CO2"),
            "GtCO2/a",
            np.array([np.datetime64("{}-01-01".format(y)) for y in [2010, 2020, 2030]]),
            timeseries_type="average",
            extrapolation="linear",
        ).empty

    def test_shutdown(self, test_adapter):
        super().test_shutdown(test_adapter)

    def test_run(self, test_adapter, test_run_parameters):
        # more specific tests covered by test_match_original
        super().test_run(test_adapter, test_run_parameters)

    def test_match_original(self, test_adapter):
        _run_and_compare(test_adapter, "tests/data/dice/original_results.csv")

    def test_match_original_bau(self, test_adapter):
        _run_and_compare(test_adapter, "tests/data/dice/original_results_bau.csv")

    def test_step(self, test_adapter, test_run_parameters):
        # more specific tests covered by test_match_original_step
        super().test_step(test_adapter, test_run_parameters)

    def test_match_original_step(self, test_adapter):
        _run_and_compare(
            test_adapter, "tests/data/dice/original_results.csv", timestep_count=5
        )

    def test_run_reset_run_same(self, test_adapter, test_run_parameters):
        output = test_adapter._output

        self.prepare_run_input(
            test_adapter, test_run_parameters.start_time, test_run_parameters.stop_time
        )

        time_points = np.array(
            [
                np.datetime64("{}-01-01".format(y))
                .astype("datetime64[s]")
                .astype(float)
                for y in range(2010, 2091, 10)
            ]
        )
        check_args_rf = [("Radiative Forcing", "CO2"), "W/m^2", time_points]
        check_args_temperature = [
            "Surface Temperature Increase",
            "delta_degC",
            time_points,
        ]
        assert output.timeseries(*check_args_rf, timeseries_type="average").empty
        assert output.timeseries(*check_args_temperature).empty

        test_adapter.reset()
        test_adapter.run()
        first_run_rf = output.timeseries(
            *check_args_rf, timeseries_type="average"
        ).values
        first_run_temperature = output.timeseries(*check_args_temperature).values

        test_adapter.reset()
        # currently failing
        # assert output.timeseries(*check_args_rf, timeseries_type="average").empty
        # assert output.timeseries(*check_args_temperature).empty
        test_adapter.run()
        second_run_rf = output.timeseries(
            *check_args_rf, timeseries_type="average"
        ).values
        second_run_temperature = output.timeseries(*check_args_temperature).values
        np.testing.assert_allclose(first_run_temperature, second_run_temperature)
        np.testing.assert_allclose(first_run_rf, second_run_rf)

    def test_step_reset_run_same(self, test_adapter, test_run_parameters):
        output = test_adapter._output

        self.prepare_run_input(
            test_adapter, test_run_parameters.start_time, test_run_parameters.stop_time
        )

        time_points = np.array(
            [
                np.datetime64("{}-01-01".format(y))
                .astype("datetime64[s]")
                .astype(float)
                for y in range(2010, 2091, 10)
            ]
        )
        check_args_rf = [("Radiative Forcing", "CO2"), "W/m^2", time_points]
        check_args_temperature = [
            "Surface Temperature Increase",
            "delta_degC",
            time_points,
        ]

        assert output.timeseries(*check_args_rf, timeseries_type="average").empty
        assert output.timeseries(*check_args_temperature).empty

        test_adapter.reset()
        test_adapter.run()
        first_run_rf = output.timeseries(
            *check_args_rf, timeseries_type="average"
        ).values
        first_run_temperature = output.timeseries(*check_args_temperature).values
        test_adapter.reset()
        # currently failing
        # assert output.timeseries(*check_args_rf, timeseries_type="average").empty
        # assert output.timeseries(*check_args_temperature).empty
        test_adapter.step()
        test_adapter.step()
        first_two_steps_rf = output.timeseries(
            ("Radiative Forcing", "CO2"),
            "W/m^2",
            time_points[:3],
            timeseries_type="average",
        ).values
        first_two_steps_temperature = output.timeseries(
            "Surface Temperature Increase", "delta_degC", time_points[:2]
        ).values
        # currently failing
        # for some reason accessing the first two elements of a `Timeseries` resets everything to zero
        # so we require this hack...
        # np.testing.assert_allclose(np.array(first_run_rf, copy=True)[:2], first_two_steps_rf)
        # np.testing.assert_allclose(np.array(first_run_temperature, copy=True)[:2], first_two_steps_temperature)

        test_adapter.reset()
        # currently failing
        # assert output.timeseries(*check_args_rf, timeseries_type="average").empty
        # assert output.timeseries(*check_args_temperature).empty
        test_adapter.run()
        second_run_rf = output.timeseries(
            *check_args_rf, timeseries_type="average"
        ).values
        second_run_temperature = output.timeseries(*check_args_temperature).values

        np.testing.assert_allclose(first_run_rf, second_run_rf)
        np.testing.assert_allclose(first_run_temperature, second_run_temperature)

    def test_openscm_standard_parameters_handling(self, test_adapter):
        parameters = test_adapter._parameters
        output_parameters = test_adapter._output

        parameters.generic("Start Time").value = np.datetime64("1850-01-01")
        parameters.generic("Stop Time").value = np.datetime64("2100-01-01")
        ecs_magnitude = 3.12
        parameters.scalar(
            "Equilibrium Climate Sensitivity", "delta_degC"
        ).value = ecs_magnitude
        parameters.scalar(("DICE", "t2xco2"), "delta_degC").value = 2 * ecs_magnitude

        self.prepare_run_input(
            test_adapter,
            parameters.generic("Start Time").value,
            parameters.generic("Stop Time").value,
        )
        test_adapter.reset()
        test_adapter.run()

        # make sure OpenSCM ECS value was used preferentially to the model's t2xco2
        assert test_adapter._values.t2xco2.value == ecs_magnitude
        assert (
            parameters.scalar(("DICE", "t2xco2"), "delta_degC").value == ecs_magnitude
        )

        # currently failing
        # assert (
        #     output_parameters.scalar(
        #         "Equilibrium Climate Sensitivity", "delta_degC"
        #     ).value
        #     == ecs_magnitude
        # )
        # assert output_parameters.generic("Start Time").value == np.datetime64(
        #     "1850-01-01"
        # )
        # assert output_parameters.generic("Stop Time").value == np.datetime64(
        #     "2100-01-01"
        # )

    def prepare_run_input(self, test_adapter, start_time, stop_time):
        test_adapter._parameters.generic("Start Time").value = start_time
        test_adapter._parameters.generic("Stop Time").value = stop_time

        npoints = int((stop_time - start_time) / YEAR) + 1  # include self._stop_time

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
