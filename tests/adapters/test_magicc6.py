import numpy as np
import pytest
from base import _AdapterTester

from openscm.adapters.magicc6 import MAGICC6
from openscm.core.parameters import ParameterType
from openscm.core.parameterset import ParameterSet
from openscm.core.time import create_time_points
from openscm.errors import DimensionalityError

class TestMAGICC6(_AdapterTester):
    tadapter = MAGICC6
    def test_initialize(self, test_adapter):
        super().test_initialize(test_adapter)

        assert (
            test_adapter._parameters.scalar(("MAGICC6", "core_climatesensitivity"), "delta_degC").value
            == 3.0
        )
        assert (
            test_adapter._parameters.scalar(("MAGICC6", "core_delq2xco2"), "W/m^2").value
            == 3.71
        )
        assert test_adapter._parameters.scalar(
            ("Equilibrium Climate Sensitivity"), "delta_degC"
        ).empty

        with pytest.raises(DimensionalityError):
            test_adapter._parameters.timeseries(
                ("Emissions", "CO2", "MAGICC Fossil and Industrial"),
                "GtN2O/a",
                time_points=np.array(
                    [np.datetime64("{}-01-01".format(y)) for y in [1810, 2020, 2100]]
                ),
                timeseries_type="average",
                extrapolation="linear",
            )

        assert test_adapter._parameters.timeseries(
            ("Emissions", "CO2", "MAGICC Fossil and Industrial"),
            "GtCO2/a",
            time_points=np.array(
                [np.datetime64("{}-01-01".format(y)) for y in [1810, 2020, 2100]]
            ),
            timeseries_type="average",
            extrapolation="linear",
        ).empty

    def test_shutdown(self, test_adapter):
        super().test_shutdown(test_adapter)

    def test_run(self, test_adapter, test_run_parameters):
        super().test_run(test_adapter, test_run_parameters)
        # assert default temperatures unchanged
        time_points = create_time_points(
            np.datetime64("2094-01-01"),
            np.timedelta64(365, "D"),
            3,
            ParameterType.POINT_TIMESERIES,
        )
        run_res_temperature = test_adapter._output.timeseries(
            ("Surface Temperature Increase"),
            "K",
            time_points=time_points,
            timeseries_type="point",
        ).values
        np.testing.assert_allclose(run_res_temperature, np.array([1.5695633, 1.5683754, 1.5671952]))

    def test_step(self, test_adapter):
        test_adapter.reset()
        with pytest.raises(NotImplementedError):
            test_adapter.step()

    def test_step_reset_run_same(self, test_adapter):
        test_adapter.reset()
        with pytest.raises(NotImplementedError):
            test_adapter.step()

        test_adapter.reset()
        with pytest.raises(NotImplementedError):
            test_adapter.step()

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
                for y in range(1850, 2101, 10)
            ]
        )
        check_args_rf = [("Radiative Forcing", "CO2"), "W/m^2"]
        check_args_temperature = ["Surface Temperature Increase", "delta_degC"]
        assert output.timeseries(
            *check_args_rf, time_points=time_points, timeseries_type="point"
        ).empty
        assert output.timeseries(*check_args_temperature, time_points=time_points).empty

        test_adapter.reset()
        test_adapter.run()
        first_run_rf = output.timeseries(
            *check_args_rf, time_points=time_points, timeseries_type="average"
        ).values
        first_run_temperature = output.timeseries(
            *check_args_temperature, time_points=time_points
        ).values

        test_adapter.reset()
        assert np.isnan(
            output.timeseries(
                *check_args_rf, time_points=time_points, timeseries_type="average"
            ).values
        ).all()
        assert np.isnan(
            output.timeseries(*check_args_temperature, time_points=time_points).values
        ).all()
        test_adapter.run()
        second_run_rf = output.timeseries(
            *check_args_rf, time_points=time_points, timeseries_type="average"
        ).values
        second_run_temperature = output.timeseries(
            *check_args_temperature, time_points=time_points
        ).values
        np.testing.assert_allclose(first_run_temperature, second_run_temperature)
        np.testing.assert_allclose(first_run_rf, second_run_rf)

    def test_openscm_standard_parameters_handling(self, test_adapter):
        parameters = test_adapter._parameters

        parameters.generic("Start Time").value = np.datetime64("1850-01-01")
        parameters.generic("Stop Time").value = np.datetime64("2100-01-01")
        ecs_magnitude = 3.12
        parameters.scalar(
            "Equilibrium Climate Sensitivity", "delta_degC"
        ).value = ecs_magnitude

        parameters.scalar(("MAGICC6", "core_climatesensitivity"), "delta_degC").value = 2 * ecs_magnitude

        self.prepare_run_input(
            test_adapter,
            parameters.generic("Start Time").value,
            parameters.generic("Stop Time").value,
        )
        test_adapter.reset()
        test_adapter.run()

        # make sure OpenSCM ECS value was used preferentially to the model's
        # core_climatesensitivity
        assert test_adapter._output.scalar(("MAGICC6", "core_climatesensitivity"), "delta_degC").value == ecs_magnitude
        assert (
            parameters.scalar(("MAGICC6", "core_climatesensitivity"), "delta_degC").value == 2*ecs_magnitude
        )

        # double check the model didn't do anything funky
        assert (
            parameters.scalar("Equilibrium Climate Sensitivity", "delta_degC").value
            == ecs_magnitude
        )
        assert parameters.generic("Start Time").value == np.datetime64("1850-01-01")
        assert parameters.generic("Stop Time").value == np.datetime64("2100-01-01")

    def prepare_run_input(self, test_adapter, start_time, stop_time):
        pass
        # test_adapter._parameters.generic("Start Time").value = start_time
        # test_adapter._parameters.generic("Stop Time").value = stop_time

        # npoints = int((stop_time - start_time) / YEAR) + 1  # include self._stop_time

        # time_points_for_averages = create_time_points(
        #     start_time,
        #     stop_time - start_time,
        #     npoints,
        #     ParameterType.AVERAGE_TIMESERIES,
        # )
        # test_adapter._parameters.timeseries(
        #     ("Emissions", "CO2"),
        #     "GtCO2/a",
        #     time_points=time_points_for_averages,
        #     timeseries_type="average",
        # ).values = np.zeros(npoints)

    def test_openscm_standard_parameters_handling_on_init(self):
        parameters = ParameterSet()
        output_parameters = ParameterSet()

        ecs_magnitude = 2.76
        parameters.scalar(
            "Equilibrium Climate Sensitivity", "delta_degC"
        ).value = ecs_magnitude

        tadapter = self.tadapter(parameters, output_parameters)

        # will need to update this test
        assert (
            # OpenSCM ECS value passed into parameters
            tadapter._parameters.scalar("Equilibrium Climate Sensitivity", "delta_degC").value
            == ecs_magnitude
        )
        assert (
            # OpenSCM ECS value doesn't overwrite MAGICC value (may want to update in
            # future, true test is what comes when `run` is called which is tested
            # elsewhere)
            tadapter._parameters.scalar(("MAGICC6", "core_climatesensitivity"), "delta_degC").value
            == 3.0
        )

    def test_defaults(self, test_adapter):
        assert test_adapter._start_time == np.datetime64("1500-01-01")
        assert test_adapter._end_time == np.datetime64("4200-01-01")
        assert test_adapter._timestep_count == 2701

    def test_timeseries_time_points_require_update(self, test_adapter):
        assert test_adapter._timeseries_time_points_require_update()
        test_adapter._set_model_from_parameters()
        assert not test_adapter._timeseries_time_points_require_update()
        test_adapter._parameter_views["Start Time"].value = np.datetime64("1900-01-01")
        assert test_adapter._timeseries_time_points_require_update()
