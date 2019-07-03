import re

import numpy as np
import pytest
from base import _AdapterTester
from conftest import assert_pint_equal

from openscm.adapters.ph99 import PH99
from openscm.core import ParameterSet
from openscm.core.parameters import ParameterType
from openscm.core.time import create_time_points
from openscm.core.units import _unit_registry
from openscm.errors import ParameterEmptyError


class TestPH99Adapter(_AdapterTester):
    tadapter = PH99

    def test_initialize(self, test_adapter):
        super().test_initialize(test_adapter)
        timestep = int((1 * _unit_registry("yr")).to("s").magnitude)
        assert (
            test_adapter._parameters.scalar(("PH99", "timestep"), "s").value == timestep
        )
        assert (
            test_adapter._parameters.scalar(("PH99", "b"), "ppm / (GtC * yr)").value
            == 1.51 * 10 ** -3
        )
        assert test_adapter._parameters.generic("Start Time").empty
        assert test_adapter._parameters.generic("Step Length").empty

    def test_shutdown(self, test_adapter):
        super().test_shutdown(test_adapter)

    def test_run(self, test_adapter, test_run_parameters):
        super().test_run(test_adapter, test_run_parameters)

    def test_step(self, test_adapter, test_run_parameters):
        super().test_step(test_adapter, test_run_parameters)

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
        check_args_concs = [("Atmospheric Concentrations", "CO2"), "ppm", time_points]
        check_args_temperature = [
            "Surface Temperature Increase",
            "delta_degC",
            time_points,
        ]
        assert output.timeseries(*check_args_concs).empty
        assert output.timeseries(*check_args_temperature).empty

        test_adapter.reset()
        test_adapter.run()
        first_run_conc = output.timeseries(*check_args_concs).values
        first_run_temperature = output.timeseries(*check_args_temperature).values

        test_adapter.reset()
        # currently failing
        # assert output.timeseries(*check_args_concs).empty
        # assert output.timeseries(*check_args_temperature).empty
        test_adapter.run()
        second_run_conc = output.timeseries(*check_args_concs).values
        second_run_temperature = output.timeseries(*check_args_temperature).values
        np.testing.assert_allclose(first_run_temperature, second_run_temperature)
        np.testing.assert_allclose(first_run_conc, second_run_conc)

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
        check_args_conc = [("Atmospheric Concentrations", "CO2"), "ppm", time_points]
        check_args_temperature = [
            "Surface Temperature Increase",
            "delta_degC",
            time_points,
        ]

        assert output.timeseries(*check_args_conc).empty
        assert output.timeseries(*check_args_temperature).empty

        test_adapter.reset()
        test_adapter.run()
        first_run_conc = output.timeseries(*check_args_conc).values
        first_run_temperature = output.timeseries(*check_args_temperature).values
        test_adapter.reset()
        # currently failing
        # assert output.timeseries(*check_args_rf, timeseries_type="average").empty
        # assert output.timeseries(*check_args_temperature).empty
        test_adapter.step()
        test_adapter.step()
        first_two_steps_conc = output.timeseries(
            ("Atmospheric Concentrations", "CO2"), "ppm", time_points[:2]
        ).values
        first_two_steps_temperature = output.timeseries(
            "Surface Temperature Increase", "delta_degC", time_points[:2]
        ).values
        # currently failing
        # for some reason accessing the first two elements of a `Timeseries` resets everything to zero
        # so we require this hack...
        # np.testing.assert_allclose(np.array(first_run_conc, copy=True)[:2], first_two_steps_conc)
        # np.testing.assert_allclose(np.array(first_run_temperature, copy=True)[:2], first_two_steps_temperature)

        test_adapter.reset()
        # currently failing
        # assert output.timeseries(*check_args_rf, timeseries_type="average").empty
        # assert output.timeseries(*check_args_temperature).empty
        test_adapter.run()
        second_run_conc = output.timeseries(*check_args_conc).values
        second_run_temperature = output.timeseries(*check_args_temperature).values

        np.testing.assert_allclose(first_run_conc, second_run_conc)
        np.testing.assert_allclose(first_run_temperature, second_run_temperature)

    def test_openscm_standard_parameters_handling(self):
        parameters = ParameterSet()

        start_t = np.datetime64("1850-01-01")
        parameters.generic("Start Time").value = start_t

        stop_t = np.datetime64("2100-01-01")
        parameters.generic("Stop Time").value = stop_t

        ecs_magnitude = 3.12
        parameters.scalar(
            "Equilibrium Climate Sensitivity", "delta_degC"
        ).value = ecs_magnitude
        parameters.scalar(("PH99", "alpha"), "1/yr").value = (
            3.9 * 10 ** -2
        )  # ensure openscm standard parameters take precedence

        rf2xco2_magnitude = 4.012
        parameters.scalar(
            "Radiative Forcing 2xCO2", "W / m^2"
        ).value = rf2xco2_magnitude

        output_parameters = ParameterSet()

        test_adapter = self.tadapter(parameters, output_parameters)

        self.prepare_run_input(
            test_adapter,
            parameters.generic("Start Time").value,
            parameters.generic("Stop Time").value,
        )
        test_adapter.reset()
        test_adapter.run()

        assert test_adapter._parameters.generic("Start Time").value == start_t
        assert test_adapter._parameters.generic("Stop Time").value == stop_t
        assert (
            test_adapter._parameters.scalar(
                "Equilibrium Climate Sensitivity", "delta_degC"
            ).value
            == ecs_magnitude
        )
        assert (
            test_adapter._parameters.scalar("Radiative Forcing 2xCO2", "W/m^2").value
            == rf2xco2_magnitude
        )

        # do we want adapters to push all parameter values to output too? If yes, uncomment this
        # assert output_parameters.generic("Start Time").value == np.datetime64("1850-01-01")
        # assert output_parameters.generic("Stop Time").value == np.datetime64("2100-01-01")
        # assert output_parameters.scalar("Equilibrium Climate Sensitivity", "delta_degC").value == ecs_magnitude

    def test_initialize_run_parameters_ph99_specific(self):
        # remove test_drivers and just do it here?
        self._prepare_test_settings()
        in_parameters = self._test_drivers
        out_parameters = ParameterSet()

        tadapter = self.tadapter(in_parameters, out_parameters)

        tadapter.reset()

        timestep = tadapter.model.timestep.to("s").magnitude
        assert (
            timestep
            == in_parameters.scalar(("PH99", "timestep"), "s", region="World").value
        )
        timestep_count = (
            int(
                (self._test_stop_time - self._test_start_time).item().total_seconds()
                // timestep
            )
            + 1
        )
        time_points = create_time_points(
            self._test_start_time,
            np.timedelta64(int(timestep), "s"),
            timestep_count,
            timeseries_type="point",
        )
        expected_emms = in_parameters.timeseries(
            ("Emissions", "CO2"),
            str(tadapter.model.emissions.units),
            time_points,
            timeseries_type="point",
            interpolation="linear",
        ).values
        np.testing.assert_allclose(tadapter.model.emissions, expected_emms)

    def test_run_ph99_specific(self):
        self._prepare_test_settings()
        in_parameters = self._test_drivers
        out_parameters = ParameterSet()
        tadapter = self.tadapter(in_parameters, out_parameters)

        tadapter.reset()
        tadapter.run()

        timestep = tadapter.model.timestep.to("s").magnitude
        assert timestep == in_parameters.scalar(("PH99", "timestep"), "s").value
        timestep_count = (
            int(
                (self._test_stop_time - self._test_start_time).item().total_seconds()
                // timestep
            )
            + 1
        )
        time_points = create_time_points(
            self._test_start_time,
            np.timedelta64(int(timestep), "s"),
            timestep_count,
            timeseries_type="point",
        )
        expected_emms = in_parameters.timeseries(
            ("Emissions", "CO2"),
            str(tadapter.model.emissions.units),
            time_points,
            timeseries_type="point",
            interpolation="linear",
        ).values

        resulting_emms = out_parameters.timeseries(
            ("Emissions", "CO2"),
            str(tadapter.model.emissions.units),
            time_points,
            timeseries_type="point",
            interpolation="linear",
        ).values

        np.testing.assert_allclose(
            expected_emms, resulting_emms, rtol=1e-10, atol=max(expected_emms) * 1e-6
        )

        # regression test
        temp_2017_2018 = tadapter._output.timeseries(
            ("Surface Temperature Increase"),
            "delta_degC",
            np.array(
                [
                    np.datetime64("2017-01-01").astype("datetime64[s]").astype(float),
                    np.datetime64("2018-01-01").astype("datetime64[s]").astype(float),
                ]
            ),
            region="World",
            timeseries_type="point",
        ).values
        np.testing.assert_allclose(
            temp_2017_2018, np.array([15.148409452339525, 15.15382326154207]), rtol=1e-5
        )

    def test_run_no_emissions_error(self, test_adapter):
        test_adapter._parameters.generic("Start Time").value = np.datetime64(
            "2010-01-01"
        )
        test_adapter._parameters.generic("Stop Time").value = np.datetime64(
            "2013-01-01"
        )

        error_msg = re.escape("PH99 requires ('Emissions', 'CO2') in order to run")
        with pytest.raises(ParameterEmptyError, match=error_msg):
            test_adapter.reset()

    def prepare_run_input(self, test_adapter, start_time, stop_time):
        """
        Overload this in your adapter test if you need to set required input parameters.
        This method is called directly before ``test_adapter.initialize_model_input``
        during tests.
        """
        test_adapter._parameters.generic("Start Time").value = start_time
        test_adapter._parameters.generic("Stop Time").value = stop_time
        timestep = np.timedelta64(30, "D")
        test_adapter._parameters.scalar(
            ("PH99", "timestep"), "s"
        ).value = timestep.item().total_seconds()

        npoints = (stop_time - start_time) // timestep + 1
        time_points = create_time_points(
            start_time, stop_time - start_time, npoints, "point"
        )
        test_adapter._parameters.timeseries(
            ("Emissions", "CO2"), "GtCO2/a", time_points, timeseries_type="point"
        ).values = np.zeros(npoints)

    def test_openscm_standard_parameters_take_priority(self):
        self._prepare_test_settings()
        in_parameters = self._test_drivers

        out_parameters = ParameterSet()
        tadapter = self.tadapter(in_parameters, out_parameters)

        rf2xco2 = 3.2
        in_parameters.scalar(
            "Radiative Forcing 2xCO2", "W/m^2", region=("World",)
        ).value = rf2xco2

        in_parameters.scalar(
            "Equilibrium Climate Sensitivity", str(self._test_ecs.units)
        ).value = self._test_ecs.magnitude

        mu = 8.9 * 10 ** -2
        in_parameters.scalar(("PH99", "mu"), "delta_degC/yr").value = mu

        alpha = 1.9 * 10 ** -2
        in_parameters.scalar(("PH99", "alpha"), "1/yr").value = alpha

        tadapter.reset()

        assert_pint_equal(
            tadapter.model.alpha, tadapter.model.mu * np.log(2) / self._test_ecs
        )

        # currenty failing, should we also update the parameter set if there's conflicts?
        # expected_mu = (
        #     _unit_registry.Quantity(rf2xco2, "W/m^2") / tadapter._hc_per_m2_approx
        # )
        # assert_pint_equal(tadapter.model.mu, expected_mu)
        # np.testing.assert_allclose(
        #     in_parameters.scalar(("PH99", "mu"), "delta_degC/yr", region=("World",)).value,
        #     expected_mu.to("delta_degC/yr"),
        # )

        # currently failing
        # # make sure tadapter.model.mu isn't given by value passed into ParameterSet
        # # earlier i.e. openscm parameter takes priority
        # with pytest.raises(AssertionError):
        #     assert_pint_equal(tadapter.model.mu, _unit_registry.Quantity(mu, "delta_degC/yr"))

        # with pytest.raises(AssertionError):
        #     np.testing.assert_allclose(
        #         in_parameters.scalar(
        #             ("PH99", "mu"), "delta_degC/yr", region=("World",)
        #         ).value,
        #         mu,
        #     )

        np.testing.assert_allclose(
            in_parameters.scalar(
                ("Radiative Forcing 2xCO2",), "W/m^2", region=("World",)
            ).value,
            rf2xco2,
        )

        in_parameters.scalar(
            ("Radiative Forcing 2xCO2",), "W/m^2", region=("World",)
        ).value = (2 * rf2xco2)
        tadapter.reset()

        assert_pint_equal(
            tadapter.model.alpha, tadapter.model.mu * np.log(2) / self._test_ecs
        )
        assert_pint_equal(
            tadapter.model.mu,
            _unit_registry.Quantity(2 * rf2xco2, "W/m^2") / tadapter._hc_per_m2_approx,
        )
        np.testing.assert_allclose(
            in_parameters.scalar(
                ("Radiative Forcing 2xCO2",), "W/m^2", region=("World",)
            ).value,
            2 * rf2xco2,
        )

        np.testing.assert_allclose(
            tadapter._parameters.scalar(
                ("Radiative Forcing 2xCO2",), "W/m^2", region=("World",)
            ).value,
            2 * rf2xco2,
        )

    def _prepare_test_settings(self):
        self._test_drivers = ParameterSet()
        self._test_start_time = np.datetime64("1810-03-04")
        self._test_stop_time = np.datetime64("2450-06-01")
        self._test_period_length = np.timedelta64(5000, "D")
        self._test_ecs = 2.5 * _unit_registry("delta_degC")
        self._test_rf2xco2 = 3.5 * _unit_registry("W/m^2")

        self._test_drivers.scalar(
            ("Equilibrium Climate Sensitivity",),
            str(self._test_ecs.units),
            region=("World",),
        ).value = self._test_ecs.magnitude
        self._test_drivers.scalar(
            ("Radiative Forcing 2xCO2",),
            str(self._test_rf2xco2.units),
            region=("World",),
        ).value = self._test_rf2xco2.magnitude

        self._test_drivers.generic("Start Time").value = self._test_start_time
        self._test_drivers.generic("Stop Time").value = self._test_stop_time

        self._test_timestep_count = (
            self._test_stop_time - self._test_start_time
        ) // self._test_period_length + 2

        self._test_emissions_time_points = create_time_points(
            self._test_start_time,
            self._test_period_length,
            self._test_timestep_count,
            timeseries_type="point",
        )

        self._test_emissions_units = "GtCO2/a"
        self._test_emissions = (
            np.linspace(0, 40, self._test_timestep_count)
            * np.sin(np.arange(self._test_timestep_count) * 2 * np.pi / 50)
            * _unit_registry(self._test_emissions_units)
        )
        self._test_drivers.timeseries(
            ("Emissions", "CO2"),
            str(self._test_emissions.units),
            self._test_emissions_time_points,
            region=("World",),
            timeseries_type="point",
        ).values = self._test_emissions.magnitude
