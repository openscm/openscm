from abc import ABCMeta, abstractmethod, abstractproperty

import pytest

from openscm.core.parameterset import ParameterSet


class _AdapterTester(metaclass=ABCMeta):
    """
    Base class for adapter testing.

    At minimum, a new adapter should define a subclass of this class called,
    ``AdapterXTester`` which has ``tadapter`` set to the adapter to be tested. This
    ensures that the new adapter is subject to all of OpenSCM's minimum requirements
    whilst giving authors the ability to tweak the tests as necessary for their specific
    adapter.
    """

    @abstractproperty
    def tadapter(self):
        """
        Adapter to test
        """
        pass

    @abstractmethod
    def test_initialize(self, test_adapter):
        """
        Test the adapter is initiated as intended.

        Also tests that passed in ``ParameterSet``'s are filled in.

        Extra tests should be added for different adapters, to check any other
        expected behaviour as part of ``__init__`` calls.
        """
        parameters = ParameterSet()
        output_parameters = ParameterSet()
        tadapter = self.tadapter(parameters, output_parameters)

    @abstractmethod
    def test_shutdown(self, test_adapter):
        """
        Test the adapter can be shutdown.

        Extra tests should be added depending on what the adapter should actually
        do on shutdown.
        """
        del test_adapter

    @abstractmethod
    def test_run(self, test_adapter, test_run_parameters):
        """
        Test that running the model does as intended.

        Extra tests should be added depending on what the adapter should actually
        do when run with the parameters provided by `test_run_parameters`.
        """
        self.prepare_run_input(
            test_adapter, test_run_parameters.start_time, test_run_parameters.stop_time
        )
        test_adapter.reset()
        test_adapter.run()

    @abstractmethod
    def test_step(self, test_adapter, test_run_parameters):
        """
        Test that stepping the model does as intended.

        Extra tests should be added depending on what the adapter should do when
        stepped.
        """
        # if your model cannot step, override this method with the below
        """
        pytest.skip("Step unavailable for {}".format(type(test_adapter)))
        """
        # otherwise, use the below as a base
        self.prepare_run_input(
            test_adapter, test_run_parameters.start_time, test_run_parameters.stop_time
        )
        test_adapter.reset()
        assert test_adapter._current_time == test_run_parameters.start_time

        new_time = test_adapter.step()
        assert new_time > test_run_parameters.start_time

    @abstractmethod
    def test_run_reset_run_same(self, test_adapter, test_run_parameters):
        """
        Test that running, resetting and running the model again does as intended.

        Below we give an example of how this could look. Implementers should write their
        own implementation which tests key output for their model to make this test more
        robust.
        """
        output = test_adapter._output

        self.prepare_run_input(
            test_adapter, test_run_parameters.start_time, test_run_parameters.stop_time
        )

        check_args = [
            "Surface Temperature",
            "K",
            np.array(
                [
                    np.datetime64(
                        "{}-01-01".format(y).astype("datetime64[s]").astype(float)
                    )
                    for y in range(2010, 2090, 10)
                ]
            ),
        ]
        assert output.timeseries(*check_args).empty

        test_adapter.reset()
        test_adapter.run()
        first_run_temperature = output.timeseries(*check_args).values

        test_adapter.reset()
        assert output.timeseries(*check_args).empty
        test_adapter.run()
        second_run_temperature = output.timeseries(*check_args).values

        np.testing.assert_allclose(first_run_temperature, second_run_temperature)

    @abstractmethod
    def test_step_reset_run_same(self, test_adapter, test_run_parameters):
        """
        Test that running, resetting, stepping, resetting and running the model again does as intended.

        Below we give an example of how this could look. Implementers should write their
        own implementation which tests key output for their model to make this test more
        robust.
        """
        output = test_adapter._output

        self.prepare_run_input(
            test_adapter, test_run_parameters.start_time, test_run_parameters.stop_time
        )

        check_args = [
            "Surface Temperature",
            "K",
            np.array(
                [
                    np.datetime64(
                        "{}-01-01".format(y).astype("datetime64[s]").astype(float)
                    )
                    for y in range(2010, 2091, 10)
                ]
            ),
        ]
        assert output.timeseries(*check_args).empty

        test_adapter.reset()
        test_adapter.run()
        first_run_temperature = output.timeseries(*check_args).values

        test_adapter.reset()
        assert output.timeseries(*check_args).empty
        test_adapter.step()
        test_adapter.step()
        first_two_steps_temperature = output.timeseries(
            "Surface Temperature",
            "K",
            np.array(
                [
                    np.datetime64(
                        "{}-01-01".format(y).astype("datetime64[s]").astype(float)
                    )
                    for y in range(2010, 2031, 10)
                ]
            ),
        )
        np.testing.assert_allclose(
            first_run_temperature[:2], first_two_steps_temperature
        )

        test_adapter.reset()
        assert output.timeseries(*check_args).empty
        test_adapter.run()
        second_run_temperature = output.timeseries(*check_args).values

        np.testing.assert_allclose(first_run_temperature, second_run_temperature)

    @abstractmethod
    def test_openscm_standard_parameters_handling(self, test_adapter):
        """
        Test how the adapter handles OpenSCM's standard parameters.

        Implementers must implement this method to check what the user would get when
        OpenSCM's standard parameters are passed to the adapter. It might be that they
        get used, that they are re-mapped to a different name, that they are not
        supported and hence nothing is done. All these behaviours are valid, they just
        need to be tested and validated.

        We give an example of how such a test might look below.
        """
        parameters = test_adapter._parameters
        output_parameters = test_adapter._output

        parameters.generic("Start Time").value = np.datetime64("1850-01-01")
        parameters.generic("Stop Time").value = np.datetime64("2100-01-01")
        ecs_magnitude = 3.12
        parameters.scalar(
            "Equilibrium Climate Sensitivity", "delta_degC"
        ).value = ecs_magnitude

        self.prepare_run_input(
            test_adapter,
            parameters.generic("Start Time").value,
            parameters.generic("Stop Time").value,
        )
        test_adapter.reset()
        test_adapter.run()

        # From here onwards you can test whether e.g. the parameters have been used as
        # intended, an error was thrown or the parameters were not used.
        # If you're testing the parameters are used as intended, it might look
        # something like:
        assert (
            parameters.scalar(("Model name", "model ecs parameter"), "delta_degC").value
            == ecs_magnitude  # make sure OpenSCM ECS value was used preferentially to the model's ecs
        )

        assert (
            output_parameters.scalar(
                "Equilibrium Climate Sensitivity", "delta_degC"
            ).value
            == ecs_magnitude
        )
        assert output_parameters.generic("Start Time").value == np.datetime64(
            "1850-01-01"
        )
        assert output_parameters.generic("Stop Time").value == np.datetime64(
            "2100-01-01"
        )
