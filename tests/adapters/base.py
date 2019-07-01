from abc import ABCMeta, abstractmethod


class _AdapterTester(metaclass=ABCMeta):
    """
    Base class for adapter testing.

    At minimum, a new adapter should define a subclass of this class called,
    ``AdapterXTester`` which has ``tadapter`` set to the adapter to be tested. This
    ensures that the new adapter is subject to all of OpenSCM's minimum requirements
    whilst giving authors the ability to tweak the tests as necessary for their specific
    adapter.
    """

    tadapter = None
    """
    Adapter to test
    """

    def test_initialize(self, test_adapter):
        test_adapter._initialize_model()

    def test_shutdown(self, test_adapter):
        """
        Test the adapter can be shutdown.

        Extra tests can be adapted depending on what the adapter should actually
        do on shutdown.
        """
        del test_adapter

    def test_initialize_model_input(self, test_adapter):
        """
        Test that initalizing model input does as intended.
        """
        assert not test_adapter._initialized
        test_adapter.initialize_model_input()
        assert test_adapter._initialized

    def test_initialize_run_parameters(self, test_adapter, test_run_parameters):
        """
        Test that initalizing run parameters does as intended.
        """
        self.prepare_run_input(
            test_adapter, test_run_parameters.start_time, test_run_parameters.stop_time
        )
        assert not test_adapter._initialized
        test_adapter.initialize_run_parameters()
        assert test_adapter._initialized

    def test_run1(self, test_adapter, test_run_parameters):
        self.prepare_run_input(
            test_adapter, test_run_parameters.start_time, test_run_parameters.stop_time
        )
        test_adapter.initialize_model_input()
        test_adapter.initialize_run_parameters()
        test_adapter.reset()
        test_adapter.run()

    def test_run2(self, test_adapter, test_run_parameters):
        self.prepare_run_input(
            test_adapter, test_run_parameters.start_time, test_run_parameters.stop_time
        )
        test_adapter.initialize_run_parameters()
        test_adapter.initialize_model_input()
        test_adapter.reset()
        test_adapter.run()

    def test_step(self, test_adapter, test_run_parameters):
        self.prepare_run_input(
            test_adapter, test_run_parameters.start_time, test_run_parameters.stop_time
        )
        test_adapter.initialize_model_input()
        test_adapter.initialize_run_parameters()
        test_adapter.reset()
        assert test_adapter._current_time == test_run_parameters.start_time
        try:
            new_time = test_adapter.step()
            assert new_time > test_run_parameters.start_time
        except NotImplementedError:
            pass

    @abstractmethod
    def test_openscm_standard_parameters_handling(self):
        """
        Test how the adapter handles OpenSCM's standard parameters.

        Implementers must implement this method to check what the user would get when
        OpenSCM's standard parameters are passed to the adapter. It might be that they
        get used, that they are re-mapped to a different name, that they are not
        supported and hence nothing is done. All these behaviours are valid, they just
        need to be tested and validated.

        We give an example of how such a test might look below.
        """
        pass  # TODO: implement once parameter usage can be checked
