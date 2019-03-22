class _AdapterTester:
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
        assert not test_adapter._initialized
        test_adapter.initialize_run_parameters(
            test_run_parameters.start_time, test_run_parameters.stop_time
        )
        assert test_adapter._initialized

    def test_run(self, test_adapter, test_run_parameters):
        test_adapter.initialize_model_input()
        test_adapter.initialize_run_parameters(
            test_run_parameters.start_time, test_run_parameters.stop_time
        )
        self.prepare_run_input(
            test_adapter, test_run_parameters.start_time, test_run_parameters.stop_time
        )
        test_adapter.reset()
        test_adapter.run()

    def test_step(self, test_adapter, test_run_parameters):
        test_adapter.initialize_model_input()
        test_adapter.initialize_run_parameters(
            test_run_parameters.start_time, test_run_parameters.stop_time
        )
        self.prepare_run_input(
            test_adapter, test_run_parameters.start_time, test_run_parameters.stop_time
        )
        test_adapter.reset()
        assert test_adapter._current_time == test_run_parameters.start_time
        try:
            new_time = test_adapter.step()
            assert new_time > test_run_parameters.start_time
        except NotImplementedError:
            pass

    def prepare_run_input(self, test_adapter, start_time, stop_time):
        """
        Overload this in your adapter test if you need to set required input parameters.
        This method is called directly after ``test_adapter.initialize_run_parameters``
        and before ``test_adapter.run`` or ``test_adapter.step`` during tests.
        """
        pass
