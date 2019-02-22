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
        # TODO test for missing but mandatory parameter
        test_adapter.initialize_model_input()
        assert test_adapter._initialized

    def test_initialize_model_input_non_model_parameter(self, test_adapter):
        tname = "junk"
        test_adapter._parameters.get_writable_scalar_view(tname, ("World",), "K").set(4)
        test_adapter.initialize_model_input()
        # TODO test that "junk" has not been used

    def test_initialize_run_parameters(self, test_adapter, test_run_parameters):
        """
        Test that initalizing run parameters does as intended.
        """
        assert not test_adapter._initialized
        # TODO see test_initialize_model_input
        test_adapter.initialize_run_parameters(
            test_run_parameters.start_time, test_run_parameters.stop_time
        )
        assert test_adapter._initialized

    def test_initialize_run_parameters_non_model_parameter(
        self, test_adapter, test_run_parameters
    ):
        tname = "junk"
        test_adapter._parameters.get_writable_scalar_view(tname, ("World",), "K").set(4)
        test_adapter.initialize_run_parameters(
            test_run_parameters.start_time, test_run_parameters.stop_time
        )
        # TODO see test_initialize_model_input_non_model_parameter

    def test_run(self, test_adapter, test_run_parameters):
        test_adapter.initialize_model_input()
        test_adapter.initialize_run_parameters(
            test_run_parameters.start_time, test_run_parameters.stop_time
        )

        res = test_adapter.run()

        assert (
            res.parameters.get_scalar_view(
                name=("ecs",), region=("World",), unit="K"
            ).get()
            == 3
        )

        assert (
            res.parameters.get_scalar_view(
                name=("rf2xco2",), region=("World",), unit="W / m^2"
            ).get()
            == 4.0
        )
