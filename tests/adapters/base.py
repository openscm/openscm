class _AdapterTester(object):
    """
    Base class for adapter testing.

    At minimum, a new adapter should define a subclass of this class called,
    ``AdapterXTester`` which has ``tadapter`` equal to the adapter to be tested.
    This ensures that the new adapter is subject to all of OpenSCM's minimum
    requirements whilst giving authors the ability to tweak the tests as necessary
    for their specific adapter.
    """

    tadapter = None
    """
    Adapter to test
    """

    @classmethod
    def test_initialize(cls):
        tadapter = cls.tadapter
        assert not tadapter.initialized
        tadapter._initialize_model()
        assert tadapter.initialized

    @classmethod
    def test_shutdown(cls, test_adapter):
        """
        Test the adapter can be shutdown

        Extra tests can be adapted depending on what the adapter should actually
        do on shutdown.
        """
        test_adapter.shutdown()

    @classmethod
    def test_initialise_model_input(cls, test_adapter, test_config_paraset):
        """
        Test that initalising model input does as intended
        """
        # @swillner I am not sure how we want this to work. I still don't understand
        # how the model can distinguish between 'model_input' and 'run_parameters' if
        # it is always reading from one ParameterSet i.e. how does it know which
        # parameters in the ParameterSet to set when `initialise_model_input` is
        # called vs. `intialise_run_paramters`?
        test_adapter.initialize_model_input(test_config_paraset)
        # some test here that model input was set as intended (will have to be model
        # specific I think).

    @classmethod
    def test_initialise_model_input_non_model_parameter(
        cls, test_adapter, test_config_paraset
    ):
        tname = "junk"
        test_config_paraset.get_writable_scalar_view(tname, ("World",), "K").set(4)
        # What should happen here when we try to write a parameter which the model
        # does not recognise? Warning? Error?
        test_adapter.initialize_model_input(test_config_paraset)

    @classmethod
    def test_initialise_run_parameters(cls, test_adapter, test_config_paraset):
        """
        Test that initalising run parameters does as intended
        """
        # blocked by questions about initialize_model_input above
        test_adapter.initialize_run_parameters(test_config_paraset)

    @classmethod
    def test_initialise_run_parameters_non_model_parameter(
        cls, test_adapter, test_config_paraset
    ):
        # blocked by questions about initialize_model_input above
        tname = "junk"
        test_config_paraset.get_writable_scalar_view(tname, ("World",), "K").set(4)
        # What should happen here when we try to write a parameter which the model
        # does not recognise? Warning? Error?
        test_adapter.initialize_run_parameters(test_config_paraset)

    @classmethod
    def test_run(cls, test_adapter, test_config_paraset, test_drivers_core):
        test_adapter.initialize_model_input(test_drivers_core)
        test_adapter.initialize_run_parameters(test_config_paraset)

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
