class _AdapterTester(object):
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

    @classmethod
    def test_initialize(cls):
        tadapter = cls.tadapter
        assert not tadapter.initialized
        tadapter._initialize_model()
        assert tadapter.initialized

    @classmethod
    def test_shutdown(cls, test_adapter):
        """
        Test the adapter can be shutdown.

        Extra tests can be adapted depending on what the adapter should actually
        do on shutdown.
        """
        test_adapter.shutdown()

    @classmethod
    def test_initialize_model_input(cls, test_adapter, test_config_paraset):
        """
        Test that initalizing model input does as intended.
        """
        # TODO test for missing but mandatory parameter
        test_adapter.initialize_model_input(test_config_paraset)

    @classmethod
    def test_initialize_model_input_non_model_parameter(
        cls, test_adapter, test_config_paraset
    ):
        tname = "junk"
        test_config_paraset.get_writable_scalar_view(tname, ("World",), "K").set(4)
        test_adapter.initialize_model_input(test_config_paraset)
        # TODO test that "junk" has not been used

    @classmethod
    def test_initialize_run_parameters(cls, test_adapter, test_config_paraset):
        """
        Test that initalizing run parameters does as intended.
        """
        # TODO see test_initialize_model_input
        test_adapter.initialize_run_parameters(test_config_paraset)

    @classmethod
    def test_initialize_run_parameters_non_model_parameter(
        cls, test_adapter, test_config_paraset
    ):
        tname = "junk"
        test_config_paraset.get_writable_scalar_view(tname, ("World",), "K").set(4)
        test_adapter.initialize_run_parameters(test_config_paraset)
        # TODO see test_initialize_model_input_non_model_parameter

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
