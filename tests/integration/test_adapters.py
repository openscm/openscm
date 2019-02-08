import datetime
from abc import ABCMeta, abstractmethod
from unittest.mock import Mock, MagicMock
import re


import pytest
import numpy as np


from openscm.adapters import MAGICC6, Hector
from openscm.core import ParameterSet
from openscm.errors import ModelNotInitialisedError
from openscm.scenarios import rcps
from openscm.highlevel import convert_scmdataframe_to_core
from openscm.errors import NotAnScmParameterError
from openscm.utils import convert_datetime_to_openscm_time
from openscm.constants import ONE_YEAR_IN_S_INTEGER


from conftest import assert_core


@pytest.fixture(scope="function")
def test_adapter(request):
    try:
        yield request.cls.tadapter()
    except TypeError:
        pytest.skip("{} cannot be instantiated".format(str(request.cls.tadapter)))


@pytest.fixture(scope="function")
def test_config_paraset():
    parameters = ParameterSet()
    ecs_writable = parameters.get_writable_scalar_view("ecs", ("World",), "K")
    ecs_writable.set(3)
    rf2x_writable = parameters.get_writable_scalar_view("rf2xco2", ("World",), "W / m^2")
    rf2x_writable.set(4.0)

    yield parameters


@pytest.fixture(scope="function")
def test_drivers_core():
    core = convert_scmdataframe_to_core(rcps.filter(scenario="RCP26"))

    yield core


class _AdapterTester(object):
    @property
    @abstractmethod
    def tadapter(self):
        pass

    def test_initialize(self, test_adapter):
        assert not test_adapter.initialized
        test_adapter.initialize()
        assert test_adapter.initialized

    def test_shutdown(self, test_adapter):
        test_adapter.initialize()
        test_adapter.shutdown()

    def test_set_config(self, test_adapter, test_config_paraset):
        with pytest.raises(ModelNotInitialisedError):
            test_adapter.set_config(test_config_paraset)

        test_adapter.initialize()
        test_adapter.set_config(test_config_paraset)

    def test_junk_config(self, test_adapter, test_config_paraset):
        test_adapter.initialize()
        tname = "junk"
        junk_w = test_config_paraset.get_writable_scalar_view(tname, ("World",), "K")
        junk_w.set(4)
        error_msg = re.escape(
            "{} is not a {} parameter".format(tname, self.tadapter.__name__)
        )
        with pytest.raises(NotAnScmParameterError, match=error_msg):
            test_adapter.set_config(test_config_paraset)

    def test_run(self, test_adapter, test_config_paraset, test_drivers_core):
        test_adapter.initialize()
        test_adapter.set_config(test_config_paraset)
        test_adapter.set_drivers(test_drivers_core)
        res = test_adapter.run()

        pview = res.parameters.get_scalar_view(
            name=("ecs",),
            region=("World",),
            unit="K"
        )
        assert pview.get() == 3

        pview = res.parameters.get_scalar_view(
            name=("rf2xco2",),
            region=("World",),
            unit="W / m^2"
        )
        assert pview.get() == 4.0


class TestMAGICCAdapter(_AdapterTester):
    tadapter = MAGICC6

    def test_initialize(self, test_adapter):
        assert test_adapter.magicc is None
        super().test_initialize(test_adapter)
        assert test_adapter.magicc is not None

    def test_initialize_arg_passing(self, test_adapter):
        mock_magicc = MagicMock()
        test_adapter._magicc_class = Mock(return_value=mock_magicc)
        tkwargs = {"here": "there", "everywhere": "nowhere"}
        test_adapter.initialize(**tkwargs)

        test_adapter._magicc_class.assert_called_with(**tkwargs)
        mock_magicc.__enter__.assert_called()

    def test_shutdown(self, test_adapter):
        super().test_shutdown(test_adapter)

        test_adapter.magicc = MagicMock()
        test_adapter.shutdown()
        test_adapter.magicc.__exit__.assert_called()

    def test_set_config(self, test_adapter, test_config_paraset):
        super().test_set_config(test_adapter, test_config_paraset)

        tf2x = 3.8
        rf2x_writable = test_config_paraset.get_writable_scalar_view(
            "rf2xco2", ("World",), "mW / m^2"
        )
        rf2x_writable.set(tf2x * 1000)

        tco2_tempfeedback_switch = False
        co2_tempfeedback_switch_writable = test_config_paraset.get_writable_boolean_view(
            "co2_tempfeedback_switch", ("World",)
        )
        co2_tempfeedback_switch_writable.set(tco2_tempfeedback_switch)

        tgen_sresregions2nh = np.array([0.95, 1.0, 1.0, 0.4])
        gen_sresregions2nh_writable = test_config_paraset.get_writable_array_view(
            "gen_sresregions2nh", ("World",), "dimensionless"
        )
        gen_sresregions2nh_writable.set(tgen_sresregions2nh)

        test_adapter.initialize()
        test_adapter.set_config(test_config_paraset)

        magicc_config = test_adapter.magicc.update_config()

        np.testing.assert_allclose(
            magicc_config["nml_allcfgs"]["core_climatesensitivity"], 3
        )
        np.testing.assert_allclose(magicc_config["nml_allcfgs"]["core_delq2xco2"], tf2x)
        assert (
            magicc_config["nml_allcfgs"]["co2_tempfeedback_switch"]
            == tco2_tempfeedback_switch
        )
        assert (
            magicc_config["nml_allcfgs"]["gen_sresregions2nh"] == tgen_sresregions2nh
        ).all()

    def test_run(self, test_adapter, test_config_paraset, test_drivers_core):
        super().test_run(test_adapter, test_config_paraset, test_drivers_core)

        test_adapter.initialize()
        test_adapter.set_config(test_config_paraset)
        test_adapter.set_drivers(test_drivers_core)
        res = test_adapter.run()

        def get_comparison_time_for_year(yr):
            return convert_datetime_to_openscm_time(
                datetime.datetime(yr, 1, 1,)
            )

        assert_core(
            9.1478,
            get_comparison_time_for_year(2017),
            res,
            ("Emissions", "CO2", "MAGICC Fossil and Industrial"),
            "World",
            "GtC / yr",
            res.start_time,
            ONE_YEAR_IN_S_INTEGER,
        )

        assert_core(
            1.5833606,  # MAGICC6 should be stabe
            get_comparison_time_for_year(2100),
            res,
            ("Surface Temperature"),
            "World",
            "K",
            res.start_time,
            ONE_YEAR_IN_S_INTEGER,
        )


class TestHectorAdapter(_AdapterTester):
    tadapter = Hector
