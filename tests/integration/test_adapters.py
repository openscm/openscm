from abc import ABCMeta, abstractmethod
from unittest.mock import Mock, MagicMock


import pytest
import numpy as np


from openscm.adapters import MAGICC6, Hector
from openscm.core import ParameterSet
from openscm.errors import ModelNotInitialisedError


@pytest.fixture(scope="function")
def test_adapter(request):
    try:
        yield request.cls.tadapter()
    except TypeError:
        pytest.skip("{} cannot be instantiated".format(str(request)))


@pytest.fixture(scope="function")
def test_config_paraset():
	parameters = ParameterSet()
	ecs_writable = parameters.get_writable_scalar_view(
		"ecs",
		"World",
		"K"
	)
	ecs_writable.set(3)

	yield parameters


# @pytest.fixture(scope="function")
# def test_drivers_paraset():
# 	parameters = ParameterSet()
# 	# convert e.g. scenario drivers to parameterset

# 	yield parameters


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

	# def test_junk_config(self, test_adapter, test_config_paraset):
		# test_adapter.initialize()
		# what to do here
		# with pytest.raises(Error):
		# 	test_adapter.set_config(junk para set)

	# def test_run(self, test_adapter, test_drivers_paraset):
	# 	test_adapter.initialize()
	# 	test_adapter.set_drivers(test_drivers_paraset)
	# 	test_adapter.run()


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
		f2x_writable = test_config_paraset.get_writable_scalar_view(
			"f2xco2",
			"World",
			"mW / m^2"
		)
		f2x_writable.set(tf2x * 1000)

		# tco2_tempfeedback_switch = False
		# co2_tempfeedback_switch_writable = test_config_paraset.get_boolean_view(
		# 	"co2_tempfeedback_switch",
		# 	"World",
		# )
		# co2_tempfeedback_switch_writable.set(tco2_tempfeedback_switch)

		# tco2_tempfeedback_switch = False
		# co2_tempfeedback_switch_writable = test_config_paraset.get_boolean_view(
		# 	"co2_tempfeedback_switch",
		# 	"World",
		# )
		# co2_tempfeedback_switch_writable.set(tco2_tempfeedback_switch)

		test_adapter.initialize()
		test_adapter.set_config(test_config_paraset)

		magicc_config = test_adapter.magicc.update_config()

		np.testing.assert_allclose(magicc_config["nml_allcfgs"]["core_climatesensitivity"], 3)
		np.testing.assert_allclose(magicc_config["nml_allcfgs"]["core_delq2xco2"], tf2x)
		# assert magicc_config["nml_allcfgs"]["co2_tempfeedback_switch"] == tco2_tempfeedback_switch

	# def test_run(self, test_adapter):
	# 	test_adapter.initialize()
	# 	test_adapter.set_drivers()
	# 	test_adapter.run()

	# 	assert on results

class TestHectorAdapter(_AdapterTester):
	tadapter = Hector
