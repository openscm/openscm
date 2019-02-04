from abc import ABCMeta, abstractmethod
from unittest.mock import Mock, MagicMock


import pytest


from openscm.adapters import MAGICC6, Hector
from openscm.core import ParameterSet


@pytest.fixture(scope="function")
def test_adapter(request):
    yield request.cls.tadapter()


@pytest.fixture(scope="function")
def test_config_paraset():
	parameters = ParameterSet()
	ecs_writable = parameters.get_writable_scalar_view(
		"ecs",
		"World",
		"K"
	)
	ecs_writable.set(3)


class _AdapterTester(object):
	@property
	@abstractmethod
	def tadapter(self):
		pass

	def test_initialize(self, test_adapter):
		test_adapter.initialize()

	def test_shutdown(self, test_adapter):
		test_adapter.initialize()
		test_adapter.shutdown()

	def test_set_config(self, test_adapter, test_config_paraset):
		with pytest.raises(ModelUninitialisedError):
			test_adapter.set_config(test_config_paraset)

		test_adapter.initialize()
		test_adapter.set_config(test_config_paraset)


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

		test_adapter.initialize()
		test_adapter.set_config(test_config_paraset)

		magicc_config = test_adapter.magicc.update_config()

		assert magicc_config["core_climatesensitivity"] == 3


class TestHectorAdapter(_AdapterTester):
	tadapter = Hector
