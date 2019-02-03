from abc import ABCMeta, abstractmethod
from unittest.mock import Mock


from openscm.adapters import MAGICC6


class _AdapterTester(object):
	@property
	@abstractmethod
	def tadapter(self):
		pass

	def test_init(self, test_adapter):
		test_adapter.initialize()


class TestMAGICCAdapter(_AdapterTester):
	tadapter = MAGICC6

	def test_init(self, test_adapter):
		assert test_adapter.magicc is None
		super().test_init(test_adapter)
		assert test_adapter.magicc is not None

	def test_init_arg_passing(self, test_adapter):
		test_adapter._magicc_class = Mock()
		tkwargs = {"here": "there", "everywhere": "nowhere"}
		test_adapter.initialize(**tkwargs)

		test_adapter._magicc_class.assert_called_with(**tkwargs)