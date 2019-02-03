from abc import ABCMeta, abstractmethod
from unittest.mock import Mock, MagicMock


from openscm.adapters import MAGICC6, Hector


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


class TestHectorAdapter(_AdapterTester):
	tadapter = Hector
