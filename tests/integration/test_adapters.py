from abc import ABCMeta, abstractmethod


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