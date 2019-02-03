import pymagicc


from ..internal import Adapter
from ..core import Core, ParameterSet


class MAGICC6(Adapter):
    def __init__(self):
    	self._magicc_class = pymagicc.MAGICC6 
    	self.magicc = None

    def initialize(self, **kwargs) -> None:
    	"""Initialise the model.

		Parameters
		----------
		kwargs
			Passed to ``pymagicc.MAGICC6.__init__``
    	"""
    	self.magicc = self._magicc_class(**kwargs)
    	self.magicc.__enter__()

    def set_drivers(self, parameters: ParameterSet) -> None:
        raise NotImplementedError

    def set_config(self, parameters: ParameterSet) -> None:
        raise NotImplementedError

    def run(self) -> Core:
    	raise NotImplementedError

    def step(self) -> None:
    	raise NotImplementedError

    def shutdown(self) -> None:
    	self.magicc.__exit__()