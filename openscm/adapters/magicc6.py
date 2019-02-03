import pymagicc


from ..internal import Adapter
from ..core import Core


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

    def run(self) -> Core:
    	raise NotImplementedError

    def step(self) -> None:
    	raise NotImplementedError