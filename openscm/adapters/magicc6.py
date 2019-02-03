import pymagicc


from ..internal import Adapter
from ..core import Core


class MAGICC6(Adapter):
    def __init__(self):
    	self.magicc = pymagicc.MAGICC6()

    def initialize(self) -> None:
    	raise NotImplementedError

    def run(self) -> Core:
    	raise NotImplementedError

    def step(self) -> None:
    	raise NotImplementedError