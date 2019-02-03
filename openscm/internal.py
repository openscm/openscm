"""
Internally used classes and functions including unit handling and the
model adapter.
"""

from .core import ParameterSet
from abc import ABCMeta, abstractmethod


class Adapter(metaclass=ABCMeta):
    """
    Base class for model adapters which wrap specific SCMs.

    A model adapter is responsible for requesting the expected input
    parameters (in the expected time format and units) for the
    particular SCM from a :class:`openscm.core.ParameterSet`.  It also
    runs its wrapped SCM and writes the output data back to a
    :class:`openscm.core.ParameterSet`.
    """

    def __init__(self, parameters: ParameterSet):
        self.parameters = parameters

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the model.
        """
        pass

    @abstractmethod
    def run(self) -> None:
        """
        Run the model over the full time range.
        """
        pass

    @abstractmethod
    def step(self) -> None:
        """
        Do a single time step.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        Shut the model down, cleaning up any artefacts.
        """
        pass