"""
Internally used classes and functions including unit handling and the
model adapter.
"""
from abc import ABCMeta, abstractmethod


from .core import ParameterSet
from .errors import ModelNotInitialisedError


class Adapter(metaclass=ABCMeta):
    """
    Base class for model adapters which wrap specific SCMs.

    A model adapter is responsible for requesting the expected input
    parameters (in the expected time format and units) for the
    particular SCM from a :class:`openscm.core.ParameterSet`.  It also
    runs its wrapped SCM and writes the output data back to a
    :class:`openscm.core.ParameterSet`.
    """

    def __init__(self):
        self.initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the model.
        """
        self.initialized = True

    @abstractmethod
    def set_drivers(self, parameters: ParameterSet) -> None:
        """
        Set the drivers (emissions, concentrations etc.) for the model to run.
        """
        pass

    @abstractmethod
    def set_config(self, parameters: ParameterSet) -> None:
        """
        Run the model over the full time range.
        """
        if not self.initialized:
            raise ModelNotInitialisedError

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
