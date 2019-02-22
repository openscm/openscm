"""
All model adapters in OpenSCM are implemented as subclasses of the
:class:`openscm.adapter.Adapter` base class.

:ref:`writing-adapters` provides a how-to on implementing an adapter.
"""

from abc import ABCMeta, abstractmethod

from .core import ParameterSet


class Adapter(metaclass=ABCMeta):
    """
    Base class for model adapters which wrap specific SCMs.

    A model adapter is responsible for requesting the expected input parameters (in the
    expected time format and units) for the particular SCM from a
    :class:`openscm.core.ParameterSet`. It also runs its wrapped SCM and writes the
    output data back to a :class:`openscm.core.ParameterSet`.
    """

    _current_time: int
    """
    Current time when using `step` (seconds since
    ``1970-01-01 00:00:00``)
    """

    _initialized: bool
    """True if model has been initialized via :func:`_initialize_model`"""

    _output: ParameterSet
    """Output parameter set"""

    _parameters: ParameterSet
    """Input parameter set"""

    _start_time: int
    """
    Beginning of the time range to run over (seconds since
    ``1970-01-01 00:00:00``)
    """

    _stop_time: int
    """
    End of the time range to run over (including; seconds since
    ``1970-01-01 00:00:00``)
    """

    def __init__(self, parameters: ParameterSet, output: ParameterSet):
        """
        Initialize.

        Parameters
        ----------
        parameters
            Input parameter set to use
        output
            Output parameter set to use
        """
        self._parameters = parameters
        self._output = output
        self._initialized = False

    def __del__(self):
        """
        Destructor.
        """
        self._shutdown()

    def initialize_model_input(self) -> None:
        """
        Initialize the model input.

        Called before the adapter is used in any way and at most once before a call to
        `run` or `step`.
        """
        if not self._initialized:
            self._initialize_model()
            self._initialized = True
        self._initialize_model_input()

    def initialize_run_parameters(self, start_time: int, stop_time: int) -> None:
        """
        Initialize parameters for the run.

        Called before the adapter is used in any way and at most once before a call to
        `run` or `step`.

        Parameters
        ----------
        start_time
            Beginning of the time range to run over (seconds since
            ``1970-01-01 00:00:00``)
        stop_time
            End of the time range to run over (including; seconds since
            ``1970-01-01 00:00:00``)
        """
        if not self._initialized:
            self._initialize_model()
            self._initialized = True
        self._start_time = start_time
        self._stop_time = stop_time
        self._initialize_run_parameters()

    def reset(self) -> None:
        """
        Reset the model to prepare for a new run.

        Called once after each call of `run` and to reset the model after several calls
        to `step`.
        """
        self._current_time = self._start_time
        self._reset()

    def run(self) -> None:
        """
        Run the model over the full time range.
        """
        self._run()

    def step(self) -> int:
        """
        Do a single time step.

        Returns
        -------
        int
            Current time (seconds since ``1970-01-01 00:00:00``)
        """
        self._step()
        return self._current_time

    @abstractmethod
    def _initialize_model(self) -> None:
        """To be implemented by specific adapters"""
        pass

    @abstractmethod
    def _initialize_model_input(self) -> None:
        """To be implemented by specific adapters"""
        pass

    @abstractmethod
    def _initialize_run_parameters(self) -> None:
        """To be implemented by specific adapters"""
        pass

    @abstractmethod
    def _reset(self) -> None:
        """To be implemented by specific adapters"""
        pass

    @abstractmethod
    def _run(self) -> None:
        """To be implemented by specific adapters"""
        pass

    @abstractmethod
    def _shutdown(self) -> None:
        """To be implemented by specific adapters"""
        pass

    @abstractmethod
    def _step(self) -> None:
        """To be implemented by specific adapters"""
        pass
