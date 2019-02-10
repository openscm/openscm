"""
All model adapters in OpenSCM are implemented as subclasses of the
:class:`openscm.adapter.Adapter` base class.

:ref:`writing-adapters` provides a how-to on implementing an adapter.
"""

from .core import ParameterSet


class Adapter():
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

    _end_time: int
    """
    End of the time range to run over (including; seconds since
    ``1970-01-01 00:00:00``)
    """

    _parameters: ParameterSet
    """Parameter set"""

    _start_time: int
    """
    Beginning of the time range to run over (seconds since
    ``1970-01-01 00:00:00``)
    """

    def __init__(self, parameters: ParameterSet):
        """
        Initialize.

        Parameters
        ----------
        parameters
            Parameter set to use
        """
        self._parameters = parameters

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
        end_time
            End of the time range to run over (including; seconds since
            ``1970-01-01 00:00:00``)
        """
        self._start_time = start_time
        self._stop_time = stop_time

    def initialize_model_input(self) -> None:
        """
        Initialize the model input.

        Called before the adapter is used in any way and at most once before a call to
        `run` or `step`.
        """
        pass

    def reset(self) -> None:
        """
        Reset the model to prepare for a new run.

        Called once after each call of `run` and to reset the model after several calls
        to `step`.
        """
        self._current_time = self._start_time

    def run(self) -> None:
        """
        Run the model over the full time range.
        """
        raise NotImplementedError

    def step(self) -> int:
        """
        Do a single time step.

        Returns
        -------
        int
            Current time (seconds since ``1970-01-01 00:00:00``)
        """
        raise NotImplementedError
