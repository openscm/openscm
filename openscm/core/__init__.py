"""
The OpenSCM Core API includes the basic functionality to run a particular simple
climate model with OpenSCM as well as setting/getting its :ref:`parameter <parameters>`
values. Mapping of :ref:`parameter names <parameter-hierarchy>` and :ref:`units <units>`
is done internally.
"""

from ..adapters import Adapter, load_adapter
from .parameterset import ParameterSet


class OpenSCM:
    """
    OpenSCM core class.

    Represents a particular simple climate model to be run.
    """

    _model: Adapter
    """Adapter of the SCM to run"""

    _model_name: str
    """Name of the SCM to run"""

    _output: ParameterSet
    """Set of output :ref:`parameters <parameters>` of the model"""

    _parameters: ParameterSet
    """Set of input :ref:`parameters <parameters>` for the model"""

    def __init__(self, model_name: str):
        """
        Initialize.

        Parameters
        ----------
        model
            Name of the SCM to run

        Raises
        ------
        KeyError
            No adapter for SCM named ``model`` found
        """
        self._input_parameters = ParameterSet()
        self._output_parameters = ParameterSet()
        self._model_name = model_name
        self._model = load_adapter(model_name)(
            self._input_parameters, self._output_parameters
        )

    @property
    def model(self) -> str:
        """
        Name of the SCM
        """
        return self._model_name

    @property
    def output(self) -> ParameterSet:
        """
        Set of output parameters of the model
        """
        return self._output_parameters

    @property
    def parameters(self) -> ParameterSet:
        """
        Set of input parameters for the model
        """
        return self._input_parameters

    def reset_stepping(self, start_time: int, stop_time: int) -> None:
        """
        Reset the model before starting stepping.

        Parameters
        ----------
        start_time
            Beginning of the time range to run over (seconds since
            ``1970-01-01 00:00:00``)
        stop_time
            End of the time range to run over (including; seconds since
            ``1970-01-01 00:00:00``)
        """
        self._model.initialize_model_input()
        self._model.initialize_run_parameters(start_time, stop_time)
        self._model.reset()

    def run(self, start_time: int, stop_time: int) -> None:
        """
        Run the model over the full time range.

        Parameters
        ----------
        start_time
            Beginning of the time range to run over (seconds since
            ``1970-01-01 00:00:00``)
        stop_time
            End of the time range to run over (including; seconds since
            ``1970-01-01 00:00:00``)
        """
        self.reset_stepping(start_time, stop_time)
        self._model.run()

    def step(self) -> int:
        """
        Do a single time step.

        Returns
        -------
        int
            Current time (seconds since ``1970-01-01 00:00:00``)
        """
        # TODO check if reset_stepping has been called
        return self._model.step()
