"""
Module including all model adapters shipped with OpenSCM.
"""

from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

from ..core.parameterset import ParameterSet
from ..errors import AdapterNeedsModuleError

_loaded_adapters: Dict[str, type] = {}


class Adapter(metaclass=ABCMeta):
    """
    All model adapters in OpenSCM are implemented as subclasses of the
    :class:`openscm.adapter.Adapter` base class.

    :ref:`writing-adapters` provides a how-to on implementing an adapter.

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

    def __init__(self, input_parameters: ParameterSet, output_parameters: ParameterSet):
        """
        Initialize.

        Parameters
        ----------
        input_parameters
            Input parameter set to use
        output_parameters
            Output parameter set to use
        """
        self._parameters = input_parameters
        self._output = output_parameters
        self._initialized = False
        self._current_time = 0

    def __del__(self) -> None:
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
        """
        To be implemented by specific adapters.

        Initialize the model. Called only once but as late as possible before a call to
        `_run` or `_step`.
        """

    @abstractmethod
    def _initialize_model_input(self) -> None:
        """
        To be implemented by specific adapters.

        Initialize the model input. Called before the adapter is used in any way and at
        most once before a call to `_run` or `_step`.
        """

    @abstractmethod
    def _initialize_run_parameters(self) -> None:
        """
        To be implemented by specific adapters.

        Initialize parameters for the run. Called before the adapter is used in any way
        and at most once before a call to `_run` or `_step`.
        """

    @abstractmethod
    def _reset(self) -> None:
        """
        To be implemented by specific adapters.

        Reset the model to prepare for a new run. Called once after each call of `_run`
        and to reset the model after several calls to `_step`.
        """

    @abstractmethod
    def _run(self) -> None:
        """
        To be implemented by specific adapters.

        Run the model over the full time range.
        """

    @abstractmethod
    def _shutdown(self) -> None:
        """
        To be implemented by specific adapters.

        Shut the model down.
        """

    @abstractmethod
    def _step(self) -> None:
        """
        To be implemented by specific adapters.

        Do a single time step.
        """


def load_adapter(name: str) -> type:
    """
    Load adapter with a given name.

    Parameters
    ----------
    name
        Name of the adapter/model

    Returns
    -------
    type
        Requested adapter class

    Raises
    ------
    AdapterNeedsModuleError
        Adapter needs a module that is not installed
    KeyError
        Adapter/model not found
    """
    if name in _loaded_adapters:
        return _loaded_adapters[name]

    adapter: Optional[type] = None

    try:
        if name in ["DICE", "unspecified", "a_model"]:  # TODO
            adapter = Adapter

        # TODO remove once we have at least one real adapter in here
        elif name == "MODELNAME":
            from .modelname import MODELNAME

            adapter = MODELNAME

        """
        When implementing an additional adapter, include your adapter NAME here as:
        ```
        elif name == "NAME":
            from .NAME import NAME

            adapter = NAME
        ```
        """
    except ImportError:
        raise AdapterNeedsModuleError(
            "To run '{name}' you need to install additional dependencies. Please "
            "install them using `pip install openscm[model-{name}]`.".format(name=name)
        )

    if adapter is None:
        raise KeyError("Unknown model '{}'".format(name))

    _loaded_adapters[name] = adapter
    return adapter
