"""
Module including all model adapters shipped with OpenSCM.
"""

from abc import ABCMeta, abstractmethod

from ..errors import AdapterNeedsModuleError

_loaded_adapters = {}


class Adapter(metaclass=ABCMeta):
    """
    All model adapters in OpenSCM are implemented as subclasses of the
    :class:`openscm.adapter.Adapter` base class.

    :ref:`writing-adapters` provides a how-to on implementing an adapter.

    A model adapter is responsible for running the model based on the inputs from
    OpenSCM and writing the data back into the OpenSCM format [TODO the OpenSCM
    format].
    """

    def __init__(self):
        """
        Initialize.

        Attributes
        ----------
        _initialized : bool
            Has the model been initialised via :func:`_initialize_model`?

        _initialized_inputs : bool
            Have the model inputs been initialised via :func:`initialize_model_input`?
        """
        self._initialized = False
        self._initialized_inputs = False

    def __del__(self):
        """
        Destructor.
        """
        self._shutdown()

    def initialize_model_input(self):
        """
        Initialize the model input.

        Called before the adapter is used in any way and at most once before a call to
        :func:`run`.
        """
        if not self._initialized:
            self._initialize_model()
            self._initialized = True

        self._initialize_model_input()

    def initialize_run_parameters(self):
        """
        Initialize parameters for the run.

        Called before the adapter is used in any way and at most once before a call to
        :func:`run`.
        """
        if not self._initialized:
            self._initialize_model()
            self._initialized = True

        self._initialize_run_parameters()

    def reset(self):
        """
        Reset the model to prepare for a new run.

        Called once after each call of :func:`run`.
        """
        self._reset()

    def run(self):
        """
        Run the model over the full time range.
        """
        self._run()

    @abstractmethod
    def _initialize_model(self):
        """
        To be implemented by specific adapters.

        Initialize the model. Called only once but as late as possible before a call to
        :func:`_run`.
        """

    @abstractmethod
    def _initialize_model_input(self):
        """
        To be implemented by specific adapters.

        Initialize the model input. Called before the adapter is used in any way and at
        most once before a call to :func:`_run`.
        """

    @abstractmethod
    def _initialize_run_parameters(self):
        """
        To be implemented by specific adapters.

        Initialize parameters for the run. Called before the adapter is used in any way
        and at most once before a call to :func:`_run`.
        """

    @abstractmethod
    def _reset(self):
        """
        To be implemented by specific adapters.

        Reset the model to prepare for a new run. Called once after each call of
        :func:`_run`.
        """

    @abstractmethod
    def _run(self):
        """
        To be implemented by specific adapters.

        Run the model over the full time range.
        """

    @abstractmethod
    def _shutdown(self):
        """
        To be implemented by specific adapters.

        Shut the model down.
        """


def load_adapter(name):
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

    adapter = None

    try:
        if name == "DICE":
            from .dice import (  # pylint: disable=cyclic-import,import-outside-toplevel
                DICE,
            )

            adapter = DICE

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
