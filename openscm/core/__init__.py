"""
The OpenSCM Core API includes the basic functionality to run a particular simple
climate model with OpenSCM as well as setting/getting its :ref:`parameter <parameters>`
values. Mapping of :ref:`parameter names <parameter-hierarchy>` and :ref:`units <units>`
is done internally.
"""

from typing import Optional

import numpy as np

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

    def __init__(
        self,
        model_name: str,
        input_parameters: Optional[ParameterSet] = None,
        output_parameters: Optional[ParameterSet] = None,
    ):
        """
        Initialize.

        Parameters
        ----------
        model
            Name of the SCM to run
        input_parameters
            Input :class:`ParameterSet` to use (or a new one is used when this is
            ``None``)
        output_parameters
            Output :class:`ParameterSet` to use (or a new one is used when this is
            ``None``)

        Raises
        ------
        KeyError
            No adapter for SCM named :obj:`model` found
        """
        self._input_parameters = (
            input_parameters if input_parameters is not None else ParameterSet()
        )
        self._output_parameters = (
            output_parameters if output_parameters is not None else ParameterSet()
        )
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
        raise NotImplementedError  # noqa
        # return self._output_parameters

    @property
    def parameters(self) -> ParameterSet:
        """
        Set of input parameters for the model
        """
        return self._input_parameters

    def reset_stepping(self) -> None:
        """
        Reset the model before starting stepping.
        """
        raise NotImplementedError  # noqa
        # self._model.initialize_model_input()
        # self._model.initialize_run_parameters()
        # self._model.reset()

    def run(self) -> None:
        """
        Run the model over the full time range.
        """
        raise NotImplementedError  # noqa
        # self.reset_stepping()
        # self._model.run()

    def step(self) -> np.datetime64:
        """
        Do a single time step.

        Returns
        -------
        int
            Current time
        """
        raise NotImplementedError  # noqa
        # # TODO: check if reset_stepping has been called
        # return self._model.step()
