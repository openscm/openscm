"""
Module including all model adapters shipped with OpenSCM.
"""
import warnings
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import namedtuple
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..core.parameters import HierarchicalName, ParameterInfo, ParameterType
from ..core.parameterset import ParameterSet
from ..core.time import ExtrapolationType, InterpolationType, create_time_points
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

    _current_time: np.datetime64
    """Current time when using :func:`step`"""

    _parameters: ParameterSet
    """Input parameter set"""

    _output: ParameterSet
    """Output parameter set"""

    _parameter_views: Dict[HierarchicalName, ParameterInfo]
    """Parameter views available with this model"""

    _parameter_versions: Dict[HierarchicalName, int]
    """Parameter versions (last time parameter was passed to the model)"""

    _openscm_standard_parameter_mappings: Dict[HierarchicalName, str]
    """
    Mapping from OpenSCM parameters to model parameters.

    If required, use property setters to add extra behaviour (like calculating a model 
    parameter based on the value of the OpenSCM parameter) when setting a model 
    parameter from an OpenSCM parameter.
    """

    _direct_set: bool
    """Be careful about overwriting other parameters when setting a model parameter?"""

    def __init__(self, input_parameters: ParameterSet, output_parameters: ParameterSet):
        """
        Initialize the adapter as well as the model sitting underneath it.

        *Note:* as part of this process, all available model parameters are added to 
        ``input_parameters`` (if they're not already there).

        Parameters
        ----------
        input_parameters
            Input parameter set to use

        output_parameters
            Output parameter set to use
        """
        self._parameters = input_parameters
        self._output = output_parameters
        self._parameter_views = {}
        self._parameter_versions = {}
        self._direct_set = False
        self._initialize_model()

    def __del__(self) -> None:
        """
        Destructor.
        """
        self._shutdown()

    def _add_parameter_view(
        self,
        full_name: HierarchicalName,
        value: Optional[Any] = None,
        overwrite: bool = False,
        unit: Optional[str] = None,
        timeseries_type: Optional[Union[ParameterType, str]] = None,
        time_points: Optional[np.ndarray] = None,
        region: HierarchicalName = ("World",),
        interpolation: Union[InterpolationType, str] = "linear",
        extrapolation: Union[ExtrapolationType, str] = "none",
    ):
        """
        Add parameter view to ``self._parameter_views``

        This method also allows default values to be set if the parameter is empty (or 
        ``overwrite`` is True) and stores the version of parameters too.
        """
        if unit is None:
            p = self._parameters.generic(full_name, region=region)
        elif timeseries_type is None:
            p = self._parameters.scalar(full_name, unit, region=region)
        else:
            p = self._parameters.timeseries(
                full_name, unit, region=region, timeseries_type=timeseries_type
            )

        self._parameter_views[full_name] = p

        if value is not None and (p.empty or overwrite):
            # match parameterview to value
            self._set_parameter_value(p, value)
        elif not p.empty and (time_points is not None or timeseries_type is None):
            # match model to parameter view
            self._update_model(full_name, p)

        self._parameter_versions[full_name] = p.version

    def _set_parameter_value(self, p, value):
        if p.parameter_type in (ParameterType.SCALAR, ParameterType.GENERIC):
            p.value = value
        else:
            p.values = value

    def reset(self) -> None:
        """
        Reset the model to prepare for a new run.

        Called once after each call of :func:`run` and to reset the model after several calls
        to :func:`step`.

        *Note:* this method sets the model configuration to match the values in
        `self._parameters``, which is not necessarily the same as the state which was 
        used at the start of the last run.
        """
        self._current_time = self._start_time
        self._set_model_from_parameters()
        self._reset()

    def run(self) -> None:
        """
        Run the model over the full time range.
        """
        self._run()

    def step(self) -> np.datetime64:
        """
        Do a single time step.

        Returns
        -------
        np.datetime64
            Current time
        """
        self._step()
        return self._current_time

    def _set_model_from_parameters(self):
        update_time_points = self._timeseries_time_points_require_update()
        for name, view in self._get_view_iterator():
            update_time = update_time_points and self._parameter_views[
                name
            ].parameter_type in (
                ParameterType.POINT_TIMESERIES,
                ParameterType.AVERAGE_TIMESERIES,
            )
            if update_time:
                current_view = self._parameter_views[name]
                self._parameter_views[name] = self._parameters.timeseries(
                    name,
                    current_view.unit,
                    time_points=self._get_time_points(current_view.parameter_type),
                    region=current_view.region,
                    timeseries_type=current_view.parameter_type,
                    interpolation="linear",  # TODO: take these from ParameterSet
                    extrapolation="none",
                )
            update_para = self._parameter_versions[name] < view.version or update_time
            if update_para:
                if isinstance(name, tuple) and name[0] == self.name:
                    self._direct_set = True
                self._update_model_parameter(name)
                self._parameter_versions[name] = self._parameter_views[name].version
                self._direct_set = False

    def _get_view_iterator(self):
        view_iterator = self._parameter_views.items()
        view_iterator = sorted(view_iterator, key=lambda s: len(s[0]))
        generic_views = [
            v for v in view_iterator if v[1].parameter_type == ParameterType.GENERIC
        ]
        scalar_views = [
            v for v in view_iterator if v[1].parameter_type == ParameterType.SCALAR
        ]
        other_views = [
            v
            for v in view_iterator
            if v[1].parameter_type not in (ParameterType.GENERIC, ParameterType.SCALAR)
        ]
        return generic_views + scalar_views + other_views

    def _update_model_parameter(self, name: HierarchicalName):
        para = self._parameter_views[name]
        self._update_model(name, para)

    def _get_parameter_value(self, p):
        if p.parameter_type in (ParameterType.SCALAR, ParameterType.GENERIC):
            return p.value
        else:
            return p.values

    def _check_derived_paras(self, paras: List[ParameterInfo], name: HierarchicalName):
        for para in paras:
            p = self._parameter_views[para]
            if p.version > 1:
                warnings.warn(
                    "Setting {} overrides setting with {}".format(name, p.name)
                )

    @property
    def _inverse_openscm_standard_parameter_mappings(self):
        return {v: k for k, v in self._openscm_standard_parameter_mappings.items()}

    @abstractmethod
    def _initialize_model(self) -> None:
        """
        Initialize the model, including collecting all relevant ParameterViews. 

        Called only once, during :func:`__init__`.
        """

    @abstractmethod
    def _reset(self) -> None:
        """
        Perform model specific steps to reset after a run.
        """

    @abstractmethod
    def _run(self) -> None:
        """
        Run the model over the full time range.
        """

    @abstractmethod
    def _shutdown(self) -> None:
        """
        Shut the model down.
        """

    @abstractmethod
    def _step(self) -> None:
        """
        Do a single time step.
        """

    @abstractmethod
    def _get_time_points(
        self, timeseries_type: Union[ParameterType, str]
    ) -> np.ndarray:
        """
        Get time points for timeseries views.

        Parameters
        ----------
        timeseries_type
            Type of timeseries for which to get points
        """

    @abstractmethod
    def _update_model(self, name: HierarchicalName, para: ParameterInfo) -> None:
        """
        Update a model value

        Parameters
        ----------
        name
            Name of parameter to update

        para
            Parameter view to use for the update
        """

    @abstractmethod
    def _timeseries_time_points_require_update(self) -> None:
        """
        Determine if the timeseries view time points require updating
        """

    @abstractproperty
    def name(self):
        """
        Name of the model as used in OpenSCM parameters
        """

    @abstractproperty
    def _start_time(self):
        """
        Start time of the run
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
        if name == "DICE":
            from .dice import DICE  # pylint: disable=cyclic-import

            adapter = DICE

        elif name == "PH99":
            from .ph99 import PH99  # pylint: disable=cyclic-import

            adapter = PH99

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
