"""
The OpenSCM low-level API includes the basic functionality to run a particular
simple climate model with OpenSCM as well as setting/getting its :ref:`parameter
<parameters>` values. Mapping of :ref:`parameter names <parameter-hierarchy>` and
:ref:`units <units>` is done internally.

Parts of this API definition seems unpythonic as it is designed to be easily
implementable in several programming languages.
"""

from enum import Enum
from typing import Any, Dict, Sequence, Tuple


class ParameterType(Enum):
    """
    Parameter type.
    """

    SCALAR = 1
    TIMESERIES = 2


class _Parameter:
    """
    Represents a :ref:`parameter <parameters>` in the :ref:`parameter hierarchy
    <parameter-hierarchy>`.
    """

    _children: Dict[str, "_Parameter"]
    """Child parameters"""

    _data: Any
    """Data"""

    _has_been_aggregated: bool
    """
    If True, parameter has already been read in an aggregated way, i.e., aggregating
    over child parameters
    """

    _has_been_written_to: bool
    """If True, parameter data has already been changed"""

    _name: str
    """Name"""

    _parent: "_Parameter"
    """Parent parameter"""

    _type: ParameterType
    """Parameter type"""

    _unit: str
    """Unit"""

    def __init__(self, name: str):
        """
        Initialize.

        Parameters
        ----------
        name
            Name
        """
        self._children = {}
        self._has_been_aggregated = False
        self._has_been_written_to = False
        self._name = name
        self._parent = None
        self._type = None
        self._unit = None

    def get_or_create_child_parameter(
        self, name: str, unit: str, parameter_type: ParameterType
    ) -> "_Parameter":
        """
        Get a (direct) child parameter of this parameter. Create and add it if not
        found.

        Parameters
        ----------
        name
            Name
        unit
            Unit
        parameter_type
            Parameter type

        Raises
        ------
        ParameterAggregatedError
            If the child paramater would need to be added, but this parameter has
            already been read in an aggregated way. In this case a child parameter
            cannot be added.
        ParameterWrittenError
            If the child paramater would need to be added, but this parameter has
            already been written to. In this case a child parameter cannot be added.
        """
        res = self._children.get(name, None)
        if res is None:
            if self._has_been_written_to:
                raise ParameterWrittenError
            if self._has_been_aggregated:
                raise ParameterAggregatedError
            res = _Parameter(name)
            res._parent = self
            res._type = parameter_type
            res._unit = unit
            self._children[name] = res
        return res

    def attempt_aggregate(self, parameter_type: ParameterType) -> None:
        """
        Tell parameter that it will be read from in an aggregated way, i.e., aggregating
        over child parameters.

        Parameters
        ----------
        parameter_type
            Parameter type to be read

        Raises
        ------
        ParameterTypeError
            If parameter has already been read from or written to in a different type.
        """
        if self._type is not None and self._type != parameter_type:
            raise ParameterTypeError
        self._type = parameter_type
        self._has_been_aggregated = True

    def attempt_write(self, parameter_type: ParameterType) -> None:
        """
        Tell parameter that its data will be written to.

        Parameters
        ----------
        parameter_type
            Parameter type to be written

        Raises
        ------
        ParameterReadonlyError
            If parameter is read-only because it has child parameters.
        """
        if self._children:
            raise ParameterReadonlyError
        self.attempt_aggregate(parameter_type)
        self._has_been_written_to = True

    @property
    def full_name(self) -> Tuple[str]:
        """
        Full :ref:`hierarchical name <parameter-hierarchy>`
        """
        p = self
        r = []
        while p is not None:
            r.append(p.name)
            p = p._parent
        return tuple(reversed(r))

    @property
    def name(self) -> str:
        """
        Name
        """
        return self._name

    @property
    def parameter_type(self) -> ParameterType:
        """
        Parameter type
        """
        return self._type

    @property
    def parent(self) -> "_Parameter":
        """
        Parent parameter
        """
        return self._parent

    @property
    def unit(self) -> str:
        """
        Unit
        """
        return self._unit


class _Region:
    """
    Represents a region in the region hierarchy.
    """

    _children: Dict[str, "_Region"]
    """Subregions"""

    _has_been_aggregated: bool
    """
    If True, a parameter of this region has already been read in an aggregated way,
    i.e., aggregating over subregions
     """

    _name: str
    """Name"""

    _parameters: Dict[str, _Parameter]
    """Parameters"""

    _parent: "_Region"
    """Parent region (or `None` if root region)"""

    def __init__(self, name: str):
        """
        Initialize

        Parameters
        ----------
        name
            Name
        """
        self._name = name
        self._children = {}
        self._has_been_aggregated = False
        self._parameters = {}
        self._parent = None

    def get_or_create_subregion(self, name: str) -> "_Region":
        """
        Get a (direct) subregion of this region. Create and add it if not found.

        Parameters
        ----------
        name
            Name

        Raises
        ------
        RegionAggregatedError
            If the subregion would need to be added and a parameter of this region has
            already been read in an aggregated way. In this case a subregion cannot be
            created.
        """
        res = self._children.get(name, None)
        if res is None:
            if self._has_been_aggregated:
                raise RegionAggregatedError
            res = _Region(name)
            res._parent = self
            self._children[name] = res
        return res

    def get_or_create_parameter(self, name: str) -> _Parameter:
        """
        Get a root parameter for this region. Create and add it if not found.

        Parameters
        ----------
        name
            Name
        """
        res = self._parameters.get(name, None)
        if res is None:
            res = _Parameter(name)
            self._parameters[name] = res
        return res

    def attempt_aggregate(self) -> None:
        """
        Tell region that one of its parameters will be read from in an aggregated way,
        i.e., aggregating over subregions.
        """
        self._has_been_aggregated = True

    @property
    def name(self) -> str:
        """
        Name
        """
        return self._name

    @property
    def parent(self) -> "_Region":
        """
        Parent region (or `None` if root region)
        """
        return self._parent


class ParameterError(Exception):
    """
    Exception relating to a parameter. Used as super class.
    """


class ParameterLengthError(ParameterError):
    """
    Exception raised when sequences in timeseries do not match run
    size.
    """


class ParameterReadonlyError(ParameterError):
    """
    Exception raised when a requested parameter is read-only.

    This can happen, for instance, if a parameter's parent parameter
    in the parameter hierarchy has already been requested as writable.
    """


class ParameterTypeError(ParameterError):
    """
    Exception raised when a parameter is of a different type than
    requested (scalar or timeseries).
    """


class ParameterAggregatedError(ParameterError):
    """
    Exception raised when a parameter has been read from (raised, e.g., when attempting
    to create a child parameter).
    """


class ParameterWrittenError(ParameterError):
    """
    Exception raised when a parameter has been written to (raised, e.g., when attempting
    to create a child parameter).
    """


class RegionAggregatedError(Exception):
    """
    Exception raised when a region has been read from in a region-aggregated way.
    """


class ParameterView:
    """
    Generic view to a :ref:`parameter <parameters>` (scalar or timeseries).
    """

    _name: Tuple[str]
    """:ref:`Hierarchical name <parameter-hierarchy>`"""

    _region: Tuple[str]
    """Hierarchical region name"""

    _unit: str
    """Unit"""

    def __init__(self, name: Tuple[str], region: Tuple[str], unit: str):
        """
        Initialize.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>`
        region
            Hierarchical region name
        unit
            Unit
        """
        self._name = name
        self._region = region
        self._unit = unit

    @property
    def name(self) -> Tuple[str]:
        """
        :ref:`Hierarchical name <parameter-hierarchy>`
        """
        return self._name

    @property
    def region(self) -> Tuple[str]:
        """
        Hierarchical region name
        """
        return self._region

    @property
    def unit(self) -> str:
        """
        Unit
        """
        return self._unit

    @property
    def is_empty(self) -> bool:
        """
        Check if parameter is empty, i.e. has not yet been written to.
        """
        raise NotImplementedError


class ParameterInfo(ParameterView):
    """
    Provides information about a :ref:`parameter <parameters>`.
    """

    _type: ParameterType
    """Parameter type"""

    def __init__(self, parameter: _Parameter):
        """
        Initialize.

        Parameters
        ----------
        parameter
            Parameter
        """
        self._type = parameter.parameter_type

    @property
    def parameter_type(self) -> ParameterType:
        """Parameter type"""
        return self._type


class ScalarView(ParameterView):
    """
    Read-only view of a scalar parameter.
    """

    def get(self) -> float:
        """
        Get current value of scalar parameter.
        """
        raise NotImplementedError


class WritableScalarView(ScalarView):
    """
    View of a scalar parameter whose value can be changed.
    """

    def set(self, value: float) -> None:
        """
        Set current value of scalar parameter.

        Parameters
        ----------
        value
            Value
        """
        raise NotImplementedError


class TimeseriesView(ParameterView):
    """
    Read-only :class:`ParameterView` of a timeseries.
    """

    def get_series(self) -> Sequence[float]:
        """
        Get values of the full timeseries.
        """
        raise NotImplementedError

    def get(self, index: int) -> float:
        """
        Get value at a particular time.

        Parameters
        ----------
        index
            Time step index

        Raises
        ------
        IndexError
            ``time`` is out of run time range.
        """
        raise NotImplementedError

    def length(self) -> int:
        """
        Get length of time series.
        """
        raise NotImplementedError


class WritableTimeseriesView(TimeseriesView):
    """
    View of a timeseries whose values can be changed.
    """

    def set_series(self, values: Sequence[float]) -> None:
        """
        Set value for whole time series.

        Parameters
        ----------
        values
            Values to set. The length of this sequence (list/1-D
            array/...) of ``float`` values must equal size.

        Raises
        ------
        ParameterLengthError
            Length of ``values`` does not equal size.
        """
        raise NotImplementedError

    def set(self, value: float, index: int) -> None:
        """
        Set value for a particular time in the time series.

        Parameters
        ----------
        value
            Value
        index
            Time step index

        Raises
        ------
        IndexError
            ``index`` is out of range.
        """
        raise NotImplementedError


class ParameterSet:
    """
    Collates a set of :ref:`parameters <parameters>`.
    """

    _world: _Region
    """Root region (contains all parameters)"""

    def __init__(self):
        """
        Initialize.
        """
        self._world = _Region(None)

    def _get_or_create_region(self, name: Tuple[str]) -> _Region:
        """
        Get a region. Create and add it if not found.

        Parameters
        ----------
        name
            Hierarchy name of the region
        """
        if len(name) > 0:
            p = self._get_or_create_region(name[:-1])
            return p.get_or_create_subregion(name[-1])
        else:
            return self._world

    def _get_or_create_parameter(
        self,
        name: Tuple[str],
        region: _Region,
        unit: str,
        parameter_type: ParameterType,
    ) -> _Parameter:
        """
        Get a parameter. Create and add it if not found.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the parameter
        region
            Region
        unit
            Unit for the values in the view
        parameter_type
            Parameter type

        Raises
        ------
        ValueError
            Name not given
        """
        if len(name) > 1:
            p = self._get_or_create_parameter(name[:-1], region, unit, parameter_type)
            return p.get_or_create_child_parameter(name[-1], unit, parameter_type)
        elif len(name) == 1:
            return region.get_or_create_parameter(name[0])
        else:  # len(name) == 0
            raise ValueError

    def get_scalar_view(
        self, name: Tuple[str], region: Tuple[str], unit: str
    ) -> ScalarView:
        """
        Get a read-only view to a scalar parameter.

        The parameter is created as a scalar if not viewed so far.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the parameter
        region
            Hierarchical region name
        unit
            Unit for the values in the view

        Raises
        ------
        ParameterTypeError
            Parameter is not scalar
        ValueError
            Name not given or invalid region
        """
        raise NotImplementedError

    def get_writable_scalar_view(
        self, name: Tuple[str], region: Tuple[str], unit: str
    ) -> WritableScalarView:
        """
        Get a writable view to a scalar parameter.

        The parameter is created as a scalar if not viewed so far.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the parameter
        region
            Hierarchical region name
        unit
            Unit for the values in the view

        Raises
        ------
        ParameterReadonlyError
            Parameter is read-only (e.g. because its parent has been written to)
        ParameterTypeError
            Parameter is not scalar
        ValueError
            Name not given or invalid region
        """
        raise NotImplementedError

    def get_timeseries_view(
        self,
        name: Tuple[str],
        region: Tuple[str],
        unit: str,
        start_time: int,
        period_length: int,
    ) -> TimeseriesView:
        """
        Get a read-only view to a timeseries parameter.

        The parameter is created as a timeseries if not viewed so far.
        The length of the returned ParameterView's timeseries is adjusted such
        that its last value corresponds to a time not exceeding the ``end_time``
        of the underlying run (i.e. ``Core`` object).

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the parameter
        region
            Hierarchical region name
        unit
            Unit for the values in the view
        start_time
            Time for first point in timeseries (seconds since 1970-01-01 00:00:00)
        period_length
            Length of single time step in seconds

        Raises
        ------
        ParameterTypeError
            Parameter is not timeseries
        ValueError
            Name not given or invalid region
        """
        raise NotImplementedError

    def get_writable_timeseries_view(
        self,
        name: Tuple[str],
        region: Tuple[str],
        unit: str,
        start_time: int,
        period_length: int,
    ) -> WritableTimeseriesView:
        """
        Get a writable view to a timeseries parameter.

        The parameter is created as a timeseries if not viewed so far.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the parameter
        region
            Hierarchical region name
        unit
            Unit for the values in the view
        start_time
            Time for first point in timeseries (seconds since 1970-01-01 00:00:00)
        period_length
            Length of single time step in seconds

        Raises
        ------
        ParameterReadonlyError
            Parameter is read-only (e.g. because its parent has been written to)
        ParameterTypeError
            Parameter is not timeseries
        ValueError
            Name not given or invalid region
        """
        raise NotImplementedError

    def get_parameter_info(self, name: Tuple[str]) -> ParameterInfo:
        """
        Get information about a parameter.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the parameter

        Raises
        ------
        ValueError
            Name not given

        Returns
        -------
        ParameterInfo
            Information about the parameter or ``None`` if the parameter has not been
            created yet.
        """
        raise NotImplementedError

    def has_parameter(self, name: Tuple[str]) -> bool:
        """
        Query if parameter set has a specific parameter.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the parameter

        Raises
        ------
        ValueError
            Name not given
        """
        raise NotImplementedError


class Core:
    """
    OpenSCM core class.

    Represents a model run with a particular simple climate model.
    """

    _end_time: int
    """
    End of the time range to run over (including; seconds since 1970-01-01 00:00:00)
    """

    _model: str
    """Name of the SCM to run"""

    _parameters: ParameterSet
    """Set of :ref:`parameters <parameters>` for the run"""

    _start_time: int
    """Beginning of the time range to run over (seconds since 1970-01-01 00:00:00)"""

    def __init__(self, model: str, start_time: int, end_time: int):
        """
        Initialize.

        Attributes
        ----------
        model
            Name of the SCM to run
        start_time
            Beginning of the time range to run over (seconds since 1970-01-01 00:00:00)
        end_time
            End of the time range to run over (including; seconds since 1970-01-01
            00:00:00)

        Raises
        ------
        KeyError
            No adapter for SCM named ``model`` found
        ValueError
            ``end_time`` before ``start_time``
        """
        self._model = model
        self._start_time = start_time
        self._end_time = end_time
        self._parameters = ParameterSet()

    @property
    def end_time(self) -> int:
        """
        End of the time range to run over (including; seconds since 1970-01-01 00:00:00)
        """
        return self._end_time

    @property
    def model(self) -> str:
        """
        Name of the SCM to run
        """
        return self._model

    @property
    def parameters(self) -> ParameterSet:
        """
        Set of parameters for the run
        """
        return self._parameterset

    def run(self) -> None:
        """
        Run the model over the full time range.
        """
        raise NotImplementedError

    @property
    def start_time(self) -> int:
        """
        Beginning of the time range to run over (seconds since 1970-01-01 00:00:00)
        """
        return self._start_time

    def step(self) -> int:
        """
        Do a single time step.

        Returns
        -------
        int
            Current time (seconds since 1970-01-01 00:00:00)
        """
        raise NotImplementedError
