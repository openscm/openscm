"""
Parameter handling.
"""

# pylint: disable=unused-import,protected-access

from enum import Enum
from typing import cast, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .errors import (
    ParameterAggregationError,
    ParameterReadError,
    ParameterReadonlyError,
    ParameterTypeError,
    ParameterWrittenError,
)
from .utils import ensure_input_is_tuple
from . import regions  # needed for type annotations


class ParameterType(Enum):
    """
    Parameter type.
    """

    SCALAR = 1
    AVERAGE_TIMESERIES = 2
    POINT_TIMESERIES = 3
    BOOLEAN = 4
    STRING = 5


class ParameterInfo:
    """
    Information for a :ref:`parameter <parameters>`.
    """

    _name: str
    """Name"""

    _region: "regions._Region"
    """Region this parameter belongs to"""

    _time_points: Optional[np.ndarray]
    """Timeseries time points; only for timeseries parameters"""

    _type: Optional[ParameterType]
    """Parameter type"""

    _unit: Optional[str]
    """Unit"""

    def __init__(self, name: str, region: "regions._Region"):
        """
        Initialize.

        Parameters
        ----------
        name
            Name
        region
            Region
        """
        self._name = name
        self._region = region
        self._time_points = None
        self._type = None
        self._unit = None

    @property
    def name(self) -> str:
        """
        Name
        """
        return self._name

    @property
    def parameter_type(self) -> Optional[ParameterType]:
        """
        Parameter type
        """
        return self._type

    @property
    def region(self) -> Tuple[str]:
        """
        Hierarchichal name of the region this parameter belongs to
        """
        return self._region.full_name

    @property
    def unit(self) -> Optional[str]:
        """
        Unit
        """
        return self._unit


class _Parameter:
    """
    Represents a :ref:`parameter <parameters>` in the :ref:`parameter hierarchy
    <parameter-hierarchy>`.
    """

    _children: Dict[str, "_Parameter"]
    """Child parameters"""

    _data: Union[None, bool, float, str, Sequence[float]]
    """Data"""

    _has_been_read_from: bool
    """If True, parameter has already been read from"""

    _has_been_written_to: bool
    """If True, parameter data has already been changed"""

    _info: ParameterInfo
    """Information about the parameter"""

    _parent: Optional["_Parameter"]
    """Parent parameter"""

    def __init__(self, name: str, region: "regions._Region"):
        """
        Initialize.

        Parameters
        ----------
        name
            Name
        """
        self._children = {}
        self._has_been_read_from = False
        self._has_been_written_to = False
        self._info = ParameterInfo(name, region)
        self._parent = None

    def get_or_create_child_parameter(self, name: str) -> "_Parameter":
        """
        Get a (direct) child parameter of this parameter. Create and add it if not
        found.

        Parameters
        ----------
        name
            Name
        unit
            Unit for the parameter if it is going to be created
        parameter_type
            Parameter type if it is going to be created

        Raises
        ------
        ParameterReadError
            If the child parameter would need to be added, but this parameter has
            already been read from. In this case a child parameter cannot be added.
        ParameterWrittenError
            If the child parameter would need to be added, but this parameter has
            already been written to. In this case a child parameter cannot be added.
        """
        res = self._children.get(name, None)
        if res is None:
            if self._has_been_written_to:
                raise ParameterWrittenError
            if self._has_been_read_from:
                raise ParameterReadError
            res = _Parameter(name, self._info._region)
            res._parent = self
            self._children[name] = res
        return res

    def get_subparameter(self, name: Tuple[str, ...]) -> Optional["_Parameter"]:
        """
        Get a sub parameter of this parameter or ``None`` if not found.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the subparameter below this
            parameter or ``()`` for this parameter
        """
        name = ensure_input_is_tuple(name)
        if name:
            res = self._children.get(name[0], None)
            if res is not None:
                res = res.get_subparameter(name[1:])
            return res

        return self

    def attempt_read(
        self,
        parameter_type: ParameterType,
        unit: Optional[str] = None,
        time_points: Optional[np.ndarray] = None,
    ) -> None:
        """
        Tell parameter that it will be read from. If the parameter has child parameters
        it will be read in in an aggregated way, i.e., aggregating over child
        parameters.

        Parameters
        ----------
        parameter_type
            Parameter type to be read
        unit
            Unit to be read; only for scalar and timeseries parameters
        time_points
            Timeseries time points; only for timeseries parameters

        Raises
        ------
        ParameterTypeError
            If parameter has already been read from or written to in a different type.
        ParameterAggregationError
            If parameter has child parameters which cannot be aggregated (for boolean
            and string parameters).
        """
        # TODO aggregate
        if self._info._type is not None and self._info._type != parameter_type:
            raise ParameterTypeError
        if self._info._type is None:
            self._info._unit = unit
            self._info._type = parameter_type
            if parameter_type == ParameterType.SCALAR:
                self._data = float("NaN")
            elif parameter_type == ParameterType.BOOLEAN:
                if self._children:
                    raise ParameterAggregationError
                self._data = False
            elif parameter_type == ParameterType.STRING:
                if self._children:
                    raise ParameterAggregationError
                self._data = ""
            else:  # parameter is a timeseries
                self._data = np.full(
                    (
                        (len(time_points) - 1)  # type: ignore
                        if parameter_type == ParameterType.AVERAGE_TIMESERIES
                        else len(time_points)  # type: ignore
                    ),
                    float("NaN"),
                )
                self._info._time_points = np.array(time_points, copy=True)
        self._has_been_read_from = True

    def attempt_write(
        self,
        parameter_type: ParameterType,
        unit: Optional[str] = None,
        time_points: Optional[np.ndarray] = None,
    ) -> None:
        """
        Tell parameter that its data will be written to.

        Parameters
        ----------
        parameter_type
            Parameter type to be written
        unit
            Unit to be written; only for scalar and timeseries parameters
        time_points
            Timeseries time points; only for timeseries parameters

        Raises
        ------
        ParameterReadonlyError
            If parameter is read-only because it has child parameters.
        """
        if self._children:
            raise ParameterReadonlyError
        self.attempt_read(parameter_type, unit, time_points)
        self._has_been_written_to = True

    @property
    def full_name(self) -> Tuple[str]:
        """
        Full :ref:`hierarchical name <parameter-hierarchy>`
        """
        p: Optional["_Parameter"] = self
        r = []
        while p is not None:
            r.append(p._info._name)
            p = p._parent
        return cast(Tuple[str], tuple(reversed(r)))

    @property
    def info(self) -> ParameterInfo:
        """
        Parameter information
        """
        return self._info

    @property
    def parent(self) -> Optional["_Parameter"]:
        """
        Parent parameter
        """
        return self._parent
