"""
Parameter handling.
"""

from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..errors import (
    ParameterAggregationError,
    ParameterReadError,
    ParameterReadonlyError,
    ParameterTypeError,
    ParameterWrittenError,
)
from .utils import HierarchicalName, hierarchical_name_as_sequence

if TYPE_CHECKING:  # pragma: no cover
    from . import regions  # pylint: disable=cyclic-import

# pylint: disable=protected-access
# pylint: disable=too-many-instance-attributes


class ParameterType(Enum):
    """
    Parameter type.
    """

    SCALAR = 1
    AVERAGE_TIMESERIES = 2
    POINT_TIMESERIES = 3
    GENERIC = 4

    @classmethod
    def from_timeseries_type(
        cls, timeseries_type: Union["ParameterType", str]
    ) -> "ParameterType":
        if isinstance(timeseries_type, str):
            if timeseries_type.lower() == "average":
                return cls.AVERAGE_TIMESERIES
            if timeseries_type.lower() == "point":
                return cls.POINT_TIMESERIES
            raise ValueError("Unknown timeseries type '{}'".format(timeseries_type))
        if timeseries_type not in [cls.AVERAGE_TIMESERIES, cls.POINT_TIMESERIES]:
            raise ValueError("Timeseries type expected")
        return timeseries_type


class ParameterInfo:
    """
    Information for a :ref:`parameter <parameters>`.
    """

    _name: str
    """Name"""

    _region: "regions._Region"
    """Region this parameter belongs to"""

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
    def region(self) -> Tuple[str, ...]:
        """
        Hierarchichal name of the region this parameter belongs to
        """
        return self._region.full_name

    @property
    def unit(self) -> Optional[str]:
        """
        Parameter unit
        """
        return self._unit


class _Parameter:
    """
    Represents a :ref:`parameter <parameters>` in the :ref:`parameter hierarchy
    <parameter-hierarchy>`.
    """

    children: Dict[str, "_Parameter"]
    """Child parameters"""

    data: Union[None, bool, float, str, Sequence[float]]
    """Data"""

    has_been_read_from: bool
    """If True, parameter has already been read from"""

    has_been_written_to: bool
    """If True, parameter data has already been changed"""

    info: ParameterInfo
    """Information about the parameter"""

    parent: Optional["_Parameter"]
    """Parent parameter"""

    time_points: Optional[np.ndarray]
    """Timeseries time points; only for timeseries parameters"""

    version: int

    def __init__(self, name: str, region: "regions._Region"):
        """
        Initialize.

        Parameters
        ----------
        name
            Name
        """
        self.children = {}
        self.has_been_read_from = False
        self.has_been_written_to = False
        self.info = ParameterInfo(name, region)
        self.parent = None
        self.version = 0

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

        Returns
        -------
        _Parameter
            Parameter found or newly created

        Raises
        ------
        ParameterReadError
            If the child parameter would need to be added, but this parameter has
            already been read from. In this case a child parameter cannot be added.
        ParameterWrittenError
            If the child parameter would need to be added, but this parameter has
            already been written to. In this case a child parameter cannot be added.
        """
        res = self.children.get(name, None)
        if res is None:
            if self.has_been_written_to:
                raise ParameterWrittenError
            if self.has_been_read_from:
                raise ParameterReadError
            res = _Parameter(name, self.info._region)
            res.parent = self
            self.children[name] = res
        return res

    def get_subparameter(self, name: HierarchicalName) -> Optional["_Parameter"]:
        """
        Get a sub parameter of this parameter or ``None`` if not found.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the subparameter below this
            parameter or ``()`` for this parameter

        Returns
        -------
        Optional[_Parameter]
            Parameter of ``None`` if not found
        """
        name = hierarchical_name_as_sequence(name)
        if name:
            res = self.children.get(name[0], None)
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
            If parameter has already been read from or written to in a different type
        ParameterAggregationError
            If parameter has child parameters which cannot be aggregated (for boolean
            and string parameters)
        """
        if self.info._type is not None and self.info._type != parameter_type:
            raise ParameterTypeError
        if self.info._type is None:
            self.info._unit = unit
            self.info._type = parameter_type
            if parameter_type == ParameterType.SCALAR:
                self.data = float("NaN")
            elif parameter_type == ParameterType.GENERIC:
                if self.children:
                    raise ParameterAggregationError
                self.data = None
            else:  # parameter is a timeseries
                self.data = np.full(
                    (
                        (len(time_points) - 1)  # type: ignore
                        if parameter_type == ParameterType.AVERAGE_TIMESERIES
                        else len(time_points)  # type: ignore
                    ),
                    float("NaN"),
                )
                self.time_points = np.array(time_points, copy=True)
        self.has_been_read_from = True

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
            If parameter is read-only because it has child parameters
        """
        if self.children:
            raise ParameterReadonlyError
        self.attempt_read(parameter_type, unit, time_points)
        self.has_been_written_to = True

    @property
    def full_name(self) -> Tuple[str, ...]:
        """
        Full :ref:`hierarchical name <parameter-hierarchy>`
        """
        p: Optional["_Parameter"] = self
        r = []
        while p is not None:
            r.append(p.info._name)
            p = p.parent
        return tuple(reversed(r))
