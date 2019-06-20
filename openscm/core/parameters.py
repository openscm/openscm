"""
Parameter handling.
"""

from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..errors import (
    ParameterAggregationError,
    ParameterEmptyError,
    ParameterReadError,
    ParameterReadonlyError,
    ParameterTypeError,
    ParameterWrittenError,
)

if TYPE_CHECKING:  # pragma: no cover
    from . import regions  # pylint: disable=cyclic-import

# pylint: disable=protected-access
# pylint: disable=too-many-instance-attributes

HierarchicalName = Union[str, Sequence[str]]

HIERARCHY_SEPARATOR: str = "|"
"""
String used to define different levels in our data hierarchies

By default we follow pyam and use "|". In such a case, emissions of CO2 for energy from
coal would be "Emissions|CO2|Energy|Coal".
"""


class ParameterType(Enum):
    """
    Parameter type.
    """

    SCALAR = 1
    AVERAGE_TIMESERIES = 2
    POINT_TIMESERIES = 3
    GENERIC = 4

    @classmethod
    def timeseries_type_to_string(cls, timeseries_type: "ParameterType") -> str:
        """
        Get time series type (i.e. :attr:`ParameterType.AVERAGE_TIMESERIES` or
        :attr:`ParameterType.POINT_TIMESERIES`) from :class:`ParameterType` as string
        value.

        Parameters
        ----------
        timeseries_type
            Value to convert to string value

        Returns
        -------
        str
            String value

        Raises
        ------
        ValueError
            If :obj:`timeseries_type` is invalid enum value
        """
        if timeseries_type == ParameterType.AVERAGE_TIMESERIES:
            return "average"
        if timeseries_type == ParameterType.POINT_TIMESERIES:
            return "point"
        raise ValueError("Timeseries type expected")

    @classmethod
    def from_timeseries_type(
        cls, timeseries_type: Union["ParameterType", str]
    ) -> "ParameterType":
        """
        Get time series type (i.e. :attr:`ParameterType.AVERAGE_TIMESERIES` or
        :attr:`ParameterType.POINT_TIMESERIES`) from :class:`ParameterType` or string
        value.

        Parameters
        ----------
        timeseries_type
            Value to convert to enum value (can be
            ``"average"``/:attr:`ParameterType.AVERAGE_TIMESERIES` or
            ``"point"``/:attr:`ParameterType.POINT_TIMESERIES`)

        Returns
        -------
        ParameterType
            Enum value

        Raises
        ------
        ValueError
            If :obj:`timeseries_type` is unknown string or invalid enum value
        """
        if isinstance(timeseries_type, str):
            if timeseries_type.lower() == "average":
                return cls.AVERAGE_TIMESERIES
            if timeseries_type.lower() == "point":
                return cls.POINT_TIMESERIES
            raise ValueError("Unknown timeseries type '{}'".format(timeseries_type))
        if timeseries_type not in [cls.AVERAGE_TIMESERIES, cls.POINT_TIMESERIES]:
            raise ValueError("Timeseries type expected")
        return timeseries_type


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

    name: str
    """Name"""

    parameter_type: Optional[ParameterType]
    """Parameter type"""

    parent: Optional["_Parameter"]
    """Parent parameter"""

    region: "regions._Region"
    """Region this parameter belongs to"""

    time_points: Optional[np.ndarray]
    """Timeseries time points; only for timeseries parameters"""

    unit: Optional[str]
    """Unit"""

    version: int
    """Internal version (incremented by each write operation)"""

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
        self.name = name
        self.parameter_type = None
        self.parent = None
        self.region = region
        self.unit = None
        self.version = 0

    def get_or_create_child_parameter(self, name: str) -> "_Parameter":
        """
        Get a (direct) child parameter of this parameter. Create and add it if not
        found.

        Parameters
        ----------
        name
            Name

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
            res = _Parameter(name, self.region)
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
        if name:
            if isinstance(name, str):
                name = name.split(HIERARCHY_SEPARATOR)
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
        if self.parameter_type is not None and self.parameter_type != parameter_type:
            raise ParameterTypeError
        if self.parameter_type is None:
            self.unit = unit
            self.parameter_type = parameter_type
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
            r.append(p.name)
            p = p.parent
        return tuple(reversed(r))

    def __str__(self) -> str:
        """
        Return string representation / description.
        """
        region_name = self.region.full_name
        return "{} in {}{}".format(
            "|".join(self.full_name),
            self.unit if self.unit is not None else "undefined",
            " for {}".format("|".join(region_name)) if len(region_name) > 1 else "",
        )


class ParameterInfo:
    """
    Information for a :ref:`parameter <parameters>`.
    """

    _parameter: _Parameter
    """Parameter"""

    def __init__(self, parameter: _Parameter):
        """
        Initialize.

        Parameters
        ----------
        parameter
            Parameter
        """
        self._parameter = parameter

    @property
    def name(self) -> Tuple[str, ...]:
        """
        Hierarchical name of the parameter
        """
        return self._parameter.full_name

    @property
    def parameter_type(self) -> Optional[ParameterType]:
        """
        Parameter type
        """
        return self._parameter.parameter_type

    @property
    def region(self) -> Tuple[str, ...]:
        """
        Hierarchichal name of the region this parameter belongs to
        """
        return self._parameter.region.full_name

    @property
    def unit(self) -> Optional[str]:
        """
        Parameter unit
        """
        return self._parameter.unit

    @property
    def empty(self) -> bool:
        """
        Check if parameter is empty, i.e. has not yet been written to.
        """
        return not self._parameter.has_been_written_to

    @property
    def version(self) -> int:
        """
        Version number of parameter (used internally)
        """
        return self._parameter.version

    def ensure(self) -> None:
        """
        Ensure that parameter is not empty.

        Raises
        ------
        ParameterEmptyError
            If parameter is empty
        """
        if self.empty:
            raise ParameterEmptyError(
                "Parameter {} is required but empty".format(str(self._parameter))
            )
