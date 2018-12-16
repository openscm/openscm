from enum import Enum
from typing import Any, Dict, Tuple
from .errors import (
    ParameterAggregatedError,
    ParameterReadonlyError,
    ParameterTypeError,
    ParameterWrittenError,
)


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

    def get_subparameter(self, name: Tuple[str]) -> "_Parameter":
        """
        Get a sub parameter of this parameter or ``None`` if not found.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the subparameter below this
            parameter or ``()`` for this parameter
        """
        if len(name) > 0:
            res = self._children.get(name[0], None)
            if res is not None:
                res = res.get_subparameter(name[1:])
            return res
        else:
            return self

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
