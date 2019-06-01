"""
Handling of region information.
"""

from typing import Dict, Optional, Tuple

from ..errors import RegionAggregatedError
from . import parameters

# pylint: disable=protected-access


class _Region:
    """
    Represents a region in the region hierarchy.
    """

    _children: Dict[str, "_Region"]
    """Subregions"""

    _has_been_aggregated: bool
    """
    If ``True``, a parameter of this region has already been read in an aggregated way,
    i.e., aggregating over subregions
     """

    _name: str
    """Name"""

    _parameters: Dict[str, "parameters._Parameter"]
    """Parameters"""

    _parent: Optional["_Region"]
    """Parent region (or ``None`` if root region)"""

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

        Returns
        -------
        _Region
            Region found or newly created

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

    def get_subregion(self, name: "parameters.HierarchicalName") -> Optional["_Region"]:
        """
        Get a subregion of this region or ``None`` if not found.

        Parameters
        ----------
        name
            Hierarchical name of the region below this region or ``()`` for this region

        Returns
        -------
        Optional[_Region]
            Subregion or ``None`` if not found
        """
        if name:
            if isinstance(name, str):
                name = name.split(parameters.HIERARCHY_SEPARATOR)
            res = self._children.get(name[0], None)
            if res is not None:
                return res.get_subregion(name[1:])
            return None

        return self

    def get_or_create_parameter(self, name: str) -> "parameters._Parameter":
        """
        Get a root parameter for this region. Create and add it if not found.

        Parameters
        ----------
        name
            Name

        Returns
        -------
        parameters._Parameter
            Root parameter found or newly created
        """
        res = self._parameters.get(name, None)
        if res is None:
            res = parameters._Parameter(name, self)
            self._parameters[name] = res
        return res

    def get_parameter(
        self, name: "parameters.HierarchicalName"
    ) -> Optional["parameters._Parameter"]:
        """
        Get a (root or sub-) parameter for this region or ``None`` if not found.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the parameter

        Returns
        -------
        Optional[parameters._Parameter]
            Parameter of ``None`` if not found

        Raises
        ------
        ValueError
            Name not given
        """
        if not name:
            raise ValueError("No parameter name given")
        if isinstance(name, str):
            name = name.split(parameters.HIERARCHY_SEPARATOR)
        root_parameter = self._parameters.get(name[0], None)
        if root_parameter is not None and len(name) > 1:
            return root_parameter.get_subparameter(name[1:])
        return root_parameter

    def attempt_aggregate(self) -> None:
        """
        Tell region that one of its parameters will be read from in an aggregated way,
        i.e., aggregating over subregions.
        """
        self._has_been_aggregated = True

    @property
    def full_name(self) -> Tuple[str, ...]:
        """
        Full hierarchical name
        """
        p = self
        r = []
        while p._parent is not None:
            r.append(p.name)
            p = p._parent
        r.append(p.name)
        return tuple(reversed(r))

    @property
    def name(self) -> str:
        """
        Name
        """
        return self._name

    @property
    def parent(self) -> Optional["_Region"]:
        """
        Parent region (or ``None`` if root region)
        """
        return self._parent
