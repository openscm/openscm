"""
Bundling parameters in a parameter set.
"""
from typing import Optional, Union

import numpy as np

from .parameters import (
    HIERARCHY_SEPARATOR,
    HierarchicalName,
    ParameterInfo,
    ParameterType,
    _Parameter,
)
from .regions import _Region
from .time import ExtrapolationType, InterpolationType
from .views import GenericView, ScalarView, TimeseriesView


class ParameterSet:
    """
    Collates a set of :ref:`parameters <parameters>`.
    """

    _root: _Region
    """Root region (contains all parameters)"""

    def __init__(self, name_root: str = "World"):
        """
        Initialize.

        Parameters
        ----------
        name_root
            Name of root region, default is "World"
        """
        self._root = _Region(name_root)

    def _get_or_create_region(self, name: HierarchicalName) -> _Region:
        """
        Get a region. Create and add it if not found.

        Parameters
        ----------
        name
            Hierarchical name of the region

        Returns
        -------
        _Region
            Found or created region

        Raises
        ------
        ValueError
            If parent region could not be found
        """
        if isinstance(name, str):
            name = name.split(HIERARCHY_SEPARATOR)

        if len(name) > 1:
            p = self._get_or_create_region(name[:-1])
            return p.get_or_create_subregion(name[-1])

        if len(name) == 1:
            name_str = name[0]
            root_name = self._root._name  # pylint: disable=protected-access
            if name_str != root_name:
                error_msg = (
                    "Cannot access region {}, root region for this parameter set "
                    "is {}"
                ).format(name_str, root_name)
                raise ValueError(error_msg)

            return self._root

        # len(name) == 0
        raise ValueError("No region name given")

    def _get_region(self, name: HierarchicalName) -> Optional[_Region]:
        """
        Get a region by its hierarchichal name.

        Parameters
        ----------
        name
            Hierarchical name of the region.

        Returns
        -------
        Optional[_Region]
            Region or ``None`` if not found
        """
        if isinstance(name, str):
            name = name.split(HIERARCHY_SEPARATOR)

        if name[0] != self._root.name:
            return None

        return self._root.get_subregion(name[1:])

    def _get_or_create_parameter(
        self, name: HierarchicalName, region: _Region
    ) -> _Parameter:
        """
        Get a parameter. Create and add it if not found.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the parameter
        region
            Region

        Returns
        -------
        _Parameter
            Parameter found or newly created.

        Raises
        ------
        ValueError
            Name not given
        """
        if isinstance(name, str):
            name = name.split(HIERARCHY_SEPARATOR)

        if len(name) > 1:
            p = self._get_or_create_parameter(name[:-1], region)
            return p.get_or_create_child_parameter(name[-1])

        if len(name) == 1:
            return region.get_or_create_parameter(name[0])

        # len(name) == 0
        raise ValueError("No parameter name given")

    def scalar(
        self, name: HierarchicalName, unit: str, region: HierarchicalName = ("World",)
    ) -> ScalarView:
        """
        Get a view to a scalar parameter.

        The parameter is created as a scalar if not viewed so far.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the parameter
        unit
            Unit for the values in the view
        region
            Hierarchical name of the region

        Returns
        -------
        ScalarView
            View to the parameter

        Raises
        ------
        ParameterTypeError
            Parameter is not scalar
        ValueError
            Name not given or invalid region
        """
        parameter = self._get_or_create_parameter(
            name, self._get_or_create_region(region)
        )

        parameter.attempt_read(ParameterType.SCALAR, unit)
        return ScalarView(parameter, unit)

    def timeseries(  # pylint: disable=too-many-arguments
        self,
        name: HierarchicalName,
        unit: str,
        time_points: np.ndarray,
        region: HierarchicalName = ("World",),
        timeseries_type: Union[ParameterType, str] = ParameterType.POINT_TIMESERIES,
        interpolation: Union[InterpolationType, str] = InterpolationType.LINEAR,
        extrapolation: Union[ExtrapolationType, str] = ExtrapolationType.NONE,
    ) -> TimeseriesView:
        """
        Get a view to a timeseries parameter.

        The parameter is created as a timeseries if not viewed so far. 

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the parameter
        unit
            Unit for the values in the view
        time_points
            Time points of the timeseries
        region
            Hierarchical name of the region
        timeseries_type
            Time series type
        interpolation_type
            Interpolation type
        extrapolation_type
            Extrapolation type

        Returns
        -------
        TimeseriesView
            Read-only view to the parameter

        Raises
        ------
        ParameterTypeError
            Parameter is not timeseries
        ValueError
            Name not given or invalid region
        """
        timeseries_type = ParameterType.from_timeseries_type(timeseries_type)
        interpolation = InterpolationType.from_interpolation_type(interpolation)
        extrapolation = ExtrapolationType.from_extrapolation_type(extrapolation)
        parameter = self._get_or_create_parameter(
            name, self._get_or_create_region(region)
        )

        parameter.attempt_read(timeseries_type, unit, time_points)
        return TimeseriesView(
            parameter, unit, time_points, timeseries_type, interpolation, extrapolation
        )  # TimeseriesView

    def generic(
        self, name: HierarchicalName, region: HierarchicalName = ("World",)
    ) -> GenericView:
        """
        Get a view to a generic parameter.

        The parameter is created as a generic if not viewed so far.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the parameter
        region
            Hierarchical name of the region

        Returns
        -------
        GenericView
            Read-only view to the parameter

        Raises
        ------
        ParameterAggregationError
            If parameter has child parameters and thus cannot be aggregated
        ParameterTypeError
            Parameter is not generic (scalar or timeseries)
        ValueError
            Name not given or invalid region
        """
        parameter = self._get_or_create_parameter(
            name, self._get_or_create_region(region)
        )

        parameter.attempt_read(ParameterType.GENERIC)
        return GenericView(parameter)

    def info(
        self, name: HierarchicalName, region: HierarchicalName = ("World",)
    ) -> Optional[ParameterInfo]:
        """
        Get parameter information or ``None`` if not found.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the parameter
        region
            Hierarchical name of the region

        Raises
        ------
        ValueError
            Name not given

        Returns
        -------
        Optional[ParameterInfo]
            Parameter information or ``None`` if the parameter has not been created yet
        """
        region_ = self._get_region(region)
        if region_ is not None:
            parameter = region_.get_parameter(name)
            if parameter is not None:
                return ParameterInfo(parameter)
        return None
