"""
The OpenSCM Core API includes the basic functionality to run a particular simple
climate model with OpenSCM as well as setting/getting its :ref:`parameter <parameters>`
values. Mapping of :ref:`parameter names <parameter-hierarchy>` and :ref:`units <units>`
is done internally.

Parts of this API definition seems unpythonic as it is designed to be easily
implementable in several programming languages.
"""

from typing import Optional, Tuple, Union, cast

from .parameter_views import (
    ScalarView,
    TimeseriesView,
    WritableScalarView,
    WritableTimeseriesView,
    BooleanView,
    WritableBooleanView,
    StringView,
    WritableStringView,
)
from .parameters import ParameterInfo, ParameterType, _Parameter
from .regions import _Region
from .timeseries_converter import ExtrapolationType, InterpolationType
from .utils import ensure_input_is_tuple

# pylint: disable=too-many-arguments


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
        name_root : str
            Name of root region, default is "World".
        """
        self._root = _Region(name_root)

    def _get_or_create_region(self, name: Union[str, Tuple[str, ...]]) -> _Region:
        """
        Get a region. Create and add it if not found.

        Parameters
        ----------
        name
            Hierarchical name of the region or ``()`` for "World".
        """
        name_tuple = ensure_input_is_tuple(name)
        if len(name_tuple) > 1:
            p = self._get_or_create_region(name_tuple[:-1])
            return p.get_or_create_subregion(name_tuple[-1])

        if len(name_tuple) == 1:
            name_str = name_tuple[0]
            root_name = self._root._name  # pylint: disable=protected-access
            if name_str != root_name:
                error_msg = (
                    "Cannot access region {}, root region for this parameter set "
                    "is {}"
                ).format(name_str, root_name)
                raise ValueError(error_msg)

            return self._root

        # len(name_tuple) == 0
        raise ValueError("No region name given")

    def _get_region(self, name: Tuple[str]) -> Optional[_Region]:
        """
        Get a region or ``None`` if not found.

        Parameters
        ----------
        name
            Hierarchical name of the region.
        """
        name_tuple = ensure_input_is_tuple(name)
        if name_tuple[0] != self._root.name:
            return None

        return self._root.get_subregion(name_tuple[1:])

    def _get_or_create_parameter(self, name: Tuple[str], region: _Region) -> _Parameter:
        """
        Get a parameter. Create and add it if not found.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the parameter
        region
            Region

        Raises
        ------
        ValueError
            Name not given
        """
        name_tuple = ensure_input_is_tuple(name)
        if len(name_tuple) > 1:
            p = self._get_or_create_parameter(cast(Tuple[str], name_tuple[:-1]), region)
            return p.get_or_create_child_parameter(name_tuple[-1])

        if len(name_tuple) == 1:
            return region.get_or_create_parameter(name_tuple[0])

        # len(name_tuple) == 0
        raise ValueError("No parameter name given")

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
            Hierarchical name of the region or ``()`` for "World".
        unit
            Unit for the values in the view

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
            Hierarchical name of the region or ``()`` for "World".
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
        parameter = self._get_or_create_parameter(
            name, self._get_or_create_region(region)
        )
        parameter.attempt_write(ParameterType.SCALAR, unit)
        return WritableScalarView(parameter, unit)

    def get_timeseries_view(
        self,
        name: Tuple[str],
        region: Tuple[str],
        unit: str,
        time_points: Tuple[int],
        timeseries_type: ParameterType,
        interpolation_type: InterpolationType,
        extrapolation_type: ExtrapolationType,
    ) -> TimeseriesView:
        """
        Get a read-only view to a timeseries parameter.

        The parameter is created as a timeseries if not viewed so far. The length of the
        returned ParameterView's timeseries is adjusted such that its last value is
        equal to or less than ``stop_time``.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the parameter
        region
            Hierarchical name of the region or ``()`` for "World".
        unit
            Unit for the values in the view
        time_points
            Time points of the timeseries (seconds since ``1970-01-01 00:00:00``)
        timeseries_type
            Time series type
        interpolation_type
            Interpolation type
        extrapolation_type
            Extrapolation type

        Raises
        ------
        ParameterTypeError
            Parameter is not timeseries
        ValueError
            Name not given or invalid region
        """
        parameter = self._get_or_create_parameter(
            name, self._get_or_create_region(region)
        )
        parameter.attempt_read(timeseries_type, unit, time_points)
        return TimeseriesView(
            parameter,
            unit,
            time_points,
            timeseries_type,
            interpolation_type,
            extrapolation_type,
        )  # TimeseriesView

    def get_writable_timeseries_view(
        self,
        name: Tuple[str],
        region: Tuple[str],
        unit: str,
        time_points: Tuple[int],
        timeseries_type: ParameterType,
        interpolation_type: InterpolationType,
        extrapolation_type: ExtrapolationType,
    ) -> WritableTimeseriesView:
        """
        Get a writable view to a timeseries parameter.

        The parameter is created as a timeseries if not viewed so far.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the parameter
        region
            Hierarchical name of the region or ``()`` for "World".
        unit
            Unit for the values in the view
        time_points
            Time points of the timeseries (seconds since ``1970-01-01 00:00:00``)
        timeseries_type
            Time series type
        interpolation_type
            Interpolation type
        extrapolation_type
            Extrapolation type

        Raises
        ------
        ParameterReadonlyError
            Parameter is read-only (e.g. because its parent has been written to)
        ParameterTypeError
            Parameter is not timeseries
        ValueError
            Name not given or invalid region
        """
        parameter = self._get_or_create_parameter(
            name, self._get_or_create_region(region)
        )
        parameter.attempt_write(timeseries_type, unit, time_points)
        return WritableTimeseriesView(
            parameter,
            unit,
            time_points,
            timeseries_type,
            interpolation_type,
            extrapolation_type,
        )  # WritableTimeseriesView

    def get_boolean_view(self, name: Tuple[str], region: Tuple[str]) -> BooleanView:
        """
        Get a read-only view to a boolean parameter.

        The parameter is created as a boolean if not viewed so far.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the parameter
        region
            Hierarchical name of the region or ``()`` for "World".

        Raises
        ------
        ParameterTypeError
            Parameter is not boolean
        ValueError
            Name not given or invalid region
        """
        parameter = self._get_or_create_parameter(
            name, self._get_or_create_region(region)
        )
        parameter.attempt_read(ParameterType.BOOLEAN)
        return BooleanView(parameter)

    def get_writable_boolean_view(
        self, name: Tuple[str], region: Tuple[str]
    ) -> WritableBooleanView:
        """
        Get a writable view to a boolean parameter.

        The parameter is created as a boolean if not viewed so far.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the parameter
        region
            Hierarchical name of the region or ``()`` for "World".

        Raises
        ------
        ParameterReadonlyError
            Parameter is read-only (e.g. because its parent has been written to)
        ParameterTypeError
            Parameter is not boolean
        ValueError
            Name not given or invalid region
        """
        parameter = self._get_or_create_parameter(
            name, self._get_or_create_region(region)
        )
        parameter.attempt_write(ParameterType.BOOLEAN)
        return WritableBooleanView(parameter)

    def get_string_view(self, name: Tuple[str], region: Tuple[str]) -> StringView:
        """
        Get a read-only view to a string parameter.

        The parameter is created as a string if not viewed so far.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the parameter
        region
            Hierarchical name of the region or ``()`` for "World".

        Raises
        ------
        ParameterTypeError
            Parameter is not string
        ValueError
            Name not given or invalid region
        """
        parameter = self._get_or_create_parameter(
            name, self._get_or_create_region(region)
        )
        parameter.attempt_read(ParameterType.STRING)
        return StringView(parameter)

    def get_writable_string_view(
        self, name: Tuple[str], region: Tuple[str]
    ) -> WritableStringView:
        """
        Get a writable view to a string parameter.

        The parameter is created as a string if not viewed so far.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the parameter
        region
            Hierarchical name of the region or ``()`` for "World".

        Raises
        ------
        ParameterReadonlyError
            Parameter is read-only (e.g. because its parent has been written to)
        ParameterTypeError
            Parameter is not boolean
        ValueError
            Name not given or invalid region
        """
        parameter = self._get_or_create_parameter(
            name, self._get_or_create_region(region)
        )
        parameter.attempt_write(ParameterType.STRING)
        return WritableStringView(parameter)

    def get_parameter_info(
        self, name: Tuple[str], region_name: Tuple[str]
    ) -> Optional[ParameterInfo]:
        """
        Get a parameter or ``None`` if not found.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>` of the parameter
        region_name
            Hierarchical name of the region".

        Raises
        ------
        ValueError
            Name not given

        Returns
        -------
        _Parameter
            Parameter or ``None`` if the parameter has not been created yet.
        """
        region = self._get_region(region_name)
        if region is not None:
            parameter = region.get_parameter(name)
            if parameter is not None:
                return parameter.info
        return None


class Core:
    """
    OpenSCM core class.

    Represents a model run with a particular simple climate model.
    """

    _model: str
    """Name of the SCM to run"""

    _parameters: ParameterSet
    """Set of :ref:`parameters <parameters>` for the run"""

    _start_time: int
    """
    Beginning of the time range to run over (seconds since
    ``1970-01-01 00:00:00``)
    """

    _stop_time: int
    """
    End of the time range to run over (including; seconds since
    ``1970-01-01 00:00:00``)
    """

    def __init__(self, model: str, start_time: int, stop_time: int):
        """
        Initialize.

        Attributes
        ----------
        model
            Name of the SCM to run
        start_time
            Beginning of the time range to run over (seconds since
            ``1970-01-01 00:00:00``)
        stop_time
            End of the time range to run over (including; seconds since
            ``1970-01-01 00:00:00``)

        Raises
        ------
        KeyError
            No adapter for SCM named ``model`` found
        ValueError
            ``stop_time`` before ``start_time``
        """
        self._model = model
        self._start_time = start_time
        self._stop_time = stop_time
        self._parameters = ParameterSet()

    @property
    def stop_time(self) -> int:
        """
        End of the time range to run over (including; seconds since
        ``1970-01-01 00:00:00``)
        """
        return self._stop_time

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
        return self._parameters

    def run(self) -> None:
        """
        Run the model over the full time range.
        """
        raise NotImplementedError

    @property
    def start_time(self) -> int:
        """
        Beginning of the time range to run over (seconds since
        ``1970-01-01 00:00:00``)
        """
        return self._start_time

    def step(self) -> int:
        """
        Do a single time step.

        Returns
        -------
        int
            Current time (seconds since ``1970-01-01 00:00:00``)
        """
        raise NotImplementedError
