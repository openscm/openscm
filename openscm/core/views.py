"""
Parameter views provide ways to read and write parameter data with a defined unit
and time information.
"""

import numbers
from typing import Any, Optional, Sequence, Tuple, cast

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from pandas.core.arrays.base import ExtensionOpsMixin

from ..errors import ParameterEmptyError, TimeseriesPointsValuesMismatchError
from .parameters import ParameterInfo, ParameterType, _Parameter
from .time import ExtrapolationType, InterpolationType, TimeseriesConverter
from .units import UnitConverter

# pylint: disable=protected-access


# not sure how to get code coverage up here...
class _Timeseries(ExtensionOpsMixin, NDArrayOperatorsMixin):  # type: ignore
    """
    Internal class which wraps numpy to make sure data is buffered and up-to-date
    """

    _HANDLED_TYPES = (np.ndarray, numbers.Number)
    __array_priority__ = 1000
    _ndarray: np.ndarray

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        args = [
            i.__read__()._ndarray if isinstance(i, type(self)) else i for i in inputs
        ]
        outputs = kwargs.pop("out", None)
        if outputs:
            kwargs["out"] = tuple(
                i._ndarray if isinstance(i, type(self)) else i for i in outputs
            )
            results = self._ndarray.__array_ufunc__(ufunc, method, *args, **kwargs)
            if results is NotImplemented:
                return NotImplemented
            if ufunc.nout == 1:
                results = (results,)
            results = tuple(
                (output.__write__() if isinstance(output, type(self)) else result)
                for result, output in zip(results, outputs)
            )
            return results[0] if len(results) == 1 else results
        return self._ndarray.__array_ufunc__(ufunc, method, *args, **kwargs)

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Shape of the 1-dimensional timeseries array
        """
        return cast(Tuple[int, ...], self._ndarray.shape)

    @classmethod
    def _create_arithmetic_method(cls, op):
        def arithmetic_method(self, other):
            if isinstance(other, cls):
                other = other._ndarray

            with np.errstate(all="ignore"):
                return op(self._ndarray, other)

        arithmetic_method.__name__ = "__{}__".format(op.__name__)
        arithmetic_method.__qualname__ = "{cl}.__{name}__".format(
            cl=cls.__name__, name=op.__name__
        )
        arithmetic_method.__module__ = cls.__module__
        return arithmetic_method

    _create_comparison_method = _create_arithmetic_method

    def __init__(self, input_array: np.ndarray, parameter_view: "TimeseriesView"):
        """
        Initialize.

        Parameters
        ----------
        input_array
            Array to handle
        parameter_view
            :class:`TimeseriesView` this timeseries belongs to
        """
        self._ndarray = np.asarray(input_array)
        self._parameter_view = parameter_view

    def __read__(self):
        self._parameter_view._read()
        return self

    def __write__(self):
        self._parameter_view._write()
        return self

    def __getitem__(self, item):
        self.__read__()
        result = self._ndarray[item]
        if not isinstance(item, int):
            result = type(self)(result, self._parameter_view)
        return result

    def __setitem__(self, key, val):
        self._ndarray[key] = val
        self.__write__()

    def __iter__(self):
        self.__read__()
        for i in self._ndarray:
            yield i

    def __repr__(self):
        return "timeseries({})".format(repr(self._ndarray))

    def __array__(self, dtype=None):
        return np.asarray(self._ndarray, dtype=dtype)

    def __len__(self):
        return len(self._ndarray)

    @property
    def dtype(self) -> np.dtype:
        """
        :class:`np.dtype` of the timeseries
        """
        return self._ndarray.dtype

    @property
    def ndim(self):
        """
        Dimension of the timeseries (``==1``)
        """
        return 1

    @property
    def nbytes(self):
        """
        Bytes block of the unterlying timeseries
        """
        self.__read__()
        return self._ndarray.nbytes


_Timeseries._add_arithmetic_ops()
_Timeseries._add_comparison_ops()


class ScalarView(ParameterInfo):
    """
    View of a scalar parameter.
    """

    _child_data_views: Sequence["ScalarView"]
    """List of views to the child parameters for aggregated reads"""

    _unit_converter: UnitConverter
    """Unit converter"""

    _writable: bool
    """Is this view writable? Is set to ``True`` once this view is written to."""

    def __init__(self, parameter: _Parameter, unit: str):
        """
        Initialize.

        Parameters
        ----------
        parameter
            Parameter to handle
        unit
            Unit for the values in the view
        """
        super().__init__(parameter)
        self._unit_converter = UnitConverter(cast(str, parameter.unit), unit)
        self._writable = False

        def get_data_views_for_children_or_parameter(
            parameter: _Parameter
        ) -> Sequence["ScalarView"]:
            if parameter.children:
                return sum(
                    (
                        get_data_views_for_children_or_parameter(p)
                        for p in parameter.children.values()
                    ),
                    [],
                )
            return [ScalarView(parameter, self._unit_converter.target)]

        if self._parameter.children:
            self._child_data_views = get_data_views_for_children_or_parameter(
                self._parameter
            )

    @property
    def value(self) -> float:
        """
        Value of scalar parameter

        If the parameter has child parameters, the returned value will be the sum of the
        values of all of the child parameters.

        Raises
        ------
        ParameterEmptyError
            Parameter is empty, i.e. has not yet been written to
        ParameterReadonlyError
            Parameter is read-only (e.g. because its parent has been written to)
        """
        if self._parameter.children:
            return sum(v.value for v in self._child_data_views)
        if self.empty:
            raise ParameterEmptyError

        return self._unit_converter.convert_from(cast(float, self._parameter.data))

    @value.setter
    def value(self, v: float) -> None:
        """
        Value of scalar parameter
        """
        if not self._writable:
            self._parameter.attempt_write(
                ParameterType.SCALAR, self._unit_converter.target
            )
            self._writable = True
        self._parameter.data = self._unit_converter.convert_to(v)
        self._parameter.version += 1

    def __str__(self) -> str:
        """
        Return string representation / description.
        """
        return "View of scalar {}".format(str(self._parameter))


class TimeseriesView(ParameterInfo):  # pylint: disable=too-many-instance-attributes
    """
    View of a timeseries.
    """

    _child_data_views: Sequence["TimeseriesView"]
    """List of views to the child parameters for aggregated reads"""

    _data: Optional[np.ndarray]
    """Chache for underlying data"""

    _locked: bool
    """
    Is this view locked (i.e., does it not update the underlying parameter on every
    write)?
    """

    _timeseries: Optional[_Timeseries]
    """Time series handler"""

    _timeseries_converter: TimeseriesConverter
    """Timeseries converter"""

    _unit_converter: UnitConverter
    """Unit converter"""

    _version: int
    """Version of cache"""

    _writable: bool
    """Is this view writable? Is set to ``True`` once this view is written to."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        parameter: _Parameter,
        unit: str,
        time_points: np.ndarray,
        timeseries_type: ParameterType,
        interpolation_type: InterpolationType,
        extrapolation_type: ExtrapolationType,
    ):
        """
        Initialize.

        Parameters
        ----------
        parameter
            Parameter
        unit
            Unit for the values in the view
        time_points
            Timeseries time points
        timeseries_type
            Time series type
        interpolation_type
            Interpolation type
        extrapolation_type
            Extrapolation type
        """
        super().__init__(parameter)
        self._unit_converter = UnitConverter(cast(str, parameter.unit), unit)
        self._timeseries_converter = TimeseriesConverter(
            parameter.time_points,
            time_points,
            timeseries_type,
            interpolation_type,
            extrapolation_type,
        )
        self._data = None
        self._locked = False
        self._timeseries = None
        self._writable = False

        def get_data_views_for_children_or_parameter(
            parameter: _Parameter
        ) -> Sequence["TimeseriesView"]:
            if parameter.children:
                return sum(
                    (
                        get_data_views_for_children_or_parameter(p)
                        for p in parameter.children.values()
                    ),
                    [],
                )
            return [
                TimeseriesView(
                    parameter,
                    self._unit_converter.target,
                    self._timeseries_converter._target,  # pylint: disable=protected-access
                    timeseries_type,
                    interpolation_type,
                    extrapolation_type,
                )
            ]

        if self._parameter.children:
            self._child_data_views = get_data_views_for_children_or_parameter(
                self._parameter
            )

    def _read(self):
        if self._data is None:
            self._data = self._get_values()
        elif self._version != self._parameter.version:
            import pdb
            pdb.set_trace()
            np.copyto(self._data, self._get_values())
        self._version = self._parameter.version

    def _check_write(self):
        if not self._writable:
            self._parameter.attempt_write(
                self._timeseries_converter._timeseries_type,
                self._unit_converter.target,
                self._timeseries_converter._target,
            )
            self._writable = True

    def _write(self):
        if not self._locked:
            self._parameter.data = self._timeseries_converter.convert_to(
                self._unit_converter.convert_to(self._data)
            )
            self._parameter.version += 1

    def _get_values(self) -> np.ndarray:
        if self._parameter.children:
            return cast(
                Sequence[float],
                np.sum((v.values for v in self._child_data_views), axis=0),
            )
        if self.empty:
            raise ParameterEmptyError

        return cast(
            Sequence[float],
            self._timeseries_converter.convert_from(
                self._unit_converter.convert_from(
                    cast(Sequence[float], self._parameter.data)
                )
            ),
        )

    @property
    def values(self) -> _Timeseries:
        """
        Values of the full timeseries

        If the parameter has child parameters, the returned value will be the sum of the
        values of all of the child parameters.

        Raises
        ------
        ParameterEmptyError
            Parameter is empty, i.e. has not yet been written to
        ParameterReadonlyError
            Parameter is read-only (e.g. because its parent has been written to)
        TimeseriesPointsValuesMismatchError
            Lengths of set value and the time points number mismatch
        """
        if self._timeseries is None:
            self._read()
            self._timeseries = _Timeseries(self._data, self)
        return self._timeseries

    @values.setter
    def values(self, v: np.ndarray) -> None:
        """
        Value of the full timeseries
        """
        self._check_write()
        if len(v) != self._timeseries_converter.target_length:
            raise TimeseriesPointsValuesMismatchError
        if self._data is None:
            self._data = np.asarray(v).copy()
            self._timeseries = _Timeseries(self._data, self)
        else:
            np.copyto(self._data, np.asarray(v))
        self._write()

    @property
    def length(self) -> int:
        """
        Length of timeseries
        """
        return self._timeseries_converter.target_length

    def __str__(self) -> str:
        """
        Return string representation / description.
        """
        return "View of timeseries {}".format(str(self._parameter))

    def lock(self) -> None:
        """
        Lock this view (i.e., do not update the underlying parameter on every write).
        """
        self._locked = True

    def unlock(self) -> None:
        """
        Unlock this view (i.e., update the underlying parameter on every write).

        Updates the underlying parameter.
        """
        self._locked = False
        self._write()


class GenericView(ParameterInfo):
    """
    View of a generic parameter.
    """

    _writable: bool
    """Is this view writable? Is set to ``True`` once this view is written to."""

    def __init__(self, parameter: _Parameter):
        """
        Initialize.

        Parameters
        ----------
        parameter
           Parameter to handle
        """
        super().__init__(parameter)
        self._writable = False

    @property
    def value(self) -> Any:
        """
        Value of generic parameter

        Raises
        ------
        ParameterEmptyError
            Parameter is empty, i.e. has not yet been written to
        ParameterReadonlyError
            Parameter is read-only (e.g. because its parent has been written to)
        """
        if self.empty:
            raise ParameterEmptyError

        return self._parameter.data

    @value.setter
    def value(self, v: Any) -> None:
        """
        Value of generic parameter
        """
        if not self._writable:
            self._parameter.attempt_write(ParameterType.GENERIC)
            self._writable = True
        self._parameter.data = v
        self._parameter.version += 1

    def __str__(self) -> str:
        """
        Return string representation / description.
        """
        return "Read-only view of {}".format(str(self._parameter))
