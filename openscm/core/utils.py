"""
Utility functions for openscm.
"""
import datetime
from typing import Any, Sequence, Tuple, Union

import numpy as np
from dateutil.relativedelta import relativedelta
from numpy.lib.mixins import NDArrayOperatorsMixin
from pandas.core.arrays.base import ExtensionOpsMixin

HierarchicalName = Union[str, Sequence[str]]

# TODO get rid of:
OPENSCM_REFERENCE_TIME = datetime.datetime(1970, 1, 1, 0, 0, 0)


class NumpyArrayHandler(ExtensionOpsMixin, NDArrayOperatorsMixin):  # type: ignore
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


def hierarchical_name_as_sequence(inp: HierarchicalName) -> Sequence[str]:
    """
    TODO Return parameter as a tuple.

    Parameters
    ----------
    inp
        String or tuple to return as a tuple

    Returns
    -------
    Sequence[str]
        A sequence with a single string `inp` if `inp` is a string, otherwise return
        `inp`
    """
    if isinstance(inp, str):
        return inp.split("|")

    return inp


def is_floatlike(f: Any) -> bool:
    """
    Check if input can be cast to a float.

    This includes strings such as "6.03" which can be cast to a float

    Parameters
    ----------
    f
        Input

    Returns
    -------
    bool
        True if f can be cast to a float
    """
    if isinstance(f, (int, float)):
        return True

    try:
        float(f)
        return True
    except (TypeError, ValueError):
        return False
