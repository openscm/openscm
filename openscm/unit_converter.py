"""
Unit converter which provides simplified conversion of units
"""
import warnings
from typing import Union

import numpy as np
from pint.errors import (  # noqa: F401 # pylint: disable=unused-import
    DimensionalityError,
    UndefinedUnitError,
)

from .units import unit_registry


class UnitConverter:
    """
    Converts numbers between two units.

    """

    _source: str
    """Source unit"""

    _target: str
    """Target unit"""

    _offset: float
    """Offset for units (e.g. for temperature units)"""

    _scaling: float
    """Scaling factor between units"""

    def __init__(self, source: str, target: str):
        """
        Initialize.

        Parameters
        ----------
        source
            Unit to convert **from**
        target
            Unit to convert **to**

        Raises
        ------
        pint.errors.DimensionalityError
            Units cannot be converted into each other.
        pint.errors.UndefinedUnitError
            Unit undefined.
        """
        self._source = source
        self._target = target

        source_unit = unit_registry.Unit(source)
        target_unit = unit_registry.Unit(target)

        s1 = unit_registry.Quantity(1, source_unit)
        s2 = unit_registry.Quantity(-1, source_unit)

        t1 = s1.to(target_unit)
        t2 = s2.to(target_unit)
        if np.isnan(t1) or np.isnan(t2):
            warn_msg = (
                "No conversion from {} to {} available, nan will be returned "
                "upon conversion".format(source, target)
            )
            warnings.warn(warn_msg)

        self._scaling = float(t2.m - t1.m) / float(s2.m - s1.m)
        self._offset = t1.m - self._scaling * s1.m

    def convert_from(self, v: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert value **from** source unit to target unit.

        Parameters
        ----------
        value
            Value in source unit

        Returns
        -------
        Union[float, np.ndarray]
            Value in target unit
        """
        return self._offset + v * self._scaling

    def convert_to(self, v: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert value from target unit **to** source unit.

        Parameters
        ----------
        value
            Value in target unit

        Returns
        -------
        Union[float, np.ndarray]
            Value in source unit
        """
        return (v - self._offset) / self._scaling
