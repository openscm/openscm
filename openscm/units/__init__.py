"""
Unit handling.

Our unit handling library is `Pint <https://github.com/hgrecco/pint>`_. This allows us
to define units simply as well as providing us with the ability to define contexts.


**A note on emissions units**

Emissions are a flux composed of three parts: mass, the species being emitted and the
time period e.g. "t CO2 / yr". As mass and time are part of SI units, all we need to
define here are emissions units i.e. the stuff. Here we include as many of the canonical
emissions units, and their conversions, as possible.

For emissions units, there are a few cases to be considered:

- fairly obvious ones e.g. carbon dioxide emissions can be provided in 'C' or 'CO2' and
  converting between the two is possible
- less obvious ones e.g. nitrous oxide emissions can be provided in 'N', 'N2O' or
  'N2ON', we provide conversions
- case-sensitivity. In order to provide a simplified interface, using all uppercase
  versions of any unit is also valid e.g. ``unit_registry("HFC4310mee")`` is the same as
  ``unit_registry("HFC4310MEE")``
- hyphens and underscores in units. In order to be Pint compatible and to simplify
  things, we strip all hyphens and underscores from units.

As a convenience, we allow users to combine the mass and the type of emissions to make a
'joint unit' e.g. "tCO2" but it should be recognised that this joint unit is a derived
unit and not a base unit.

By defining these three separate components, it is much easier to track what conversions
are valid and which are not. For example, as the emissions units are all defined as
emissions units, and not as atomic masses, we are able to prevent invalid conversions.
If emissions units were simply atomic masses, it would be possible to convert between
e.g. C and N2O which would be a problem. Conventions such as allowing carbon dioxide
emissions to be reported in C or CO2, despite the fact that they are fundamentally
different chemical species, is a convention which is particular to emissions (as far as
we can tell).

Finally, contexts are particularly useful for emissions as they facilitate much easier
metric conversions. With a context, a conversion which wouldn't normally be allowed
(e.g. tCO2 --> tN2O) is allowed and will use whatever metric conversion is appropriate
for that context (e.g. AR4GWP100).

Finally, we discuss namespace collisions.

*CH4*

Methane emissions are defined as 'CH4'. In order to prevent inadvertent conversions of
'CH4' to e.g. 'CO2' via 'C', the conversion 'CH4' <--> 'C' is by default forbidden.
However, it can be performed within the context 'CH4_conversions' as shown below:

.. code:: python

    >>> from openscm.units import unit_registry
    >>> CH4 = unit_registry("CH4")
    >>> CH4.to("C")
    pint.errors.DimensionalityError: Cannot convert from 'CH4' ([methane]) to 'C' ([carbon])

    # with a context, the conversion becomes legal again
    >>> CH4.to("C", "CH4_conversions")
    <Quantity(0.75, 'C')>

    # as an unavoidable side effect, this also becomes possible
    >>> CH4.to("CO2", "CH4_conversions")
    <Quantity(2.75, 'CO2')>

*NOx*

Like for methane, NOx emissions also suffer from a namespace collision. In order to
prevent inadvertent conversions from 'NOx' to e.g. 'N2O', the conversion 'NOx' <-->
'N' is by default forbidden. It can be performed within the 'NOx_conversions' context:

.. code:: python

    >>> from openscm.units import unit_registry
    >>> NOx = unit_registry("NOx")
    >>> NOx.to("N")
    pint.errors.DimensionalityError: Cannot convert from 'NOx' ([NOx]) to 'N' ([nitrogen])

    # with a context, the conversion becomes legal again
    >>> NOx.to("N", "NOx_conversions")
    <Quantity(0.30434782608695654, 'N')>

    # as an unavoidable side effect, this also becomes possible
    >>> NOx.to("N2O", "NOx_conversions")
    <Quantity(0.9565217391304348, 'N2O')>
"""
from os.path import abspath, dirname, join
from typing import Dict, Sequence, Union

import pandas as pd
from pint import Context, UnitRegistry
from pint.errors import (  # noqa: F401 # pylint: disable=unused-import
    DimensionalityError,
    UndefinedUnitError,
)

import warnings
from typing import Union

import numpy as np
from pint.errors import (  # noqa: F401 # pylint: disable=unused-import
    DimensionalityError,
    UndefinedUnitError,
)
from pint.registry import UnitRegistry  # noqa: F401 # pylint: disable=unused-import

class _Registry:
    _ur: UnitRegistry = None
    """Unit registry which is used for conversions"""

    @property
    def _unit_registry(self) -> UnitRegistry:
        """
        Unit registry which is used for conversions.
        """
        if self._ur is None:
            from ._unit_registry import _unit_registry
            self._ur = _unit_registry

        return self._ur


_register = _Registry()


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

    _unit_registry: UnitRegistry = None
    """Unit registry which is used for conversions"""

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

        source_unit = self.unit_registry.Unit(source)
        target_unit = self.unit_registry.Unit(target)

        s1 = self.unit_registry.Quantity(1, source_unit)
        s2 = self.unit_registry.Quantity(-1, source_unit)

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

    @property
    def unit_registry(self) -> UnitRegistry:
        """
        Unit registry which is used for conversions.
        """
        return _register._unit_registry
