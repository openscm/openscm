"""
Unit handling.

Our unit handling library is `Pint <https://github.com/hgrecco/pint>`_. This allows us
to define units simply as well as providing us with the ability to define contexts.


**A note on emissions units**

Emissions are a flux composed of three parts: mass, the stuff being emitted and the
time period e.g. "t CO2 / yr". As mass and time are part of SI units, all we need to
define here are emissions units i.e. the stuff. Here we include as many of the
canonical emissions units, and their conversions, as possible.

For emissions units, there are a few cases to be considered:

- fairly obvious ones e.g. carbon dioxide emissions can be provided in 'C' or 'CO2' and converting between the two is possible
- less obvious ones e.g. nitrous oxide emissions can be provided in 'N', 'N2O' or 'N2ON', we provide conversions
- namespace collisions e.g. methane emissions can be provided in 'CH4' or 'C' and the conversions are valid. However, that also means that methane emissions can be converted directly to 'CO2' which isn't something which should actually be valid.
- case-sensitivity. In order to provide a simplified interface, using all uppercase versions of any unit is also valid e.g. ``unit_registry("HFC4310mee") == unit_registry("HFC4310MEE")``
- hyphens and underscores in units. In order to be Pint compatible and to simplify things, we strip all hyphens and underscores from units.

As a convenience, we allow users to combine the first two to make a 'joint unit' e.g.
"tCO2" but it should be recognised that this joint unit is a derived unit and not a
base unit.

By definining these three separate components, it is much easier to track what
conversions are valid and which are not. For example, as the emissions units are all
defined as emissions units, and not as atomic masses, we are able to prevent invalid
conversions. If emissions units were simply atomic masses, it would be possible to
convert between e.g. C and N2O which would be a problem. Conventions such as allowing
carbon dioxide emissions to be reported in C or CO2, despite the fact that they are
fundamentally different chemical species, is a convention which is particular to
emissions (as far as we can tell).

Finally, contexts are particularly useful for emissions as they facilitate much easier
metric conversions. With a context, a conversion which wouldn't normally be allowed (
e.g. tCO2 --> tN2O) is allowed and will use whatever metric conversion is appropriate
for that context (e.g. AR4GWP100).
"""

from pint import Context, UnitRegistry
from pint.errors import DimensionalityError, UndefinedUnitError


# Start a unit registry using the default variables:
unit_registry = UnitRegistry()
"""`obj`:`pint.registry.UnitRegistry`: OpenSCM's unit registry

The unit registry contains all of our recognised units. A couple of examples

.. code:: python

    >>> from openscm.definitions import unit_registry
    >>> unit_registry("CO2")
    <Quantity(1, 'CO2')>

    >>> emissions_aus = 0.34 * unit_registry("Gt C / yr")
    >>> emissions_aus
    <Quantity(0.34, 'C * gigametric_ton / a')>

    >>> emissions_aus.to("Mt C / week")
    <Quantity(6.516224050620789, 'C * megametric_ton / week')>
"""

# Define gases. If the value is:
# - str: this entry defines a base gas unit
# - list: this entry defines a derived unit
#    - the first entry defines how to convert from base units
#    - other entries define other names i.e. aliases
_gases = {
    "C": "carbon",
    "CO2": ["12/44 * C", "carbon_dioxide"],
    "N": "nitrogen",
    "N2O": ["14/44 * N", "nitrous_oxide"],
    "N2ON": ["14/28 * N", "nitrous_oxide_farming_style"],
    # "NOx": ["14/46 * N", "nox"],
    "NOx": "NOx",
    "nox": ["NOx"],
    "NH3": ["14/17 * N", "ammonia"],
    "S": ["sulfur"],
    "SO2": ["32/64 * S", "sulfur_dioxide"],
    "SOx": ["SO2"],
    "VOC": "VOC",
    "NMVOC": ["VOC", "non_methane_volatile_organic_compounds"],
    "BC": "black_carbon",
    "C2F6": "C2F6",
    "CCl4": "CCl4",
    "CF4": "CF4",
    "CFC11": "CFC11",
    "CFC113": "CFC113",
    "CFC114": "CFC114",
    "CFC115": "CFC115",
    "CFC12": "CFC12",
    "CH3Br": "CH3Br",
    "CH3CCl3": "CH3CCl3",
    "CH3Cl": "CH3Cl",
    "CH4": "CH4",
    "CO": "carbon_monoxide",
    "HCFC141b": "HCFC141b",
    "HCFC142b": "HCFC142b",
    "HCFC22": "HCFC22",
    "HFC125": "HFC125",
    "HFC134a": "HFC134a",
    "HFC143a": "HFC143a",
    "HFC152a": "HFC152a",
    "HFC227ea": "HFC227ea",
    "HFC23": "HFC23",
    "HFC236fa": "HFC236fa",
    "HFC245fa": "HFC245fa",
    "HFC32": "HFC32",
    "HFC365mfc": "HFC365mfc",
    "HFC4310mee": "HFC4310mee",
    "HFC4310": ["HFC4310mee"],
    "Halon1202": "Halon1202",
    "Halon1211": "Halon1211",
    "Halon1301": "Halon1301",
    "Halon2402": "Halon2402",
    "OC": "OC",
    "SF6": "SF6",
}


def _add_mass_emissions_joint_version(symbol: str):
    """Add a unit which is the combination of mass and emissions.

    This allows users to access e.g. ``unit_registry("tC")`` rather than requiring a space between the mass and the emissions i.e. ``unit_registry("t C")``

    Parameters
    ----------
    symbol
        The unit to add a joint version for.
    """
    unit_registry.define("g{} = g * {}".format(symbol, symbol))
    unit_registry.define("t{} = t * {}".format(symbol, symbol))


for symbol, value in _gases.items():
    if isinstance(value, str):
        # symbol is base unit
        unit_registry.define("{} = [{}]".format(symbol, value))
    else:
        # symbol has conversion and aliases
        unit_registry.define("{} = {}".format(symbol, value[0]))
        for alias in value[1:]:
            unit_registry.define("{} = {}".format(alias, symbol))

    _add_mass_emissions_joint_version(symbol)

    # Add alias for upper case symbol:
    if symbol.upper() != symbol:
        unit_registry.define("{} = {}".format(symbol.upper(), symbol))
        _add_mass_emissions_joint_version(symbol.upper())


# Other definitions:

unit_registry.define("a = 1 * year = annum = yr")
unit_registry.define("h = hour")
unit_registry.define("degreeC = degC")
unit_registry.define("degreeF = degF")
unit_registry.define("kt = 1000 * t")  # since kt is used for "knot" in the defaults

unit_registry.define("ppt = [concentrations]")
unit_registry.define("ppb = 1000*ppt")
unit_registry.define("ppm = 1000*ppb")

# Contexts:

c = Context("AR4GWP12")
c.add_transformation(
    "[carbon]",
    "[nitrogen]",
    lambda unit_registry, x: 20 * unit_registry.N * x / unit_registry.C,
)
c.add_transformation(
    "[nitrogen]",
    "[carbon]",
    lambda unit_registry, x: x * unit_registry.C / unit_registry.N / 20,
)
unit_registry.add_context(c)


class UnitConverter:
    """
    Converts numbers between two units.

    """

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
        DimensionalityError
            Units cannot be converted into each other.
        UndefinedUnitError
            Unit undefined.
        """
        source_unit = unit_registry.Unit(source)
        target_unit = unit_registry.Unit(target)
        s1 = unit_registry.Quantity(1, source_unit)
        s2 = unit_registry.Quantity(-1, source_unit)
        t1 = s1.to(target_unit)
        t2 = s2.to(target_unit)
        self._scaling = float(t2.m - t1.m) / float(s2.m - s1.m)
        self._offset = t1.m - self._scaling * s1.m

    def convert_from(self, v):
        """
        Convert value **from** source unit to target unit.

        Parameters
        ----------
        value
            Value
        """
        return self._offset + v * self._scaling

    def convert_to(self, v):
        """
        Convert value from target unit **to** source unit.

        Parameters
        ----------
        value
            Value
        """
        return (v - self._offset) / self._scaling
