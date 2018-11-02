"""
Unit handling.

The units are defined using `Pint <https://github.com/hgrecco/pint>`.
"""

from pint import Context, UnitRegistry
from pint.errors import DimensionalityError, UndefinedUnitError

# Start a unit repository using the default variables:
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
# make private
_gases = {
    "C": "carbon",
    "CO2": ["12/44 * C", "carbon_dioxide"],
    "N": "nitrogen",
    "N2O": ["14/44 * N", "nitrous_oxide"],
    "NOx": ["14/46 * N", "nox"],
    "N2ON": ["14/28 * N", "nitrous_oxide_farming_style"],
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


def add_short_weight_version(symbol: str):
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

    add_short_weight_version(symbol)

    # Add alias for upper case symbol:
    if symbol.upper() != symbol:
        unit_registry.define("{} = {}".format(symbol.upper(), symbol))
        add_short_weight_version(symbol.upper())


# Other definitions:

unit_registry.define("a = 1 * year = annum = yr")
unit_registry.define("h = hour")
unit_registry.define("degreeC = degC")
unit_registry.define("degreeF = degF")
unit_registry.define("kt = 1000 * t")  # since kt is used for "knot" in the defaults


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
