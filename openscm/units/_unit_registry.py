"""
Unit registry for OpenSCM, based on `Pint <https://github.com/hgrecco/pint>`_

.. code:: python

    >>> from openscm.units._unit_registry import _unit_registry
    >>> _unit_registry("CO2")
    <Quantity(1, 'CO2')>

    >>> emissions_aus = 0.34 * _unit_registry("Gt C / yr")
    >>> emissions_aus
    <Quantity(0.34, 'C * gigametric_ton / a')>

    >>> emissions_aus.to("Mt C / week")
    <Quantity(6.516224050620789, 'C * megametric_ton / week')>
"""
from os.path import abspath, dirname, join
from typing import Dict, Sequence, Union

import pandas as pd
from pint import Context, UnitRegistry

from pint.errors import (  # noqa: F401 # pylint: disable=unused-import; noqa: F401 # pylint: disable=unused-import
    DimensionalityError,
    UndefinedUnitError,
)

here = dirname(abspath(__file__))

# Start a unit registry using the default variables:
_unit_registry = UnitRegistry()
"""`obj`:`pint.registry.UnitRegistry`: OpenSCM's unit registry

The unit registry contains all of our recognised units. A couple of examples
"""

# Define gases. If the value is:
# - str: this entry defines a base gas unit
# - list: this entry defines a derived unit
#    - the first entry defines how to convert from base units
#    - other entries define other names i.e. aliases
_gases = {
    # CO2, CH4, N2O
    "C": "carbon",
    "CO2": ["12/44 * C", "carbon_dioxide"],
    "CH4": "methane",
    "N": "nitrogen",
    "N2O": ["14/44 * N", "nitrous_oxide"],
    "N2ON": ["14/28 * N", "nitrous_oxide_farming_style"],
    # aerosol precursors
    "NOx": "NOx",
    "nox": ["NOx"],
    "NH3": ["14/17 * N", "ammonia"],
    "S": ["sulfur"],
    "SO2": ["32/64 * S", "sulfur_dioxide"],
    "SOx": ["SO2"],
    "BC": "black_carbon",
    "OC": "OC",
    "CO": "carbon_monoxide",
    "VOC": "VOC",
    "NMVOC": ["VOC", "non_methane_volatile_organic_compounds"],
    # CFCs
    "CFC11": "CFC11",
    "CFC12": "CFC12",
    "CFC13": "CFC13",
    "CFC113": "CFC113",
    "CFC114": "CFC114",
    "CFC115": "CFC115",
    # HCFCs
    "HCFC21": "HCFC21",
    "HCFC22": "HCFC22",
    "HCFC123": "HCFC123",
    "HCFC124": "HCFC124",
    "HCFC141b": "HCFC141b",
    "HCFC142b": "HCFC142b",
    "HCFC225ca": "HCFC225ca",
    "HCFC225cb": "HCFC225cb",
    # HFCs
    "HFC23": "HFC23",
    "HFC32": "HFC32",
    "HFC41": "HFC41",
    "HFC125": "HFC125",
    "HFC134": "HFC134",
    "HFC134a": "HFC134a",
    "HFC143": "HFC143",
    "HFC143a": "HFC143a",
    "HFC152": "HFC152",
    "HFC152a": "HFC152a",
    "HFC161": "HFC161",
    "HFC227ea": "HFC227ea",
    "HFC236cb": "HFC236cb",
    "HFC236ea": "HFC236ea",
    "HFC236fa": "HFC236fa",
    "HFC245ca": "HFC245ca",
    "HFC245fa": "HFC245fa",
    "HFC365mfc": "HFC365mfc",
    "HFC4310mee": "HFC4310mee",
    "HFC4310": ["HFC4310mee"],
    # Halogenated gases
    "Halon1201": "Halon1201",
    "Halon1202": "Halon1202",
    "Halon1211": "Halon1211",
    "Halon1301": "Halon1301",
    "Halon2402": "Halon2402",
    # PFCs
    "CF4": "CF4",
    "C2F6": "C2F6",
    "cC3F6": "cC3F6",
    "C3F8": "C3F8",
    "cC4F8": "cC4F8",
    "C4F10": "C4F10",
    "C5F12": "C5F12",
    "C6F14": "C6F14",
    "C7F16": "C7F16",
    "C8F18": "C8F18",
    "C10F18": "C10F18",
    # Fluorinated ethers
    "HFE125": "HFE125",
    "HFE134": "HFE134",
    "HFE143a": "HFE143a",
    "HCFE235da2": "HCFE235da2",
    "HFE245cb2": "HFE245cb2",
    "HFE245fa2": "HFE245fa2",
    "HFE347mcc3": "HFE347mcc3",
    "HFE347pcf2": "HFE347pcf2",
    "HFE356pcc3": "HFE356pcc3",
    "HFE449sl": "HFE449sl",
    "HFE569sf2": "HFE569sf2",
    "HFE4310pccc124": "HFE4310pccc124",
    "HFE236ca12": "HFE236ca12",
    "HFE338pcc13": "HFE338pcc13",
    "HFE227ea": "HFE227ea",
    "HFE236ea2": "HFE236ea2",
    "HFE236fa": "HFE236fa",
    "HFE245fa1": "HFE245fa1",
    "HFE263fb2": "HFE263fb2",
    "HFE329mcc2": "HFE329mcc2",
    "HFE338mcf2": "HFE338mcf2",
    "HFE347mcf2": "HFE347mcf2",
    "HFE356mec3": "HFE356mec3",
    "HFE356pcf2": "HFE356pcf2",
    "HFE356pcf3": "HFE356pcf3",
    "HFE365mcf3": "HFE365mcf3",
    "HFE374pc2": "HFE374pc2",
    # Perfluoropolyethers
    "PFPMIE": "PFPMIE",
    # Misc
    "CCl4": "CCl4",
    "CHCl3": "CHCl3",
    "CH2Cl2": "CH2Cl2",
    "CH3CCl3": "CH3CCl3",
    "CH3Cl": "CH3Cl",
    "CH3Br": "CH3Br",
    "SF5CF3": "SF5CF3",
    "SF6": "SF6",
    "NF3": "NF3",
}


def _add_mass_emissions_joint_version(symbol: str) -> None:
    """
    Add a unit which is the combination of mass and emissions.

    This allows users to access e.g. ``_unit_registry("tC")`` rather than requiring a
    space between the mass and the emissions i.e. ``_unit_registry("t C")``

    Parameters
    ----------
    symbol
        The unit to add a joint version for.
    """
    _unit_registry.define("g{symbol} = g * {symbol}".format(symbol=symbol))
    _unit_registry.define("t{symbol} = t * {symbol}".format(symbol=symbol))


def _add_gases_to_unit_registry(gases: Dict[str, Union[str, Sequence[str]]]) -> None:
    for symbol, value in gases.items():
        if isinstance(value, str):
            # symbol is base unit
            _unit_registry.define("{} = [{}]".format(symbol, value))
            if value != symbol:
                _unit_registry.define("{} = {}".format(value, symbol))
        else:
            # symbol has conversion and aliases
            _unit_registry.define("{} = {}".format(symbol, value[0]))
            for alias in value[1:]:
                _unit_registry.define("{} = {}".format(alias, symbol))

        _add_mass_emissions_joint_version(symbol)

        # Add alias for upper case symbol:
        if symbol.upper() != symbol:
            _unit_registry.define("{} = {}".format(symbol.upper(), symbol))
            _add_mass_emissions_joint_version(symbol.upper())


_add_gases_to_unit_registry(_gases)

# Other definitions:

_unit_registry.define("a = 1 * year = annum = yr")
_unit_registry.define("h = hour")
_unit_registry.define("d = day")
_unit_registry.define("degreeC = degC")
_unit_registry.define("degreeF = degF")
_unit_registry.define("kt = 1000 * t")  # since kt is used for "knot" in the defaults

_unit_registry.define("ppt = [concentrations]")
_unit_registry.define("ppb = 1000 * ppt")
_unit_registry.define("ppm = 1000 * ppb")

# Contexts:

_ch4_context = Context("CH4_conversions")
_ch4_context.add_transformation(
    "[carbon]",
    "[methane]",
    lambda _unit_registry, x: 16 / 12 * _unit_registry.CH4 * x / _unit_registry.C,
)
_ch4_context.add_transformation(
    "[methane]",
    "[carbon]",
    lambda _unit_registry, x: x * _unit_registry.C / _unit_registry.CH4 / (16 / 12),
)
_unit_registry.add_context(_ch4_context)

_n2o_context = Context("NOx_conversions")
_n2o_context.add_transformation(
    "[nitrogen]",
    "[NOx]",
    lambda _unit_registry, x: (14 + 2 * 16)
    / 14
    * _unit_registry.NOx
    * x
    / _unit_registry.nitrogen,
)
_n2o_context.add_transformation(
    "[NOx]",
    "[nitrogen]",
    lambda _unit_registry, x: x
    * _unit_registry.nitrogen
    / _unit_registry.NOx
    / ((14 + 2 * 16) / 14),
)
_unit_registry.add_context(_n2o_context)

# TODO: ask how to cache this properly so we don't read from disk unless we have to
_metric_conversions = pd.read_csv(
    join(here, "metric_conversions.csv"),
    skiprows=1,  # skip source row
    header=0,
    index_col=0,
).iloc[
    1:, :
]  # drop out 'OpenSCM base unit' row


def _get_transform_func(ureg_unit, conversion_factor, forward=True):
    if forward:

        def result_forward(ur, strt):
            return strt * ur.carbon / ureg_unit * conversion_factor

        return result_forward

    def result_backward(ur, strt):
        return strt * ureg_unit / ur.carbon / conversion_factor

    return result_backward


for col in _metric_conversions:
    tc = Context(col)
    for label, val in _metric_conversions[col].iteritems():
        conv_val = (
            val
            * (_unit_registry("CO2").to_base_units()).magnitude
            / (_unit_registry(label).to_base_units()).magnitude
        )
        base_unit = [
            s
            for s, _ in _unit_registry._get_dimensionality(  # pylint: disable=protected-access
                _unit_registry(label)  # pylint: disable=protected-access
                .to_base_units()
                ._units
            ).items()
        ][0]

        unit_reg_unit = getattr(
            _unit_registry, base_unit.replace("[", "").replace("]", "")
        )
        tc.add_transformation(
            base_unit, "[carbon]", _get_transform_func(unit_reg_unit, conv_val)
        )
        tc.add_transformation(
            "[carbon]",
            base_unit,
            _get_transform_func(unit_reg_unit, conv_val, forward=False),
        )
        tc.add_transformation(
            "[mass] * {} / [time]".format(base_unit),
            "[mass] * [carbon] / [time]",
            _get_transform_func(unit_reg_unit, conv_val),
        )
        tc.add_transformation(
            "[mass] * [carbon] / [time]",
            "[mass] * {} / [time]".format(base_unit),
            _get_transform_func(unit_reg_unit, conv_val, forward=False),
        )
        tc.add_transformation(
            "[mass] * {}".format(base_unit),
            "[mass] * [carbon]",
            _get_transform_func(unit_reg_unit, conv_val),
        )
        tc.add_transformation(
            "[mass] * [carbon]",
            "[mass] * {}".format(base_unit),
            _get_transform_func(unit_reg_unit, conv_val, forward=False),
        )
        tc.add_transformation(
            "{} / [time]".format(base_unit),
            "[carbon] / [time]",
            _get_transform_func(unit_reg_unit, conv_val),
        )
        tc.add_transformation(
            "[carbon] / [time]",
            "{} / [time]".format(base_unit),
            _get_transform_func(unit_reg_unit, conv_val, forward=False),
        )

    _unit_registry.add_context(tc)
