"""
Functionality for guessing the parameter types from variable names and unit
"""
import re
from typing import Optional

from openscm.parameters import ParameterType
from openscm.units import UndefinedUnitError, _unit_registry

"""
List of regex patterns for matching variable names to :obj:`ParameterType`
"""
parameter_matches = [
    (re.compile(r"^Emissions"), ParameterType.AVERAGE_TIMESERIES),
    (re.compile(r".*_(GWP)?EMIS"), ParameterType.AVERAGE_TIMESERIES),
    (re.compile(r".*Flux"), ParameterType.AVERAGE_TIMESERIES),
    (re.compile(r".*_(PEFF|EFF|S)?RF"), ParameterType.AVERAGE_TIMESERIES),
    (re.compile(r"^Radiative Forcing"), ParameterType.AVERAGE_TIMESERIES),
    (re.compile("HEATUPTAKE_EBALANCE_TOTAL"), ParameterType.AVERAGE_TIMESERIES),
    (re.compile(r"^Atmospheric Concentrations"), ParameterType.POINT_TIMESERIES),
    (re.compile(r".*_CONC"), ParameterType.POINT_TIMESERIES),
    (re.compile(r"(Surface Temperature|.*TEMP)"), ParameterType.POINT_TIMESERIES),
]


def guess_parameter_type(  # pylint: disable=missing-return-doc
    variable_name: str, unit: Optional[str]
) -> ParameterType:
    """
    Attempt to guess the parameter of timeseries from a variable name and unit.

    This ``ParameterType`` is used to ensure the right assumptions are being made when interpolating. If the units are available
    and the units include a `time` dimension, then ``ParameterType.AVERAGE_TIMESERIES`` is always returned.

    Parameters
    ----------
    variable_name
        The full name of the variable of interest, including level separators.
    unit
        Unit corresponding to the variable.

    Returns
    -------
    :obj:`ParameterType`
    """
    if unit:
        # try and determine if the unit contains a time dimension
        try:
            pint_unit = _unit_registry(unit).units
            if "[time]" in str(pint_unit.dimensionality):
                return ParameterType.AVERAGE_TIMESERIES

            return ParameterType.POINT_TIMESERIES
        except UndefinedUnitError:
            # default to trying to parse from variable name
            pass

    for r, t in parameter_matches:
        if r.match(variable_name):
            return t

    # Default to Point time series if unknown
    return ParameterType.POINT_TIMESERIES
