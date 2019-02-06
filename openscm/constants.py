"""Helpful constants.
"""

from .units import unit_registry

ONE_YEAR_IN_S_INTEGER = int(round((1 * unit_registry("yr")).to("s").magnitude, 0))
