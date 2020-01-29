"""
Adapter for the climate component from the Dynamic Integrated Climate-Economy (DICE)
model by William Nordhaus (DICE 2013).

Original source: https://sites.google.com/site/williamdnordhaus/dice-rice

http://www.econ.yale.edu/~nordhaus/homepage/homepage/DICE2013R_100413_vanilla.gms
http://www.econ.yale.edu/~nordhaus/homepage/homepage/DICE2016R-091916ap.gms

This implementation follows the original DICE code closely, especially regarding
variable naming. Original comments are marked by "Original:".
"""

from . import Adapter


class DICE(Adapter):
    """
    Adapter for the climate component from the Dynamic Integrated Climate-Economy (DICE)
    model.

    TODO: re-write
    """

    def __init__(self):  # pylint:disable=super-init-not-called
        raise NotImplementedError
