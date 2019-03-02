"""
Pythonic convenience interface.
"""

from .core import Core

# pylint: disable=abstract-method # TODO remove once Core is complete


class OpenSCM(Core):
    """
    OpenSCM class.

    Represents model runs with a particular simple climate model.
    """
