"""
OpenSCM - unified access to simple climate models.
"""

from .core import Core
from .openscm import OpenSCM

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
