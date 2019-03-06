"""
OpenSCM - unified access to simple climate models.
"""

from .core import Core
from .openscm import OpenSCM

from ._version import get_versions

__version__: str = get_versions()["version"]  # type: ignore
del get_versions
