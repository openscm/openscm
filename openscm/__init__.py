"""
OpenSCM - unified access to simple climate models.
"""

from ._version import get_versions
from .core import OpenSCM  # noqa: F401

__version__: str = get_versions()["version"]
del get_versions
