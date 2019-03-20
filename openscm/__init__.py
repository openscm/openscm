"""
OpenSCM - unified access to simple climate models.
"""

from ._version import get_versions

__version__: str = get_versions()["version"]
del get_versions
