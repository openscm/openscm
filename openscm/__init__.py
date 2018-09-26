from .core import Core
from .highlevel import OpenSCM

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
