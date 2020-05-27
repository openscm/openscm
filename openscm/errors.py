"""
Errors/Exceptions defined and used in OpenSCM.
"""


class AdapterNeedsModuleError(Exception):
    """Exception raised when an adapter needs a module that is not installed."""
