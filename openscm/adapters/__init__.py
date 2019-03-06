"""
Module including all model adapters shipped with OpenSCM.
"""

from typing import Dict, Optional


from ..errors import AdapterNeedsModuleError


_loaded_adapters: Dict[str, type] = {}


def load_adapter(name: str) -> Optional[type]:
    """
    Load adapter with a given name.

    Parameters
    ----------
    name
        Name of the adapter/model

    Returns
    -------
    Adapter
        Instance of the requested adapter

    Raises
    ------
    AdapterNeedsModuleError
        Adapter needs a module that is not installed
    KeyError
        Adapter/model not found
    """
    if name in _loaded_adapters:
        return _loaded_adapters[name]

    adapter: Optional[type] = None

    try:
        # TODO remove once we have at least one real adapter in here
        if name == "MODELNAME":
            from .modelname import MODELNAME

            adapter = MODELNAME

        """
        When implementing an additional adapter, include your adapter NAME here as:
        ```
        elif name == "NAME":
            from .NAME import NAME

            adapter = NAME
        ```
        """
    except ImportError:
        raise AdapterNeedsModuleError(
            "To run '{name}' you need to install additional dependencies. Please "
            "install them using `pip install openscm[model-{name}]`.".format(name=name)
        )

    if adapter is None:
        raise KeyError("Unknown model '{}'".format(name))

    _loaded_adapters[name] = adapter
    return adapter
