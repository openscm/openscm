from ..adapter import Adapter
from ..errors import AdapterNeedsModuleError

_loaded_adapters = {}

def load_adapter(name: str) -> Adapter:
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
    else:
        adapter = None

        """
        When implementing an additional adapter, include your adapter NAME here as:
        ```
        elif name == "NAME":
            from .NAME import NAME
            adapter = NAME
        ```

        Make sure to throw a `AdapterNeedsModuleError` with an appropriate message how
        to install the module when your adapter needs a module (such as the model
        itself) that is not installed.
        """

        if adapter is None:
            raise KeyError("Unknown model '{}'".format(name))
        else:
            _loaded_adapters[name] = adapter
            return adapter
