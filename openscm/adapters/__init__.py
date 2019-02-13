from ..errors import AdapterNeedsModuleError

_loaded_adapters = {}


def load_adapter(name: str) -> type:
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

        try:
            if name == "MODELNAME":
                from .modelname import MODELNAME

                adapter = MODELNAME(parameters)

            """
            When implementing an additional adapter, include your adapter NAME here as:
            ```
            elif name == "NAME":
                from .NAME import NAME
                adapter = NAME
            ```
            """
        except ImportError:  # pragma: no cover
            raise AdapterNeedsModuleError(
                """
                To run '{}' you need to install additional dependencies. Please install
                them using `pip install openscm[model-{}]`.
                """.format(
                    name, name
                )
            )

        if adapter is None:
            raise KeyError("Unknown model '{}'".format(name))
        else:
            _loaded_adapters[name] = adapter
            return adapter
