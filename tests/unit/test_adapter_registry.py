import re
import sys


import pytest
from unittest.mock import patch, MagicMock


from openscm.adapters import load_adapter
from openscm.errors import AdapterNeedsModuleError


@patch("openscm.adapters._loaded_adapters", new={"stub": 1})
def test_adapter_registry():
    assert load_adapter("stub") == 1


# avoid caching anything in openscm.adapters._loaded_adapters
@patch("openscm.adapters._loaded_adapters", new={})
def test_load_model():
    with patch.dict(sys.modules, {"openscm.adapters.modelname": MagicMock()}):
        load_adapter("MODELNAME")


def test_adapter_registry_unknown_model():
    # make sure we didn't break _loaded_adapters in previous test
    with pytest.raises(KeyError, match="Unknown model 'stub'"):
        load_adapter("stub")

    with pytest.raises(KeyError, match="Unknown model 'unknown'"):
        load_adapter("unknown")


def test_adapter_registry_import_error():
    error_msg = re.escape(
        "To run 'MODELNAME' you need to install additional dependencies. Please "
        "install them using `pip install openscm[model-MODELNAME]`."
    )
    with patch.dict(sys.modules, {"openscm.adapters.modelname": None}):
        with pytest.raises(AdapterNeedsModuleError, match=error_msg):
            load_adapter("MODELNAME")
