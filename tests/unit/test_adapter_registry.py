import re
import sys
from unittest.mock import MagicMock, patch

import pytest

from openscm.adapters import load_adapter
from openscm.errors import AdapterNeedsModuleError


@patch("openscm.adapters._loaded_adapters", new={"stub": 1})
def test_adapter_registry():
    assert load_adapter("stub") == 1


# avoid caching anything in openscm.adapters._loaded_adapters
@patch("openscm.adapters._loaded_adapters", new={})
def test_load_model():
    with patch.dict(sys.modules, {"openscm.adapters.modelname": MagicMock()}):
        load_adapter("DICE")


def test_adapter_registry_unknown_model():
    # make sure we didn't break _loaded_adapters in previous test
    with pytest.raises(KeyError, match="Unknown model 'stub'"):
        load_adapter("stub")

    with pytest.raises(KeyError, match="Unknown model 'unknown'"):
        load_adapter("unknown")


def test_adapter_registry_import_error():
    error_msg = re.escape(
        "To run 'DICE' you need to install additional dependencies. Please "
        "install them using `pip install openscm[model-DICE]`."
    )
    with patch.dict(sys.modules, {"openscm.adapters.dice": None}):
        with pytest.raises(AdapterNeedsModuleError, match=error_msg):
            load_adapter("DICE")
