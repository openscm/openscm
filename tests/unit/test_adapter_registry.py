import re


import pytest
from unittest.mock import patch


from openscm.adapters import load_adapter


@patch("openscm.adapters._loaded_adapters", new={"stub": 1})
def test_adapter_registry():
    assert load_adapter("stub") == 1


def test_adapter_registry_unknown_model():
    # make sure we didn't break _loaded_adapters in previous test
    with pytest.raises(KeyError, match="Unknown model 'stub'"):
        load_adapter("stub")

    with pytest.raises(KeyError, match="Unknown model 'unknown'"):
        load_adapter("unknown")
