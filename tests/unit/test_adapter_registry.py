from openscm.adapters import _loaded_adapters, load_adapter
import pytest


def test_adapter_registry():
    assert _loaded_adapters == {}
    stub = 1
    _loaded_adapters["stub"] = stub
    assert load_adapter("stub") == stub
    with pytest.raises(KeyError, match="Unknown model 'unknown'"):
        load_adapter("unknown")
