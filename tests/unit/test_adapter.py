from openscm.adapter import Adapter
import pytest

def test_adapter():
    parametersstub = "Parameters"
    adapter = Adapter(parametersstub)
    assert(adapter._parameters == parametersstub)
    start_time = 0
    stop_time = 1
    adapter.initialize_run_parameters(start_time, stop_time)
    assert(adapter._start_time == start_time)
    assert(adapter._stop_time == stop_time)
    adapter.initialize_model_input()
    adapter.reset()
    assert(adapter._current_time == start_time)
    with pytest.raises(NotImplementedError):
        adapter.run()
    with pytest.raises(NotImplementedError):
        adapter.step()
