from openscm.adapter import Adapter


def test_adapter_base_class():
    parametersstub = "Parameters"
    outputstub = "Parameters"
    Adapter.__abstractmethods__ = set()
    adapter = Adapter(parametersstub, outputstub)  # pylint: disable=abstract-class-instantiated
    assert adapter._parameters == parametersstub
    assert adapter._output == outputstub
    start_time = 0
    stop_time = 1
    adapter.initialize_run_parameters(start_time, stop_time)
    assert adapter._start_time == start_time
    assert adapter._stop_time == stop_time
    adapter.initialize_model_input()
    adapter.reset()
    assert adapter._current_time == start_time
    adapter.run()
    assert adapter.step() == start_time
