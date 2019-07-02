from openscm.adapters import Adapter
from openscm.core.parameterset import ParameterSet


def test_adapter_base_class_init():
    parametersstub = "Parameters"
    outputstub = "Parameters"
    Adapter.__abstractmethods__ = set()
    adapter = Adapter(  # pylint: disable=abstract-class-instantiated
        parametersstub, outputstub
    )
    assert adapter._parameters == parametersstub
    assert adapter._output == outputstub


def test_adapter_base_class_run():
    start_time = 20

    Adapter._start_time = start_time
    Adapter.__abstractmethods__ = set()
    adapter = Adapter(  # pylint: disable=abstract-class-instantiated
        ParameterSet(), ParameterSet()
    )

    adapter.reset()
    assert adapter._current_time == start_time
    adapter.run()
    assert adapter.step() == start_time
