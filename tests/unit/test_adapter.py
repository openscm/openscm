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


def test_adapter_base_class_initialize_model_input():
    Adapter.__abstractmethods__ = set()
    adapter = Adapter(  # pylint: disable=abstract-class-instantiated
        ParameterSet(), ParameterSet()
    )

    adapter.initialize_model_input()
    assert adapter._initialized


def test_adapter_base_class_initialize_run_parameters():
    Adapter.__abstractmethods__ = set()
    adapter = Adapter(  # pylint: disable=abstract-class-instantiated
        ParameterSet(), ParameterSet()
    )
    adapter.initialize_run_parameters()
    assert adapter._initialized


def test_adapter_base_class_run():
    start_time = 20

    Adapter.__abstractmethods__ = set()
    in_parameters = ParameterSet()
    in_parameters.generic("Start Time").value = start_time
    adapter = Adapter(  # pylint: disable=abstract-class-instantiated
        in_parameters, ParameterSet()
    )

    adapter.initialize_model_input()
    adapter.initialize_run_parameters()
    adapter.reset()
    assert adapter._current_time == start_time
    adapter.run()
    assert adapter.step() == start_time
