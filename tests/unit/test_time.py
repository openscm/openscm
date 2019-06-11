import pytest

from openscm.core.time import ExtrapolationType, InterpolationType

@pytest.mark.parametrize("input,output",[
    ("constant", ExtrapolationType.CONSTANT),
    ("nOne", ExtrapolationType.NONE),
    ("LINEAR", ExtrapolationType.LINEAR),
])
def test_init_extrapolation(input, output):
    res = ExtrapolationType.from_extrapolation_type(input)
    assert res == output

@pytest.mark.parametrize("input,output",[
    ("linear", InterpolationType.LINEAR),
])
def test_init_interpolation(input, output):
    res = InterpolationType.from_interpolation_type(input)
    assert res == output
