import warnings


import numpy as np
import pytest


from openscm.units import unit_registry


from conftest import assert_pint_equal


def test_pint_array_comparison():
    a = np.array([0, 2]) * unit_registry("GtC")
    b = np.array([0, 2]) * unit_registry("MtC")

    # no error but does raise warning about stripping units
    with warnings.catch_warnings(record=True):
        np.testing.assert_allclose(a, b)

    # actually gives an error as we want
    with pytest.raises(AssertionError):
        assert_pint_equal(a, b)
