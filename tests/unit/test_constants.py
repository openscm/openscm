from openscm.constants import ONE_YEAR_IN_S_INTEGER


def test_one_year():
    assert ONE_YEAR_IN_S_INTEGER == 31556926
    assert isinstance(ONE_YEAR_IN_S_INTEGER, int)
