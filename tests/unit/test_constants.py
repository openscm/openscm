from openscm.constants import ONE_YEAR


def test_one_year():
    assert ONE_YEAR.to("s").magnitude == 31556925.9747
