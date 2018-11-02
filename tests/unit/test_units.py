from openscm.units import unit_registry


def test_unit_registry():
    CO2 = unit_registry("CO2")
    assert CO2.to("C").magnitude == 12. / 44.
