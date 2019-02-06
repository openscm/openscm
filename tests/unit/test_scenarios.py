import numpy as np


from openscm.scenarios import rcps


def test_rcp26():
    value = (
        rcps.filter(
            variable="Emissions|CH4", region="World", year=1994, scenario="RCP26"
        )
        .timeseries()
        .values
    )

    assert len(value)
    np.testing.assert_allclose(value, 324.46375)


def test_rcp45():
<<<<<<< HEAD
    value = (
        rcps.filter(
            variable="Emissions|BC", region="World", year=1765, scenario="RCP45"
        )
        .timeseries()
        .values
    )
=======
    value = rcps.filter(
        variable="Emissions|BC",
        region="World",
        year=1765,
        scenario="RCP45"
    ).timeseries().values
>>>>>>> Add get_times to TimeseriesView

    assert len(value)
    np.testing.assert_allclose(value, 0)


def test_rcp60():
<<<<<<< HEAD
    value = (
        rcps.filter(
            variable="Emissions|CO2|MAGICC Fossil and Industrial",
            region="World",
            year=2034,
            scenario="RCP60",
        )
        .timeseries()
        .values
    )
=======
    value = rcps.filter(
        variable="Emissions|CO2|MAGICC Fossil and Industrial",
        region="World",
        year=2034,
        scenario="RCP60"
    ).timeseries().values
>>>>>>> Add get_times to TimeseriesView

    assert len(value)
    np.testing.assert_allclose(value, 10.61878)


def test_rcp85():
<<<<<<< HEAD
    value = (
        rcps.filter(
            variable="Emissions|N2O", region="World", year=2500, scenario="RCP85"
        )
        .timeseries()
        .values
    )
=======
    value = rcps.filter(
        variable="Emissions|N2O",
        region="World",
        year=2500,
        scenario="RCP85"
    ).timeseries().values
>>>>>>> Add get_times to TimeseriesView

    assert len(value)
    np.testing.assert_allclose(value, 13.344)
