"""
This module contains all of the relevant definitions for handling SCM data.

The units are defined using Pint [add link to Pint]. To this end, we maintain a units
definitions file, ``openscm_units.txt``, which contains all of the relevant emissions
which are not already contained within the Pint library.

When we store the data in csv's, we use `Data Packages <https://frictionlessdata.io/
docs/creating-tabular-data-packages-in-python/>`_. These store the data in an easy to
read csv file whilst providing comprehensive metadata describing the data (column
meanings and expected types) in the accompanying ``datapackage.json`` file. Please see
this metadata for further details.
"""
from os.path import join, dirname


import pint


# start a unit repository using the default variables
unit_registry = pint.UnitRegistry()
"""`obj`:`pint.registry.UnitRegistry`: OpenSCM's unit registry

The unit registry contains all of our recognised units. A couple of examples

.. code:: python

    >>> from openscm.definitions import unit_registry
    >>> unit_registry("CO2")
    <Quantity(1, 'CO2')>

    >>> emissions_aus = 0.34 * unit_registry("Gt C / yr")
    >>> emissions_aus
    <Quantity(0.34, 'C * gigametric_ton / a')>

    >>> emissions_aus.to("Mt C / week")
    <Quantity(6.516224050620789, 'C * megametric_ton / week')>
"""

unit_registry.load_definitions(join(dirname(__file__), "openscm_units.txt"))
