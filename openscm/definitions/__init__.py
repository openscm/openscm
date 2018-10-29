from os.path import join, dirname


import pint

unit_registry = pint.UnitRegistry()  # start a unit repository using the default variables
# should move the definitions somewhere more stable
# openscm/definitions ?
unit_registry.load_definitions(join(dirname(__file__), 'emissions_units.txt'))
