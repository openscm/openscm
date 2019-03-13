.. _standard-parameters:

Parameters
==========

OpenSCM follows a number of conventions related to parameters, we document these
In OpenSCM a 'parameter' is anything which is passed to a model e.g. |CO2| emissions, equilibrium climate sensitivity, aerosol forcing scaling.

Simple climate models come in many different shapes and forms hence we do not expect them to all be able to do everything.
However, to be included in OpenSCM they must be able to understand all of OpenSCM's key parameters where here, 'understand' means that their adapters should either be able to use them or should throw sensible warning/error messages if they are used.

Conventions
-----------

'Pre-industrial' refers to an unperturbed state of the climate.
Individual adapters can translate this into whatever year they need to for their model, but they should do such translations with this definition in mind.

'Reference period' refers to the period a given variable is reported relative to the mean of.
For example, 'surface temperature relative to a 1961-1990 reference period' refers to surface temperatures relative to the mean of the period 1961-1990.
Adapters can report variables without reference periods if these are not used internally by the models.
However, if the model uses an internal reference period then this should be indicated in the reported variable names by appending ``(rel. to XXXX-YYYY)`` to the variable name.
Hence, 'surface temperature relative to a 1961-1990 reference period' would become ``Surface Temperature (rel. to 1961-1990)`` [TODO: discuss with other authors how we want this to play out].


Variables
---------

Variables in OpenSCM come as part of a hierarchy, separated by the ``|`` character.
For example, ``Emissions|CO2|Energy``.
``Emissions|CO2|Energy`` is emissions of |CO2| from the energy sub-sector (whatever 'energy' happens to mean in this context).
To date, variables which are higher in the hierarchy, e.g. ``Emissions|CO2`` is 'higher' than ``Emissions|CO2|Energy``, are the sum of all the variables which are one level below them in the hierarchy.
For example, if we had ``Emissions|CO2|Energy``, ``Emissions|CO2|Transport`` and ``Emissions|CO2|Agriculture`` then ``Emissions|CO2`` would be the sum of ``Emissions|CO2|Energy``, ``Emissions|CO2|Transport`` and ``Emissions|CO2|Agriculture``.

Below we provide a list of variables which all OpenSCM adapters must understand, alongside an example unit.
As OpenSCM can handle unit conversion any unit which can be converted to the example unit should be usable with OpenSCM.

Emissions
*********

All climate models included in OpenSCM must be able to understand and take the following emissions variables as input. Note that adding up all sub-categories of ``Emissions`` will fail as the units will all be different.

- ``Emissions|CO2`` (mass carbon / time)
- ``Emissions|CH4`` (mass methane / time)
- ``Emissions|N2O`` (mass nitrous_oxide / time)
- ``Emissions|SOx`` (mass sulfur_oxide / time)
- ``Emissions|CO`` (mass carbon_monoxide / time)
- ``Emissions|NMVOC`` (mass volatile_organic_compound / time)
- ``Emissions|NOx`` (mass nitrogen_oxide / time)
- ``Emissions|BC`` (mass black_carbon / time)
- ``Emissions|OC`` (mass organic_carbon / time)
- ``Emissions|NH3`` (mass nitrogen / time)
- ``Emissions|NF3`` (mass NF3 / time)
- ``Emissions|CF4`` (mass CF4 / time)
- ``Emissions|C2F6`` (mass C2F6 / time)
- ``Emissions|C3F8`` (mass C3F8 / time)
- ``Emissions|cC4F8`` (mass cC4F8 / time)
- ``Emissions|C4F10`` (mass C4F10 / time)
- ``Emissions|C5F12`` (mass C5F12 / time)
- ``Emissions|C6F14`` (mass C6F14 / time)
- ``Emissions|C7F16`` (mass C7F16 / time)
- ``Emissions|C8F18`` (mass C8F18 / time)
- ``Emissions|CCl4`` (mass CCl4 / time)
- ``Emissions|CHCl3`` (mass CHCl3 / time)
- ``Emissions|CH2Cl2`` (mass CH2Cl2 / time)
- ``Emissions|CH3CCl3`` (mass CH3CCl3 / time)
- ``Emissions|CH3Cl`` (mass CH3Cl / time)
- ``Emissions|CH3Br`` (mass CH3Br / time)
- ``Emissions|HFC23`` (mass HFC23 / time)
- ``Emissions|HFC32`` (mass HFC32 / time)
- ``Emissions|HFC4310`` (mass HFC4310 / time)
- ``Emissions|HFC125`` (mass HFC125 / time)
- ``Emissions|HFC134a`` (mass HFC134a / time)
- ``Emissions|HFC143a`` (mass HFC143a / time)
- ``Emissions|HFC152a`` (mass HFC152a / time)
- ``Emissions|HFC227ea`` (mass HFC227ea / time)
- ``Emissions|HFC236fa`` (mass HFC236fa / time)
- ``Emissions|HFC245fa`` (mass HFC245fa / time)
- ``Emissions|HFC365mfc`` (mass HFC365mfc / time)
- ``Emissions|CFC11`` (mass CFC11 / time)
- ``Emissions|CFC12`` (mass CFC12 / time)
- ``Emissions|CFC113`` (mass CFC113 / time)
- ``Emissions|CFC114`` (mass CFC114 / time)
- ``Emissions|CFC115`` (mass CFC115 / time)
- ``Emissions|HCFC22`` (mass HCFC22 / time)
- ``Emissions|HCFC141b`` (mass HCFC141b / time)
- ``Emissions|HCFC142b`` (mass HCFC142b / time)
- ``Emissions|SF6`` (mass SF6 / time)
- ``Emissions|SO2F2`` (mass SO2F2 / time)
- ``Emissions|Halon1202`` (mass Halon1202 / time)
- ``Emissions|Halon1211`` (mass Halon1211 / time)
- ``Emissions|Halon1301`` (mass Halon1301 / time)
- ``Emissions|Halon2402`` (mass Halon2402 / time)


Concentrations
**************

Atmospheric concentrations should be of the form ``Atmospheric Concentrations|XXX`` e.g. ``Atmospheric Concentrations|CO2``.
OpenSCM models should understand all of the following variables.
Note that adding up all sub-categories of ``Atmospheric Concentrations`` will either fail as the units will not all be the same or will produce a basically nonsense number (but still be possible).

- ``Atmospheric Concentrations|CO2`` (ppm)
- ``Atmospheric Concentrations|CH4`` (ppb)
- ``Atmospheric Concentrations|N2O`` (ppb)
- ``Atmospheric Concentrations|NF3`` (ppt)
- ``Atmospheric Concentrations|CF4`` (ppt)
- ``Atmospheric Concentrations|C2F6`` (ppt)
- ``Atmospheric Concentrations|C3F8`` (ppt)
- ``Atmospheric Concentrations|cC4F8`` (ppt)
- ``Atmospheric Concentrations|C4F10`` (ppt)
- ``Atmospheric Concentrations|C5F12`` (ppt)
- ``Atmospheric Concentrations|C6F14`` (ppt)
- ``Atmospheric Concentrations|C7F16`` (ppt)
- ``Atmospheric Concentrations|C8F18`` (ppt)
- ``Atmospheric Concentrations|CCl4`` (ppt)
- ``Atmospheric Concentrations|CHCl3`` (ppt)
- ``Atmospheric Concentrations|CH2Cl2`` (ppt)
- ``Atmospheric Concentrations|CH3CCl3`` (ppt)
- ``Atmospheric Concentrations|CH3Cl`` (ppt)
- ``Atmospheric Concentrations|CH3Br`` (ppt)
- ``Atmospheric Concentrations|HFC23`` (ppt)
- ``Atmospheric Concentrations|HFC32`` (ppt)
- ``Atmospheric Concentrations|HFC4310`` (ppt)
- ``Atmospheric Concentrations|HFC125`` (ppt)
- ``Atmospheric Concentrations|HFC134a`` (ppt)
- ``Atmospheric Concentrations|HFC143a`` (ppt)
- ``Atmospheric Concentrations|HFC152a`` (ppt)
- ``Atmospheric Concentrations|HFC227ea`` (ppt)
- ``Atmospheric Concentrations|HFC236fa`` (ppt)
- ``Atmospheric Concentrations|HFC245fa`` (ppt)
- ``Atmospheric Concentrations|HFC365mfc`` (ppt)
- ``Atmospheric Concentrations|CFC11`` (ppt)
- ``Atmospheric Concentrations|CFC12`` (ppt)
- ``Atmospheric Concentrations|CFC113`` (ppt)
- ``Atmospheric Concentrations|CFC114`` (ppt)
- ``Atmospheric Concentrations|CFC115`` (ppt)
- ``Atmospheric Concentrations|HCFC22`` (ppt)
- ``Atmospheric Concentrations|HCFC141b`` (ppt)
- ``Atmospheric Concentrations|HCFC142b`` (ppt)
- ``Atmospheric Concentrations|SF6`` (ppt)
- ``Atmospheric Concentrations|SO2F2`` (ppt)
- ``Atmospheric Concentrations|Halon1202`` (ppt)
- ``Atmospheric Concentrations|Halon1211`` (ppt)
- ``Atmospheric Concentrations|Halon1301`` (ppt)
- ``Atmospheric Concentrations|Halon2402`` (ppt)


Radiative Forcing
*****************

Radiative forcing should be of the form ``Radiative Forcing|XXX`` e.g. ``Radiative Forcing|CO2``.
OpenSCM models should understand all of the following variables.
Note that adding up all sub-categories of ``Radiative Forcing`` will give total forcing which is a sensible number, unlike emissions and concentrations.
However, the adapters have to be careful to ensure that they don't double report by e.g. providing ``Radiative Forcing|Aerosols|Direct Effect`` and ``Radiative Forcing|Aerosols|NOx``.

- ``Radiative Forcing`` (W/m\*\*2)
- ``Radiative Forcing|Aerosols`` (W/m\*\*2)
- ``Radiative Forcing|Aerosols|Direct Effect`` (W/m\*\*2)
- ``Radiative Forcing|Aerosols|Indirect Effect`` (W/m\*\*2)
- ``Radiative Forcing|Aerosols|SOx`` (W/m\*\*2)
- ``Radiative Forcing|Aerosols|NOx`` (W/m\*\*2)
- ``Radiative Forcing|Aerosols|OC`` (W/m\*\*2)
- ``Radiative Forcing|Aerosols|BC`` (W/m\*\*2)
- ``Radiative Forcing|Land-use Change`` (W/m\*\*2)
- ``Radiative Forcing|Black Carbon on Snow`` (W/m\*\*2)
- ``Radiative Forcing|Volcanic`` (W/m\*\*2)
- ``Radiative Forcing|Solar`` (W/m\*\*2)
- ``Radiative Forcing|External`` (W/m\*\*2)
- ``Radiative Forcing|CO2`` (W/m\*\*2)
- ``Radiative Forcing|CH4`` (W/m\*\*2)
- ``Radiative Forcing|N2O`` (W/m\*\*2)
- ``Radiative Forcing|NF3`` (W/m\*\*2)
- ``Radiative Forcing|CF4`` (W/m\*\*2)
- ``Radiative Forcing|C2F6`` (W/m\*\*2)
- ``Radiative Forcing|C3F8`` (W/m\*\*2)
- ``Radiative Forcing|cC4F8`` (W/m\*\*2)
- ``Radiative Forcing|C4F10`` (W/m\*\*2)
- ``Radiative Forcing|C5F12`` (W/m\*\*2)
- ``Radiative Forcing|C6F14`` (W/m\*\*2)
- ``Radiative Forcing|C7F16`` (W/m\*\*2)
- ``Radiative Forcing|C8F18`` (W/m\*\*2)
- ``Radiative Forcing|CCl4`` (W/m\*\*2)
- ``Radiative Forcing|CHCl3`` (W/m\*\*2)
- ``Radiative Forcing|CH2Cl2`` (W/m\*\*2)
- ``Radiative Forcing|CH3CCl3`` (W/m\*\*2)
- ``Radiative Forcing|CH3Cl`` (W/m\*\*2)
- ``Radiative Forcing|CH3Br`` (W/m\*\*2)
- ``Radiative Forcing|HFC23`` (W/m\*\*2)
- ``Radiative Forcing|HFC32`` (W/m\*\*2)
- ``Radiative Forcing|HFC4310`` (W/m\*\*2)
- ``Radiative Forcing|HFC125`` (W/m\*\*2)
- ``Radiative Forcing|HFC134a`` (W/m\*\*2)
- ``Radiative Forcing|HFC143a`` (W/m\*\*2)
- ``Radiative Forcing|HFC152a`` (W/m\*\*2)
- ``Radiative Forcing|HFC227ea`` (W/m\*\*2)
- ``Radiative Forcing|HFC236fa`` (W/m\*\*2)
- ``Radiative Forcing|HFC245fa`` (W/m\*\*2)
- ``Radiative Forcing|HFC365mfc`` (W/m\*\*2)
- ``Radiative Forcing|CFC11`` (W/m\*\*2)
- ``Radiative Forcing|CFC12`` (W/m\*\*2)
- ``Radiative Forcing|CFC113`` (W/m\*\*2)
- ``Radiative Forcing|CFC114`` (W/m\*\*2)
- ``Radiative Forcing|CFC115`` (W/m\*\*2)
- ``Radiative Forcing|HCFC22`` (W/m\*\*2)
- ``Radiative Forcing|HCFC141b`` (W/m\*\*2)
- ``Radiative Forcing|HCFC142b`` (W/m\*\*2)
- ``Radiative Forcing|SF6`` (W/m\*\*2)
- ``Radiative Forcing|SO2F2`` (W/m\*\*2)
- ``Radiative Forcing|Halon1202`` (W/m\*\*2)
- ``Radiative Forcing|Halon1211`` (W/m\*\*2)
- ``Radiative Forcing|Halon1301`` (W/m\*\*2)
- ``Radiative Forcing|Halon2402`` (W/m\*\*2)


Material Fluxes
***************

These variables can be used to store the flux of material within the model.
They should be of the form ``X to Y Flux`` where the material is flowing from ``X`` into ``Y`` (and hence negative values represent flows from ``Y`` into ``X``).
OpenSCM models should understand all of the following variables.


- ``Land to Air Flux|CO2|Permafrost`` (mass carbon / time) - land to air flux of |CO2| from permafrost
- ``Land to Air Flux|CH4|Permafrost`` (mass methane / time)


Other
*****

Other variables which should be recognised by OpenSCM adapters are given below.

- ``Surface Temperature`` (K) - surface air temperature i.e. tas
- ``Ocean Temperature`` (K) - surface ocean temperature i.e. tos
- ``Ocean Heat Content`` (J)
- ``Sea Level Rise`` (mm)


Regions
-------

Similarly to variables, regions are also provided in a hierarchy separated by the ``|`` character.
To date, regions which are higher in the hierarchy are the sum of all the regions which are one level below them in the hierarchy (be careful of this when looking at e.g. |CO2| concentration data at a regional level).

All OpenSCM adapaters must understand the following regions:

- ``World``
- ``World|Northern Hemisphere``
- ``World|Northern Hemisphere|Ocean``
- ``World|Northern Hemisphere|Land``
- ``World|Southern Hemisphere``
- ``World|Southern Hemisphere|Ocean``
- ``World|Southern Hemisphere|Land``
- ``World|Ocean``
- ``World|Land``
- ``World|R5ASIA``
- ``World|R5REF``
- ``World|R5MAF``
- ``World|R5OECD``
- ``World|R5LAM``
- ``World|R5.2ASIA``
- ``World|R5.2REF``
- ``World|R5.2MAF``
- ``World|R5.2OECD``
- ``World|R5.2LAM``
- ``World|Bunkers``


Configuration
-------------

Each model will have its own set of configuration parameters and conventions.
In OpenSCM we allow the user to pass these to and from the model via the adapter, following the model's own internal conventions for naming.
However, we also insist that models understand the following configuration options.

- ``ecs`` (K) - equilibrium climate sensitivity
- ``tcr`` (K) - transient climate response
- ``f2xco2`` (W/m\*\*2) - radiative forcing due to a doubling of atmospheric |CO2| concentrations from pre-industrial level
