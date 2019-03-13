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

- ``Emissions|CO2`` (GtC / yr)
- ``Emissions|CH4`` (MtCH4 / yr)
- ``Emissions|N2O`` (MtN / yr)
- ``Emissions|SOx`` (MtS / yr)
- ``Emissions|CO`` (MtCO / yr)
- ``Emissions|NMVOC`` (MtNMVOC / yr)
- ``Emissions|NOx`` (MtN / yr)
- ``Emissions|BC`` (MtBC / yr)
- ``Emissions|OC`` (MtOC / yr)
- ``Emissions|NH3`` (MtN / yr)
- ``Emissions|NF3`` (MtNF3 / yr)
- ``Emissions|CF4`` (ktCF4 / yr)
- ``Emissions|C2F6`` (ktC2F6 / yr)
- ``Emissions|C3F8`` (ktC3F8 / yr)
- ``Emissions|cC4F8`` (ktcC4F8 / yr)
- ``Emissions|C4F10`` (ktC4F10 / yr)
- ``Emissions|C5F12`` (ktC5F12 / yr)
- ``Emissions|C6F14`` (ktC6F14 / yr)
- ``Emissions|C7F16`` (ktC7F16 / yr)
- ``Emissions|C8F18`` (ktC8F18 / yr)
- ``Emissions|CCl4`` (ktCCl4 / yr)
- ``Emissions|CHCl3`` (ktCHCl3 / yr)
- ``Emissions|CH2Cl2`` (ktCH2Cl2 / yr)
- ``Emissions|CH3CCl3`` (ktCH3CCl3 / yr)
- ``Emissions|CH3Cl`` (ktCH3Cl / yr)
- ``Emissions|CH3Br`` (ktCH3Br / yr)
- ``Emissions|HFC23`` (ktHFC23 / yr)
- ``Emissions|HFC32`` (ktHFC32 / yr)
- ``Emissions|HFC4310`` (ktHFC4310 / yr)
- ``Emissions|HFC125`` (ktHFC125 / yr)
- ``Emissions|HFC134a`` (ktHFC134a / yr)
- ``Emissions|HFC143a`` (ktHFC143a / yr)
- ``Emissions|HFC152a`` (ktHFC152a / yr)
- ``Emissions|HFC227ea`` (ktHFC227ea / yr)
- ``Emissions|HFC236fa`` (ktHFC236fa / yr)
- ``Emissions|HFC245fa`` (ktHFC245fa / yr)
- ``Emissions|HFC365mfc`` (ktHFC365mfc / yr)
- ``Emissions|CFC11`` (ktCFC11 / yr)
- ``Emissions|CFC12`` (ktCFC12 / yr)
- ``Emissions|CFC113`` (ktCFC113 / yr)
- ``Emissions|CFC114`` (ktCFC114 / yr)
- ``Emissions|CFC115`` (ktCFC115 / yr)
- ``Emissions|HCFC22`` (ktHCFC22 / yr)
- ``Emissions|HCFC141b`` (ktHCFC141b / yr)
- ``Emissions|HCFC142b`` (ktHCFC142b / yr)
- ``Emissions|SF6`` (ktSF6 / yr)
- ``Emissions|SO2F2`` (ktSO2F2 / yr)
- ``Emissions|Halon1202`` (ktHalon1202 / yr)
- ``Emissions|Halon1211`` (ktHalon1211 / yr)
- ``Emissions|Halon1301`` (ktHalon1301 / yr)
- ``Emissions|Halon2402`` (ktHalon2402 / yr)


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


- ``Land to Air Flux|CO2|Permafrost`` (GtC / yr) - land to air flux of |CO2| from permafrost
- ``Land to Air Flux|CH4|Permafrost`` (MtCH4 / yr)


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
