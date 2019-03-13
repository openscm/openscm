.. _standard-parameters:

Parameters
==========

OpenSCM follows a number of conventions related to parameters, we document these
In OpenSCM a 'parameter' is anything which is passed to a model e.g. |CO2| emissions, equilibrium climate sensitivity, aerosol forcing scaling.

Simple climate models come in many different shapes and forms hence we do not expect them to all be able to do everything.
However, to be included in OpenSCM they must be able to understand all of OpenSCM's key parameters where here, 'understand' means that their adapters should either be able to use them or should throw sensible warning/error messages if they are used.

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

All climate models included in OpenSCM must be able to understand and take the following emissions variables as input:

- ``Emissions|CO2`` (GtC / yr)
- ``Emissions|CO2|MAGICC Fossil and Industrial`` (GtC / yr) i.e. the ``FossilCO2`` column in a MAGICC ``.SCEN`` file
- ``Emissions|CO2|MAGICC AFOLU`` (GtC / yr) i.e. the ``OtherCO2`` column in a MAGICC ``.SCEN`` file
- ``Emissions|CH4`` (MtCH4 / yr)
- ``Emissions|CH4|MAGICC Fossil and Industrial`` (MtCH4 / yr)
- ``Emissions|CH4|MAGICC AFOLU`` (MtCH4 / yr)
- ``Emissions|N2O`` (MtN / yr)
- ``Emissions|N2O|MAGICC Fossil and Industrial`` (MtN / yr)
- ``Emissions|N2O|MAGICC AFOLU`` (MtN / yr)
- ``Emissions|SOx`` (MtS / yr)
- ``Emissions|SOx|MAGICC Fossil and Industrial`` (MtS / yr)
- ``Emissions|SOx|MAGICC AFOLU`` (MtS / yr)
- ``Emissions|CO`` (MtCO / yr)
- ``Emissions|CO|MAGICC Fossil and Industrial`` (MtCO / yr)
- ``Emissions|CO|MAGICC AFOLU`` (MtCO / yr)
- ``Emissions|NMVOC`` (MtNMVOC / yr)
- ``Emissions|NMVOC|MAGICC Fossil and Industrial`` (MtNMVOC / yr)
- ``Emissions|NMVOC|MAGICC AFOLU`` (MtNMVOC / yr)
- ``Emissions|NOx`` (MtN / yr)
- ``Emissions|NOx|MAGICC Fossil and Industrial`` (MtN / yr)
- ``Emissions|NOx|MAGICC AFOLU`` (MtN / yr)
- ``Emissions|BC`` (MtBC / yr)
- ``Emissions|BC|MAGICC Fossil and Industrial`` (MtBC / yr)
- ``Emissions|BC|MAGICC AFOLU`` (MtBC / yr)
- ``Emissions|OC`` (MtOC / yr)
- ``Emissions|OC|MAGICC Fossil and Industrial`` (MtOC / yr)
- ``Emissions|OC|MAGICC AFOLU`` (MtOC / yr)
- ``Emissions|NH3`` (MtN / yr)
- ``Emissions|NH3|MAGICC Fossil and Industrial`` (MtN / yr)
- ``Emissions|NH3|MAGICC AFOLU`` (MtN / yr)
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
