Usage
=====

OpenSCM defines and provides three interfaces for its users.

Its **low-level** interface is targeted to users who want to include OpenSCM in their model, e.g. integrated assessment modellers. This interface provides the basic functionality necessary to run all SCMs included in OpenSCM. It includes functions for getting and setting parameters as well as to run and reset the model.

The **high-level* interface adds several convinience functions on top of the low-level interface. It allows for reading and writing parameters from and to a standardized and other file formats, including whole scenario definitions. Also, it provides functions for running ensembles of model runs and doing stochastic analyses and model tuning.

The **command line* interface lets users run models with specified parameters and model input directly from the command line without coding themselves. Please see :ref:`tool` for its usage documentation.


Parameters
----------

**Parameter** here refers to any named input or output variable of a model. A parameter can be either scalar (i.e. a single number) or a time series and has a unique name in a hierarchy of arbitrary depth. That means every parameter either is a root parameter without a parent parameter ("level 0") or belongs to a parent parameter.

For example, the parameter for industrial carbon emissions belongs to the parameter for carbon emissions, which in turn belongs to the root parameter emission. Thus, it is identified by

``Emissions`` -> ``CO2`` -> ``Industrial``.

In the low-level API parameters are expected to be identified by tuples of strings describing their position in the hierarchy, i.e. in this example ``("Emissions", "CO2", "Industrial")``. The high-level API also allows for giving it as a string with the hierarchical levels separated by ``|``, e.g. ``"Emissions|CO2|Industrial"``.

See :ref:`parameter-hierarchy` for the standard parameters in OpenSCM.


Time frames
-----------



Low-level interface
-------------------

(see :ref:`low-level-reference` for an API reference)

Setting up a model run
**********************

Setting input parameters
************************

Running the model
*****************

Getting output parameters
*************************



High-level interface
--------------------

(see :ref:`high-level-reference` for an API reference)
