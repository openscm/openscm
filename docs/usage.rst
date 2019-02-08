Usage
=====

OpenSCM defines and provides three interfaces for its users.

Its **low-level** interface is targeted to users who want to include
OpenSCM in their model, e.g. integrated assessment modellers. This
interface provides the basic functionality necessary to run all SCMs
included in OpenSCM. It includes functions for getting and setting
parameters as well as to run and reset the model.

The **high-level** interface adds several convinience functions on top
of the low-level interface. It allows for reading and writing
parameters from and to a standardized and other file formats,
including whole scenario definitions. Also, it provides functions for
running ensembles of model runs and doing stochastic analyses and
model tuning.

The **command line** interface lets users run models with specified
parameters and model input directly from the command line without
coding themselves. Please see :ref:`tool` for its usage documentation.


.. _parameters:

Parameters
----------

**Parameter** here refers to any named input or output variable of a
model. A parameter can be either scalar (i.e. a single number) or a
timeseries and has a unique name in a hierarchy of arbitrary depth.
That means every parameter either is a root parameter without a parent
parameter ("level 0") or belongs to a parent parameter.

For example, the parameter for industrial carbon emissions belongs to
the parameter for carbon emissions, which in turn belongs to the root
parameter emission. Thus, it is identified by

    ``Emissions`` -> ``CO2`` -> ``Industrial``.

In the low-level API parameters are expected to be identified by
tuples of strings describing their position in the hierarchy, i.e. in
this example ``("Emissions", "CO2", "Industrial")``. The high-level
API also allows for giving it as a string with the hierarchical levels
separated by ``|``, e.g. ``"Emissions|CO2|Industrial"``.

See :ref:`parameter-hierarchy` for the standard parameters in OpenSCM.


Time frames
-----------

Timeseries parameters are always given with a corresponding time
frame, which consists of a time point and a period length. The time
point gives the start of the timeseries; the period length gives the
length of the period between consecutive values in the timeseries.
Each parameter value is assumed to be the **average value** for its
corresponding period. This implies that additive values such as
emissions need to be given as a rate, e.g. ``tC/a`` rather than
``tC``.

In the low-level API time points are given in seconds since
``1970-01-01 00:00:00``, while time period lengths are specified in
seconds. The high-level API additionally accepts string values and
``datetime.datetime`` objects.


Low-level interface
-------------------

(see :ref:`low-level-reference` for an API reference)

Setting up a model run
**********************

A model run is represented by a :class:`openscm.core.Core` object
specifying the underlying SCM and start and end time:

.. code:: python

    from openscm.core import Core as ModelRun
    from datetime import datetime, timedelta

    start_time = datetime(2006, 1, 1).timestamp()
    year_seconds = timedelta(365).total_seconds()
    # use year_seconds since models do not account for leap years
    end_time = start_time + (2100 - 2006) * year_seconds
    model_run = ModelRun("DICE", start_time, end_time)

Setting input parameters
************************

In the low-level API parameters are get and set through of subclasses
of :class:`ParameterView`. While the values of the parameters are
stored internaly, a :class:`ParameterView` provides an (always
up-to-date) "view" of the corresponding parameter giving the parameter
values in a specific unit and, in the case of timeseries, a specific
time frame.

Unit and time frame have to be specified when requesting a
:class:`~openscm.core.ParameterView` from the
:class:`~openscm.core.Core`'s :class:`~openscm.core.ParameterSet`
property called ``parameters`` using one of the following functions:

- :func:`~openscm.core.ParameterSet.get_scalar_view` returns a
  read-only view to a scalar parameter
  (:class:`~openscm.core.ScalarView`)
- :func:`~openscm.core.ParameterSet.get_timeseries_view` returns a
  read-only view to a timeseries parameter
  (:class:`~openscm.core.TimeseriesView`)
- :func:`~openscm.core.ParameterSet.get_writable_scalar_view` returns
  a writable view to a scalar parameter
  (:class:`~openscm.core.WritableScalarView`)
- :func:`~openscm.core.ParameterSet.get_writable_timeseries_view`
  returns a writable view to a timeseries parameter
  (:class:`~openscm.core.WritableTimeseriesView`)

Each of these functions take the hierarchical name of the parameter
(as described under :ref:`parameters`) and, in a similar fashion, the
hierarchical name of the region it applies to or an empty tuple,
``()``, in case the parameter applies to all regions.

Values can be get and set using ``get`` and ``set`` (also,
``get_series`` and ``set_series`` for whole timeseries), respectively.
Conversion, if necessary, is done internally by the object. There is
no standard for the unit and time frame for internal storage, but
those of the first :class:`ParameterView` requested are used. If a
scalar view for a time series is requested (or vice-versa), or if the
units are not convertible, an error is raised.

:class:`~openscm.core.ParameterView` objects also convert between
hierarchical levels if possible: a view to a higher level parameter
yields the sum of its child parameters. This implies that, once a
*writable* view to a parameter is requested, there cannot be a view to
one of its children. Otherwise consistency cannot be guaranteed, so an
error is raised. The same holds if a child parameter has already been
set and the user tries to set values for one of its parent parameters.
A similar logic applies to the hierarchy of regions.

Using :class:`~openscm.core.ParameterView` as proxy objects rather
than directly setting/returning parameter valus allows for efficient
parameter handling in the expected units and time frames without
specifying these for each value (e.g. seeting a timeseries step-wise
would create large overhead).

.. code:: python

    climate_sensitivity = model_run.parameters.get_writable_scalar_view(
        ("Equilibrium Climate Sensitivity"), (), "degC"
    )
    climate_sensitivity.set(3)

    carbon_emissions_raw = [10 for _ in range(2100 - 2006)]
    carbon_emissions = model_run.parameters.get_writable_timeseries_view(
        ("Emissions", "CO2"), (), "GtCO2/a", start_time, year_seconds
    )
    carbon_emissions.set_series(carbon_emissions_raw)

Running the model
*****************

The model is simply run by calling the :func:`~openscm.core,Core.run`
function:

.. code:: python

    model_run.run()

This tells the adapter for the particular SCM to get the necessary
parameters in the format as expected by the model, while conversion
for units and time frames is done by the corresponding
:class:`ParameterView` objects. It then runs the model itself.

After the run the model is reset, so the
:func:`~openscm.core.Core.run` function can be called again (setting
parameters to new values before, if desired).

Getting output parameters
*************************

During the run the model adapter sets the output parameters just like
the input parameters were set above. Thus, these can be read using
read-only :class:`~openscm.core.ParameterView` objects:

.. code:: python

    gmt = model_run.parameters.get_timeseries_view(
        ("Temperature", "Surface"), (), "degC", start_time, year_seconds
    )
    print(gmt.get_series())


High-level interface
--------------------

(see :ref:`high-level-reference` for an API reference)
