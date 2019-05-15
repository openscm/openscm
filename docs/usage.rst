Usage
=====

OpenSCM defines and provides three interfaces for its users.

Its **core** interface is targeted to users who want to include
OpenSCM in their model, e.g. integrated assessment modellers. This
interface provides the basic functionality necessary to run all SCMs
included in OpenSCM. It includes functions for getting and setting
parameters as well as to run and reset the model. Additionally, it
allows for reading and writing parameters from and to standardized and
other file formats, including whole scenario definitions.

The **ensemble** interface provides functions for running ensembles of
model runs.

TODO describe ScmDataFrame

The **command line** interface lets users run models with specified
parameters and model input directly from the command line without
coding themselves. Please see :ref:`tool` for its usage documentation.


.. _parameters:

Parameters
----------

.. _parameter-hierarchy:

**Parameter** here refers to any named input or output variable of a
model. A parameter can be either scalar (i.e. a single number), a
timeseries, a boolean, or a string value and has a unique name in a
hierarchy of arbitrary depth. That means every parameter either is a
root parameter without a parent parameter ("level 0") or belongs to a
parent parameter.

For example, the parameter for industrial carbon emissions belongs to
the parameter for carbon emissions, which in turn belongs to the root
parameter emission. Thus, it is identified by

    ``Emissions`` -> ``CO2`` -> ``Industrial``.

In the core API parameters are expected to be identified by tuples of
strings describing their position in the hierarchy, i.e. in this
example ``("Emissions", "CO2", "Industrial")``.

See :ref:`standard-parameters` for the standard parameters in OpenSCM.


.. _timeframes:

Time frames
-----------

Timeseries parameters are always given with a corresponding time
frame, which consists of a time point and a period length. The time
point gives the start of the timeseries; the period length gives the
length of the period between consecutive values in the timeseries.
Each parameter value is assumed to be the **average value** for its
corresponding period. This implies that values that are not averages
but, for instance, absolute values, such as emissions need to be given
as a rate, e.g. ``tC/a`` rather than ``tC``.

In the core API time points are given in seconds since ``1970-01-01
00:00:00``, while time period lengths are specified in seconds.


Main interface
--------------

(see :ref:`core-reference` for an API reference)

Setting up a model run
**********************

A model run is represented by a :class:`openscm.OpenSCM` object
specifying the underlying SCM and start and end time:

.. code:: python

    from openscm import OpenSCM
    from datetime import datetime, timedelta

    start_time = datetime(2006, 1, 1).timestamp()
    year_seconds = timedelta(365).total_seconds()
    # use year_seconds since models do not account for leap years
    stop_time = start_time + (2100 - 2006) * year_seconds
    model_run = OpenSCM("DICE", start_time, stop_time)

.. _get-set-parameters:

Setting input parameters
************************

In the core API parameters are get and set through subclasses of
:class:`openscm.parameter_views.ParameterView`. While the values of
the parameters are stored internaly, a
:class:`openscm.parameter_views.ParameterView` provides an (always
up-to-date) "view" of the corresponding parameter and will always
return the parameter values in a specific unit and, in the case of
timeseries, a specific time frame.

Unit and time frame have to be specified when requesting a
:class:`~openscm.parameter_views.ParameterView` from the
:class:`~openscm.OpenSCM`'s :class:`~openscm.core.ParameterSet`
property called ``parameters`` using one of the following functions:

- :func:`~openscm.core.ParameterSet.get_scalar_view` returns a
  read-only view to a scalar ("number") parameter
  (:class:`~openscm.parameter_views.ScalarView`)
- :func:`~openscm.core.ParameterSet.get_writable_scalar_view` returns
  a writable view to a scalar ("number") parameter
  (:class:`~openscm.parameter_views.WritableScalarView`)
- :func:`~openscm.core.ParameterSet.get_timeseries_view` returns a
  read-only view to a timeseries parameter
  (:class:`~openscm.parameter_views.TimeseriesView`)
- :func:`~openscm.core.ParameterSet.get_writable_timeseries_view`
  returns a writable view to a timeseries parameter
  (:class:`~openscm.parameter_views.WritableTimeseriesView`)
- :func:`~openscm.core.ParameterSet.get_boolean_view` returns a
  read-only view to a boolean parameter
  (:class:`~openscm.parameter_views.BooleanView`)
- :func:`~openscm.core.ParameterSet.get_writable_boolean_view` returns
  a writable view to a boolean parameter
  (:class:`~openscm.parameter_views.WritableBooleanView`)
- :func:`~openscm.core.ParameterSet.get_string_view` returns a
  read-only view to a string parameter
  (:class:`~openscm.parameter_views.StringView`)
- :func:`~openscm.core.ParameterSet.get_writable_string_view` returns
  a writable view to a string parameter
  (:class:`~openscm.parameter_views.WritableStringView`)

Each of these functions take the hierarchical name of the parameter
(as described under :ref:`parameters`) and, in a similar fashion, the
hierarchical name of the region it applies to. The "root" region, i.e.
the region of which all others are subregions and which applies to
parameters for all regions, is by default named ``"World"``.

Values can be get and set using ``get`` and ``set``, respectively.
Conversion, if necessary, is done internally by the object. There is
no standard for the unit and time frame for internal storage, but
those of the first :class:`openscm.parameter_views.ParameterView`
requested are used. If a scalar view for a time series is requested
(or vice-versa), or if the units are not convertible, an error is
raised.

:class:`~openscm.parameter_views.ParameterView` objects also convert
between hierarchical levels if possible: a view to a higher level
parameter yields the sum of its child parameters. This implies that,
once a *writable* view to a parameter is requested, there cannot be a
view to one of its children. Otherwise consistency cannot be
guaranteed, so an error is raised. The same holds if a child parameter
has already been set and the user tries to set values for one of its
parent parameters. A similar logic applies to the hierarchy of
regions.

Using :class:`~openscm.parameter_views.ParameterView` as proxy objects
rather than directly setting/returning parameter values allows for
efficient parameter handling in the expected units and time frames
without specifying these for each value (e.g. seeting a timeseries
step-wise would create large overhead).

.. code:: python

    climate_sensitivity = model_run.parameters.get_writable_scalar_view(
        ("Equilibrium Climate Sensitivity",), ("World",), "degC"
    )
    climate_sensitivity.set(3)

    carbon_emissions_raw = [10 for _ in range(2100 - 2006)]
    time_points = create_time_points(
        start_time,
        year_seconds,
        len(carbon_emissions_raw),
        ParameterType.AVERAGE_TIMESERIES,
    )
    carbon_emissions = model_run.parameters.get_writable_timeseries_view(
        ("Emissions", "CO2"),
        ("World",),
        "GtCO2/a",
        time_points,
        ParameterType.AVERAGE_TIMESERIES,
        InterpolationType.LINEAR,
        ExtrapolationType.NONE,
    )
    carbon_emissions.set(carbon_emissions_raw)

Running the model
*****************

The model is simply run by calling the :func:`~openscm.OpenSCM.run`
function:

.. code:: python

    model_run.run()

This tells the adapter for the particular SCM to get the necessary
parameters in the format as expected by the model, while conversion
for units and time frames is done by the corresponding
:class:`openscm.parameter_views.ParameterView` objects. It then runs
the model itself.

After the run the model is reset, so the
:func:`~openscm.OpenSCM.run` function can be called again (setting
parameters to new values before, if desired).

Getting output parameters
*************************

During the run the model adapter sets the output parameters just like
the input parameters were set above. Thus, these can be read using
read-only :class:`~openscm.parameter_views.ParameterView` objects:

.. code:: python

    gmt = model_run.parameters.get_timeseries_view(
        ("Temperature", "Surface"), ("World",), "degC", start_time, year_seconds
    )
    print(gmt.get())
