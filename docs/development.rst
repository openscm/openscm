.. include:: ../README.rst
    :start-after: sec-begin-development
    :end-before: sec-end-development

.. _writing-adapters:

Writing a model adapter
=======================

Creating an :class:`~openscm.adapter.Adapter` subclass
******************************************************

- Create your adapter source file in ``openscm/adapters/``, e.g.
  ``myadpater.py``

- If your adapter needs additional dependencies add them to the
  ``requirements_extras`` dictionary in ``setup.py`` (see comment
  there).

- Subclass the :class:`openscm.adapter.Adapter` class:

  .. code:: python

      from ..adapter import Adapter

      class MyAdapter(Adapter):

- Implement the relevant methods (you are not obliged to implement any
  method, but should at least implement the
  :func:`~openscm.adapter.Adapter.run` function to do the actual model
  run. An adapter should ONLY interface with the :ref:`low-level API
  <low-level-reference>`.

  Make sure to always call the original implementation of each
  implemented method using ``super.FUNCTION_NAME(PARAMETERS)`` (see
  examples below). The implemented functions should NOT accept
  additional parameters (e.g. using ``*args``/``**kwargs``).

  - The :func:`~openscm.adapter.Adapter.__init__` method initializes
    the adapter. It should set the default values of mandatory
    *model-specific* (not :ref:`standard OpenSCM parameters
    <standard-parameters>`!) parameters in the
    :class:`~openscm.core.ParameterSet`. Their :ref:`hierarchical
    names <parameter-hierarchy>` should start with the model/adapter
    name.

    .. code:: python

        def __init__(self, parameters: ParameterSet):
            super.__init__(parameters)
            # TODO Initialize the model
            # TODO Set default parameter values:
            self._parameters.get_writable_scalar_view(
                ("MyModel", "Specific Parameter"), ("World",), "Unit"
            ).set(DEFAULT_VALUE)

  - The :func:`~openscm.adapter.Adapter.initialize_run_parameters`
    method initializes a particular run. It is called before the
    adapter is used in any way and at most once before a call to
    :func:`~openscm.adapter.Adapter.run` or
    :func:`~openscm.adapter.Adapter.step`.

    .. code:: python

        def initialize_run_parameters(
            self, start_time: int, stop_time: int
        ) -> None:
            super.initialize_run_parameters(start_time, stop_time)
            """
            TODO Initialize run parameters by reading model parameters
            from `self._parameters` (see below).
            """

    You should later use the start and stop time of the run using the
    :attr:`self._start_time <openscm.adapter.Adapter._start_time>` and
    :attr:`self._stop_time <openscm.adapter.Adapter._stop_time>`
    attributes.

  - The :func:`~openscm.adapter.Adapter.initialize_model_input` method
    initializes the input and model parameters of a particular run. It
    is also called before the adapter is used in any way and at most
    once before a call to :func:`~openscm.adapter.Adapter.run` or
    :func:`~openscm.adapter.Adapter.step`.

    This and the
    :func:`~openscm.adapter.Adapter.initialize_run_parameters` method
    are separated for higher efficiency when doing ensemble runs for
    models that have additional overhead for changing run parameters.

    .. code:: python

        def initialize_model_input(self) -> None:
            super.initialize_model_input()
            """
            TODO Initialize model input by reading input parameters
            from ``self._parameters`` (see below).
            """

  - The :func:`~openscm.adapter.Adapter.reset` method resets the model
    to prepare for a new run. It is called once after each call of
    :func:`~openscm.adapter.Adapter.run` and to reset the model after
    several calls to :func:`~openscm.adapter.Adapter.step`.

    .. code:: python

        def reset(self) -> None:
            super.reset()
            # TODO Reset the model

  - The :func:`~openscm.adapter.Adapter.run` method runs the model
    over the full time range (as given by the times set by the
    previous call to
    :func:`~openscm.adapter.Adapter.initialize_run_parameters`). You
    should at least implement this function.

    .. code:: python

        def run(self) -> None:
            # Do not call ``super.run()`` as we override this method
            """
            TODO Run the model and write output parameters to
            `self._parameters` (see below).
            """

  - The :func:`~openscm.adapter.Adapter.step` method does a single
    time step. You can get the current time from
    :attr:`self._current_time
    <openscm.adapter.Adapter._current_time>`, which you should
    increase by the time step length and return its value.

    .. code:: python

        def step(self) -> int:
            # Do not call ``super.step()`` as we override this method
            """
            TODO Do a single time step and write corresponding output
            parameters to ``self._parameters`` (see below).
            """
            self._current_time += YEAR
            return self._current_time

  - The :func:`~openscm.adapter.Adapter.__del__` method cleans up the
    adapter. If you need to shutdown the model, please implement this
    method.

    .. code:: python

        def __del__(self):
            # TODO Shut down model


Reading model and input parameters
**********************************

Model parameters and input data (referred to as general "parameters"
in OpenSCM) are **pulled** from the :func:`~openscm.core.ParameterSet`
provided by the OpenSCM Core. From this, you read the parameters in
the :ref:`time frame <timeframes>` and :ref:`unit <units>` as needed
by the specific model; conversion is done interally if possible.

OpenSCM defines a :ref:`set of standard parameters
<standard-parameters>` to be shared between different SCMs. You should
use these where appropriate.

Refer to the :ref:`low-level-interface` documentation on how to do so.


Writing output parameters
*************************

In the :func:`~openscm.adapter.Adapter.run` and
:func:`~openscm.adapter.Adapter.step` function you should directly
write the relevant output parameters. Refer to the
:ref:`low-level-interface` documentation on how to do so.


Adding the adapter to the model registry
****************************************

Once ready, add a lookup for your adapter in
``openscm/adapters/__init__.py`` (where marked in the file) according
to:

.. code:: python

    elif name == "MyAdapter":
        from .MyAdapter import MyAdapter
        adapter = MyAdapter


.. include:: ../CONTRIBUTING.rst

.. _code_of_conduct:

.. include:: ../CODE_OF_CONDUCT.rst
