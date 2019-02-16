.. include:: ../README.rst
    :start-after: sec-begin-development
    :end-before: sec-end-development

.. _writing-adapters:

Writing a model adapter
=======================

Creating an :class:`~openscm.adapter.Adapter` subclass
******************************************************

Create your adapter source file in ``openscm/adapters/``, e.g.
``myadapter.py``, and subclass the :class:`openscm.adapter.Adapter`
class:

.. code:: python

    from ..adapter import Adapter
    
    YEAR = 365 * 24 * 60 * 60  # example time step length as used below

    class MyAdapter(Adapter):

Implement the relevant methods (or just do ``pass`` if you do not
need to do anything in the particular method). An adapter should not
use parts of OpenSCM other than ``ParameterSet``.

- The :func:`~openscm.adapter.Adapter._initialize_model` method
  initializes the adapter and is called only once. It should set the
  default values of mandatory *model-specific* (not :ref:`standard
  OpenSCM parameters <standard-parameters>`!) parameters in the in the
  :class:`~openscm.core.ParameterSet`
  :attr:`~openscm.adapter.Adapter._parameter`. The :ref:`hierarchical
  names <parameter-hierarchy>` of these model-specific parameters should
  start with the model/adapter name (as you set it in the model registry,
  see below).

  .. code:: python

      def _initialize_model(self) -> None:
          # TODO Initialize the model
          # TODO Set default parameter values:
          self._parameters.get_writable_scalar_view(
              ("MyModel", "Specific Parameter"), ("World",), "Unit"
          ).set(DEFAULT_VALUE)

- The :func:`~openscm.adapter.Adapter._initialize_run_parameters`
  method initializes a particular run. It is called before the
  adapter is used in any way and at most once before a call to
  :func:`~openscm.adapter.Adapter.run` or
  :func:`~openscm.adapter.Adapter.step`.

  .. code:: python

      def _initialize_run_parameters(self) -> None:
          """
          TODO Initialize run parameters by reading model parameters
          from `self._parameters` (see below).
          """

  The adapter should later use the start and stop time of the run as
  stored in the
  :attr:`self._start_time <openscm.adapter.Adapter._start_time>` and
  :attr:`self._stop_time <openscm.adapter.Adapter._stop_time>`
  attributes.

- The :func:`~openscm.adapter.Adapter._initialize_model_input` method
  initializes the input and model parameters of a particular run. It
  is also called before the adapter is used in any way and at most
  once before a call to :func:`~openscm.adapter.Adapter.run` or
  :func:`~openscm.adapter.Adapter.step`.

  This and the
  :func:`~openscm.adapter.Adapter._initialize_run_parameters` method
  are separated for higher efficiency when doing ensemble runs for
  models that have additional overhead for changing drivers/scenario setup.

  .. code:: python

      def _initialize_model_input(self) -> None:
          """
          TODO Initialize model input by reading input parameters
          from ``self._parameters`` (see below).
          """

- The :func:`~openscm.adapter.Adapter._reset` method resets the model
  to prepare for a new run. It is called once after each call of
  :func:`~openscm.adapter.Adapter.run` and to reset the model after
  several calls to :func:`~openscm.adapter.Adapter.step`.

  .. code:: python

      def _reset(self) -> None:
          # TODO Reset the model

- The :func:`~openscm.adapter.Adapter._run` method runs the model
  over the full time range (as given by the times set by the
  previous call to
  :func:`~openscm.adapter.Adapter._initialize_run_parameters`). You
  should at least implement this function.

  .. code:: python

      def _run(self) -> None:
          """
          TODO Run the model and write output parameters to
          `self._parameters` (see below).
          """

- The :func:`~openscm.adapter.Adapter._step` method does a single
  time step. You can get the current time from
  :attr:`self._current_time
  <openscm.adapter.Adapter._current_time>`, which you should
  increase by the time step length and return its value. If your
  model does not support stepping just do ``raise NotImplementedError``
  here.

  .. code:: python

      def _step(self) -> None:
          """
          TODO Do a single time step and write corresponding output
          parameters to ``self._parameters`` (see below).
          """
          self._current_time += YEAR

- The :func:`~openscm.adapter.Adapter._shutdown` method cleans up the
  adapter.

  .. code:: python

      def _shutdown(self) -> None:
          # TODO Shut down model


Reading model and input parameters
**********************************

Model parameters and input data (referred to as general "parameters"
in OpenSCM) are **pulled** from the :class:`~openscm.core.ParameterSet`
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
:func:`~openscm.adapter.Adapter.step` methods you should directly
write the relevant output parameters. Refer to the
:ref:`low-level-interface` documentation on how to do so.


Adding the adapter to the model registry
****************************************

Once ready, add a lookup for your adapter in
``openscm/adapters/__init__.py`` (where marked in the file) according
to:

.. code:: python

    elif name == "MyAdapter":
        from .myadapter import MyAdapter

        adapter = MyAdapter


Writing adapter tests
*********************

To ensure your adapter is properly tested, create a file
``test_myadapter.py`` in ``tests/adapters/`` and subclass the
``AdapterTester`` (this ensures that some standard tests are run on
your adapter). Tests are done using `pytest
<https://docs.pytest.org/en/latest>`__ on all methods starting
with ``test_``. Only pull requests with adapters with full test
coverage will be merged (see, for instance, the coverage on the end
of the PR page).

.. code:: python

    from . import AdapterTester
    from openscm.adapters.myadapter import MyAdapter


    class TestMyAdapter(AdapterTester):
        tadapter = MyAdapter
        
        test_my_special_feature(self, test_adapter):
            # TODO do tests on ``test_adapter``


Additional module dependencies
******************************

If your adapter needs additional dependencies add them to the
``REQUIREMENTS_MODELS`` dictionary in ``setup.py`` (see comment
there).


.. include:: ../CONTRIBUTING.rst

.. _code_of_conduct:

.. include:: ../CODE_OF_CONDUCT.rst
