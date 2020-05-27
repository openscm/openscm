.. include:: ../README.rst
    :start-after: sec-begin-development
    :end-before: sec-end-development

.. _writing-adapters:

Writing a model adapter
=======================

Writing adapter tests
*********************

To help ensure your adapter works as intended, we provide a number of
standard tests. To run these, create a file ``test_myadapter.py`` in
``tests/adapters/`` and subclass the ``AdapterTester`` (this ensures
that the standard tests are run on your adapter). Tests are done using
`pytest <https://docs.pytest.org/en/latest>`__ on all methods starting
with ``test_``. Only pull requests with adapters with full test
coverage will be merged (see, for instance, the coverage on the end of
the PR page).

.. code:: python

    # tests/adapters/test_myadapter.py

    from openscm.adapters.myadapter import MyAdapter

    from base import _AdapterTester


    class TestMyAdapter(_AdapterTester):
        tadapter = MyAdapter

        # if necessary, you can extend the tests e.g.
        def test_run(self, test_adapter, test_run_parameters):
            super().test_run(test_adapter, test_run_parameters)
            # TODO some specific test of your adapter here

        def test_my_special_feature(self, test_adapter):
            # TODO test some special feature of your adapter class


Creating an :class:`~openscm.adapter.Adapter` subclass
******************************************************

Create your adapter source file in ``openscm/adapters/``, e.g.
``myadapter.py``, and subclass the :class:`openscm.adapter.Adapter`
class:

.. code:: python

    # openscm/adapters/myadapter.py

    from ..adapter import Adapter

    YEAR = 365 * 24 * 60 * 60  # example time step length as used below

    class MyAdapter(Adapter):

Implement the relevant methods (or just do ``pass`` if you do not need
to do anything in the particular method).

- The :func:`~openscm.adapter.Adapter._initialize_model` method
  initializes the adapter and is called only once just before the
  first call to the functions below initializing the first run.

- The :func:`~openscm.adapter.Adapter._initialize_run_parameters`
  method initializes a particular run. It is called before the adapter
  is used in any way and at most once before a call to
  :func:`~openscm.adapter.Adapter._run`.

  .. code:: python

      def _initialize_run_parameters(self) -> None:
          """
          TODO Initialize run parameters by reading model parameters
          from `self._parameters` (see below).
          """

- The :func:`~openscm.adapter.Adapter._initialize_model_input` method
  initializes the input and model parameters of a particular run. It
  is also called before the adapter is used in any way and at most
  once before a call to :func:`~openscm.adapter.Adapter._run`.

  This and the
  :func:`~openscm.adapter.Adapter._initialize_run_parameters` method
  are separated for higher efficiency when doing ensemble runs for
  models that have additional overhead for changing drivers/scenario
  setup.

  .. code:: python

      def _initialize_model_input(self) -> None:
          """
          TODO Initialize model input by reading input parameters from
          :class:`self._parameters
          <~openscm.adapter.Adapter._parameters>` (see below).
          """

- The :func:`~openscm.adapter.Adapter._reset` method resets the model
  to prepare for a new run. It is called once after each call of
  :func:`~openscm.adapter.Adapter._run`.

  .. code:: python

      def _reset(self) -> None:
          # TODO Reset the model

- The :func:`~openscm.adapter.Adapter._run` method runs the model for the given inputs
  (in future we aim to add a :func:`~openscm.adapter.Adapter._step` method that will
  allow models to be run a single timestep at a time). All adapters must implement
  this function.

  .. code:: python

      def _run(self) -> None:
          """
          TODO Run the model and return output (see below).
          """

- The :func:`~openscm.adapter.Adapter._shutdown` method cleans up the
  adapter.

  .. code:: python

      def _shutdown(self) -> None:
          # TODO Shut down model


Reading model and input parameters and writing output
*****************************************************

TODO: re-write


Adding the adapter to the model registry
****************************************

Once done with your implementation, add a lookup for your adapter in
``openscm/adapters/__init__.py`` (where marked in the file) according
to:

.. code:: python

    elif name == "MyAdapter":
        from .myadapter import MyAdapter

        adapter = MyAdapter

(make sure to set ``adapter`` to your *class* not an instance of your
adapter)


Additional module dependencies
******************************

If your adapter needs additional dependencies add them to the
``REQUIREMENTS_MODELS`` dictionary in ``setup.py`` (see comment
there).

.. include:: ../CONTRIBUTING.rst


Creating a release
==================

OpenSCM uses designated Github Actions to upload the package to PyPI
(and, in the future, also to Conda). To create a release:

1. Change the "master" header in ``CHANGELOG.rst`` to the release
   version number (not starting with "v", e.g. "1.2.3") and create a
   new, empty "master" header above. Commit these changes with the
   message, "Prepare for release of vVERSIONNUMBER'' e.g. "Prepare for
   release of v1.2.3".
2. Tag the commit as "vVERSIONNUMBER", e.g. "v1.2.3", on the "master"
   branch. Push the tag.
3. The Github Actions workflow should now create a release with the
   corresponding description in ``CHANGELOG.rst`` and upload the
   release to PyPI.


.. _code_of_conduct:

.. include:: ../CODE_OF_CONDUCT.rst
