OpenSCM
=======

+----------------+-----------+--------+
| |Build Status| | |Codecov| | |Docs| |
+----------------+-----------+--------+

.. sec-begin-index

The **Open Simple Climate Model framework** unifies access to several
simple climate models (SCMs). It defines a standard interface for
getting and setting model parameters, input and output data as well as
for running the models. Additionally, OpenSCM provides a standardized
file format for these parameters and scenarios including functions for
reading and writing such files. Its high-level interface further adds
convenience functions and easily enables stochastic ensemble runs,
e.g. for model tuning.

OpenSCM comes with a command line tool ``openscm``.

.. sec-end-index
.. sec-begin-installation

Installation
------------

.. sec-end-installation
.. sec-begin-quickstart

Quickstart
----------

.. sec-end-quickstart

Documentation
-------------

Detailed documentation is given on `ReadTheDocs <https://openscm.readthedocs.io/en/latest/>`_.

.. sec-begin-development

Development
-----------

.. code:: bash

    git clone git@github.com:openclimatedata/openscm.git
    pip install -e .

Tests can be run locally with

::

    python setup.py test

.. sec-end-development

.. |Build Status| image:: https://img.shields.io/travis/openclimatedata/openscm.svg
    :target: https://travis-ci.org/openclimatedata/openscm
.. |Docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
    :target: https://openscm.readthedocs.io/en/latest/
.. |Codecov| image:: https://img.shields.io/codecov/c/github/openclimatedata/openscm.svg
    :target: https://codecov.io/gh/openclimatedata/openscm
