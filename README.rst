OpenSCM
=======

|WIP| |Docs| |GithubActions|

.. sec-begin-long-description
.. sec-begin-index

**Warning: OpenSCM is still work in progress and cannot be fully used
yet! However, we are very grateful for suggestions and critique on how
you would like to use this framework. Please have a look at the issues
and feel free to create new ones/upvote ones that would really help
you.**

The **Open Simple Climate Model framework** unifies access to several
simple climate models (SCMs). It defines a standard interface for
getting and setting model parameters, input and output data as well as
for running the models. Additionally, OpenSCM provides a standardized
file format for these parameters and scenarios including functions for
reading and writing such files. It further adds convenience functions
and easily enables ensemble runs, e.g. for scenario assessment or
model tuning.

This OpenSCM implementation comes with a command line tool
``openscm``.

.. sec-end-index

Documentation
-------------

Detailed documentation is given on `ReadTheDocs
<https://openscm.readthedocs.io/en/latest/>`_.

.. sec-end-long-description

Schema
------

.. image:: docs/static/schema_small.png
    :align: center

.. sec-begin-installation

Installation
------------

To install OpenSCM run

.. code:: bash

    pip install openscm

If you also want to run the example notebooks install additional
dependencies using

.. code:: bash

    pip install openscm[notebooks]

OpenSCM comes with model adapters only for some very simple SCMs. If
you want to run other models, you will also need to install their
dependencies (see `ReadTheDocs
<https://openscm.readthedocs.io/en/latest/models.html>`_ for a list).

.. sec-end-installation
.. sec-begin-quickstart

Quickstart
----------

.. sec-end-quickstart
.. sec-begin-development

Development
-----------

.. code:: bash

    git clone git@github.com:openclimatedata/openscm.git
    pip install -e .

Tests can be run locally with

.. code:: bash

    python setup.py test

.. sec-end-development

Maintainers
-----------

Current maintainers of OpenSCM are:

-  `Robert Gieseke <http://github.com/rgieseke>`__
   <`robert.gieseke@pik-potsdam.de
   <mailto:robert.gieseke@pik-potsdam.de>`__>
-  `Zeb Nicholls <http://github.com/znicholls>`__
   <`zebedee.nicholls@climate-energy-college.org
   <mailto:zebedee.nicholls@climate-energy-college.org>`__>
-  `Sven Willner <http://github.com/swillner>`__
   <`sven.willner@pik-potsdam.de
   <mailto:sven.willner@pik-potsdam.de>`__>

.. |Docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
    :target: https://openscm.readthedocs.io/en/latest/
.. |WIP| image:: https://img.shields.io/badge/state-work%20in%20progress-red.svg?style=flat
    :target: https://github.com/openclimatedata/openscm/milestone/1
.. |GithubActions| image:: https://wdp9fww0r9.execute-api.us-west-2.amazonaws.com/production/badge/openclimatedata/openscm?style=flat
    :target: https://github.com/openclimatedata/openscm/actions
