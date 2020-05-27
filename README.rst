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

Use guidelines
--------------

We encourage use of OpenSCM as much as possible and are open to collaboration.
If you plan to publish using OpenSCM, please be respectful of the work and the `Maintainers`_' willingness to open source their code.

In particular, when using OpenSCM, please cite the DOI of the precise version of the package used and consider citing our package description paper [when it's written, which it's not yet :)].
As appropriate, please consider also citing the wrappers and models that OpenSCM relies on.
A way to cite OpenSCM alongside the references to the wrappers and original models can be found in the documentation and are available in bibtex format in the ``CITATION`` file.

Of course, there is a balance, and no single rule will fit all situations.
If in doubt, don't hestiate to contact the `Maintainers`_ and ask.

Maintainers
-----------

Current maintainers of OpenSCM are:

-  `Robert Gieseke <http://github.com/rgieseke>`__
   <`robert.gieseke@pik-potsdam.de
   <mailto:robert.gieseke@pik-potsdam.de>`__>
-  `Jared Lewis <http://github.com/lewisjared>`__
   <`jared.lewis@climate-energy-college.org
   <mailto:jared.lewis@climate-energy-college.org>`__>
-  `Zebedee Nicholls <http://github.com/znicholls>`__
   <`zebedee.nicholls@climate-energy-college.org
   <mailto:zebedee.nicholls@climate-energy-college.org>`__>
-  `Sven Willner <http://github.com/swillner>`__
   <`sven.willner@pik-potsdam.de
   <mailto:sven.willner@pik-potsdam.de>`__>

.. sec-end-index

Documentation
-------------

Detailed documentation is available at `ReadTheDocs
<https://openscm.readthedocs.io/en/latest/>`_.

.. sec-end-long-description

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

To be written

.. sec-end-quickstart
.. sec-begin-development

Development
-----------

.. code:: bash

    git clone git@github.com:openclimatedata/openscm.git
    pip install -e .

Tests can be run locally with

.. code:: bash

    pytest tests/

.. sec-end-development

.. |Docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
    :target: https://openscm.readthedocs.io/en/latest/
.. |WIP| image:: https://img.shields.io/badge/state-work%20in%20progress-red.svg?style=flat
    :target: https://github.com/openclimatedata/openscm/milestone/1
.. |GithubActions| image:: https://wdp9fww0r9.execute-api.us-west-2.amazonaws.com/production/badge/openclimatedata/openscm?style=flat
    :target: https://github.com/openclimatedata/openscm/actions
