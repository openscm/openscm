OpenSCM
=======

The **Open Simple Climate Model framework** unifies access to several
simple climate models (SCMs). It defines a standard interface for
getting and setting model parameters, input and output data as well as
for running the models. Additionally, OpenSCM provides a standardized
file format for these parameters and scenarios including functions for
reading and writing such files. Its high-level interface further adds
convenience functions and easily enables stochastic ensemble runs,
e.g. for model tuning.

OpenSCM comes with a command line tool :doc:`openscm </tool>`.


Schema
------

.. image:: static/schema.png
   :width: 650px
   :align: center


Contents
--------

.. toctree::
   :maxdepth: 3

   usage
   tool
   lowlevel
   highlevel
   internals


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
