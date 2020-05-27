"""
Scenarios included as part of OpenSCM.

These are limited to a select few scenarios which are widely used. They are included
because they provide a useful set of scenarios with which we can run tests, make
example notebooks and allow users to easily get started.

The RCP data originally came from `PIK <http://www.pik-potsdam.de/~mmalte/rcps/>`_ and
has since been re-written into a format which can be read by OpenSCM using the
`pymagicc <https://github.com/openclimatedata/pymagicc>`_ package. We are not
currently planning on importing Pymagicc's readers into OpenSCM by default, please
raise an issue `here <https://github.com/openclimatedata/openscm/issues>`_ if you
would like us to consider doing so.
"""
from os.path import dirname, join, realpath

from scmdata import df_append

_here = dirname(realpath(__file__))

"""
ScmDataFrame: RCP emissions data
"""
rcps = df_append(
    [
        join(_here, "{}_emissions.csv".format(rcp))
        for rcp in ["rcp26", "rcp45", "rcp60", "rcp85"]
    ]
)
