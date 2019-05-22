"""
Scenarios included as part of OpenSCM.

These are limited to a select few scenarios which are widely used. They are included
because they provide a useful set of scenarios with which we can run tests, make
example notebooks and allow users to easily get started.
"""
from os.path import dirname, join, realpath

from ..scmdataframe import df_append

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
