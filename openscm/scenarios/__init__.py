"""
Scenarios included as part of OpenSCM.

These are largely limited to a select few scenarios which are widely used. This module
also provides a useful set of scenarios with which we can run tests and make example
notebooks.
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
