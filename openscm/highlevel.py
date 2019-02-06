"""
The OpenSCM high-level API provides high-level functionality around
single model runs.  This includes reading/writing input and output
data, easy setting of parameters and stochastic ensemble runs.
"""
from .core import Core
from .scmdataframebase import ScmDataFrameBase


class OpenSCM(Core):
    """
    High-level OpenSCM class.

    Represents model runs with a particular simple climate model.
    """

    pass


class ScmDataFrame(ScmDataFrameBase):
    """OpenSCM's custom DataFrame implementation.

    The ScmDataFrame implements a subset of the functionality provided by `pyam`'s
    IamDataFrame, but is focused on the providing a performant way of storing
    time series data and the metadata associated with those time series.

    The ScmDataFrame provides a number of diagnostic features (including validation
    of data, completeness of variables provided, running of simple climate models)
    as well as a number of visualization and plotting tools.
    """

    pass
