from .scmdataframebase import ScmDataFrameBase


class ScmDataFrame(ScmDataFrameBase):
    """OpenSCM's custom DataFrame implementation.

    The ScmDataFrame wraps around `pyam`'s IamDataFrame, which itself wraps around Pandas.

    The ScmDataFrame provides a number of diagnostic features (including validation
    of data, completeness of variables provided, running of simple climate models)
    as well as a number of visualization and plotting tools.
    """

    pass
