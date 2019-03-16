"""
OpenSCM's custom DataFrame implementation.
"""


class ScmDataFrame():
    """
    OpenSCM's custom DataFrame implementation.

    The ScmDataFrame provides a number of diagnostic features (including validation
    of data, completeness of variables provided, running of simple climate models)
    as well as a number of visualization and plotting tools.

    It's ``to_iamdataframe`` method, once implemented, will allow the user to generate
    a ``IamDataFrame`` instance which adds additional diagnostic and analysis
    features. Such features require that the ``pyam`` library is installed.
    """
