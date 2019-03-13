# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "../openscm"))
from _version import get_versions  # isort:skip # append path before


# -- Project information -----------------------------------------------------

project = "OpenSCM"
copyright = "2018, Robert Gieseke, Zebedee Nicholls, Sven Willner"
author = "Robert Gieseke, Zebedee Nicholls, Sven Willner"
version = get_versions()["version"]  # The short X.Y version
release = version  # The full version, including alpha/beta/rc tags


# -- General configuration ---------------------------------------------------

exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]
language = "en"
master_doc = "index"
needs_sphinx = "1.4"
# nitpicky = True  # warn about all references where the target cannot be found
nitpick_ignore = [
    ("py:class", "Exception"),
    ("py:class", "bool"),
    ("py:class", "enum.Enum"),
    ("py:class", "float"),
    ("py:class", "int"),
    ("py:class", "np.ndarray"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "object"),
    ("py:class", "str"),
    ("py:class", "typing.Any"),
    ("py:class", "typing.Callable"),
    ("py:class", "typing.Optional"),
    ("py:class", "typing.Tuple"),
    ("py:class", "typing.Union"),
    ("py:exc", "KeyError"),
    ("py:exc", "ValueError"),
]
pygments_style = "sphinx"
source_suffix = ".rst"  # ['.rst', '.md']
templates_path = ["templates"]


def skip_init(app, what, name, obj, skip, options):
    if name == "__init__":
        return False
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_init)


# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["static"]
html_context = {
    "display_github": False,
    "github_user": "openclimatedata",
    "github_repo": "openscm",
    "github_version": "master",
    "conf_py_path": "/docs/",
}


# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "OpenSCMdoc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {}
# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "OpenSCM.tex",
        "OpenSCM Documentation",
        "Robert Gieseke, Zebedee Nicholls, Sven Willner",
        "manual",
    )
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "openscm", "OpenSCM Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "OpenSCM",
        "OpenSCM Documentation",
        author,
        "OpenSCM",
        "A unifying interface for Simple Climate Models.",
        "Miscellaneous",
    )
]


# -- Extension configuration -------------------------------------------------

autodoc_default_flags = [ # TODO deprecated since 1.8
    "inherited-members",
    "members",
    "private-members",
    "show-inheritance",
    "undoc-members",
]
coverage_write_headline = False  # do not write headlines.
napoleon_google_docstring = False
napoleon_numpy_docstring =  True
