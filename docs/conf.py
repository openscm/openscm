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
import warnings

from sphinx.ext.napoleon import NumpyDocstring

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "../openscm"))
from _version import get_versions  # isort:skip # append path before


# -- Project information -----------------------------------------------------

project = "OpenSCM"
copyright = "2018-2019, Robert Gieseke, Zebedee Nicholls, Sven Willner"
author = "Robert Gieseke, Zebedee Nicholls, Sven Willner"
version = get_versions()["version"]  # The short X.Y version
release = version  # The full version, including alpha/beta/rc tags


# -- General configuration ---------------------------------------------------

exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",  # must be after sphinx.ext.napoleon
]
language = "en"
master_doc = "index"
needs_sphinx = "1.8"
nitpick_ignore = [
    ("py:class", "Callable"),
    ("py:class", "Optional"),
    ("py:class", "Sequence"),
    ("py:class", "Union"),
    ("py:class", "np.ndarray"),
    ("py:class", "typing.Tuple"),
    ("py:class", "typing.Union"),
    ("py:exc", "pint.errors.DimensionalityError"),
    ("py:exc", "pint.errors.UndefinedUnitError"),
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

autodoc_default_options = {
    "inherited-members": None,
    "members": None,
    "private-members": None,
    "show-inheritance": None,
    "undoc-members": None,
}
coverage_write_headline = False  # do not write headlines.
intersphinx_mapping = {
    "numpy": ("https://docs.scipy.org/doc/numpy", None),
    "python": ("https://docs.python.org/3", None),
    # "pint": ("https://pint.readthedocs.io/en/latest", None), # no full API doc here, unfortunately
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True
set_type_checking_flag = True


# -- Hack to get return type automatically filled from type hints -----------

def _openscm_consume_returns_section(self):
    self._consume_empty()
    fields = []
    while not self._is_section_break():
        _name, _type, _desc = self._consume_returns_field()
        if _name or _type or _desc:
            fields.append((_name, _type, _desc,))

    return fields

def _openscm_consume_returns_field(self):
    lines = self._consume_to_next_section()
    indented_bit = any([self._get_indent(l) for l in lines])
    if indented_bit:
        _type, _, _name = self._partition_field_on_colon(lines[0])
        _desc = self._dedent(lines[1:])
    else:
        _name, _type = '', ''
        _desc = self._dedent(lines)

    _name, _type = _name.strip(), _type.strip()
    _desc = self.__class__(_desc, self._config).lines()

    return _name, _type, _desc

NumpyDocstring._consume_returns_section = _openscm_consume_returns_section
NumpyDocstring._consume_returns_field = _openscm_consume_returns_field
