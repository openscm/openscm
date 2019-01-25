"""OpenSCM
"""

import versioneer

from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="openscm",
    version=versioneer.get_version(),
    description="A unifying interface for Simple Climate Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/openclimatedata/openscm",
    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
    ],
    keywords="simple climate model",
    license="GNU Affero General Public License v3.0 or later",
    packages=["openscm"],
    install_requires=[
        "numpy",
        "pint",
        "pyam-iamc @ git+https://github.com/znicholls/pyam.git@allow-custom-time-formatting"
    ],
    project_urls={
        "Bug Reports": "https://github.com/openclimatedata/openscm/issues",
        "Source": "https://github.com/openclimatedata/openscm/",
    },
    extras_require={
        "docs": ["sphinx>=1.4", "sphinx_rtd_theme", "sphinx-autodoc-typehints"],
        "tests": ["codecov", "matplotlib", "nbval", "notebook", "pytest", "pytest-cov"],
        "dev": [
            "setuptools>=38.6.0",
            "twine>=1.11.0",
            "wheel>=0.31.0",
            "black",
            "flake8",
            "pandas",
            "matplotlib",
        ],
    },
)
