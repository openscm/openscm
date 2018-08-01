"""OpenSCM
"""

import versioneer

from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="openscm",
    version=versioneer.get_version(),
    description="An API for Simple Climate Models",
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
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    keywords="simple climate model",
    packages=["openscm"],
    install_requires=["pymagicc", "fair", "pyhector"],
    project_urls={  # Optional
        "Bug Reports": "https://github.com/openclimatedata/openscm/issues",
        "Source": "https://github.com/openclimatedata/openscm/",
    },
)
