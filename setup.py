"""
OpenSCM
-------

A unifying interface for Simple Climate Models.

"""

import versioneer

from setuptools import setup
from setuptools.command.test import test as TestCommand


REQUIREMENTS = [
    "numpy",
    "pint",
    "pandas",
    # TODO can be moved into notebooks dependencies once Jared's new backend is in place
    "pyam-iamc @ git+https://github.com/IAMconsortium/pyam.git@master",
]
REQUIREMENTS_NOTEBOOKS = ["matplotlib", "notebook", "seaborn"]
REQUIREMENTS_TESTS = ["codecov", "nbval", "pytest", "pytest-cov"]
REQUIREMENTS_DOCS = ["sphinx>=1.4", "sphinx_rtd_theme", "sphinx-autodoc-typehints"]
REQUIREMENTS_DEPLOY = ["setuptools>=38.6.0", "twine>=1.11.0", "wheel>=0.31.0"]

REQUIREMENTS_EXTRAS = {
    "notebooks": REQUIREMENTS_NOTEBOOKS,
    "docs": REQUIREMENTS_DOCS,
    "tests": REQUIREMENTS_TESTS,
    "deploy": REQUIREMENTS_DEPLOY,
    "dev": [
        *["flake8", "black"],
        *REQUIREMENTS_NOTEBOOKS,
        *REQUIREMENTS_TESTS,
        *REQUIREMENTS_DOCS,
        *REQUIREMENTS_DEPLOY,
    ],
}

REQUIREMENTS_MODELS = {}
"""
When implementing an additional adapter, include your adapter NAME here as:
```
"NAME": [ ... additional pip modules you need ... ],
```
"""

for k, v in REQUIREMENTS_MODELS.items():
    REQUIREMENTS_EXTRAS["model-{}".format(k)] = v
    REQUIREMENTS_TESTS += v

# Get the long description from the README file
with open("README.rst", "r", encoding="utf-8") as f:
    README_LINES = ["OpenSCM", "=======", ""]
    add_line = False
    for line in f:
        if line == ".. sec-begin-long-description":
            add_line = True
        elif line == ".. sec-end-long-description":
            break
        elif add_line:
            README_LINES.append(line)


class OpenSCMTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        pytest.main(self.test_args)


CMDCLASS = versioneer.get_cmdclass()
CMDCLASS.update({"test": OpenSCMTest})

setup(
    name="openscm",
    version=versioneer.get_version(),
    description="A unifying interface for Simple Climate Models",
    long_description="\n".join(README_LINES),
    long_description_content_type="text/x-rst",
    author="Robert Gieseke, Jared Lewis, Zeb Nicholls, Sven Willner",
    author_email="robert.gieseke@pik-potsdam.de, jared.lewis@climate-energy-college.org, zebedee.nicholls@climate-energy-college.org, sven.willner@pik-potsdam.de",
    url="https://github.com/openclimatedata/openscm",
    license="GNU Affero General Public License v3.0 or later",
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
    keywords=["simple climate model"],
    packages=["openscm"],
    install_requires=REQUIREMENTS,
    extras_require=REQUIREMENTS_EXTRAS,
    cmdclass=CMDCLASS,
    project_urls={
        "Bug Reports": "https://github.com/openclimatedata/openscm/issues",
        "Documentation": "https://openscm.readthedocs.io/en/latest",
        "Source": "https://github.com/openclimatedata/openscm",
    },
)
