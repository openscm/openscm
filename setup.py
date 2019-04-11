"""
OpenSCM
-------

A unifying interface for Simple Climate Models.
"""

import versioneer
from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand

NAME = "openscm"
SHORT_DESCRIPTION = "A unifying interface for Simple Climate Models"
KEYWORDS = ["simple climate model"]
AUTHORS = [
    ("Robert Gieseke", "robert.gieseke@pik-potsdam.de"),
    ("Jared Lewis", "jared.lewis@climate-energy-college.org"),
    ("Zeb Nicholls", "zebedee.nicholls@climate-energy-college.org"),
    ("Sven Willner", "sven.willner@pik-potsdam.de"),
]
URL = "https://github.com/openclimatedata/openscm"
PROJECT_URLS = {
    "Bug Reports": "https://github.com/openclimatedata/openscm/issues",
    "Documentation": "https://openscm.readthedocs.io/en/latest",
    "Source": "https://github.com/openclimatedata/openscm",
}
LICENSE = "GNU Affero General Public License v3.0 or later"
CLASSIFIERS = [
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
]
REQUIREMENTS_INSTALL = ["numpy", "scipy", "pint", "pandas", "python-dateutil"]
REQUIREMENTS_NOTEBOOKS = [
    "matplotlib",
    "notebook",
    "seaborn",
    # TODO use pypi version
    "pyam-iamc @ git+https://github.com/IAMconsortium/pyam.git@a6ac0c574c3d5363fb8dba7d8b42a845fd3dbe87",
]
REQUIREMENTS_TESTS = ["codecov", "nbval", "pytest", "pytest-cov"]
REQUIREMENTS_DOCS = ["sphinx>=1.8", "sphinx_rtd_theme", "sphinx-autodoc-typehints"]
REQUIREMENTS_DEPLOY = ["setuptools>=38.6.0", "twine>=1.11.0", "wheel>=0.31.0"]
REQUIREMENTS_DEV = (
    ["black", "bandit", "coverage", "flake8", "isort", "mypy", "pydocstyle", "pylint"]
    + REQUIREMENTS_NOTEBOOKS
    + REQUIREMENTS_TESTS
    + REQUIREMENTS_DOCS
    + REQUIREMENTS_DEPLOY
)


"""
When implementing an additional adapter, include your adapter NAME here as:
```
"NAME": [ ... additional pip modules you need ... ],
```
"""
REQUIREMENTS_MODELS = {}

REQUIREMENTS_EXTRAS = {
    "notebooks": REQUIREMENTS_NOTEBOOKS,
    "docs": REQUIREMENTS_DOCS,
    "tests": REQUIREMENTS_TESTS,
    "deploy": REQUIREMENTS_DEPLOY,
    "dev": REQUIREMENTS_DEV,
}

for k, v in REQUIREMENTS_MODELS.items():
    REQUIREMENTS_EXTRAS["model-{}".format(k)] = v
    REQUIREMENTS_EXTRAS["tests"] += v
    REQUIREMENTS_EXTRAS["dev"] += v

PACKAGE_DATA = {"openscm": ["units/*.csv"]}

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
    name=NAME,
    version=versioneer.get_version(),
    description=SHORT_DESCRIPTION,
    long_description="\n".join(README_LINES),
    long_description_content_type="text/x-rst",
    keywords=KEYWORDS,
    author=", ".join([author[0] for author in AUTHORS]),
    author_email=", ".join([author[1] for author in AUTHORS]),
    url=URL,
    project_urls=PROJECT_URLS,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    packages=find_packages(exclude=["tests"]),
    package_data=PACKAGE_DATA,
    install_requires=REQUIREMENTS_INSTALL,
    extras_require=REQUIREMENTS_EXTRAS,
    cmdclass=CMDCLASS,
)
