workflow "Continuous Integration" {
  on = "push"
  resolves = ["Coverage"]
}

action "Bandit" {
  uses = "swillner/actions/python-run@master"
  args = [
    "bandit -c .bandit.yml -r openscm"
  ]
  env = {
    PYTHON_VERSION = "3.7"
    PIP_PACKAGES = "bandit"
  }
}

action "Docs" {
  uses = "swillner/actions/python-run@master"
  args = [
    "sphinx-build -M html docs docs/build -qW", # treat warnings as errors (-W)...
    "sphinx-build -M html docs docs/build -Eqn -b coverage", # ...but not when being nitpicky (-n)
    "if [[ ! -s docs/build/html/python.txt ]]",
    "then",
    "    echo",
    "    echo \"Error: Documentation missing:"",
    "    echo",
    "    cat docs/build/html/python.txt",
    "    exit 1",
    "fi"
  ]
  env = {
    PYTHON_VERSION = "3.7"
    PIP_PACKAGES = "sphinx sphinx_rtd_theme sphinx-autodoc-typehints"
  }
}

action "Formatting" {
  uses = "swillner/actions/python-run@master"
  args = [
    "black --check openscm tests setup.py --exclude openscm/_version.py",
    "isort --check-only --recursive openscm tests setup.py",
    "pydocstyle openscm",
  ]
  env = {
    PYTHON_VERSION = "3.7"
    PIP_PACKAGES = "black isort pydocstyle"
  }
}

action "Mypy" {
  uses = "swillner/actions/python-run@master"
  args = [
    "mypy openscm"
  ]
  env = {
    PYTHON_VERSION = "3.7"
    PIP_PACKAGES = "mypy"
  }
}

action "Pylint" {
  uses = "swillner/actions/python-run@master"
  args = [
    "pylint openscm"
  ]
  env = {
    PYTHON_VERSION = "3.7"
    PIP_PACKAGES = "pylint ."
  }
  needs = ["Bandit", "Docs", "Formatting", "Mypy"]
}

action "Tests" {
  uses = "swillner/actions/python-run@master"
  args = [
    "pytest tests -r a --cov=openscm --cov-report=''",
    "pytest notebooks -r a --nbval --sanitize notebooks/tests_sanitize.cfg"
  ]
  env = {
    PYTHON_VERSION = "3.7"
    PIP_PACKAGES = ".[tests]"
  }
  needs = ["Pylint"]
}

action "Coverage" {
  uses = "swillner/actions/python-run@master"
  args = [
    "if ! coverage report --fail-under=\"$MIN_COVERAGE\ --show-missing"",
    "then",
    "    echo",
    "    echo \"Error: Coverage has to be at least ${MIN_COVERAGE}%\"",
    "    exit 1",
    "fi"
  ]
  env = {
    PYTHON_VERSION = "3.7"
    MIN_COVERAGE = "100"
    PIP_PACKAGES = "coverage"
  }
  needs = ["Tests"]
}


workflow "Deployment" {
  on = "create"
  resolves = ["Create release"]
}

action "Filter tag" {
  uses = "actions/bin/filter@master"
  args = "tag 'v*'"
}

action "Publish on PyPi" {
  uses = "swillner/actions/python-run@master"
  args = [
    "rm -rf build dist",
    "python setup.py sdist",
    "twine upload dist/*"
  ]
  env = {
    PYTHON_VERSION = "3.7"
    PIP_PACKAGES = "twine ."
  }
  needs = ["Filter tag", "Coverage"]
  secrets = ["TWINE_USERNAME", "TWINE_PASSWORD"]
}

action "Create release" {
  uses = "swillner/actions/create-release@master"
  needs = ["Publish on PyPi"]
  secrets = ["GITHUB_TOKEN"]
}
