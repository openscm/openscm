workflow "Continuous Integration" {
  on = "push"
  resolves = ["Tests", "Documentation", "Notebooks"]
}

action "Documentation" {
  uses = "swillner/actions/python-run@master"
  args = [
    "sphinx-build -M html docs docs/build -qW", # treat warnings as errors (-W)...
    "sphinx-build -M html docs docs/build -Eqn -b coverage", # ...but not when being nitpicky (-n)
    "if [[ -s docs/build/html/python.txt ]]",
    "then",
    "    echo",
    "    echo \"Error: Documentation missing:\"",
    "    echo",
    "    cat docs/build/html/python.txt",
    "    exit 1",
    "fi"
  ]
  env = {
    PYTHON_VERSION = "3.7"
    PIP_PACKAGES = ".[docs]"
  }
}

action "Formatting" {
  uses = "swillner/actions/python-run@master"
  args = [
    "black --check openscm tests setup.py --exclude openscm/_version.py",
    "isort --check-only --quiet --recursive openscm tests setup.py",
    "pydocstyle openscm"
  ]
  env = {
    PYTHON_VERSION = "3.7"
    PIP_PACKAGES = "black isort pydocstyle"
  }
}

action "Linters" {
  uses = "swillner/actions/python-run@master"
  args = [
    "bandit -c .bandit.yml -r openscm",
    "flake8 openscm tests setup.py",
    "mypy openscm",
    "pylint openscm"
  ]
  env = {
    PYTHON_VERSION = "3.7"
    PIP_PACKAGES = ".[dev]"
  }
}

action "Tests" {
  uses = "swillner/actions/python-run@master"
  args = [
    "pytest tests -r a --cov=openscm --cov-report=''",
    "cat .coverage",
    "if ! coverage report --fail-under=\"$MIN_COVERAGE\" --show-missing",
    "then",
    "    echo",
    "    echo \"Error: Test coverage has to be at least ${MIN_COVERAGE}%\"",
    "    exit 1",
    "fi"
  ]
  env = {
    PYTHON_VERSION = "3.7"
    MIN_COVERAGE = "100"
    PIP_PACKAGES = ".[tests]"
  }
  needs = ["Formatting", "Linters"]
}

action "Notebooks" {
  uses = "swillner/actions/python-run@master"
  args = [
    "pytest notebooks -r a --nbval --sanitize tests/notebook-tests.cfg"
  ]
  env = {
    PYTHON_VERSION = "3.7"
    PIP_PACKAGES = ".[tests,notebooks]"
  }
  needs = ["Documentation", "Formatting", "Linters"]
}

workflow "Deployment" {
  on = "create"
  resolves = ["Create release"]
}

action "Filter tag" {
  uses = "actions/bin/filter@master"
  args = "tag 'v*'"
}

action "Filter master branch" {
  uses = "swillner/actions/filter-branch@master"
  args = "master"
  needs = "Filter tag"
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
  needs = ["Filter master branch"]
  secrets = ["TWINE_USERNAME", "TWINE_PASSWORD"]
}

action "Create release" {
  uses = "swillner/actions/create-release@master"
  needs = ["Publish on PyPi"]
  secrets = ["GITHUB_TOKEN"]
}
