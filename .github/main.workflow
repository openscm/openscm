workflow "Continuous Integration" {
  on = "push"
  resolves = ["Bandit", "Black", "Pylint", "Test coverage"]
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

action "Black" {
  uses = "swillner/actions/python-run@master"
  args = [
    "black --check openscm tests setup.py --exclude openscm/_version.py"
  ]
  env = {
    PYTHON_VERSION = "3.7"
    PIP_PACKAGES = "black"
  }
}

action "Pylint" {
  uses = "swillner/actions/python-run@master"
  args = [
    "pylint openscm"
  ]
  env = {
    PYTHON_VERSION = "3.7"
    PIP_PACKAGES = "pylint .[tests]"
  }
}

action "Test coverage" {
  uses = "swillner/actions/python-run@master"
  args = [
    "pytest -rfsxEX --cov=openscm tests --cov-report term-missing",
    "pytest -rfsxEX --nbval ./notebooks --sanitize ./notebooks/tests_sanitize.cfg",
    "if ! coverage report --fail-under=\"$MIN_COVERAGE\"",
    "then",
    "    echo",
    "    echo \"Error: Coverage has to be at least ${MIN_COVERAGE}%\"",
    "    exit 1",
    "fi"
  ]
  env = {
    PYTHON_VERSION = "3.7"
    MIN_COVERAGE = "100"
    PIP_PACKAGES = "coverage pytest pytest-cov .[tests]"
  }
}


workflow "Deployment" {
  on = "release"
  resolves = ["Create release"]
}

action "Filter tag" {
  uses = "actions/bin/filter@master"
  args = "tag v*"
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
  needs = ["Filter tag", "Bandit", "Black", "Pylint", "Test coverage"]
  secrets = ["TWINE_USERNAME", "TWINE_PASSWORD"]
}

action "Create release" {
  uses = "swillner/actions/create-release@master"
  needs = ["Publish on PyPi"]
  secrets = ["GITHUB_TOKEN"]
}
