workflow "Continuous Integration" {
  on = "push"
  resolves = ["Bandit", "Black", "Pylint", "Test coverage"]
}

action "Bandit" {
  uses = "./.github/actions/run"
  args = [
    "bandit -c .bandit.yml -r openscm"
  ]
  env = {
    PYTHON_VERSION = "3.7"
    PIP_PACKAGES = "bandit"
  }
}

action "Black" {
  uses = "./.github/actions/run"
  args = [
    "black --check openscm tests setup.py --exclude openscm/_version.py"
  ]
  env = {
    PYTHON_VERSION = "3.7"
    PIP_PACKAGES = "black"
  }
}

action "Pylint" {
  uses = "./.github/actions/run"
  args = [
    "pylint openscm"
  ]
  env = {
    PYTHON_VERSION = "3.7"
    PIP_PACKAGES = "pylint .[dev]"
  }
}

action "Test coverage" {
  uses = "./.github/actions/run"
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
    PIP_PACKAGES = "coverage pytest pytest-cov .[dev]"
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
  uses = "./.github/actions/run"
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
  uses = "./.github/actions/create-release"
  needs = ["Publish on PyPi"]
  secrets = ["GITHUB_TOKEN"]
}
