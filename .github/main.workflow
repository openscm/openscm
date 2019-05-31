workflow "Continuous Integration" {
  on = "push"
  resolves = ["Coverage", "Documentation", "Formatting", "Linters", "Notebooks", "Tests"]
}

action "Documentation" {
  uses = "swillner/actions/python-run@python-3.7"
  args = [
    "pip install -e .[docs]",
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
}

action "Formatting" {
  uses = "swillner/actions/python-run@python-3.7"
  args = [
    "black --check openscm tests setup.py --exclude openscm/_version.py",
    "isort --check-only --quiet --recursive openscm tests setup.py",
    "pydocstyle openscm"
  ]
  env = {
    PIP_PACKAGES = "black isort pydocstyle"
  }
}

action "Linters" {
  uses = "swillner/actions/python-run@python-3.7"
  args = [
    "pip install -e .[dev]",
    "bandit -c .bandit.yml -r openscm",
    "flake8 openscm tests setup.py",
    "mypy openscm",
    "pylint openscm"
  ]
  env = {
    PIP_PACKAGES = "bandit flake8 mypy" # TODO wait for fixed pylint version
  }
}

action "Tests" {
  uses = "swillner/actions/python-run@python-3.7"
  args = [
    "pip install -e .[tests]",
    "pytest tests -r a --cov=openscm --cov-report=''"
  ]
}

action "Notebooks" {
  uses = "swillner/actions/python-run@python-3.7"
  args = [
    "pip install -e .[tests,notebooks]",
    "pytest notebooks -r a --nbval --sanitize tests/notebook-tests.cfg --no-cov"
  ]
  needs = ["Coverage", "Documentation", "Formatting", "Linters", "Tests"]
}

action "Coverage" {
  uses = "swillner/actions/python-run@python-3.7"
  args = [
    "if ! coverage report --fail-under=\"$MIN_COVERAGE\" --show-missing",
    "then",
    "    echo",
    "    echo \"Error: Test coverage has to be at least ${MIN_COVERAGE}%\"",
    "    exit 1",
    "fi"
  ]
  env = {
    MIN_COVERAGE = "80"
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

action "Publish on PyPI" {
  uses = "swillner/actions/python-run@python-3.7"
  args = [
    "pip install -e .",
    "rm -rf build dist",
    "python setup.py sdist",
    "twine upload dist/*"
  ]
  env = {
    PIP_PACKAGES = "twine"
  }
  needs = ["Filter tag"]
  secrets = ["TWINE_USERNAME", "TWINE_PASSWORD"]
}

action "Test PyPI install" {
  uses = "./.github/actions/compile"
  args = [
    "sleep 15",
    "mkdir tmp",
    "cd tmp",
    "pip install openscm",
    "python -c 'import openscm'"
  ]
  needs = ["Publish on PyPI"]
}

action "Create release" {
  uses = "swillner/actions/create-release@master"
  needs = ["Test PyPI install"]
  secrets = ["GITHUB_TOKEN"]
}
