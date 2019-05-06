.PHONY: black checks clean coverage docs flake8 isort publish-on-pypi test test-all test-pypi-install
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

black: venv  ## apply black formatter to source and tests
	@status=$$(git status --porcelain openscm tests); \
	if test "x$${status}" = x; then \
		./venv/bin/black --exclude _version.py setup.py openscm tests; \
	else \
		echo Not trying any formatting. Working directory is dirty ... >&2; \
	fi;

checks: venv  ## run all the checks
	./venv/bin/bandit -c .bandit.yml -r openscm
	./venv/bin/black --check openscm tests setup.py --exclude openscm/_version.py
	./venv/bin/flake8 openscm tests setup.py
	./venv/bin/isort --check-only --quiet --recursive openscm tests setup.py
	./venv/bin/mypy openscm
	./venv/bin/pydocstyle openscm
	./venv/bin/pylint openscm
	./venv/bin/pytest notebooks -r a --nbval --sanitize tests/notebook-tests.cfg
	./venv/bin/pytest tests -r a --cov=openscm --cov-report='' \
		&& ./venv/bin/coverage report --fail-under=100
	./venv/bin/sphinx-build -M html docs docs/build -EW

check-docs: venv  ## check that the docs build successfully
	./venv/bin/sphinx-build -M html docs docs/build -En

clean:  ## remove the virtual environment
	@rm -rf venv

define clean_notebooks_code
	(.cells[] | select(has("execution_count")) | .execution_count) = 0 \
	| (.cells[] | select(has("outputs")) | .outputs[] | select(has("execution_count")) | .execution_count) = 0 \
	| .metadata = {"language_info": {"name": "python", "pygments_lexer": "ipython3"}} \
	| .cells[].metadata = {}
endef

clean-notebooks: venv  ## clean the notebooks of spurious changes to prepare for a PR
	@tmp=$$(mktemp); \
	for notebook in notebooks/*.ipynb; do \
		jq --indent 1 '${clean_notebooks_code}' "$${notebook}" > "$${tmp}"; \
		cp "$${tmp}" "$${notebook}"; \
	done; \
	rm "$${tmp}"

coverage: venv  ## run all the tests and show code coverage
	./venv/bin/pytest tests -r a --cov=openscm --cov-report='' --durations=10
	./venv/bin/coverage html
	./venv/bin/coverage report --show-missing

docs: venv  ## build the docs
	./venv/bin/sphinx-build -M html docs docs/build

isort: venv  ## format the imports in the source and tests
	./venv/bin/isort --recursive openscm tests setup.py

publish-on-pypi: venv  ## publish a release on PyPI
	-rm -rf build dist
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		./venv/bin/python setup.py bdist_wheel --universal; \
		./venv/bin/twine upload dist/*; \
	else \
		echo Working directory is dirty >&2; \
	fi;

test: venv  ## run all the tests
	./venv/bin/pytest -sx tests

test-notebooks: venv  ## test all the notebooks
	./venv/bin/pytest notebooks -r a --nbval --sanitize tests/notebook-tests.cfg

test-all: test test-notebooks  ## run the testsuite and the notebook tests

test-pypi-install: venv  ## test openscm installs from the test PyPI server
	$(eval TEMPVENV := $(shell mktemp -d))
	python3 -m venv $(TEMPVENV)
	$(TEMPVENV)/bin/pip install pip --upgrade
	$(TEMPVENV)/bin/pip install openscm
	$(TEMPVENV)/bin/python -c "import sys; sys.path.remove(''); import openscm; print(openscm.__version__)"

venv: setup.py  ## install a development virtual environment
	[ -d ./venv ] || python3 -m venv ./venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -e .[dev]
	touch venv
