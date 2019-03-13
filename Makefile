.PHONY: black checks clean coverage docs flake8 isort publish-on-pypi test test-all test-pypi-install

black: venv
	@status=$$(git status --porcelain openscm tests); \
	if test "x$${status}" = x; then \
		./venv/bin/black --exclude _version.py setup.py openscm tests; \
	else \
		echo Not trying any formatting. Working directory is dirty ... >&2; \
	fi;

checks: venv
	./venv/bin/bandit -c .bandit.yml -r openscm
	./venv/bin/black --check openscm tests setup.py --exclude openscm/_version.py
	./venv/bin/flake8 openscm tests
	./venv/bin/isort --check-only --recursive openscm tests setup.py
	./venv/bin/mypy openscm
	./venv/bin/pydocstyle openscm
	./venv/bin/pylint openscm
	./venv/bin/pytest notebooks -r a --nbval --sanitize tests/notebook-tests.cfg
	./venv/bin/pytest tests -r a --cov=openscm --cov-report='' \
		&& ./venv/bin/coverage report --fail-under=100
	./venv/bin/sphinx-build -M html docs docs/build -EW

check-docs: venv
	./venv/bin/sphinx-build -M html docs docs/build -En

clean:
	@rm -rf venv

define clean_notebooks_code
	(.cells[] | select(has("execution_count")) | .execution_count) = 0 \
	| (.cells[] | select(has("outputs")) | .outputs[] | select(has("execution_count")) | .execution_count) = 0 \
	| .metadata = {"language_info": {"name": "python", "pygments_lexer": "ipython3"}} \
	| .cells[].metadata = {}
endef

clean-notebooks: venv
	@tmp=$$(mktemp); \
	for notebook in notebooks/*.ipynb; do \
		jq --indent 1 '${clean_notebooks_code}' "$${notebook}" > "$${tmp}"; \
		cp "$${tmp}" "$${notebook}"; \
	done; \
	rm "$${tmp}"

coverage: venv
	./venv/bin/pytest tests -r a --cov=openscm --cov-report=''
	./venv/bin/coverage html
	./venv/bin/coverage report --show-missing

docs: venv
	./venv/bin/sphinx-build -M html docs docs/build

isort: venv
	./venv/bin/isort --recursive openscm tests setup.py

publish-on-pypi: venv
	-rm -rf build dist
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		./venv/bin/python setup.py bdist_wheel --universal; \
		./venv/bin/twine upload dist/*; \
	else \
		echo Working directory is dirty >&2; \
	fi;

test: venv
	./venv/bin/pytest -sx tests

test-notebooks: venv
	./venv/bin/pytest notebooks -r a --nbval --sanitize tests/notebook-tests.cfg

test-all: test test-notebooks

test-pypi-install: venv
	$(eval TEMPVENV := $(shell mktemp -d))
	python3 -m venv $(TEMPVENV)
	$(TEMPVENV)/bin/pip install pip --upgrade
	$(TEMPVENV)/bin/pip install openscm
	$(TEMPVENV)/bin/python -c "import sys; sys.path.remove(''); import openscm; print(openscm.__version__)"

venv: setup.py
	[ -d ./venv ] || python3 -m venv ./venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -e .[dev]
	touch venv
