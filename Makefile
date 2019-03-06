venv: setup.py
	[ -d ./venv ] || python3 -m venv ./venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -e .[dev]
	touch venv

test: venv
	./venv/bin/pytest -rfsxEX --cov=openscm tests --cov-report term-missing

coverage: test
	coverage html

test-notebooks: venv
	./venv/bin/pytest -rfsxEX --nbval ./notebooks --sanitize ./notebooks/tests_sanitize.cfg

test-all: test test-notebooks

define clean_notebooks_code
	(.cells[] | select(has("execution_count")) | .execution_count) = 0 \
	| .metadata = {"language_info": {"name": "python", "pygments_lexer": "ipython3"}} \
	| .cells[].metadata = {}
endef

clean-notebooks: venv
	tmp=$$(mktemp); \
	for notebook in notebooks/*.ipynb; do \
		jq --indent 1 '${clean_notebooks_code}' "$${notebook}" > "$${tmp}"; \
		cp "$${tmp}" "$${notebook}"; \
	done; \
	rm "$${tmp}"

docs: venv
	./venv/bin/sphinx-build -M html docs docs/build

flake8: venv
	./venv/bin/flake8 openscm tests

black: venv
	@status=$$(git status --porcelain openscm tests); \
	if test "x$${status}" = x; then \
		./venv/bin/black --exclude _version.py setup.py openscm tests; \
	else \
		echo Not trying any formatting. Working directory is dirty ... >&2; \
	fi;

publish-on-pypi: venv
	-rm -rf build dist
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		./venv/bin/python setup.py bdist_wheel --universal; \
		./venv/bin/twine upload dist/*; \
	else \
		echo Working directory is dirty >&2; \
	fi;

test-pypi-install: venv
	$(eval TEMPVENV := $(shell mktemp -d))
	python3 -m venv $(TEMPVENV)
	$(TEMPVENV)/bin/pip install pip --upgrade
	$(TEMPVENV)/bin/pip install openscm
	$(TEMPVENV)/bin/python -c "import sys; sys.path.remove(''); import openscm; print(openscm.__version__)"

clean:
	rm -rf venv

.PHONY: clean coverage test test-all black flake8 docs publish-on-pypi test-pypi-install
