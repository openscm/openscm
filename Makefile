.PHONY: checks check-docs clean clean-notebooks coverage docs format test test-all test-notebooks

checks: venv
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

format: venv clean-notebooks
	./venv/bin/isort --recursive openscm tests setup.py
	./venv/bin/black openscm tests setup.py --exclude openscm/_version.py

test-all: test test-notebooks

test: venv
	./venv/bin/pytest -sx tests

test-notebooks: venv
	./venv/bin/pytest notebooks -r a --nbval --sanitize tests/notebook-tests.cfg

venv: setup.py
	[ -d ./venv ] || python3 -m venv ./venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -e .[dev]
	touch venv
