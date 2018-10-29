venv: dev-requirements.txt docs/requirements.txt
	[ -d ./venv ] || python3 -m venv ./venv
	./venv/bin/pip install --upgrade pip
		./venv/bin/pip install -r dev-requirements.txt
		./venv/bin/pip install -r docs/requirements.txt
		./venv/bin/pip install -e .
	touch venv

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

clean:
	rm -rf venv

.PHONY: clean
