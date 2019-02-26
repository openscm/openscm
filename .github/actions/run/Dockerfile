ARG PYTHON_VERSION=latest
FROM python:$PYTHON_VERSION

LABEL "com.github.actions.name"="Run for Python"
LABEL "com.github.actions.description"="Run command in a Python-supporting environment"
LABEL "com.github.actions.icon"="command"
LABEL "com.github.actions.color"="blue"

RUN pip install --upgrade pip

ADD entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
