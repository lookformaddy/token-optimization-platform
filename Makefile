.PHONY: install test demo serve lint clean

export PYTHONPATH := src

PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	$(PY) -m pytest -q

demo:
	$(PY) scripts/demo.py

serve:
	$(PY) -m uvicorn tokenopt.service:app --host 0.0.0.0 --port 8000

lint:
	$(PY) -m compileall -q src tests scripts

clean:
	rm -rf $(VENV) __pycache__ src/tokenopt/__pycache__ tests/__pycache__ .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
