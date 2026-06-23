# Convenience commands for the Customer Churn Prediction & Retention project.
# Usage: `make setup`, `make run`, `make dashboard`, `make test`, `make clean`.

PY ?= python
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate

.PHONY: help setup run dashboard test lint clean

help:
	@echo "Targets:"
	@echo "  setup      Create venv and install requirements"
	@echo "  run        Run the full end-to-end pipeline (python main.py)"
	@echo "  dashboard  Launch the Streamlit dashboard"
	@echo "  test       Run the pytest suite"
	@echo "  clean      Remove generated artifacts (reports, models, caches)"

setup:
	$(PY) -m venv $(VENV)
	$(ACTIVATE) && pip install --upgrade pip && pip install -r requirements.txt

run:
	$(ACTIVATE) && $(PY) main.py

dashboard:
	$(ACTIVATE) && streamlit run dashboard/app.py

test:
	$(ACTIVATE) && pytest -q

clean:
	rm -rf reports/figures/*.png reports/*.csv reports/*.json
	rm -rf models/*.joblib
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned generated artifacts."
