# PolicyLens — common tasks. Run `make help` for the list.
#
# Everything here works with no API key. PYTHONPATH is set per-target so the repo can
# be used straight from a clone without installing it.

PY ?= python3
export PYTHONPATH := src

.DEFAULT_GOAL := help
.PHONY: help install index ask serve eval eval-dev ablation corpus test lint clean docker docker-eval

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the core dependencies (no keys, no model downloads)
	$(PY) -m pip install -r requirements.txt

index: ## Build the retrieval index and knowledge graph from the PDF corpus
	$(PY) -m policylens.cli index --force

ask: ## Ask one question, e.g. make ask Q="what is the PED waiting period under Aarogya Suraksha?"
	@test -n "$(Q)" || (echo 'Set Q="your question"'; exit 1)
	$(PY) -m policylens.cli ask "$(Q)" --trace

serve: ## Run the API on http://localhost:8000 (docs at /docs)
	$(PY) -m policylens.cli serve --port 8000

eval: ## Full evaluation on the held-out test split, with dev sweep and ablation
	$(PY) -m policylens.cli eval --split test --sweep --ablation

eval-dev: ## Evaluation on the dev split (use this while tuning; never report it)
	$(PY) -m policylens.cli eval --split dev

ablation: ## Compare retrieval configurations only
	$(PY) -m policylens.cli eval --split test --ablation

corpus: ## Re-render data/corpus_pdf from the text corpus and verify the round-trip
	$(PY) scripts/render_pdfs.py --verify

test: ## Run the test suite
	$(PY) -m pytest tests/

lint: ## Lint with ruff, if installed
	@command -v ruff >/dev/null 2>&1 && ruff check src tests scripts \
	  || echo "ruff not installed; pip install ruff"

docker: ## Build the image and start the API
	docker compose up --build

docker-eval: ## Run the evaluation inside the container, writing to ./reports
	docker compose run --rm eval

clean: ## Remove caches, the built index and timestamped eval runs
	rm -rf .pytest_cache .ruff_cache data/index reports/eval-* reports/latest.json
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
