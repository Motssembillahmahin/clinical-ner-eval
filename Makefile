.PHONY: setup train eval all clean lint

# Install dependencies via uv
setup:
	uv sync

# Fine-tune all shortlisted models on the combined (harmonized) dataset
train:
	uv run python -m src.train

# Run evaluation harness and produce the comparison table
eval:
	uv run python -m src.evaluate

# Full pipeline: train every model, then evaluate
all: train eval

# MACCROBAT-only showcase run with native (full) label set
showcase:
	uv run python -m src.train --dataset maccrobat --label-mode native

lint:
	uv run ruff check src

clean:
	rm -rf results/*.json results/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} +
