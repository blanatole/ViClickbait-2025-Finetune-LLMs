# Makefile for Clickbait Detection Project

.PHONY: help setup install test clean run-prompting run-finetuning run-all

# Default target
help:
	@echo "Available commands:"
	@echo "  setup          - Setup environment and install dependencies"
	@echo "  install        - Install Python dependencies"
	@echo "  test           - Run quick test"
	@echo "  run-prompting  - Run prompting experiments"
	@echo "  run-finetuning - Run fine-tuning experiments"
	@echo "  run-all        - Run all experiments"
	@echo "  clean          - Clean temporary files"
	@echo "  format         - Format code with black"
	@echo "  lint           - Run code linting"

# Setup environment
setup:
	python scripts/setup_environment.py

# Install dependencies
install:
	pip install -r requirements.txt

# Run quick test
test:
	python scripts/run_experiment.py

# Run prompting experiments
run-prompting:
	python src/experiments/pretrained_prompting.py
	python src/experiments/pretrained_prompting_improved.py

# Run fine-tuning experiments  
run-finetuning:
	python src/experiments/finetuning_experiment.py

# Run all experiments
run-all: run-prompting run-finetuning
	python src/experiments/evaluate_finetuned_prompting.py

# Clean temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	find . -type f -name ".DS_Store" -delete

# Format code
format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

# Lint code
lint:
	flake8 src/ scripts/ tests/
	black --check src/ scripts/ tests/

# Install development dependencies
install-dev:
	pip install -r requirements.txt
	pip install black isort flake8 pytest pytest-cov

# Run tests with coverage
test-coverage:
	pytest tests/ --cov=src/ --cov-report=html

# Build package
build:
	python setup.py sdist bdist_wheel

# Install package in development mode
install-package:
	pip install -e .
