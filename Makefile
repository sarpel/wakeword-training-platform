# ==============================================================================
# Wakeword Training Platform - Makefile
# Common development and deployment tasks
# ==============================================================================

.PHONY: help install install-dev install-gpu test lint format typecheck clean docker run

# Default target
help:
	@echo "Wakeword Training Platform - Available Commands"
	@echo "================================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install development dependencies"
	@echo "  make install-gpu   Install with CUDA GPU support"
	@echo "  make install-all   Install all dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make run           Launch the training application"
	@echo "  make test          Run all tests"
	@echo "  make test-unit     Run unit tests only"
	@echo "  make test-cov      Run tests with coverage"
	@echo "  make lint          Run all linters"
	@echo "  make format        Auto-format code"
	@echo "  make typecheck     Run type checking"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run in Docker container"
	@echo "  make docker-up     Start all services (compose)"
	@echo "  make docker-down   Stop all services"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         Remove build artifacts"
	@echo "  make clean-all     Remove all generated files"

# ==============================================================================
# Installation
# ==============================================================================

install:
	pip install --upgrade pip
	pip install -e .

install-dev:
	pip install --upgrade pip
	pip install -e ".[dev]"
	pre-commit install

install-gpu:
	pip install --upgrade pip
	pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
	pip install -e ".[gpu]"

install-all:
	pip install --upgrade pip
	pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
	pip install -e ".[all]"
	pre-commit install

# ==============================================================================
# Development
# ==============================================================================

run:
	python run.py

server:
	uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# ==============================================================================
# Testing
# ==============================================================================

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/ -v -m "unit" --tb=short

test-integration:
	pytest tests/ -v -m "integration" --tb=short

test-gpu:
	pytest tests/ -v -m "gpu" --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

# ==============================================================================
# Code Quality
# ==============================================================================

lint:
	@echo "Running flake8..."
	flake8 src/ tests/ --max-line-length=120
	@echo "Running isort check..."
	isort --check-only --diff src/ tests/
	@echo "Running black check..."
	black --check --diff src/ tests/
	@echo "All checks passed!"

format:
	@echo "Formatting with isort..."
	isort src/ tests/
	@echo "Formatting with black..."
	black src/ tests/
	@echo "Done!"

typecheck:
	mypy src/ --ignore-missing-imports

pre-commit:
	pre-commit run --all-files

# ==============================================================================
# Docker
# ==============================================================================

docker-build:
	docker build -t wakeword-trainer:latest .

docker-build-dev:
	docker build --target development -t wakeword-trainer:dev .

docker-run:
	docker run --gpus all -p 7860:7860 -v $(PWD)/data:/app/data wakeword-trainer:latest

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# ==============================================================================
# Cleanup
# ==============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage coverage.xml

clean-all: clean
	rm -rf venv/ .venv/
	rm -rf logs/*.log
	rm -rf models/checkpoints/*.pt
	rm -rf exports/*.onnx

# ==============================================================================
# Build & Release
# ==============================================================================

build:
	python -m build

publish-test:
	python -m twine upload --repository testpypi dist/*

publish:
	python -m twine upload dist/*
