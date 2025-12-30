.PHONY: help install dev test lint format build publish image-build image-run image-push image-up image-down image-logs binary clean

DOCKER_IMAGE ?= ghcr.io/smartql/smartql
DOCKER_TAG ?= latest

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development
install: ## Install dependencies
	uv pip install -e "."

dev: ## Install with dev dependencies
	uv pip install -e ".[dev]"

test: ## Run tests
	pytest tests/ -v

lint: ## Run linter
	ruff check src/

format: ## Format code
	ruff format src/

# Python packaging
build: ## Build Python package
	python -m build

publish: build ## Publish to PyPI
	twine upload dist/*

# Docker
image-build: ## Build Docker image
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

image-run: ## Run Docker container standalone
	docker run -p 5000:8000 \
		-v ./config.yml:/app/config.yml:ro \
		-e GROQ_API_KEY=$(GROQ_API_KEY) \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

image-push: ## Push Docker image to GitHub Container Registry
	docker push $(DOCKER_IMAGE):$(DOCKER_TAG)

image-up: ## Start with docker-compose
	docker compose up -d

image-down: ## Stop docker-compose
	docker compose down

image-logs: ## View docker-compose logs
	docker compose logs -f

# Binary
binary: ## Build standalone binary with PyInstaller
	pyinstaller smartql.spec
	@echo "Binary created at dist/smartql"

# CLI shortcuts
shell: ## Start interactive shell
	smartql shell -c config.yml

serve: ## Start HTTP server on port 5000
	smartql serve -c config.yml --port 5000

# Cleanup
clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
