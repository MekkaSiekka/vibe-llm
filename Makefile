# Local LLM Service Makefile

.PHONY: help build up down logs test clean dev prod

# Default target
help:
	@echo "Local LLM Service - Available commands:"
	@echo "  make build     - Build Docker images"
	@echo "  make up        - Start development environment"
	@echo "  make prod      - Start production environment"
	@echo "  make down      - Stop all services"
	@echo "  make logs      - View logs"
	@echo "  make test      - Run tests"
	@echo "  make clean     - Clean up containers and volumes"
	@echo "  make dev       - Start development with hot reload"

# Build Docker images
build:
	docker-compose build

# Start development environment
up:
	docker-compose up -d
	@echo "Service started at http://localhost:8000"
	@echo "WebSocket available at ws://localhost:8000/ws"

# Start production environment
prod:
	docker-compose --profile production up -d
	@echo "Production service started at http://localhost:8000"

# Stop all services
down:
	docker-compose down

# View logs
logs:
	docker-compose logs -f

# Run tests
test:
	docker-compose exec local-llm pytest

# Clean up
clean:
	docker-compose down -v
	docker system prune -f

# Development with hot reload
dev:
	docker-compose up --build

# Install dependencies locally
install:
	pip install -r requirements.txt

# Format code
format:
	black .
	isort .

# Type check
typecheck:
	mypy .

# Lint code
lint:
	flake8 .

# Health check
health:
	curl -f http://localhost:8000/health || echo "Service not running"

# Quick test
quick-test:
	curl -X POST http://localhost:8000/chat/simple \
		-H "Content-Type: application/json" \
		-d '{"message": "Hello, how are you?"}'

