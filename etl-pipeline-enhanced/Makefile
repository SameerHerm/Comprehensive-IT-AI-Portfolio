.PHONY: help setup build up down restart logs clean test lint format

# Variables
DOCKER_COMPOSE = docker-compose
PYTHON = python3
PIP = pip3

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
NC = \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

setup: ## Initial setup of the project
	@echo "$(GREEN)Setting up ETL Pipeline...$(NC)"
	@cp .env.example .env
	@$(PIP) install -r requirements.txt
	@./scripts/setup.sh
	@echo "$(GREEN)Setup complete!$(NC)"

build: ## Build Docker containers
	@echo "$(GREEN)Building Docker containers...$(NC)"
	@$(DOCKER_COMPOSE) build

up: ## Start all services
	@echo "$(GREEN)Starting services...$(NC)"
	@$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)Services started! Access Airflow at http://localhost:8080$(NC)"

down: ## Stop all services
	@echo "$(YELLOW)Stopping services...$(NC)"
	@$(DOCKER_COMPOSE) down

restart: down up ## Restart all services

logs: ## Show logs from all services
	@$(DOCKER_COMPOSE) logs -f

airflow-logs: ## Show Airflow logs
	@$(DOCKER_COMPOSE) logs -f airflow-webserver airflow-scheduler airflow-worker

kafka-logs: ## Show Kafka logs
	@$(DOCKER_COMPOSE) logs -f kafka zookeeper

clean: ## Clean up generated files and data
	@echo "$(YELLOW)Cleaning up...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@rm -rf logs/* data/processed/* data/staging/*
	@echo "$(GREEN)Cleanup complete!$(NC)"

test: ## Run tests
	@echo "$(GREEN)Running tests...$(NC)"
	@$(PYTHON) -m pytest tests/ -v

test-coverage: ## Run tests with coverage
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	@$(PYTHON) -m pytest tests/ --cov=src --cov-report=html

lint: ## Run linting
	@echo "$(GREEN)Running linters...$(NC)"
	@$(PYTHON) -m flake8 src/ airflow/dags/ kafka/
	@$(PYTHON) -m pylint src/ airflow/dags/ kafka/

format: ## Format code with black
	@echo "$(GREEN)Formatting code...$(NC)"
	@$(PYTHON) -m black src/ airflow/dags/ kafka/ tests/

monitoring: ## Open monitoring dashboard
	@echo "$(GREEN)Opening monitoring dashboard...$(NC)"
	@open http://localhost:3000 || xdg-open http://localhost:3000

kafka-ui: ## Open Kafka UI
	@echo "$(GREEN)Opening Kafka UI...$(NC)"
	@open http://localhost:9000 || xdg-open http://localhost:9000

produce-sample: ## Produce sample data to Kafka
	@echo "$(GREEN)Producing sample data...$(NC)"
	@$(PYTHON) kafka/producers/data_producer.py

consume-sample: ## Consume data from Kafka
	@echo "$(GREEN)Consuming data...$(NC)"
	@$(PYTHON) kafka/consumers/data_consumer.py

etl-batch: ## Run batch ETL pipeline
	@echo "$(GREEN)Running batch ETL...$(NC)"
	@./scripts/etl_processor.sh --mode batch

etl-stream: ## Run streaming ETL pipeline
	@echo "$(GREEN)Running streaming ETL...$(NC)"
	@./scripts/etl_processor.sh --mode stream

backup: ## Backup data
	@echo "$(GREEN)Backing up data...$(NC)"
	@./scripts/backup_data.sh

restore: ## Restore data from backup
	@echo "$(GREEN)Restoring data...$(NC)"
	@./scripts/restore_data.sh

status: ## Check status of all services
	@echo "$(GREEN)Checking service status...$(NC)"
	@$(DOCKER_COMPOSE) ps
	@./scripts/monitoring.sh status
