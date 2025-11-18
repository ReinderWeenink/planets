GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
NC     := \033[0m # No Color

# These variables are less critical now but kept for general context/future use
PROJECT_NAME := planet_name_generator

.PHONY: build up down clean help

# Default target
.DEFAULT_GOAL := help

help:
	@echo "$(YELLOW)Available targets:$(NC)"
	@echo "  $(GREEN)build$(NC)     - Build the service images defined in docker-compose.yml"
	@echo "  $(GREEN)up$(NC)        - Build (if needed), create, and start containers in detached mode"
	@echo "  $(GREEN)down$(NC)      - Stop and remove containers, networks, and volumes"
	@echo "  $(GREEN)clean$(NC)     - Stop, remove containers, and forcefully remove the built images"

build:
	@echo "$(YELLOW)Building Docker Compose services...$(NC)"
	docker compose build
	@echo "$(GREEN)Build complete!$(NC)"

up: build
	@echo "$(YELLOW)Starting Docker Compose services in detached mode...$(NC)"
	# Run in detached mode (background)
	docker compose up -d
	@echo "$(GREEN)Application is running! Access at http://localhost:8080$(NC)"
	
down:
	@echo "$(YELLOW)Stopping and removing containers...$(NC)"
	docker compose down
	@echo "$(GREEN)Containers stopped and removed. $(NC)"

clean: down
	@echo "$(YELLOW)Removing built images...$(NC)"
	# --rmi all removes all images built by the Compose file
	docker compose down --rmi all
	@echo "$(GREEN)Cleanup complete!$(NC)"