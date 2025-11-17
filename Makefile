GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
NC     := \033[0m # No Color

IMAGE_NAME := planets
IMAGE_TAG := latest

.PHONY: build run clean help

# Default target
.DEFAULT_GOAL := help

help:
	@echo "$(YELLOW)Available targets:$(NC)"
	@echo "  $(GREEN)build$(NC)       - Build Docker image"
	@echo "  $(GREEN)run$(NC)         - Run container in detached mode"
	@echo "  $(GREEN)clean$(NC)       - Remove container and image"

build:
	@echo "$(YELLOW) building $(IMAGE_NAME):$(IMAGE_TAG)"
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "$(GREEN)Build complete!$(NC)"

run: build
	@echo "$(YELLOW)Starting container $(IMAGE_NAME):$(IAMGE_TAG)...$(NC)"
	# adding the -t flag enables color codiding for loguru
	docker run -p 80:80 -t $(IMAGE_NAME):$(IMAGE_TAG)

clean:
	@echo "$(YELLOW)Cleaning up...$(NC)"
	-docker stop $(IMAGE_NAME) 2>/dev/null || true
	-docker rm $(IMAGE_NAME) 2>/dev/null || true
	-docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete!$(NC)"
