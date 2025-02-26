.PHONY: all test test-specific lint format clean help

# Directory containing Python source files and tests
SRC_DIR := .
TESTS_DIR := tests

# Use the virtual environment's Python (already activated)
PYTHON := python
PIP := pip

# Default test arguments (can be overridden via command line)
TEST_ARGS ?=

# Run all tests, linting, and formatting
all: test lint format

# Run all tests in tests/ with pytest
test:
	@echo "Running all tests..."
	pytest $(TEST_ARGS) $(TESTS_DIR)/

# Run tests for a specific Python file or test file
test-specific:
	@echo "Running specific tests..."
	@if [ -z "$(FILE)" ]; then \
		echo "Error: Specify a file (e.g., make test-specific FILE=data_processing.py)"; \
		exit 1; \
	fi
	@if [ -f "$(SRC_DIR)/$(FILE)" ]; then \
		pytest $(TEST_ARGS) $(TESTS_DIR)/test_$(basename $(FILE))_test.py; \
	elif [ -f "$(TESTS_DIR)/$(FILE)" ]; then \
		pytest $(TEST_ARGS) $(TESTS_DIR)/$(FILE); \
	else \
		echo "Error: File '$(FILE)' not found in $(SRC_DIR) or $(TESTS_DIR)"; \
		exit 1; \
	fi

# Run linters (flake8 and bandit) on source files
lint:
	@echo "Running linters..."
	flake8 $(SRC_DIR) --max-line-length=100
	bandit -r $(SRC_DIR)

# Format code with black
format:
	@echo "Formatting code..."
	black $(SRC_DIR)

# Clean up Python cache files
clean:
	@echo "Cleaning up..."
	find $(SRC_DIR) -type d -name "__pycache__" -exec rm -r {} +
	find $(SRC_DIR) -type f -name "*.pyc" -delete
	find $(TESTS_DIR) -type d -name "__pycache__" -exec rm -r {} +
	find $(TESTS_DIR) -type f -name "*.pyc" -delete

# Show available commands
help:
	@echo "Available commands:"
	@echo "  make all          - Run tests, linting, and formatting"
	@echo "  make test         - Run all tests (use TEST_ARGS for options, e.g., TEST_ARGS='-v')"
	@echo "  make test-specific FILE=<file> - Run tests for a specific file (e.g., FILE=data_processing.py)"
	@echo "  make lint         - Check code with flake8 and bandit"
	@echo "  make format       - Format code with black"
	@echo "  make clean        - Remove Python cache files"
	@echo "  make help         - Show this help message"