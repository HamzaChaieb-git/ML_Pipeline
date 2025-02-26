.PHONY: all test test-specific lint format clean help check-deps install-deps

# Directory containing Python source files and tests
SRC_DIR := .
TESTS_DIR := tests
REPORTS_DIR := reports

# Python files to be linted
PYTHON_FILES := app.py data_processing.py main.py model_evaluation.py model_persistence.py model_training.py streamlit_app.py

# Use the virtual environment's Python (already activated)
PYTHON := python
PIP := pip

# Default test arguments (can be overridden via command line)
TEST_ARGS ?=

# Run all tests, linting, and formatting
all: check-deps test lint format

# Check and install dependencies
check-deps:
	@echo "Checking dependencies..."
	@$(PIP) show pytest-html > /dev/null 2>&1 || (echo "Installing pytest-html plugin..." && $(PIP) install pytest-html)
	@$(PIP) show flake8-html > /dev/null 2>&1 || (echo "Installing flake8-html plugin..." && $(PIP) install flake8-html)

# Run all tests in tests/ with pytest
test:
	@echo "Running all tests..."
	@mkdir -p $(REPORTS_DIR)
	@if $(PIP) show pytest-html > /dev/null 2>&1; then \
		pytest $(TEST_ARGS) $(TESTS_DIR)/ --html=$(REPORTS_DIR)/pytest.html --self-contained-html || exit 1; \
		echo "Test report saved to $(REPORTS_DIR)/pytest.html"; \
	else \
		pytest $(TEST_ARGS) $(TESTS_DIR)/ > $(REPORTS_DIR)/pytest.txt 2>&1 || (cat $(REPORTS_DIR)/pytest.txt && exit 1); \
		cat $(REPORTS_DIR)/pytest.txt; \
		echo "Test report saved to $(REPORTS_DIR)/pytest.txt"; \
		echo "Install pytest-html for HTML reports: pip install pytest-html"; \
	fi

# Run tests for a specific Python file or test file
test-specific:
	@echo "Running specific tests..."
	@if [ -z "$(FILE)" ]; then \
		echo "Error: Specify a file (e.g., make test-specific FILE=data_processing.py)"; \
		exit 1; \
	fi
	@mkdir -p $(REPORTS_DIR)
	@if [ -f "$(SRC_DIR)/$(FILE)" ]; then \
		if $(PIP) show pytest-html > /dev/null 2>&1; then \
			pytest $(TEST_ARGS) $(TESTS_DIR)/test_$(basename $(FILE))_test.py --html=$(REPORTS_DIR)/pytest-specific.html --self-contained-html || exit 1; \
			echo "Test report saved to $(REPORTS_DIR)/pytest-specific.html"; \
		else \
			pytest $(TEST_ARGS) $(TESTS_DIR)/test_$(basename $(FILE))_test.py > $(REPORTS_DIR)/pytest-specific.txt 2>&1 || (cat $(REPORTS_DIR)/pytest-specific.txt && exit 1); \
			cat $(REPORTS_DIR)/pytest-specific.txt; \
			echo "Test report saved to $(REPORTS_DIR)/pytest-specific.txt"; \
		fi; \
	elif [ -f "$(TESTS_DIR)/$(FILE)" ]; then \
		if $(PIP) show pytest-html > /dev/null 2>&1; then \
			pytest $(TEST_ARGS) $(TESTS_DIR)/$(FILE) --html=$(REPORTS_DIR)/pytest-specific.html --self-contained-html || exit 1; \
			echo "Test report saved to $(REPORTS_DIR)/pytest-specific.html"; \
		else \
			pytest $(TEST_ARGS) $(TESTS_DIR)/$(FILE) > $(REPORTS_DIR)/pytest-specific.txt 2>&1 || (cat $(REPORTS_DIR)/pytest-specific.txt && exit 1); \
			cat $(REPORTS_DIR)/pytest-specific.txt; \
			echo "Test report saved to $(REPORTS_DIR)/pytest-specific.txt"; \
		fi; \
	else \
		echo "Error: File '$(FILE)' not found in $(SRC_DIR) or $(TESTS_DIR)"; \
		exit 1; \
	fi

# Run linters (flake8 and bandit) on specific Python files
lint:
	@echo "Running linters on specific Python files..."
	@mkdir -p $(REPORTS_DIR)
	
	@echo "Running flake8..."
	@if $(PIP) show flake8-html > /dev/null 2>&1; then \
		flake8 $(PYTHON_FILES) --max-line-length=100 --format=html --htmldir=$(REPORTS_DIR)/flake8-report || true; \
		echo "flake8 HTML report saved to $(REPORTS_DIR)/flake8-report/index.html"; \
	else \
		flake8 $(PYTHON_FILES) --max-line-length=100 > $(REPORTS_DIR)/flake8.txt 2>&1 || true; \
		cat $(REPORTS_DIR)/flake8.txt; \
		echo "flake8 report saved to $(REPORTS_DIR)/flake8.txt"; \
	fi
	
	@echo "Running bandit..."
	@mkdir -p $(REPORTS_DIR)/bandit-report
	@bandit -r $(PYTHON_FILES) -f html -o $(REPORTS_DIR)/bandit-report/index.html || true
	@echo "bandit HTML report saved to $(REPORTS_DIR)/bandit-report/index.html"

# Format code with black, output to HTML report
format:
	@echo "Formatting code..."
	@mkdir -p $(REPORTS_DIR)
	@black --check $(PYTHON_FILES) > $(REPORTS_DIR)/black-check.txt 2>&1 || true
	@black $(PYTHON_FILES)
	@echo "<html><head><title>Black Formatting Report</title><style>body{font-family:Arial,sans-serif;margin:20px}pre{background:#f5f5f5;padding:10px;border-radius:5px}</style></head><body><h1>Black Formatting Report</h1><pre>" > $(REPORTS_DIR)/black-report.html
	@cat $(REPORTS_DIR)/black-check.txt >> $(REPORTS_DIR)/black-report.html
	@echo "</pre></body></html>" >> $(REPORTS_DIR)/black-report.html
	@echo "black formatting report saved to $(REPORTS_DIR)/black-report.html"

# Clean up Python cache files
clean:
	@echo "Cleaning up..."
	find $(SRC_DIR) -type d -name "__pycache__" -exec rm -r {} +
	find $(SRC_DIR) -type f -name "*.pyc" -delete
	find $(TESTS_DIR) -type d -name "__pycache__" -exec rm -r {} +
	find $(TESTS_DIR) -type f -name "*.pyc" -delete
	@echo "Cleaning up reports directory..."
	@rm -rf $(REPORTS_DIR)/*

# Show available commands
help:
	@echo "Available commands:"
	@echo "  make all          - Run tests, linting, and formatting"
	@echo "  make check-deps   - Check and install required dependencies"
	@echo "  make test         - Run all tests with HTML reports (use TEST_ARGS for options)"
	@echo "  make test-specific FILE=<file> - Run tests for a specific file with HTML report"
	@echo "  make lint         - Check specific Python files with flake8 and bandit (HTML output)"
	@echo "  make format       - Format specific Python files with black and generate HTML report"
	@echo "  make clean        - Remove Python cache files and reports"
	@echo "  make help         - Show this help message"