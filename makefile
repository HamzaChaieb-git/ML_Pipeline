# Variables
VENV_DIR=venv
PYTHON=$(VENV_DIR)/bin/python
PIP=$(VENV_DIR)/bin/pip
REQ_FILE=requirements.txt

# Default target
.DEFAULT_GOAL := all

# 1) Install dependencies
.PHONY: install
install:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV_DIR)
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQ_FILE)

# 2) Code Verification (Linting, Formatting, Security)
.PHONY: lint format security
lint:
	@echo "Checking code quality..."
	$(PIP) install flake8
	flake8 --max-line-length=100 .

format:
	@echo "Auto-formatting code..."
	$(PIP) install black
	black .

security:
	@echo "Running security checks..."
	$(PIP) install bandit
	bandit -r .

verify: lint format security

# 3) Prepare Data (Run prepare_data function in main.py)
.PHONY: prepare_data
prepare_data:
	@echo "Preparing data..."
	$(PYTHON) main.py prepare_data

# 4) Train Model (Run train_model function in main.py)
.PHONY: train
train:
	@echo "Training the model..."
	$(PYTHON) main.py train_model

# 5) Evaluate Model
.PHONY: evaluate
evaluate:
	@echo "Evaluating the model..."
	$(PYTHON) main.py evaluate_model

# 6) Save Model
.PHONY: save
save:
	@echo "Saving the trained model..."
	$(PYTHON) main.py save_model

# 7) Load Model
.PHONY: load
load:
	@echo "Loading the trained model..."
	$(PYTHON) main.py load_model

# 8) Run Tests
.PHONY: test
test:
	@echo "Running tests..."
	$(PYTHON) -m unittest discover -s tests

# 9) Run All Steps Automatically
.PHONY: all
all: install verify prepare_data train evaluate save load test
	@echo "Full pipeline executed successfully!"

# 10) Clean Environment
.PHONY: clean
clean:
	@echo "Cleaning environment..."
	rm -rf $(VENV_DIR)
	rm -rf __pycache__
	rm -rf *.pkl

