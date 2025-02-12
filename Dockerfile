FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional tools for linting and code quality
RUN pip install --no-cache-dir flake8 black bandit

# Copy the rest of the application
COPY . .

# Command to run when starting the container
CMD ["python", "main.py"]
