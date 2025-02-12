Dockerfile
Copy
# Build stage for Python dependencies
FROM python:3.12-slim as builder
# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir flake8 black bandit
# Final stage for runtime
FROM python:3.12-slim
WORKDIR /app
# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
# Copy the rest of the application
COPY . .
# Command to run when starting the container
CMD ["python", "main.py"]
