#!/bin/bash

# Script to build, run, and access MLflow and Streamlit

echo "Starting ML Pipeline setup..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Ensure artifacts directory exists (optional: download from GitHub if needed)
if [ ! -d "artifacts" ] || [ ! -f "mlflow.db" ]; then
    echo "Artifacts or mlflow.db not found. Please ensure they are present."
    echo "You may need to download pipeline-artifacts.zip from GitHub Actions and unzip it."
    exit 1
fi

# Build Docker images
echo "Building Docker images..."
docker compose build || { echo "Build failed"; exit 1; }

# Start containers in the background
echo "Starting containers..."
docker compose up -d || { echo "Startup failed"; exit 1; }

# Wait for services to stabilize (30 seconds)
echo "Waiting 30 seconds for services to start..."
sleep 30

# Check container status
echo "Container status:"
docker compose ps

# Verify MLflow is running
echo "Checking MLflow..."
if curl -s http://localhost:5001 >/dev/null; then
    echo "MLflow is running at http://localhost:5001"
else
    echo "MLflow failed to start. Showing logs:"
    docker compose logs mlflow
    exit 1
fi

# Verify Streamlit is running
echo "Checking Streamlit..."
if curl -s http://localhost:8501 >/dev/null; then
    echo "Streamlit is running at http://localhost:8501"
else
    echo "Streamlit failed to start. Showing logs:"
    docker compose logs streamlit
    exit 1
fi

echo "Setup complete! Access MLflow at http://localhost:5001 and Streamlit at http://localhost:8501"
echo "To stop, run: docker compose down"
