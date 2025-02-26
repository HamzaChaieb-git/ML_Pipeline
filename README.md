ML Pipeline for Churn Prediction
A comprehensive machine learning pipeline for predicting customer churn, built with a modern MLOps approach.
📋 Overview
This project implements an end-to-end machine learning system for customer churn prediction with the following components:

Machine Learning Pipeline: Automated data processing, model training, and evaluation
FastAPI Backend: REST API for real-time predictions
Streamlit Dashboard: User-friendly web interface for predictions
MLflow Tracking: Experiment tracking and model registry
Docker Integration: Containerized deployment for all components
CI/CD Pipeline: Automated testing, building, and deployment using GitHub Actions

🚀 Features

Data preprocessing with automatic feature engineering
XGBoost model training with hyperparameter configurations
Comprehensive model evaluation metrics and visualizations
Model persistence and versioning
Real-time predictions via API
Interactive web dashboard
Automated testing and CI/CD workflow

🔧 Architecture
The system consists of four main services:

FastAPI: Provides a RESTful API for model predictions
Streamlit: Offers a user-friendly interface for interacting with the model
MLflow: Tracks experiments, metrics, and models
Pipeline: Orchestrates the ML workflow (data processing, training, evaluation)

📦 Installation
Prerequisites

Docker and Docker Compose
Git
Python 3.8+ (for local development)

Setup

Clone the repository:
bashCopygit clone https://github.com/yourusername/ml-pipeline-project.git
cd ml-pipeline-project

Run the pipeline using Docker Compose:
bashCopydocker compose up

Alternatively, use the provided script:
bashCopychmod +x run_pipeline.sh
./run_pipeline.sh


🔍 Usage
Accessing the Services
Once running, the services will be available at:

FastAPI: http://localhost:8000
Streamlit Dashboard: http://localhost:8501
MLflow Tracking Server: http://localhost:5001

API Examples
Make predictions using the FastAPI endpoint:
bashCopycurl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Total day minutes": [120.5],
    "Customer service calls": [3],
    "International plan": [0],
    "Total intl minutes": [10.2],
    "Total intl calls": [5],
    "Total eve minutes": [200.0],
    "Number vmail messages": [0],
    "Voice mail plan": [0]
  }'
Running the Pipeline Manually
To execute specific steps of the pipeline:
bashCopypython main.py --train-file churn-bigml-80.csv --test-file churn-bigml-20.csv --action all
Available actions: train, evaluate, all
🧪 Testing
Run the test suite with:
bashCopymake test
For linting and formatting:
bashCopymake lint
make format
📊 CI/CD Pipeline
The project includes a GitHub Actions workflow that:

Runs code quality checks (flake8, bandit, black)
Builds Docker images
Tests the services
Performs a pipeline execution
Temporarily exposes services via ngrok for review
Sends notification emails upon completion

📄 Project Structure
Copy├── .github/workflows/   # GitHub Actions workflows
├── artifacts/           # Model artifacts and MLflow data
├── tests/               # Test suite
├── app.py               # FastAPI application
├── data_processing.py   # Data preprocessing module
├── docker-compose.yml   # Docker Compose configuration
├── Dockerfile.*         # Dockerfiles for each service
├── main.py              # Pipeline orchestration
├── model_*.py           # Model training, evaluation, persistence
├── streamlit_app.py     # Streamlit dashboard
└── makefile             # Build and test automation
🛠️ Configuration
Major configuration options:

Model parameters: model_training.py
Docker settings: docker-compose.yml and Dockerfile.* files
CI/CD pipeline: .github/workflows/simple-ci-cd.yml

🔒 Security Note
The repository uses GitHub Secrets for storing sensitive information like email credentials. Make sure to set up the following secrets in your repository:

NGROK_AUTH_TOKEN: For exposing services during CI/CD
EMAIL_USERNAME: Email address for notifications
EMAIL_PASSWORD: Email password or app-specific password

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
