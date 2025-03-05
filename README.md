# ML Pipeline for Churn Prediction

![GitHub Workflow Status](https://img.shields.io/badge/CI/CD-GitHub_Actions-blue)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-green)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-red)
![Grafana](https://img.shields.io/badge/Grafana-Dashboards-orange)
![Python Version](https://img.shields.io/badge/python-3.12-blue)

A comprehensive machine learning pipeline for predicting customer churn, built with a modern MLOps approach featuring automatic model retraining, performance monitoring, and containerized deployment.

## 📋 Overview

This project implements an end-to-end MLOps system for customer churn prediction with the following components:

- **Machine Learning Pipeline**: Automated data processing, model training, evaluation, and versioning
- **FastAPI Backend**: REST API for real-time predictions and model retraining
- **Streamlit Dashboard**: User-friendly web interface for predictions, model monitoring, and retraining
- **MLflow Integration**: Experiment tracking, model registry, and versioning
- **Monitoring Stack**: Prometheus metrics collection and Grafana dashboards
- **Database Integration**: MongoDB for storing predictions and metrics
- **Docker Deployment**: Containerized multi-service architecture
- **CI/CD Pipeline**: Automated testing, building, and deployment using GitHub Actions and Makefile

## 🚀 Features

### ML Pipeline
- Data preprocessing with automatic feature engineering and encoding
- XGBoost model training with configurable hyperparameters
- Comprehensive model evaluation metrics and visualizations
- MLflow experiment tracking and model registry
- Automatic model promotion based on performance metrics

### FastAPI Service
- Real-time prediction endpoints with request validation
- Model retraining API endpoints
- Dataset upload capabilities
- Model registry management
- Prometheus metrics instrumentation

### Streamlit Dashboard
- Interactive prediction interface with visualizations
- Model performance monitoring
- System metrics visualization
- Dataset management for retraining
- Model registry and lifecycle management

### Monitoring
- Prometheus metrics collection
- Grafana dashboards for:
  - Model performance metrics
  - System resource utilization
  - Prediction patterns
  - API performance
- Alerting capabilities for model and system issues

### Containerization & Deployment
- Multi-container architecture with Docker Compose
- Separate containers for:
  - FastAPI service
  - Streamlit dashboard
  - MLflow tracking server
  - Monitoring services
  - MongoDB database
- Environment isolation and reproducibility

## 🔧 Architecture

The system consists of several integrated services:

1. **FastAPI**: Provides a RESTful API for model predictions and retraining
2. **Streamlit**: Offers a user-friendly interface for interacting with the model and monitoring experiments
3. **MLflow**: Tracks experiments, metrics, and models
4. **MongoDB**: Stores predictions and monitoring data
5. **Prometheus**: Collects metrics from all services
6. **Grafana**: Visualizes metrics with custom dashboards
7. **Pipeline**: Orchestrates the ML workflow (data processing, training, evaluation)

## 📦 Installation

### Prerequisites

- Docker and Docker Compose
- Git
- Python 3.8+ (for local development)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ml-pipeline-project.git
   cd ml-pipeline-project
   ```

2. Run the pipeline using Docker Compose:
   ```bash
   docker compose up
   ```

3. Alternatively, use the provided script:
   ```bash
   chmod +x run_pipeline.sh
   ./run_pipeline.sh
   ```

## 🔍 Usage

### Accessing the Services

Once running, the services will be available at:

- FastAPI: http://localhost:8000
- Streamlit Dashboard: http://localhost:8501
- MLflow Tracking Server: http://localhost:5001
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

### API Examples

#### Make Predictions
```bash
curl -X POST "http://localhost:8000/predict" \
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
```

#### Trigger Model Retraining
```bash
curl -X POST "http://localhost:8000/retrain" \
  -F "train_file=churn-bigml-80.csv" \
  -F "test_file=churn-bigml-20.csv" \
  -F "auto_promote=true"
```

### Running the Pipeline Manually

To execute specific steps of the pipeline:

```bash
python main.py --train-file churn-bigml-80.csv --test-file churn-bigml-20.csv --action all
```

Available actions:
- `train`: Train a new model
- `evaluate`: Evaluate an existing model
- `all`: Train and evaluate a model

With model promotion:
```bash
python main.py --train-file churn-bigml-80.csv --test-file churn-bigml-20.csv --action all --promote
```

## 🧪 Testing

Run the test suite with:

```bash
make test
```

For linting and formatting:

```bash
make lint
make format
```

## 📊 Streamlit Dashboard Features

The enhanced Streamlit dashboard includes:

1. **Dashboard Overview**: Key performance metrics and system status
2. **Model Performance**: Detailed model metrics and visualizations
3. **Make Prediction**: Interactive prediction interface with visual results
4. **Recent Predictions**: Analysis of prediction patterns and distribution
5. **System Monitoring**: Resource utilization and service status
6. **Model Retraining**: Dataset management and model lifecycle control

## 📈 Monitoring Features

This project includes comprehensive monitoring capabilities:

- **Prometheus Metrics**: 
  - Prediction counts and latencies
  - Model loading status
  - Resource utilization
  - API performance

- **Grafana Dashboards**:
  - Model performance dashboard
  - Prediction analytics dashboard
  - System performance dashboard

- **Alerting**:
  - Model unavailability
  - High prediction latency
  - Resource constraints
  - Prediction error rates

## 🔄 Model Retraining

The system supports automated model retraining:

- Upload or select training and testing datasets
- Trigger retraining via UI or API
- Monitor training progress and results
- Automatic model promotion based on performance thresholds
- Model registry integration for lifecycle management

## 📄 Project Structure

```
├── .github/workflows/   # GitHub Actions workflows
├── artifacts/           # Model artifacts and MLflow data
├── grafana/             # Grafana dashboard configurations
├── prometheus/          # Prometheus configuration
├── tests/               # Test suite
├── app.py               # FastAPI application
├── data_processing.py   # Data preprocessing module
├── db_connector.py      # MongoDB connector
├── docker-compose.yml   # Docker Compose configuration
├── Dockerfile.*         # Dockerfiles for each service
├── main.py              # Pipeline orchestration
├── model_*.py           # Model training, evaluation, persistence
├── monitoring.py        # Monitoring utilities
├── model_retrain.py     # Model retraining module
├── streamlit_app.py     # Streamlit dashboard
└── makefile             # Build and test automation
```

## 🛠️ Configuration

Major configuration options:

- Model parameters: `model_training.py`
- Docker settings: `docker-compose.yml` and `Dockerfile.*` files
- CI/CD pipeline: `.github/workflows/simple-ci-cd.yml`
- Prometheus: `prometheus/prometheus.yml`
- Grafana: `grafana/provisioning/dashboards/`

## 🔒 Security Note

The repository uses GitHub Secrets for storing sensitive information. Make sure to set up the appropriate secrets in your repository.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
