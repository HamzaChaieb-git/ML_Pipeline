"""Enhanced main module for running the ML pipeline with MLflow and OpenTelemetry tracing."""

import argparse
import os
import mlflow
import mlflow.xgboost
import mlflow.sklearn
from datetime import datetime
from typing import Tuple
import sys
import json

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor, SimpleSpanProcessor

# Import pipeline functions
from data_processing import prepare_data as process_data
from model_training import train_model as train_xgb_model
from model_evaluation import evaluate_model as evaluate_xgb_model
from model_persistence import save_model as save_xgb_model, load_model as load_xgb_model

# Check for OpenTelemetry availability
try:
    from mlflow.opentelemetry import MlflowSpanExporter
    OPENTELEMETRY_AVAILABLE = True
    print("✅ OpenTelemetry tracing enabled with MlflowSpanExporter")
except ImportError:
    print("⚠️ Warning: mlflow.opentelemetry not found. Falling back to MLflow native tracing.")
    from mlflow import tracing
    tracing.enable()
    OPENTELEMETRY_AVAILABLE = False

# Initialize OpenTelemetry if available
if OPENTELEMETRY_AVAILABLE:
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    # Export traces to console (for debugging)
    trace.get_tracer_provider().add_span_processor(
        SimpleSpanProcessor(ConsoleSpanExporter())
    )
    # Export traces to MLflow
    mlflow_exporter = MlflowSpanExporter()
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(mlflow_exporter)
    )
else:
    tracer = None

def setup_enhanced_mlflow():
    """Setup MLflow tracking with custom configuration."""
    tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")

    mlflow.xgboost.autolog()
    mlflow.sklearn.autolog()

    experiment_name = "churn_prediction"
    experiment_tags = {
        "project_name": "churn_prediction",
        "project_version": "enhanced_v1",
        "framework": "xgboost",
        "pipeline_type": "binary_classification"
    }

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment '{experiment_name}' (ID: {experiment_id})")
    else:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=os.path.abspath("./mlruns"),
            tags=experiment_tags
        )
        print(f"Created new experiment '{experiment_name}' (ID: {experiment_id})")

    mlflow.set_experiment(experiment_name)
    return experiment_id

def run_enhanced_pipeline(train_file: str, test_file: str) -> None:
    """Execute the ML pipeline with tracing."""
    print("🚀 Launching enhanced ML pipeline...")

    experiment_id = setup_enhanced_mlflow()
    model_version = datetime.now().strftime("%Y%m%d_%H%M")

    with mlflow.start_run(run_name=f"Pipeline_v{model_version}") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # Use OpenTelemetry tracing if available, otherwise fallback to MLflow tracing
        if OPENTELEMETRY_AVAILABLE:
            with tracer.start_as_current_span("pipeline_execution") as span:
                span.set_attribute("mlflow.run_id", run_id)
                _run_pipeline(train_file, test_file, span)
        else:
            with tracing.start_run(run_name="Pipeline Tracing"):
                with tracing.start_span("pipeline_execution"):
                    _run_pipeline(train_file, test_file)

def _run_pipeline(train_file: str, test_file: str, span=None):
    """Core pipeline logic with tracing for each step."""
    # Data preparation
    if OPENTELEMETRY_AVAILABLE:
        with tracer.start_as_current_span("data_preparation") as data_span:
            X_train, X_test, y_train, y_test = process_data(train_file, test_file)
            data_span.set_attribute("train_samples", len(X_train))
            data_span.set_attribute("test_samples", len(X_test))
            mlflow.log_metric("train_samples", len(X_train))
            mlflow.log_metric("test_samples", len(X_test))
    else:
        with tracing.start_span("data_preparation"):
            X_train, X_test, y_train, y_test = process_data(train_file, test_file)
            mlflow.log_metric("train_samples", len(X_train))
            mlflow.log_metric("test_samples", len(X_test))
    print("✅ Data preparation complete")

    # Model training
    model_version = datetime.now().strftime("%Y%m%d_%H%M")
    if OPENTELEMETRY_AVAILABLE:
        with tracer.start_as_current_span("model_training") as train_span:
            model = train_xgb_model(X_train, y_train, model_version=model_version)
    else:
        with tracing.start_span("model_training"):
            model = train_xgb_model(X_train, y_train, model_version=model_version)
    print("✅ Model training complete")

    # Model evaluation
    if OPENTELEMETRY_AVAILABLE:
        with tracer.start_as_current_span("model_evaluation") as eval_span:
            metrics = evaluate_xgb_model(model, X_test, y_test)
            eval_span.set_attributes({f"metric.{k}": float(v) for k, v in metrics.items()})
    else:
        with tracing.start_span("model_evaluation"):
            metrics = evaluate_xgb_model(model, X_test, y_test)
    print("✅ Model evaluation complete")

    # Log model
    mlflow.xgboost.log_model(model, "model", registered_model_name=f"churn_model_v{model_version}")
    save_xgb_model(model, f"model_v{model_version}.joblib")
    mlflow.log_param("completion_time", datetime.now().isoformat())
    mlflow.set_tag("pipeline_status", "completed")
    print(f"✨ Pipeline completed successfully - Model Version: {model_version}")

def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="Enhanced ML Pipeline")
    parser.add_argument("--train-file", type=str, default="churn-bigml-80.csv", help="Training data CSV")
    parser.add_argument("--test-file", type=str, default="churn-bigml-20.csv", help="Test data CSV")
    args = parser.parse_args()

    try:
        run_enhanced_pipeline(args.train_file, args.test_file)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
