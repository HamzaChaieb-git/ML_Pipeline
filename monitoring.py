"""Module for monitoring model performance and system metrics."""

import os
import time
import json
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sqlalchemy
from sqlalchemy import create_engine, Table, Column, Integer, Float, String, DateTime, MetaData
import mlflow
from typing import Dict, List, Any


# Configure SQLite database for metrics storage
def setup_monitoring_db(db_path="artifacts/monitoring/monitoring.db"):
    """Set up SQLite database for storing monitoring metrics."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Create database engine
    engine = create_engine(f"sqlite:///{db_path}")
    
    # Define metadata
    metadata = MetaData()
    
    # Create predictions table
    predictions = Table(
        "predictions",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("timestamp", DateTime, default=datetime.now),
        Column("model_version", String),
        Column("input_data", String),  # JSON string of input features
        Column("prediction", Float),
        Column("actual", Float, nullable=True),  # May not always have ground truth
    )
    
    # Create model metrics table
    model_metrics = Table(
        "model_metrics",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("timestamp", DateTime, default=datetime.now),
        Column("model_version", String),
        Column("metric_name", String),
        Column("metric_value", Float),
    )
    
    # Create system metrics table
    system_metrics = Table(
        "system_metrics",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("timestamp", DateTime, default=datetime.now),
        Column("cpu_percent", Float),
        Column("memory_percent", Float),
        Column("disk_percent", Float),
    )
    
    # Create tables in database
    metadata.create_all(engine)
    
    print(f"Monitoring database set up at {db_path}")
    return engine


def log_prediction(engine, model_version, input_data, prediction, actual=None):
    """Log a prediction to the database for monitoring."""
    with engine.connect() as conn:
        conn.execute(
            "INSERT INTO predictions (timestamp, model_version, input_data, prediction, actual) VALUES (?, ?, ?, ?, ?)",
            (datetime.now(), model_version, json.dumps(input_data), float(prediction), actual),
        )
        conn.commit()


def log_model_metrics(engine, model_version, metrics):
    """Log model performance metrics to the database."""
    with engine.connect() as conn:
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float, np.number)):
                conn.execute(
                    "INSERT INTO model_metrics (timestamp, model_version, metric_name, metric_value) VALUES (?, ?, ?, ?)",
                    (datetime.now(), model_version, metric_name, float(metric_value)),
                )
        conn.commit()


def log_system_metrics(engine):
    """Log system metrics (CPU, memory, disk) to the database."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        with engine.connect() as conn:
            conn.execute(
                "INSERT INTO system_metrics (timestamp, cpu_percent, memory_percent, disk_percent) VALUES (?, ?, ?, ?)",
                (datetime.now(), cpu_percent, memory_percent, disk_percent),
            )
            conn.commit()
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "disk_percent": disk_percent
        }
    except Exception as e:
        print(f"Error logging system metrics: {e}")
        return None


def generate_monitoring_report(db_path="artifacts/monitoring/monitoring.db", output_dir="artifacts/monitoring/reports"):
    """Generate monitoring reports and visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    
    # Current timestamp for report filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate system metrics report
    system_df = pd.read_sql("SELECT * FROM system_metrics ORDER BY timestamp DESC LIMIT 100", engine)
    if not system_df.empty:
        # Convert timestamp strings to datetime objects
        system_df['timestamp'] = pd.to_datetime(system_df['timestamp'])
        
        # Plot system metrics
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(system_df['timestamp'], system_df['cpu_percent'])
        plt.title('CPU Usage (%)')
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(system_df['timestamp'], system_df['memory_percent'])
        plt.title('Memory Usage (%)')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(system_df['timestamp'], system_df['disk_percent'])
        plt.title('Disk Usage (%)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"system_metrics_{timestamp}.png"))
        plt.close()
        
        # Save raw data
        system_df.to_csv(os.path.join(output_dir, f"system_metrics_{timestamp}.csv"), index=False)
    
    # Generate model metrics report
    model_df = pd.read_sql(
        "SELECT model_version, metric_name, metric_value, timestamp FROM model_metrics ORDER BY timestamp DESC LIMIT 1000", 
        engine
    )
    if not model_df.empty:
        # Convert timestamp strings to datetime objects
        model_df['timestamp'] = pd.to_datetime(model_df['timestamp'])
        
        # Pivot the data for easier plotting
        pivot_df = model_df.pivot_table(
            index=['timestamp', 'model_version'], 
            columns='metric_name', 
            values='metric_value'
        ).reset_index()
        
        # Plot key metrics over time if we have accuracy and roc_auc
        if 'accuracy' in pivot_df.columns and 'roc_auc' in pivot_df.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(pivot_df['timestamp'], pivot_df['accuracy'], label='Accuracy')
            plt.plot(pivot_df['timestamp'], pivot_df['roc_auc'], label='ROC AUC')
            plt.title('Model Performance Metrics Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel('Metric Value')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"model_metrics_{timestamp}.png"))
            plt.close()
        
        # Save raw data
        model_df.to_csv(os.path.join(output_dir, f"model_metrics_{timestamp}.csv"), index=False)
    
    print(f"Monitoring reports generated in {output_dir}")


def monitor_system(interval=60, duration=3600, db_path="artifacts/monitoring/monitoring.db"):
    """
    Monitor system metrics for a specified duration.
    
    Args:
        interval: Seconds between measurements
        duration: Total duration to monitor in seconds
        db_path: Path to SQLite database
    """
    engine = setup_monitoring_db(db_path)
    start_time = time.time()
    end_time = start_time + duration
    
    print(f"Starting system monitoring for {duration/60:.1f} minutes...")
    
    try:
        while time.time() < end_time:
            metrics = log_system_metrics(engine)
            if metrics:
                print(f"Logged system metrics: CPU {metrics['cpu_percent']}%, Memory {metrics['memory_percent']}%, Disk {metrics['disk_percent']}%")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Monitoring stopped by user")
    
    print("System monitoring complete")
    generate_monitoring_report(db_path)


if __name__ == "__main__":
    # Example usage
    monitor_system(interval=10, duration=300)  # Monitor for 5 minutes with 10-second intervals
