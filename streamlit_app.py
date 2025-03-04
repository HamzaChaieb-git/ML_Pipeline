import streamlit as st
import pandas as pd
import joblib
import requests
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import pymongo
from bson.json_util import dumps
import numpy as np
import time

# Set page configuration
st.set_page_config(
    page_title="MLOps Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional UI styles
st.markdown("""
<style>
    /* Main area styling */
    .main {
        background-color: #f9fafb;
    }
    
    /* Custom card styling */
    .card {
        border-radius: 10px;
        padding: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    
    /* Headers styling */
    h1 {
        color: #1e3a8a;
        font-weight: 700;
        font-size: 32px;
        margin-bottom: 30px;
    }
    
    h2 {
        color: #1e3a8a;
        font-weight: 600;
        font-size: 24px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    h3 {
        color: #374151;
        font-weight: 600;
        font-size: 18px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    
    /* Metric card styling */
    .metric-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        margin-bottom: 10px;
    }
    
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
        min-width: 200px;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 16px;
        color: #6b7280;
        margin-bottom: 5px;
    }
    
    .status-success {
        background-color: #ecfdf5;
        border-left: 5px solid #059669;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    
    .status-warning {
        background-color: #fffbeb;
        border-left: 5px solid #d97706;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    
    .status-error {
        background-color: #fef2f2;
        border-left: 5px solid #dc2626;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1e40af;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #1e3a8a;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1e3a8a;
    }
    
    .css-1v3fvcr {
        background-color: #1e3a8a;
        color: white;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
        background-color: #f3f4f6;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f3f4f6;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e40af;
        color: white;
        border-radius: 10px;
    }
    
    /* Table styling */
    .dataframe {
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 14px;
        font-family: sans-serif;
        min-width: 400px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        border-radius: 10px;
        overflow: hidden;
    }
    
    .dataframe thead tr {
        background-color: #1e40af;
        color: #ffffff;
        text-align: left;
    }
    
    .dataframe th, .dataframe td {
        padding: 12px 15px;
        border-bottom: 1px solid #dddddd;
    }
    
    .dataframe tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    
    .dataframe tbody tr:nth-of-type(even) {
        background-color: #f3f4f6;
    }
    
    .dataframe tbody tr:last-of-type {
        border-bottom: 2px solid #1e40af;
    }
    
    /* Specific widget styling */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
    }
    
    /* Custom success badge */
    .success-badge {
        background-color: #10b981;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 13px;
        font-weight: 500;
    }
    
    /* Custom warning badge */
    .warning-badge {
        background-color: #f59e0b;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 13px;
        font-weight: 500;
    }
    
    /* Custom error badge */
    .error-badge {
        background-color: #ef4444;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 13px;
        font-weight: 500;
    }
    
    /* Status indicator dot */
    .status-dot {
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
    }
    
    .status-dot-green {
        background-color: #10b981;
    }
    
    .status-dot-amber {
        background-color: #f59e0b;
    }
    
    .status-dot-red {
        background-color: #ef4444;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1e3a8a;
        background-color: #f3f4f6;
        border-radius: 5px;
    }
    
    /* Streamlit divider */
    hr {
        margin-top: 30px;
        margin-bottom: 30px;
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(30, 58, 138, 0.75), rgba(0, 0, 0, 0));
    }
    
    /* Card title */
    .card-title {
        color: #1e3a8a;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    /* Responsive layout */
    @media (max-width: 768px) {
        .metric-card {
            width: 100%;
            margin-bottom: 15px;
        }
    }
    
    /* Logo and title styling */
    .title-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .logo-emoji {
        font-size: 48px;
        margin-right: 15px;
    }
    
    .dashboard-title {
        font-size: 32px;
        font-weight: 700;
        color: #1e3a8a;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def create_card(title, content):
    st.markdown(f"""
    <div class="card">
        <div class="card-title">{title}</div>
        {content}
    </div>
    """, unsafe_allow_html=True)

def color_status(val):
    """Color code status labels"""
    if val == "Healthy" or val == "Online" or val == "Connected" or val == "Active":
        return f'<span class="success-badge"><span class="status-dot status-dot-green"></span>{val}</span>'
    elif val == "Warning" or val == "Degraded" or val == "Staging":
        return f'<span class="warning-badge"><span class="status-dot status-dot-amber"></span>{val}</span>'
    elif val == "Error" or val == "Offline" or val == "Disconnected" or val == "Archived":
        return f'<span class="error-badge"><span class="status-dot status-dot-red"></span>{val}</span>'
    return val

def format_percentage(val):
    """Format numbers as percentages"""
    if isinstance(val, (int, float)):
        return f"{val:.2%}"
    return val

# Safe timestamp parsing function
def safe_parse_timestamp(timestamp_data):
    """
    Safely parse MongoDB timestamp data which can come in different formats:
    - As a dict with "$date" key (from JSON serialization)
    - As a millisecond timestamp
    - As an ISO string
    """
    if not timestamp_data:
        return None
        
    if isinstance(timestamp_data, dict):
        # Handle MongoDB extended JSON format
        if "$date" in timestamp_data:
            if isinstance(timestamp_data["$date"], int):
                # Millisecond timestamp
                return pd.to_datetime(timestamp_data["$date"], unit='ms')
            elif isinstance(timestamp_data["$date"], str):
                # ISO format string
                return pd.to_datetime(timestamp_data["$date"])
    elif isinstance(timestamp_data, str):
        # Try to parse as ISO format
        try:
            return pd.to_datetime(timestamp_data)
        except:
            return None
    elif isinstance(timestamp_data, int):
        # Assume millisecond timestamp
        return pd.to_datetime(timestamp_data, unit='ms')
            
    return None

# MongoDB connection helper
def get_mongodb_connection():
    """Connect to MongoDB and return client or None if connection fails"""
    try:
        # Try Docker network connection first
        client = pymongo.MongoClient("mongodb://mongodb:27017/")
        # Check connection
        client.admin.command('ping')
        return client
    except:
        try:
            # Fallback to localhost
            client = pymongo.MongoClient("mongodb://localhost:27017/")
            client.admin.command('ping')
            return client
        except Exception as e:
            st.error(f"Could not connect to MongoDB: {e}")
            return None

# Load model from artifacts (for fallback, if FastAPI fails)
@st.cache_resource
def load_latest_model():
    models_dir = os.path.join("artifacts", "models")
    if not os.path.exists(models_dir):
        return None

    model_files = [
        f
        for f in os.listdir(models_dir)
        if f.startswith("model_") and f.endswith(".joblib")
    ]
    if not model_files:
        return None

    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
    model_path = os.path.join(models_dir, latest_model)
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        return None

# Load model info
@st.cache_data
def get_model_info():
    """Get information about deployed models"""
    try:
        response = requests.get("http://fastapi:8000/model-status", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"model_loaded": False, "features": []}
    except:
        return {"model_loaded": False, "features": []}

# Get recent predictions from API
def get_recent_predictions(limit=50):
    """Fetch recent predictions from FastAPI"""
    try:
        response = requests.get(f"http://fastapi:8000/recent-predictions?limit={limit}", timeout=5)
        if response.status_code == 200:
            return response.json().get("recent_predictions", [])
        else:
            return []
    except Exception as e:
        return []

# Get recent predictions directly from MongoDB
def get_predictions_from_mongodb(limit=50):
    """Fetch recent predictions directly from MongoDB"""
    client = get_mongodb_connection()
    if not client:
        return []
    
    try:
        db = client["ml_monitoring"]
        predictions = list(db.predictions.find().sort("timestamp", -1).limit(limit))
        # Convert MongoDB documents to JSON serializable format
        predictions_json = json.loads(dumps(predictions))
        return predictions_json
    except Exception as e:
        return []
    finally:
        if client:
            client.close()

# Process predictions into a DataFrame with proper timestamp handling
def process_predictions_dataframe(predictions):
    """Process predictions data from MongoDB into a DataFrame with proper timestamp handling"""
    if not predictions:
        return pd.DataFrame()
        
    # Convert to DataFrame
    predictions_df = pd.DataFrame([
        {
            "timestamp": safe_parse_timestamp(p.get("timestamp")),
            "model_version": p.get("model_version", "Unknown"),
            "prediction": p.get("prediction", 0),
            "features": p.get("features", {})
        }
        for p in predictions
    ])
    
    # Drop rows with None timestamps
    if not predictions_df.empty:
        predictions_df = predictions_df.dropna(subset=["timestamp"])
    
    return predictions_df

# Get model metrics from MongoDB
def get_model_metrics_from_mongodb():
    """Fetch model metrics from MongoDB"""
    client = get_mongodb_connection()
    if not client:
        return []
    
    try:
        db = client["ml_monitoring"]
        metrics = list(db.model_metrics.find().sort("timestamp", -1))
        # Convert MongoDB documents to JSON serializable format
        metrics_json = json.loads(dumps(metrics))
        return metrics_json
    except Exception as e:
        return []
    finally:
        if client:
            client.close()

# Process model metrics data
def process_model_metrics_dataframe(model_metrics):
    """Process model metrics data from MongoDB into a DataFrame with proper timestamp handling"""
    if not model_metrics:
        return pd.DataFrame()
    
    # Convert to DataFrame
    all_metrics_df = pd.DataFrame()
    
    for m in model_metrics:
        timestamp = safe_parse_timestamp(m.get("timestamp"))
        model_version = m.get("model_version", "Unknown")
        metrics_dict = m.get("metrics", {})
        
        if metrics_dict and timestamp is not None:
            # Create a row with timestamp and model_version
            row = {
                "timestamp": timestamp,
                "model_version": model_version
            }
            
            # Add all metrics to the row
            row.update(metrics_dict)
            
            # Append to the dataframe
            all_metrics_df = pd.concat([all_metrics_df, pd.DataFrame([row])], ignore_index=True)
    
    return all_metrics_df

# Trigger model evaluation via API
def trigger_model_evaluation():
    """Trigger model evaluation via API"""
    try:
        response = requests.post("http://fastapi:8000/debug-metrics", timeout=10)
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        return False

# Function to create confusion matrix
def create_confusion_matrix(metrics):
    """Create a confusion matrix visualization from metrics"""
    if not all(k in metrics for k in ["true_positives", "false_positives", "true_negatives", "false_negatives"]):
        return None
    
    confusion_data = [
        [metrics["true_negatives"], metrics["false_positives"]],
        [metrics["false_negatives"], metrics["true_positives"]]
    ]
    
    fig = px.imshow(
        confusion_data,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Negative (0)", "Positive (1)"],
        y=["Negative (0)", "Positive (1)"],
        text_auto=True,
        color_continuous_scale="Blues"
    )
    
    fig.update_layout(
        title="Confusion Matrix",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14)
    )
    
    return fig

# Get system metrics
def get_system_metrics():
    """
    Get system metrics from MongoDB or generate mock data if not available
    """
    try:
        client = get_mongodb_connection()
        if client:
            db = client["ml_monitoring"]
            system_metrics = list(db.system_metrics.find().sort("timestamp", -1).limit(24))
            system_metrics_json = json.loads(dumps(system_metrics))
            
            # Process metrics
            if system_metrics_json:
                system_df = pd.DataFrame([
                    {
                        "timestamp": safe_parse_timestamp(m.get("timestamp")),
                        "cpu_percent": m.get("cpu_percent", 0),
                        "memory_percent": m.get("memory_percent", 0),
                        "disk_percent": m.get("disk_percent", 0)
                    }
                    for m in system_metrics_json
                ])
                
                # Drop rows with None timestamps
                system_df = system_df.dropna(subset=["timestamp"])
                
                # Sort by timestamp
                system_df = system_df.sort_values("timestamp")
                
                return system_df, True
    except Exception as e:
        pass
        
    # Generate mock data if real metrics aren't available
    mock_dates = [datetime.now() - timedelta(minutes=x*10) for x in range(24)]
    mock_cpu = [30 + x*2 + (x**2)/10 for x in range(24)]
    mock_memory = [45 + x/2 for x in range(24)]
    mock_disk = [65 + x/10 for x in range(24)]
    
    # Create dataframe with mock data
    system_df = pd.DataFrame({
        'timestamp': mock_dates,
        'cpu_percent': mock_cpu,
        'memory_percent': mock_memory,
        'disk_percent': mock_disk
    })
    
    return system_df, False

# Get service status
def get_service_status():
    """
    Check the status of all services in the MLOps stack
    Returns a dictionary with status information
    """
    services = {
        "FastAPI": {"url": "http://fastapi:8000/health", "fallback_url": "http://localhost:8000/health"},
        "MLflow": {"url": "http://mlflow:5001/", "fallback_url": "http://localhost:5001/"},
    }
    
    statuses = {}
    
    for service_name, service_info in services.items():
        try:
            response = requests.get(service_info["url"], timeout=2)
            if response.status_code == 200:
                statuses[service_name] = "Healthy"
            else:
                statuses[service_name] = "Error"
        except:
            try:
                # Try fallback URL
                response = requests.get(service_info["fallback_url"], timeout=2)
                if response.status_code == 200:
                    statuses[service_name] = "Healthy"
                else:
                    statuses[service_name] = "Error"
            except:
                statuses[service_name] = "Offline"
    
    # Check MongoDB status
    mongo_client = get_mongodb_connection()
    if mongo_client:
        statuses["MongoDB"] = "Connected"
        mongo_client.close()
    else:
        statuses["MongoDB"] = "Disconnected"
    
    # Mock status for monitoring service (replace with actual check if available)
    statuses["Monitoring"] = "Healthy"
    
    return statuses

# Header with logo and navigation
def render_header():
    """Render the top header with logo and title"""
    st.markdown("""
    <div class="title-container">
        <div class="logo-emoji">🧠</div>
        <h1 class="dashboard-title">MLOps Churn Prediction Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
def render_sidebar():
    with st.sidebar:
        st.subheader("Navigation")
        
        # Navigation
        page = st.radio(
            "Choose a page:",
            ["Dashboard", "Model Performance", "Make Prediction", "Recent Predictions", "System Monitoring"]
        )
        
        # Model info section
        st.markdown("---")
        st.subheader("Model Information")
        
        # Model status
        model_info = get_model_info()
        if model_info["model_loaded"]:
            st.markdown('<div class="status-success">✅ Model is loaded and ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">❌ Model is not loaded</div>', unsafe_allow_html=True)
        
        # Load model for info
        model = load_latest_model()
        if model and hasattr(model, 'model_info'):
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Current Model</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <table style="width:100%; border-collapse: collapse;">
                <tr>
                    <td style="padding:8px; border-bottom:1px solid #eee;">Version:</td>
                    <td style="padding:8px; border-bottom:1px solid #eee; font-weight:bold;">{model.model_info.get('version', 'Unknown')}</td>
                </tr>
                <tr>
                    <td style="padding:8px; border-bottom:1px solid #eee;">Features:</td>
                    <td style="padding:8px; border-bottom:1px solid #eee; font-weight:bold;">{len(model.model_info.get('features', []))}</td>
                </tr>
                <tr>
                    <td style="padding:8px; border-bottom:1px solid #eee;">Training Samples:</td>
                    <td style="padding:8px; border-bottom:1px solid #eee; font-weight:bold;">{model.model_info.get('training_samples', 'Unknown')}</td>
                </tr>
                <tr>
                    <td style="padding:8px;">Stage:</td>
                    <td style="padding:8px; font-weight:bold;">
                        <span class="warning-badge"><span class="status-dot status-dot-amber"></span>
                            {model.model_info.get('stage', 'Staging')}
                        </span>
                    </td>
                </tr>
            </table>
            </div>
            """, unsafe_allow_html=True)
        
        # Actions section
        st.markdown("---")
        st.subheader("Actions")
        
        if st.button("Add Test Metrics"):
            with st.spinner("Adding test metrics..."):
                success = trigger_model_evaluation()
                if success:
                    st.success("✅ Test metrics added successfully!")
                    # Force refresh by clearing cache
                    st.cache_data.clear()
                    # Show feedback for 2 seconds then refresh
                    time.sleep(1)
                    st.experimental_rerun()
                else:
                    st.error("❌ Failed to add test metrics")
    
    return page

# Dashboard main page
def render_dashboard():
    # Get model metrics
    model_metrics = get_model_metrics_from_mongodb()
    metrics_df = process_model_metrics_dataframe(model_metrics)
    
    # Get latest metrics
    latest_metrics = {}
    if not metrics_df.empty:
        latest_metrics = metrics_df.iloc[0].to_dict()
    
    # Key metrics section
    st.markdown("<h2>Key Performance Metrics</h2>", unsafe_allow_html=True)
    
    # Metrics cards row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="card" style="border-top: 4px solid #3b82f6;">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{}</div>
            <div style="font-size: 12px; color: #6b7280;">Overall prediction accuracy</div>
        </div>
        """.format(format_percentage(latest_metrics.get('accuracy', 0))), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card" style="border-top: 4px solid #10b981;">
            <div class="metric-label">ROC AUC</div>
            <div class="metric-value">{}</div>
            <div style="font-size: 12px; color: #6b7280;">Area under ROC curve</div>
        </div>
        """.format(format_percentage(latest_metrics.get('roc_auc', 0))), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card" style="border-top: 4px solid #f59e0b;">
            <div class="metric-label">Precision</div>
            <div class="metric-value">{}</div>
            <div style="font-size: 12px; color: #6b7280;">True positives accuracy</div>
        </div>
        """.format(format_percentage(latest_metrics.get('precision', 0))), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="card" style="border-top: 4px solid #ef4444;">
            <div class="metric-label">Recall</div>
            <div class="metric-value">{}</div>
            <div style="font-size: 12px; color: #6b7280;">Detection rate</div>
        </div>
        """.format(format_percentage(latest_metrics.get('recall', 0))), unsafe_allow_html=True)
    
    # Display confusion matrix if available
    if latest_metrics and all(k in latest_metrics for k in ["true_positives", "false_positives", "true_negatives", "false_negatives"]):
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("<h2>Confusion Matrix</h2>", unsafe_allow_html=True)
            conf_matrix_fig = create_confusion_matrix(latest_metrics)
            if conf_matrix_fig:
                st.plotly_chart(conf_matrix_fig, use_container_width=True)
        
        with col2:
            # Class-specific metrics in nested cards
            st.markdown("<h2>Class Metrics</h2>", unsafe_allow_html=True)
            
            # Class 0 (Negative)
            st.markdown("""
            <div class="card" style="border-left: 4px solid #3b82f6; margin-bottom: 15px;">
                <div style="font-weight: 600; margin-bottom: 10px; color: #1e3a8a;">Negative Class (0)</div>
                <table style="width:100%;">
                    <tr>
                        <td style="padding:3px; color: #6b7280;">True Negatives:</td>
                        <td style="padding:3px; font-weight:bold; text-align:right;">{}</td>
                    </tr>
                    <tr>
                        <td style="padding:3px; color: #6b7280;">False Positives:</td>
                        <td style="padding:3px; font-weight:bold; text-align:right;">{}</td>
                    </tr>
                    <tr>
                        <td style="padding:3px; color: #6b7280;">Specificity:</td>
                        <td style="padding:3px; font-weight:bold; text-align:right;">{}</td>
                    </tr>
                </table>
            </div>
            """.format(
                int(latest_metrics.get('true_negatives', 0)),
                int(latest_metrics.get('false_positives', 0)),
                format_percentage(latest_metrics.get('true_negatives', 0) / (latest_metrics.get('true_negatives', 0) + latest_metrics.get('false_positives', 1)) if latest_metrics.get('true_negatives', 0) + latest_metrics.get('false_positives', 0) > 0 else 0)
            ), unsafe_allow_html=True)
            
            # Class 1 (Positive)
            st.markdown("""
            <div class="card" style="border-left: 4px solid #ef4444; margin-bottom: 15px;">
                <div style="font-weight: 600; margin-bottom: 10px; color: #1e3a8a;">Positive Class (1)</div>
                <table style="width:100%;">
                    <tr>
                        <td style="padding:3px; color: #6b7280;">True Positives:</td>
                        <td style="padding:3px; font-weight:bold; text-align:right;">{}</td>
                    </tr>
                    <tr>
                        <td style="padding:3px; color: #6b7280;">False Negatives:</td>
                        <td style="padding:3px; font-weight:bold; text-align:right;">{}</td>
                    </tr>
                    <tr>
                        <td style="padding:3px; color: #6b7280;">Recall:</td>
                        <td style="padding:3px; font-weight:bold; text-align:right;">{}</td>
                    </tr>
                </table>
            </div>
            """.format(
                int(latest_metrics.get('true_positives', 0)),
                int(latest_metrics.get('false_negatives', 0)),
                format_percentage(latest_metrics.get('recall', 0))
            ), unsafe_allow_html=True)
    
    # Recent predictions section
    st.markdown("<h2>Recent Predictions</h2>", unsafe_allow_html=True)
    
    # Get recent predictions
    predictions = get_predictions_from_mongodb(10)
    predictions_df = process_predictions_dataframe(predictions)
    
    if not predictions_df.empty:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create bar chart of recent prediction probabilities
            fig = px.bar(
                predictions_df,
                y='prediction',
                labels={'prediction': 'Churn Probability', 'index': 'Prediction ID'},
                title='Recent Prediction Probabilities',
                height=300,
                color='prediction',
                color_continuous_scale='RdYlGn_r',
            )
            fig.update_layout(
                xaxis_title="Recent Predictions",
                yaxis_title="Churn Probability",
                yaxis_range=[0, 1],
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Add prediction distribution by risk level
            predictions_df["risk_level"] = pd.cut(
                predictions_df["prediction"],
                bins=[0, 0.3, 0.7, 1.0],
                labels=["Low", "Medium", "High"]
            )
            
            # Count by risk level
            risk_counts = predictions_df["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            
            # Create pie chart
            fig = px.pie(
                risk_counts,
                values="Count",
                names="Risk Level",
                title="Distribution by Risk Level",
                color="Risk Level",
                color_discrete_map={
                    "Low": "green",
                    "Medium": "orange",
                    "High": "red"
                },
                hole=0.4
            )
            
            fig.update_layout(
                legend_title="Risk Level",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No recent predictions available. Make some predictions first.")
    
    # System status section
    st.markdown("<h2>System Status</h2>", unsafe_allow_html=True)
    
    # Get service statuses
    service_statuses = get_service_status()
    
    # Create a grid of status cards
    cols = st.columns(4)
    
    for i, (service, status) in enumerate(service_statuses.items()):
        # Color code based on status
        status_html = color_status(status)
        
        cols[i % 4].markdown(f"""
        <div class="card" style="text-align: center; padding: 15px;">
            <div style="font-size: 40px; margin-bottom: 10px;">
                {'🖥️' if service == 'FastAPI' else '📊' if service == 'MLflow' else '🗄️' if service == 'MongoDB' else '📈'}
            </div>
            <div style="font-weight: 600; margin-bottom: 10px;">{service}</div>
            <div>{status_html}</div>
        </div>
        """, unsafe_allow_html=True)

# Model Performance Page
def render_model_performance():
    st.markdown("<h2>Model Performance Analysis</h2>", unsafe_allow_html=True)
    
    # Get model metrics from MongoDB
    model_metrics = get_model_metrics_from_mongodb()
    metrics_df = process_model_metrics_dataframe(model_metrics)
    
    if not metrics_df.empty:
        # Define metric categories for better visualization
        performance_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        count_metrics = ["true_positives", "true_negatives", "false_positives", "false_negatives"]
        loss_metrics = ["log_loss"]
        
        # Latest model metrics summary
        latest_metrics = metrics_df.iloc[0].to_dict()
        model_version = latest_metrics.get("model_version", "Unknown")
        
        # Display in cards layout
        st.markdown("<h3>Latest Model Metrics</h3>", unsafe_allow_html=True)
        
        # First row of metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{latest_metrics.get('accuracy', 0):.4f}</div>
                <div style="font-size: 14px; color: #6b7280;">Model: {model_version}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="card">
                <div class="metric-label">True Positives</div>
                <div class="metric-value">{int(latest_metrics.get('true_positives', 0))}</div>
                <div style="font-size: 14px; color: #6b7280;">Correctly identified as positive</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card">
                <div class="metric-label">Precision</div>
                <div class="metric-value">{latest_metrics.get('precision', 0):.4f}</div>
                <div style="font-size: 14px; color: #6b7280;">Positive prediction accuracy</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="card">
                <div class="metric-label">True Negatives</div>
                <div class="metric-value">{int(latest_metrics.get('true_negatives', 0))}</div>
                <div style="font-size: 14px; color: #6b7280;">Correctly identified as negative</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="card">
                <div class="metric-label">Recall</div>
                <div class="metric-value">{latest_metrics.get('recall', 0):.4f}</div>
                <div style="font-size: 14px; color: #6b7280;">True positive rate</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="card">
                <div class="metric-label">False Positives</div>
                <div class="metric-value">{int(latest_metrics.get('false_positives', 0))}</div>
                <div style="font-size: 14px; color: #6b7280;">Incorrectly identified as positive</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="card">
                <div class="metric-label">F1 Score</div>
                <div class="metric-value">{latest_metrics.get('f1', 0):.4f}</div>
                <div style="font-size: 14px; color: #6b7280;">Harmonic mean of precision and recall</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="card">
                <div class="metric-label">False Negatives</div>
                <div class="metric-value">{int(latest_metrics.get('false_negatives', 0))}</div>
                <div style="font-size: 14px; color: #6b7280;">Incorrectly identified as negative</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Tabs for different visualizations
        tabs = st.tabs(["Confusion Matrix", "Metrics Over Time", "Model Comparison"])
        
        with tabs[0]:
            # Confusion Matrix visualization
            conf_matrix_fig = create_confusion_matrix(latest_metrics)
            if conf_matrix_fig:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.plotly_chart(conf_matrix_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Add confusion matrix interpretation
                st.markdown("""
                <div class="card">
                    <div class="card-title">Confusion Matrix Interpretation</div>
                    <p>The confusion matrix shows the counts of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN):</p>
                    <ul>
                        <li><strong>True Positives (TP)</strong>: Customers correctly predicted to churn</li>
                        <li><strong>True Negatives (TN)</strong>: Customers correctly predicted to stay</li>
                        <li><strong>False Positives (FP)</strong>: Customers incorrectly predicted to churn</li>
                        <li><strong>False Negatives (FN)</strong>: Customers incorrectly predicted to stay</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        with tabs[1]:
            # Metrics over time visualization - Split by category with proper scaling
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance metrics
                available_perf_metrics = [m for m in performance_metrics if m in metrics_df.columns]
                
                if available_perf_metrics:
                    fig1 = px.line(
                        metrics_df,
                        x="timestamp",
                        y=available_perf_metrics,
                        title="Performance Metrics Trend",
                        labels={"value": "Score", "timestamp": "Time", "variable": "Metric"},
                        markers=True
                    )
                    
                    fig1.update_layout(
                        yaxis_title="Metric Value (0-1)",
                        yaxis=dict(range=[0, 1]),
                        xaxis_title="Date",
                        legend_title="Metrics",
                        hovermode="x unified",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.plotly_chart(fig1, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                # Loss metrics
                available_loss_metrics = [m for m in loss_metrics if m in metrics_df.columns]
                
                if available_loss_metrics:
                    fig3 = px.line(
                        metrics_df,
                        x="timestamp",
                        y=available_loss_metrics,
                        title="Loss Metrics Trend",
                        labels={"value": "Loss", "timestamp": "Time", "variable": "Metric"},
                        markers=True
                    )
                    
                    fig3.update_layout(
                        yaxis_title="Loss Value",
                        xaxis_title="Date",
                        legend_title="Metrics",
                        hovermode="x unified",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.plotly_chart(fig3, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Confusion matrix counts over time (full width)
            available_count_metrics = [m for m in count_metrics if m in metrics_df.columns]
            
            if available_count_metrics:
                fig2 = px.line(
                    metrics_df,
                    x="timestamp",
                    y=available_count_metrics,
                    title="Confusion Matrix Counts Trend",
                    labels={"value": "Count", "timestamp": "Time", "variable": "Metric"},
                    markers=True
                )
                
                fig2.update_layout(
                    yaxis_title="Count",
                    xaxis_title="Date",
                    legend_title="Metrics",
                    hovermode="x unified",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
        with tabs[2]:
            # Model version comparison
            model_versions = metrics_df["model_version"].unique()
            if len(model_versions) > 1:
                # Create comparison charts
                version_metrics = []
                
                for version in model_versions:
                    version_df = metrics_df[metrics_df["model_version"] == version]
                    if not version_df.empty:
                        latest_version_metrics = version_df.iloc[0]
                        metrics_dict = {
                            "model_version": version,
                            "accuracy": latest_version_metrics.get("accuracy", 0),
                            "precision": latest_version_metrics.get("precision", 0),
                            "recall": latest_version_metrics.get("recall", 0),
                            "f1": latest_version_metrics.get("f1", 0),
                            "roc_auc": latest_version_metrics.get("roc_auc", 0)
                        }
                        version_metrics.append(metrics_dict)
                
                if version_metrics:
                    comparison_df = pd.DataFrame(version_metrics)
                    
                    # Bar chart comparison
                    fig = px.bar(
                        comparison_df,
                        x="model_version",
                        y=["accuracy", "precision", "recall", "f1", "roc_auc"],
                        title="Model Version Performance Comparison",
                        barmode="group",
                        labels={"value": "Score", "model_version": "Model Version", "variable": "Metric"}
                    )
                    
                    fig.update_layout(
                        yaxis_title="Metric Value",
                        yaxis=dict(range=[0, 1]),
                        xaxis_title="Model Version",
                        legend_title="Metrics",
                        hovermode="x unified",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Only one model version available. Add more models for comparison.")
        
        # Display all metrics in a data table
        with st.expander("View Raw Metrics Data"):
            # Format timestamp for display
            display_df = metrics_df.copy()
            display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # Select columns for display
            display_cols = ["timestamp", "model_version"]
            for metric_group in [performance_metrics, count_metrics, loss_metrics]:
                for col in metric_group:
                    if col in display_df.columns:
                        display_cols.append(col)
            
            # Display the filtered dataframe
            display_df = display_df[display_cols]
            st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("No model metrics available. Use the 'Add Test Metrics' button to generate test metrics.")
        
        # Add a button to add test metrics directly on this page
        if st.button("Add Test Metrics Now"):
            with st.spinner("Adding test metrics..."):
                success = trigger_model_evaluation()
                if success:
                    st.success("✅ Test metrics added successfully!")
                    time.sleep(1)
                    st.experimental_rerun()
                else:
                    st.error("❌ Failed to add test metrics")

# Make Prediction Page
def render_prediction_page():
    st.markdown("<h2>Make New Prediction</h2>", unsafe_allow_html=True)
    
    # Get the feature list from model info
    model_info = get_model_info()
    features = model_info.get("features", [
        "Total day minutes",
        "Customer service calls",
        "International plan",
        "Total intl minutes",
        "Total intl calls",
        "Total eve minutes",
        "Number vmail messages",
        "Voice mail plan",
    ])
    
    # Custom form styling
    st.markdown("""
    <style>
    .form-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='form-card'>", unsafe_allow_html=True)
    
    # Form for input
    with st.form("prediction_form"):
        st.markdown("<h3>Customer Data</h3>", unsafe_allow_html=True)
        st.markdown("Enter customer information to predict churn probability.")
        
        col1, col2 = st.columns(2)
        
        # Initialize input data dictionary
        input_data = {}
        
        # Create input fields
        with col1:
            if "Total day minutes" in features:
                input_data["Total day minutes"] = [st.number_input("Total Day Minutes", value=120.5, step=1.0, 
                                                                 help="Total minutes spent on day calls")]
            
            if "Customer service calls" in features:
                input_data["Customer service calls"] = [st.number_input("Customer Service Calls", value=1, step=1,
                                                                      help="Number of calls to customer service")]
            
            if "International plan" in features:
                input_data["International plan"] = [int(st.selectbox("International Plan", ["No", "Yes"], 
                                                                    help="Does the customer have international plan?") == "Yes")]
            
            if "Total intl minutes" in features:
                input_data["Total intl minutes"] = [st.number_input("Total International Minutes", value=10.2, step=0.1,
                                                                  help="Total minutes spent on international calls")]
        
        with col2:
            if "Total intl calls" in features:
                input_data["Total intl calls"] = [st.number_input("Total International Calls", value=3, step=1,
                                                               help="Number of international calls")]
            
            if "Total eve minutes" in features:
                input_data["Total eve minutes"] = [st.number_input("Total Evening Minutes", value=200.0, step=1.0,
                                                                help="Total minutes spent on evening calls")]
            
            if "Number vmail messages" in features:
                input_data["Number vmail messages"] = [st.number_input("Number of Voicemail Messages", value=0, step=1,
                                                                     help="Number of voicemail messages")]
            
            if "Voice mail plan" in features:
                input_data["Voice mail plan"] = [int(st.selectbox("Voice Mail Plan", ["No", "Yes"],
                                                               help="Does the customer have a voicemail plan?") == "Yes")]
        
        submitted = st.form_submit_button("Predict Churn", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if submitted:
        st.markdown("<h3>Prediction Result</h3>", unsafe_allow_html=True)
        
        try:
            # Call FastAPI endpoint
            with st.spinner("Generating prediction..."):
                response = requests.post("http://fastapi:8000/predict", json=input_data, timeout=10)
                response.raise_for_status()
                prediction = response.json()["churn_probabilities"][0]
            
            # Display prediction with gauge chart and risk assessment
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction * 100,
                    title={"text": "Churn Probability", "font": {"size": 24}},
                    gauge={
                        "axis": {"range": [0, 100], "ticksuffix": "%", "tickwidth": 1, "tickcolor": "darkblue"},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 30], "color": "#10b981"},  # Green
                            {"range": [30, 70], "color": "#f59e0b"},  # Yellow
                            {"range": [70, 100], "color": "#ef4444"}  # Red
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 70
                        }
                    },
                    number={"suffix": "%", "valueformat": ".1f", "font": {"size": 28}}
                ))
                
                fig.update_layout(
                    height=350,
                    margin=dict(l=30, r=30, t=50, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"size": 16, "color": "#333"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                risk_level = "Low" if prediction < 0.3 else "Medium" if prediction < 0.7 else "High"
                risk_color = "#10b981" if risk_level == "Low" else "#f59e0b" if risk_level == "Medium" else "#ef4444"
                
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">Churn Risk Assessment</div>
                    <div style="text-align: center; margin: 20px 0;">
                        <span style="font-size: 24px; padding: 8px 16px; background-color: {risk_color}; 
                                color: white; border-radius: 30px; font-weight: 600;">{risk_level} Risk</span>
                    </div>
                    <p style="font-size: 18px; text-align: center; margin-bottom: 20px;">
                        Probability: <strong>{prediction:.2%}</strong>
                    </p>
                    <div style="background-color: {'#ecfdf5' if risk_level == 'Low' else '#fffbeb' if risk_level == 'Medium' else '#fef2f2'}; 
                         padding: 15px; border-radius: 8px; border-left: 4px solid {risk_color};">
                """, unsafe_allow_html=True)
                
                # Recommendation based on churn risk
                if risk_level == "Low":
                    st.markdown("""
                    <p style="margin: 0;">✅ <strong>Customer likely to stay.</strong></p>
                    <p style="margin: 5px 0 0 0;">Maintain regular engagement with standard offerings.</p>
                    """, unsafe_allow_html=True)
                elif risk_level == "Medium":
                    st.markdown("""
                    <p style="margin: 0;">⚠️ <strong>Monitor closely.</strong></p>
                    <p style="margin: 5px 0 0 0;">Consider offering special promotions or personalized check-ins.</p>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <p style="margin: 0;">🚨 <strong>High churn risk!</strong></p>
                    <p style="margin: 5px 0 0 0;">Immediate retention action recommended. Consider custom retention plan.</p>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div></div>", unsafe_allow_html=True)
                
                # Additional actions card
                st.markdown("""
                <div class="card" style="margin-top: 20px;">
                    <div class="card-title">Recommended Actions</div>
                    <ul style="padding-left: 20px;">
                """, unsafe_allow_html=True)
                
                if risk_level == "Low":
                    st.markdown("""
                    <li>Schedule regular follow-up in 90 days</li>
                    <li>Send satisfaction survey</li>
                    <li>Offer standard loyalty benefits</li>
                    """, unsafe_allow_html=True)
                elif risk_level == "Medium":
                    st.markdown("""
                    <li>Schedule follow-up call within 30 days</li>
                    <li>Offer special promotional discount</li>
                    <li>Review customer usage patterns</li>
                    <li>Consider service plan optimization</li>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <li>Schedule immediate retention call</li>
                    <li>Prepare significant retention offer</li>
                    <li>Assign dedicated account manager</li>
                    <li>Customize service plan</li>
                    <li>Offer premium loyalty upgrade</li>
                    """, unsafe_allow_html=True)
                
                st.markdown("</ul></div>", unsafe_allow_html=True)
            
            # Feature analysis section
            st.markdown("<h3>Customer Profile Analysis</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Create a radar chart for input features
                categories = list(input_data.keys())
                values = [input_data[k][0] for k in categories]
                
                # Normalize values for radar chart
                min_vals = {
                    "Total day minutes": 0, "Customer service calls": 0, 
                    "International plan": 0, "Total intl minutes": 0,
                    "Total intl calls": 0, "Total eve minutes": 0, 
                    "Number vmail messages": 0, "Voice mail plan": 0
                }
                max_vals = {
                    "Total day minutes": 350, "Customer service calls": 10, 
                    "International plan": 1, "Total intl minutes": 25,
                    "Total intl calls": 20, "Total eve minutes": 350, 
                    "Number vmail messages": 50, "Voice mail plan": 1
                }
                
                normalized_values = []
                for k, v in zip(categories, values):
                    min_val = min_vals.get(k, 0)
                    max_val = max_vals.get(k, 1)
                    if max_val == min_val:
                        normalized_values.append(1 if v > 0 else 0)
                    else:
                        normalized_values.append((v - min_val) / (max_val - min_val))
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=normalized_values,
                    theta=categories,
                    fill='toself',
                    name='Customer Profile',
                    marker=dict(color="#1e40af")
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=False,
                    title="Customer Feature Profile",
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='card-title'>Key Risk Factors</div>", unsafe_allow_html=True)
                
                # Define risk thresholds
                risk_factors = []
                
                # Check for risk factors (these are simplified examples)
                if input_data.get("Customer service calls", [0])[0] > 3:
                    risk_factors.append({
                        "factor": "High customer service calls",
                        "value": f"{input_data['Customer service calls'][0]} calls",
                        "impact": "High",
                        "description": "Excessive customer service calls indicate dissatisfaction."
                    })
                
                if input_data.get("Total day minutes", [0])[0] > 250:
                    risk_factors.append({
                        "factor": "High day minutes usage",
                        "value": f"{input_data['Total day minutes'][0]} minutes",
                        "impact": "Medium",
                        "description": "High usage may lead to bill shock."
                    })
                
                if input_data.get("International plan", [0])[0] == 1 and input_data.get("Total intl minutes", [0])[0] < 5:
                    risk_factors.append({
                        "factor": "International plan underutilization",
                        "value": f"{input_data['Total intl minutes'][0]} minutes",
                        "impact": "Medium",
                        "description": "Customer is paying for an international plan but barely using it."
                    })
                
                if input_data.get("Voice mail plan", [0])[0] == 1 and input_data.get("Number vmail messages", [0])[0] == 0:
                    risk_factors.append({
                        "factor": "Voicemail plan not used",
                        "value": "0 messages",
                        "impact": "Low",
                        "description": "Customer has voicemail plan but doesn't use it."
                    })
                
                if not risk_factors:
                    risk_factors.append({
                        "factor": "No significant risk factors",
                        "value": "N/A",
                        "impact": "Low",
                        "description": "Customer profile doesn't show any obvious risk patterns."
                    })
                
                # Display risk factors
                for i, factor in enumerate(risk_factors):
                    impact_color = "#ef4444" if factor["impact"] == "High" else "#f59e0b" if factor["impact"] == "Medium" else "#10b981"
                    
                    st.markdown(f"""
                    <div style="padding: 10px; margin-bottom: 10px; border-left: 4px solid {impact_color}; background-color: #f9fafb;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <div style="font-weight: 600;">{factor["factor"]}</div>
                            <div style="color: {impact_color}; font-weight: 600;">{factor["impact"]} Impact</div>
                        </div>
                        <div style="color: #6b7280; margin-bottom: 5px;">Value: {factor["value"]}</div>
                        <div>{factor["description"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Store prediction data
                if len(risk_factors) > 0:
                    st.markdown("""
                    <div class="card" style="margin-top: 20px;">
                        <div class="card-title">Save Prediction</div>
                        <p>This prediction has been automatically saved to the database for later analysis.</p>
                        <p style="color: #6b7280; font-size: 14px;">Prediction ID: auto-generated<br>
                        Timestamp: {}</p>
                    </div>
                    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            
            # Fallback: Use local model if FastAPI fails
            model = load_latest_model()
            if model:
                st.warning("Falling back to local model...")
                input_df = pd.DataFrame(input_data)
                prediction = model.predict_proba(input_df)[:, 1][0]
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction * 100,
                    title={"text": "Churn Probability (Local Model)"},
                    gauge={
                        "axis": {"range": [0, 100], "ticksuffix": "%"},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 30], "color": "#10b981"},
                            {"range": [30, 70], "color": "#f59e0b"},
                            {"range": [70, 100], "color": "#ef4444"}
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 70
                        }
                    },
                    number={"suffix": "%", "valueformat": ".1f"}
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No model available for prediction.")

# Recent Predictions Page
def render_recent_predictions():
    st.markdown("<h2>Recent Prediction Analysis</h2>", unsafe_allow_html=True)
    
    # Number of predictions to display
    num_predictions = st.slider("Number of predictions to analyze", min_value=5, max_value=100, value=20)
    
    # Get predictions from MongoDB
    predictions = get_predictions_from_mongodb(num_predictions)
    
    # Process predictions with safe timestamp handling
    predictions_df = process_predictions_dataframe(predictions)
    
    if not predictions_df.empty:
        # Summary statistics section
        st.markdown("<h3>Prediction Statistics</h3>", unsafe_allow_html=True)
        
        # Calculate additional stats
        avg_prediction = predictions_df['prediction'].mean()
        high_risk = (predictions_df['prediction'] > 0.7).sum()
        high_risk_percent = high_risk/len(predictions_df)
        medium_risk = ((predictions_df['prediction'] > 0.3) & (predictions_df['prediction'] <= 0.7)).sum()
        medium_risk_percent = medium_risk/len(predictions_df)
        low_risk = (predictions_df['prediction'] <= 0.3).sum()
        low_risk_percent = low_risk/len(predictions_df)
        
        # Stats cards layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <div class="metric-label">Predictions Analyzed</div>
                <div class="metric-value">{len(predictions_df)}</div>
                <div style="font-size: 12px; color: #6b7280;">Total predictions</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <div class="metric-label">Average Churn Probability</div>
                <div class="metric-value">{avg_prediction:.1%}</div>
                <div style="font-size: 12px; color: #6b7280;">Across all predictions</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <div class="metric-label">High Risk Customers</div>
                <div class="metric-value">{high_risk} <span style="font-size: 18px; color: #6b7280;">({high_risk_percent:.1%})</span></div>
                <div style="font-size: 12px; color: #6b7280;">Probability > 70%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <div class="metric-label">Low Risk Customers</div>
                <div class="metric-value">{low_risk} <span style="font-size: 18px; color: #6b7280;">({low_risk_percent:.1%})</span></div>
                <div style="font-size: 12px; color: #6b7280;">Probability < 30%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Add tabs for different visualizations
        tabs = st.tabs(["Distribution", "Timeline", "Risk Breakdown"])
        
        with tabs[0]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            # Distribution of predictions
            fig = px.histogram(
                predictions_df,
                x="prediction",
                nbins=20,
                labels={"prediction": "Churn Probability"},
                title="Distribution of Prediction Probabilities",
                marginal="box",
                color_discrete_sequence=["#1e40af"]
            )
            fig.update_layout(
                xaxis_title="Churn Probability",
                yaxis_title="Count",
                plot_bgcolor="rgba(0,0,0,0)",
                bargap=0.1
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tabs[1]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            # Timeline of predictions
            fig = px.scatter(
                predictions_df,
                x="timestamp",
                y="prediction",
                color="prediction",
                color_continuous_scale="RdYlGn_r",
                labels={"prediction": "Churn Probability", "timestamp": "Time"},
                title="Predictions Over Time",
                hover_data=["model_version"]
            )
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Churn Probability",
                yaxis_range=[0, 1],
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="closest"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tabs[2]:
            # Create risk categories
            predictions_df["risk_level"] = pd.cut(
                predictions_df["prediction"],
                bins=[0, 0.3, 0.7, 1.0],
                labels=["Low", "Medium", "High"]
            )
            
            # Count by risk level
            risk_counts = predictions_df["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                # Create pie chart
                fig = px.pie(
                    risk_counts,
                    values="Count",
                    names="Risk Level",
                    title="Distribution by Risk Level",
                    color="Risk Level",
                    color_discrete_map={
                        "Low": "green",
                        "Medium": "orange",
                        "High": "red"
                    },
                    hole=0.4
                )
                
                fig.update_layout(
                    legend_title="Risk Level",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                # Create bar chart for risk breakdown
                fig = px.bar(
                    risk_counts,
                    x="Risk Level",
                    y="Count",
                    title="Risk Level Breakdown",
                    color="Risk Level",
                    text="Count",
                    color_discrete_map={
                        "Low": "green",
                        "Medium": "orange",
                        "High": "red"
                    }
                )
                
                fig.update_layout(
                    xaxis_title="Risk Level",
                    yaxis_title="Number of Customers",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Detailed predictions table
        st.markdown("<h3>Detailed Predictions</h3>", unsafe_allow_html=True)
        
        with st.expander("View detailed prediction data", expanded=False):
            # Extract feature columns if available
            if 'features' in predictions_df.columns:
                # Check if any features exist
                has_features = any(isinstance(f, dict) and len(f) > 0 for f in predictions_df['features'])
                
                if has_features:
                    try:
                        # Explode features into separate columns
                        features_df = pd.json_normalize(predictions_df['features'])
                        
                        # Combine with main dataframe
                        detailed_df = pd.concat([
                            predictions_df[['timestamp', 'model_version', 'prediction', 'risk_level']].reset_index(drop=True),
                            features_df.reset_index(drop=True)
                        ], axis=1)
                        
                        # Format timestamp for display
                        detailed_df["timestamp"] = detailed_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Display the data
                        st.markdown("<div class='card' style='overflow-x: auto;'>", unsafe_allow_html=True)
                        st.dataframe(detailed_df, use_container_width=True)
                        
                        # Add download button
                        csv = detailed_df.to_csv(index=False).encode()
                        st.download_button(
                            label="Download Predictions CSV",
                            data=csv,
                            file_name="churn_predictions.csv",
                            mime="text/csv"
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Error processing feature data: {e}")
                        st.dataframe(predictions_df[['timestamp', 'model_version', 'prediction']], use_container_width=True)
                else:
                    st.warning("No detailed feature data available in predictions")
                    st.dataframe(predictions_df[['timestamp', 'model_version', 'prediction']], use_container_width=True)
            else:
                st.warning("No detailed feature data available")
                st.dataframe(predictions_df[['timestamp', 'model_version', 'prediction']], use_container_width=True)
    else:
        st.info("No prediction data available. Make some predictions first.")

# System Monitoring Page
def render_system_monitoring():
    st.markdown("<h2>MLOps Infrastructure Monitoring</h2>", unsafe_allow_html=True)
    
    # Get system metrics
    system_df, is_real_data = get_system_metrics()
    
    if not is_real_data:
        st.info("Using simulated system metrics data for demonstration purposes.")
    
    # Current system metrics
    st.markdown("<h3>Current System Metrics</h3>", unsafe_allow_html=True)
    
    # Get latest metrics
    if not system_df.empty:
        latest_metrics = system_df.iloc[-1]
        previous_metrics = system_df.iloc[-2] if len(system_df) > 1 else latest_metrics
        
        # Display as metrics in cards
        col1, col2, col3 = st.columns(3)
        
        cpu_delta = latest_metrics['cpu_percent'] - previous_metrics['cpu_percent']
        memory_delta = latest_metrics['memory_percent'] - previous_metrics['memory_percent']
        disk_delta = latest_metrics['disk_percent'] - previous_metrics['disk_percent']
        
        # Apply custom coloring based on values
        cpu_color = "#ef4444" if latest_metrics['cpu_percent'] > 80 else "#f59e0b" if latest_metrics['cpu_percent'] > 60 else "#10b981"
        memory_color = "#ef4444" if latest_metrics['memory_percent'] > 80 else "#f59e0b" if latest_metrics['memory_percent'] > 60 else "#10b981"
        disk_color = "#ef4444" if latest_metrics['disk_percent'] > 80 else "#f59e0b" if latest_metrics['disk_percent'] > 60 else "#10b981"
        
        with col1:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">CPU Utilization</div>
                <div style="color: {cpu_color}; font-size: 28px; font-weight: 700; margin: 10px 0;">
                    {latest_metrics['cpu_percent']:.1f}%
                </div>
                <div style="color: {'#10b981' if cpu_delta <= 0 else '#ef4444'}; font-size: 14px; margin-top: 5px;">
                    {cpu_delta:.1f}% {'▼' if cpu_delta <= 0 else '▲'} from previous
                </div>
                <div style="height: 5px; background-color: #e5e7eb; border-radius: 3px; margin-top: 10px;">
                    <div style="height: 5px; width: {min(latest_metrics['cpu_percent'], 100)}%; 
                         background-color: {cpu_color}; border-radius: 3px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Memory Usage</div>
                <div style="color: {memory_color}; font-size: 28px; font-weight: 700; margin: 10px 0;">
                    {latest_metrics['memory_percent']:.1f}%
                </div>
                <div style="color: {'#10b981' if memory_delta <= 0 else '#ef4444'}; font-size: 14px; margin-top: 5px;">
                    {memory_delta:.1f}% {'▼' if memory_delta <= 0 else '▲'} from previous
                </div>
                <div style="height: 5px; background-color: #e5e7eb; border-radius: 3px; margin-top: 10px;">
                    <div style="height: 5px; width: {min(latest_metrics['memory_percent'], 100)}%; 
                         background-color: {memory_color}; border-radius: 3px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Disk Usage</div>
                <div style="color: {disk_color}; font-size: 28px; font-weight: 700; margin: 10px 0;">
                    {latest_metrics['disk_percent']:.1f}%
                </div>
                <div style="color: {'#10b981' if disk_delta <= 0 else '#ef4444'}; font-size: 14px; margin-top: 5px;">
                    {disk_delta:.1f}% {'▼' if disk_delta <= 0 else '▲'} from previous
                </div>
                <div style="height: 5px; background-color: #e5e7eb; border-radius: 3px; margin-top: 10px;">
                    <div style="height: 5px; width: {min(latest_metrics['disk_percent'], 100)}%; 
                         background-color: {disk_color}; border-radius: 3px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # System metrics over time
    st.markdown("<h3>Resource Utilization Over Time</h3>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    fig = px.line(
        system_df,
        x='timestamp',
        y=['cpu_percent', 'memory_percent', 'disk_percent'],
        title="System Resource Utilization",
        labels={'value': 'Utilization (%)', 'timestamp': 'Time', 'variable': 'Resource'},
        markers=True,
        color_discrete_map={
            'cpu_percent': '#3b82f6',
            'memory_percent': '#10b981',
            'disk_percent': '#f59e0b'
        }
    )
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Utilization (%)",
        legend_title="Resource",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Container health status
    st.markdown("<h3>Container Health Status</h3>", unsafe_allow_html=True)
    
    # Get service statuses
    service_statuses = get_service_status()
    
    # Create a grid of status cards
    cols = st.columns(4)
    
    for i, (service, status) in enumerate(service_statuses.items()):
        # Color code based on status
        status_color = "#10b981" if status in ["Healthy", "Connected", "Online"] else "#f59e0b" if status == "Degraded" else "#ef4444"
        status_icon = "✅" if status in ["Healthy", "Connected", "Online"] else "⚠️" if status == "Degraded" else "❌"
        
        cols[i % 4].markdown(f"""
        <div class="card" style="text-align: center; padding: 20px;">
            <div style="font-size: 36px; margin-bottom: 10px;">
                {'🖥️' if service == 'FastAPI' else '📊' if service == 'MLflow' else '🗄️' if service == 'MongoDB' else '📈'}
            </div>
            <div style="font-weight: 600; font-size: 18px; margin-bottom: 10px;">{service}</div>
            <div style="padding: 5px 10px; background-color: {status_color}; color: white; display: inline-block; 
                 border-radius: 20px; font-weight: 500;">
                {status_icon} {status}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add simulated API performance
    st.markdown("<h3>API Performance</h3>", unsafe_allow_html=True)
    
    # Create API latency dataframe
    api_df = pd.DataFrame({
        'timestamp': system_df['timestamp'],
        'api_latency': np.random.uniform(20, 35, size=len(system_df))
    })
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    fig = px.line(
        api_df,
        x='timestamp',
        y='api_latency',
        title="API Latency Over Time",
        labels={'api_latency': 'Latency (ms)', 'timestamp': 'Time'},
        markers=True,
        color_discrete_sequence=['#3b82f6']
    )
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Latency (ms)",
        hovermode="x",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Add deployment history section
    st.markdown("<h3>Deployment History</h3>", unsafe_allow_html=True)
    
    # This would typically come from a real monitoring system
    # For now, we'll use mock data
    deployment_data = [
        {"version": "20250304_0145", "date": "2025-03-04 01:45", "status": "Active", "metrics": {"accuracy": 0.92, "f1": 0.88}},
        {"version": "20250301_1423", "date": "2025-03-01 14:23", "status": "Archived", "metrics": {"accuracy": 0.89, "f1": 0.86}},
        {"version": "20250220_0915", "date": "2025-02-20 09:15", "status": "Archived", "metrics": {"accuracy": 0.87, "f1": 0.84}}
    ]
    
    deployment_df = pd.DataFrame(deployment_data)
    
    # Format metrics column for display
    deployment_df["metrics_display"] = deployment_df["metrics"].apply(
        lambda x: ", ".join([f"{k}: {v:.2f}" for k, v in x.items()])
    )
    
    # Use Streamlit's native dataframe
    st.dataframe(
        deployment_df[["version", "date", "status", "metrics_display"]],
        column_config={
            "version": st.column_config.TextColumn("Model Version"),
            "date": st.column_config.TextColumn("Deployment Date"),
            "status": st.column_config.TextColumn("Status"),
            "metrics_display": st.column_config.TextColumn("Key Metrics"),
        },
        use_container_width=True,
        hide_index=True
    )

# Main function
def main():
    # Render header 
    render_header()
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Render selected page
    if page == "Dashboard":
        render_dashboard()
    elif page == "Model Performance":
        render_model_performance()
    elif page == "Make Prediction":
        render_prediction_page()
    elif page == "Recent Predictions":
        render_recent_predictions()
    elif page == "System Monitoring":
        render_system_monitoring()
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #6b7280;'>MLOps Churn Prediction Dashboard | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()