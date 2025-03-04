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

# Set page configuration for a clean, professional dashboard
st.set_page_config(
    page_title="ML Model Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, professional dashboard
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
        background-color: #f5f7fa;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }
    .metric-card {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .prediction-card {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
        background-color: white;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
        st.error("No models found in artifacts/models directory.")
        return None

    model_files = [
        f
        for f in os.listdir(models_dir)
        if f.startswith("model_v") and f.endswith(".joblib")
    ]
    if not model_files:
        st.error("No model files found in artifacts/models.")
        return None

    latest_model = max(model_files, key=lambda x: x.split("v")[1].split(".joblib")[0])
    model_path = os.path.join(models_dir, latest_model)
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
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
            st.warning(f"Could not fetch recent predictions: {response.status_code}")
            return []
    except Exception as e:
        st.warning(f"Error fetching recent predictions: {e}")
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
        st.warning(f"Error fetching predictions from MongoDB: {e}")
        return []
    finally:
        if client:
            client.close()

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
        st.warning(f"Error fetching model metrics from MongoDB: {e}")
        return []
    finally:
        if client:
            client.close()

# Main Dashboard Layout
st.title("🔮 Churn Prediction Dashboard")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a page:",
        ["Dashboard", "Make Prediction", "Model Performance", "Recent Predictions", "System Monitoring"]
    )
    
    st.header("Model Information")
    model_info = get_model_info()
    if model_info["model_loaded"]:
        st.success("✅ Model is loaded and ready")
    else:
        st.error("❌ Model is not loaded")
    
    # Load model for metrics
    model = load_latest_model()
    if model and hasattr(model, 'model_info'):
        st.subheader("Current Model")
        st.write(f"Version: {model.model_info.get('version', 'Unknown')}")
        st.write(f"Features: {len(model.model_info.get('features', []))}")
        st.write(f"Training Samples: {model.model_info.get('training_samples', 'Unknown')}")
        st.write(f"Stage: {model.model_info.get('stage', 'Staging')}")

# Dashboard Overview
if page == "Dashboard":
    st.header("Overview")
    
    # Metrics layout
    col1, col2, col3, col4 = st.columns(4)
    
    # Try to get metrics from MongoDB
    model_metrics = get_model_metrics_from_mongodb()
    latest_metrics = next(iter(model_metrics), {}).get("metrics", {}) if model_metrics else {}
    
    with col1:
        st.metric(
            "Accuracy", 
            f"{latest_metrics.get('accuracy', 0):.2%}" if latest_metrics else "No data",
            delta=None
        )
    
    with col2:
        st.metric(
            "ROC AUC", 
            f"{latest_metrics.get('roc_auc', 0):.2%}" if latest_metrics else "No data",
            delta=None
        )
    
    with col3:
        st.metric(
            "Precision", 
            f"{latest_metrics.get('precision', 0):.2%}" if latest_metrics else "No data", 
            delta=None
        )
    
    with col4:
        st.metric(
            "Recall", 
            f"{latest_metrics.get('recall', 0):.2%}" if latest_metrics else "No data",
            delta=None
        )
    
    # Recent predictions summary
    st.subheader("Recent Predictions")
    
    # Get recent predictions
    predictions = get_predictions_from_mongodb(10)
    
    if predictions:
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        # Extract timestamps
        if 'timestamp' in predictions_df.columns:
            predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp']['$date'], unit='ms')
        
        # Create bar chart of recent prediction probabilities
        fig = px.bar(
            predictions_df,
            y='prediction',
            labels={'prediction': 'Churn Probability', 'index': 'Prediction ID'},
            title='Recent Prediction Probabilities',
            height=300,
        )
        fig.update_layout(
            xaxis_title="Recent Predictions",
            yaxis_title="Churn Probability",
            yaxis_range=[0, 1],
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No recent predictions available.")
    
    # System Status Card
    st.subheader("System Status")
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        try:
            fastapi_response = requests.get("http://fastapi:8000/health", timeout=2)
            if fastapi_response.status_code == 200:
                st.success("FastAPI: Online")
            else:
                st.error(f"FastAPI: Error ({fastapi_response.status_code})")
        except:
            st.error("FastAPI: Offline")
        
        try:
            mlflow_response = requests.get("http://mlflow:5001/", timeout=2)
            if mlflow_response.status_code == 200:
                st.success("MLflow: Online")
            else:
                st.error(f"MLflow: Error ({mlflow_response.status_code})")
        except:
            st.error("MLflow: Offline")
    
    with status_col2:
        mongo_client = get_mongodb_connection()
        if mongo_client:
            st.success("MongoDB: Connected")
            mongo_client.close()
        else:
            st.error("MongoDB: Disconnected")

# Make Prediction Page
elif page == "Make Prediction":
    st.header("Make a Prediction")
    
    # Get the feature list from model info
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
    
    # Form for input
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        # Initialize input data dictionary
        input_data = {}
        
        # Create input fields
        with col1:
            if "Total day minutes" in features:
                input_data["Total day minutes"] = [st.number_input("Total Day Minutes", value=120.5, step=1.0)]
            
            if "Customer service calls" in features:
                input_data["Customer service calls"] = [st.number_input("Customer Service Calls", value=1, step=1)]
            
            if "International plan" in features:
                input_data["International plan"] = [int(st.selectbox("International Plan", ["No", "Yes"]) == "Yes")]
            
            if "Total intl minutes" in features:
                input_data["Total intl minutes"] = [st.number_input("Total International Minutes", value=10.2, step=0.1)]
        
        with col2:
            if "Total intl calls" in features:
                input_data["Total intl calls"] = [st.number_input("Total International Calls", value=3, step=1)]
            
            if "Total eve minutes" in features:
                input_data["Total eve minutes"] = [st.number_input("Total Evening Minutes", value=200.0, step=1.0)]
            
            if "Number vmail messages" in features:
                input_data["Number vmail messages"] = [st.number_input("Number of Voicemail Messages", value=0, step=1)]
            
            if "Voice mail plan" in features:
                input_data["Voice mail plan"] = [int(st.selectbox("Voice Mail Plan", ["No", "Yes"]) == "Yes")]
        
        submitted = st.form_submit_button("Predict Churn")
    
    if submitted:
        st.subheader("Prediction Result")
        
        try:
            # Call FastAPI endpoint
            response = requests.post("http://fastapi:8000/predict", json=input_data, timeout=10)
            response.raise_for_status()
            prediction = response.json()["churn_probabilities"][0]
            
            # Display prediction with gauge chart
            col1, col2 = st.columns([3, 2])
            
            with col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction * 100,
                    title={"text": "Churn Probability"},
                    gauge={
                        "axis": {"range": [0, 100], "ticksuffix": "%"},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 50], "color": "lightgreen"},
                            {"range": [50, 70], "color": "yellow"},
                            {"range": [70, 100], "color": "salmon"}
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
            
            with col2:
                risk_level = "Low" if prediction < 0.5 else "Medium" if prediction < 0.7 else "High"
                st.markdown(f"### Risk Level: {risk_level}")
                st.markdown(f"**Probability:** {prediction:.2%}")
                
                # Recommendation based on churn risk
                if risk_level == "Low":
                    st.success("✅ Customer likely to stay. Regular engagement recommended.")
                elif risk_level == "Medium":
                    st.warning("⚠️ Monitor closely. Consider offering special promotions or check-ins.")
                else:
                    st.error("🚨 High churn risk! Immediate retention action recommended.")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            
            # Fallback: Use local model if FastAPI fails
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
                            {"range": [0, 50], "color": "lightgreen"},
                            {"range": [50, 70], "color": "yellow"},
                            {"range": [70, 100], "color": "salmon"}
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

# Model Performance Page
elif page == "Model Performance":
    st.header("Model Performance Metrics")
    
    # Get model metrics from MongoDB
    model_metrics = get_model_metrics_from_mongodb()
    
    if model_metrics:
        # Convert to DataFrame
        metrics_df = pd.DataFrame([
            {
                "timestamp": pd.to_datetime(m["timestamp"]["$date"], unit='ms') if "$date" in m.get("timestamp", {}) else "Unknown",
                "model_version": m.get("model_version", "Unknown"),
                **m.get("metrics", {})
            }
            for m in model_metrics
        ])
        
        # Metrics over time visualization
        st.subheader("Key Metrics Over Time")
        
        # Select metrics to display
        selected_metrics = st.multiselect(
            "Select metrics to display:",
            ["accuracy", "precision", "recall", "f1", "roc_auc"],
            default=["accuracy", "roc_auc"]
        )
        
        if selected_metrics and not metrics_df.empty:
            # Create time series plot
            fig = px.line(
                metrics_df,
                x="timestamp",
                y=selected_metrics,
                title="Model Performance Over Time",
                labels={"value": "Metric Value", "timestamp": "Date"},
                markers=True
            )
            
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Metric Value",
                legend_title="Metrics",
                hovermode="x unified",
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrix visualization from latest metrics
        st.subheader("Latest Confusion Matrix")
        
        latest_metrics = metrics_df.iloc[0] if not metrics_df.empty else {}
        
        if all(k in latest_metrics for k in ["true_positives", "false_positives", "true_negatives", "false_negatives"]):
            confusion_data = [
                [latest_metrics["true_negatives"], latest_metrics["false_positives"]],
                [latest_metrics["false_negatives"], latest_metrics["true_positives"]]
            ]
            
            # Create confusion matrix heatmap
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
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Display all metrics in a data table
        st.subheader("All Model Versions and Metrics")
        if not metrics_df.empty:
            # Filter columns for display
            display_cols = ["timestamp", "model_version", "accuracy", "precision", "recall", "f1", "roc_auc"]
            display_df = metrics_df[display_cols].copy()
            
            # Format timestamp
            display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # Format numeric columns
            for col in display_df.columns:
                if col not in ["timestamp", "model_version"]:
                    display_df[col] = display_df[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
            
            st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No model metrics available in MongoDB. Run model evaluation to generate metrics.")

# Recent Predictions Page
elif page == "Recent Predictions":
    st.header("Recent Predictions")
    
    # Number of predictions to display
    num_predictions = st.slider("Number of predictions to display", min_value=5, max_value=100, value=20)
    
    # Get predictions from MongoDB
    predictions = get_predictions_from_mongodb(num_predictions)
    
    if predictions:
        # Convert to DataFrame
        predictions_df = pd.DataFrame([
            {
                "timestamp": pd.to_datetime(p["timestamp"]["$date"], unit='ms') if "$date" in p.get("timestamp", {}) else "Unknown",
                "model_version": p.get("model_version", "Unknown"),
                "prediction": p.get("prediction", 0),
                "features": p.get("features", {})
            }
            for p in predictions
        ])
        
        # Summary statistics
        st.subheader("Prediction Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Churn Probability", f"{predictions_df['prediction'].mean():.2%}")
        
        with col2:
            high_risk = (predictions_df['prediction'] > 0.7).sum()
            st.metric("High Risk Customers", f"{high_risk} ({high_risk/len(predictions_df):.1%})")
        
        with col3:
            st.metric("Predictions Analyzed", len(predictions_df))
        
        # Distribution of predictions
        st.subheader("Distribution of Churn Probabilities")
        fig = px.histogram(
            predictions_df,
            x="prediction",
            nbins=20,
            labels={"prediction": "Churn Probability"},
            title="Distribution of Prediction Probabilities",
            marginal="box"
        )
        fig.update_layout(
            xaxis_title="Churn Probability",
            yaxis_title="Count",
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Timeline of predictions
        st.subheader("Predictions Timeline")
        fig = px.scatter(
            predictions_df,
            x="timestamp",
            y="prediction",
            color="prediction",
            color_continuous_scale="Viridis",
            labels={"prediction": "Churn Probability", "timestamp": "Time"},
            title="Predictions Over Time",
            hover_data=["model_version"]
        )
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Churn Probability",
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed predictions table
        st.subheader("Detailed Predictions")
        
        # Extract feature columns if available
        if predictions_df.empty or 'features' not in predictions_df.columns:
            st.warning("No detailed feature data available")
        else:
            # Explode features into separate columns
            features_df = pd.json_normalize(predictions_df['features'])
            
            # Combine with main dataframe
            detailed_df = pd.concat([
                predictions_df[['timestamp', 'model_version', 'prediction']].reset_index(drop=True),
                features_df.reset_index(drop=True)
            ], axis=1)
            
            # Display detailed table
            st.dataframe(detailed_df, use_container_width=True)
    else:
        st.info("No prediction data available. Make some predictions first.")

# System Monitoring Page
elif page == "System Monitoring":
    st.header("System Monitoring")
    
    # Mock system metrics (replace with real metrics if available)
    # In a production system, you would fetch these from a monitoring system
    
    st.info("System monitoring data would be displayed here. This could include:")
    
    mock_dates = [datetime.now() - timedelta(minutes=x*10) for x in range(24)]
    mock_cpu = [30 + x*2 + (x**2)/10 for x in range(24)]
    mock_memory = [45 + x/2 for x in range(24)]
    mock_api_latency = [25 + x/3 for x in range(24)]
    
    # System metrics over time
    metrics_df = pd.DataFrame({
        'timestamp': mock_dates,
        'CPU Utilization (%)': mock_cpu,
        'Memory Usage (%)': mock_memory,
        'API Latency (ms)': mock_api_latency
    })
    
    # Create system metrics visualization
    st.subheader("System Resource Utilization")
    
    fig = px.line(
        metrics_df,
        x='timestamp',
        y=['CPU Utilization (%)', 'Memory Usage (%)'],
        title="Resource Utilization Over Time",
        labels={'value': 'Utilization (%)', 'timestamp': 'Time'},
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Utilization (%)",
        legend_title="Metric",
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # API performance
    st.subheader("API Performance")
    
    fig = px.line(
        metrics_df,
        x='timestamp',
        y='API Latency (ms)',
        title="API Latency Over Time",
        labels={'API Latency (ms)': 'Latency (ms)', 'timestamp': 'Time'},
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Latency (ms)",
        hovermode="x",
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Container health status
    st.subheader("Container Health Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            response = requests.get("http://fastapi:8000/health", timeout=2)
            if response.status_code == 200:
                st.success("FastAPI: Healthy")
            else:
                st.error(f"FastAPI: Unhealthy ({response.status_code})")
        except:
            st.error("FastAPI: Offline")
    
    with col2:
        try:
            response = requests.get("http://mlflow:5001/", timeout=2)
            if response.status_code == 200:
                st.success("MLflow: Healthy")
            else:
                st.error(f"MLflow: Unhealthy ({response.status_code})")
        except:
            st.error("MLflow: Offline")
    
    with col3:
        mongo_client = get_mongodb_connection()
        if mongo_client:
            st.success("MongoDB: Healthy")
            mongo_client.close()
        else:
            st.error("MongoDB: Unhealthy")
    
    with col4:
        # This is just a mock status
        st.success("Monitoring: Healthy")
    
    st.markdown("""
    In a production environment, this page would display real-time metrics collected from:
    
    * Container resource usage (CPU, Memory, Disk)
    * API performance metrics
    * Database connection statistics
    * Model performance drift
    * System alerts and health status
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>MLOps Churn Prediction Dashboard | Last updated: "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M')}</div>",
    unsafe_allow_html=True
)