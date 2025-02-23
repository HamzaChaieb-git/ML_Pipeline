def register_model(model, metrics, run_id) -> None:
    """Register model in MLflow Model Registry."""
    model_name = "churn_prediction_model"
    
    # Get the model URI
    model_uri = f"runs:/{run_id}/model"
    
    try:
        # Register model
        model_version = mlflow.register_model(model_uri, model_name)
        print(f"Model registered as version {model_version.version}")
        
        # Add description with metrics
        description = f"Model metrics:\n"
        description += f"Accuracy: {metrics['accuracy']:.4f}\n"
        description += f"ROC AUC: {metrics['roc_auc']:.4f}\n"
        description += f"F1 Score: {metrics['f1']:.4f}"
        
        client = mlflow.tracking.MlflowClient()
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=description
        )
        
        # Transition to Production if metrics are good enough
        if metrics['accuracy'] > 0.95 and metrics['roc_auc'] > 0.90:
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production"
            )
            print("Model promoted to Production")
        else:
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )
            print("Model set to Staging")
            
    except Exception as e:
        print(f"Error registering model: {str(e)}")

def run_full_pipeline(train_file: str, test_file: str) -> None:
    """Execute the complete ML pipeline with MLflow tracking."""
    print("Running full pipeline...")
    
    # Setup MLflow
    experiment_id = setup_mlflow()
    
    # Start a new MLflow run
    with mlflow.start_run(experiment_id=experiment_id, run_name="Full Pipeline") as run:
        try:
            # Log input parameters
            mlflow.log_param("train_file", train_file)
            mlflow.log_param("test_file", test_file)
            
            # Data preparation
            print("🔹 Preparing data...")
            X_train, X_test, y_train, y_test = process_data(train_file, test_file)
            print("🔹 Data preparation complete")
            
            # Model training
            print("🔹 Training model...")
            model = train_xgb_model(X_train, y_train)
            print("🔹 Model training complete")
            
            # Model evaluation
            print("🔹 Evaluating model...")
            metrics = evaluate_xgb_model(model, X_test, y_test)
            print("🔹 Evaluation complete")
            
            # Save model
            print("🔹 Saving model...")
            save_xgb_model(model)
            mlflow.log_artifact("model.joblib")
            print("🔹 Model saved")
            
            # Register model
            print("🔹 Registering model...")
            register_model(model, metrics, run.info.run_id)
            print("🔹 Model registered")
            
            # Load and verify registered model
            client = mlflow.tracking.MlflowClient()
            latest_model = client.get_latest_versions("churn_prediction_model", stages=["Production", "Staging"])[0]
            print(f"Latest model version: {latest_model.version}")
            print(f"Current stage: {latest_model.current_stage}")
            
            return model, metrics
                
        except Exception as e:
            print(f"❌ Error in pipeline: {str(e)}")
            raise e
