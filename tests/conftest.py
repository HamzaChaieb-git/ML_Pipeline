import pytest
import os
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def mock_mlflow():
    """
    Mock all MLflow functions to prevent actual logging during tests.
    This fixture applies to all tests automatically.
    """
    # Create mock objects
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-id"

    # Create a mock experiment
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "test-experiment-id"

    # Important: End any active run that might exist from a previous test
    try:
        import mlflow

        mlflow.end_run()
    except:
        pass  # Ignore errors if no run is active

    # Create patches for common MLflow functions
    patches = [
        patch("mlflow.log_params", return_value=None),
        patch("mlflow.log_param", return_value=None),
        patch("mlflow.log_metrics", return_value=None),
        patch("mlflow.log_metric", return_value=None),
        patch("mlflow.log_artifact", return_value=None),
        patch("mlflow.log_artifacts", return_value=None),
        patch("mlflow.set_tracking_uri", return_value=None),
        patch("mlflow.set_experiment", return_value=None),
        patch("mlflow.start_run", return_value=mock_run),
        patch("mlflow.end_run", return_value=None),
        patch("mlflow.xgboost.log_model", return_value=None),
        patch("mlflow.xgboost.autolog", return_value=None),
        patch("mlflow.sklearn.autolog", return_value=None),
        patch("mlflow.log_dict", return_value=None),
        patch("mlflow.get_experiment_by_name", return_value=mock_experiment),
        patch("mlflow.create_experiment", return_value="test-experiment-id"),
        patch("mlflow.models.infer_signature", return_value=MagicMock()),
        # Special mock for `trace` decorator to make it pass-through
        patch("mlflow.trace", lambda *args, **kwargs: lambda fn: fn),
        # Mock start_span to be a working context manager
        patch(
            "mlflow.start_span",
            return_value=MagicMock(
                __enter__=MagicMock(
                    return_value=MagicMock(
                        set_inputs=MagicMock(),
                        set_outputs=MagicMock(),
                        set_attribute=MagicMock(),
                    )
                ),
                __exit__=MagicMock(),
            ),
        ),
    ]

    # Start all patches
    for p in patches:
        p.start()

    # Create artifacts directory if it doesn't exist
    os.makedirs(os.path.join("artifacts", "models"), exist_ok=True)
    os.makedirs(os.path.join("artifacts", "data"), exist_ok=True)
    os.makedirs(os.path.join("artifacts", "evaluation"), exist_ok=True)
    os.makedirs(os.path.join("artifacts", "training"), exist_ok=True)

    # Yield to allow test to run
    yield

    # Stop all patches
    for p in patches:
        p.stop()

    # Make sure any MLflow run is ended
    try:
        import mlflow

        mlflow.end_run()
    except:
        pass
