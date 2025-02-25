import unittest
import pandas as pd
import numpy as np
from model_pipeline import prepare_data, train_model, evaluate_model

class TestModelPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.test_data = pd.DataFrame({
            'feature1': np.random.random(100),
            'feature2': np.random.random(100),
            'target': np.random.randint(0, 2, 100)
        })

    def test_prepare_data(self):
        """Test if prepare_data function runs without errors"""
        try:
            result = prepare_data()
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"prepare_data() raised {type(e).__name__} unexpectedly!")

    def test_model_training(self):
        """Test if model training runs without errors"""
        try:
            X_train = np.random.random((100, 2))
            y_train = np.random.randint(0, 2, 100)
            model = train_model(X_train, y_train)
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"train_model() raised {type(e).__name__} unexpectedly!")

    def test_model_evaluation(self):
        """Test if model evaluation returns a valid accuracy score"""
        X_test = np.random.random((100, 2))
        y_test = np.random.randint(0, 2, 100)
        accuracy = evaluate_model(X_test, y_test)
        self.assertIsInstance(accuracy, float)
        self.assertTrue(0 <= accuracy <= 1)

if __name__ == '__main__':
    unittest.main()
