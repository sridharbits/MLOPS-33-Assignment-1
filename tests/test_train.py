import unittest
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        # Load data
        self.data = pd.read_csv('data/raw/data.csv')
        self.X = self.data.drop(columns=["Outcome"])
        self.y = self.data['Outcome']

    def test_model_exists(self):
        """Check if model file exists"""
        model_path = 'models/model.pkl'
        self.assertTrue(os.path.exists(model_path), "Model file does not exist")

    def test_model_prediction(self):
        """Test model prediction capabilities"""
        # Load model
        with open('models/model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)

        # Make predictions
        predictions = model.predict(self.X)

        # Verify prediction length
        self.assertEqual(len(predictions), len(self.y), "Prediction length mismatch")

        # Check model accuracy
        accuracy = accuracy_score(self.y, predictions)
        self.assertGreater(accuracy, 0.7, f"Model accuracy too low: {accuracy}")

if __name__ == '__main__':
    unittest.main()
    