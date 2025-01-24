import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.evaluate import evaluate  # Adjust the import path based on your structure

class TestEvaluateFunction(unittest.TestCase):

    @patch('src.evaluate.pd.read_csv')
    @patch('src.evaluate.pickle.load')
    @patch('src.evaluate.mlflow.log_metric')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)  # Mock open function
    def test_evaluate(self, mock_open, mock_log_metric, mock_pickle_load, mock_read_csv):

        # Mock the data
        mock_data = pd.DataFrame({
            'Pregnancies': [6, 1],
            'Glucose': [148, 85],
            'BloodPressure': [72, 66],
            'SkinThickness': [35, 29],
            'Insulin': [0, 0],
            'BMI': [33.6, 26.6],
            'DiabetesPedigreeFunction': [0.627, 0.351],
            'Age': [50, 31],
            'Outcome': [1, 0]
        })
        mock_read_csv.return_value = mock_data

        # Mock the model object
        mock_model = MagicMock()
        mock_model.predict.return_value = [1, 0]  # Mock predictions
        mock_pickle_load.return_value = mock_model

        # Call the function
        params = {'data': 'mock_data.csv', 'model': 'mock_model.pkl'}
        evaluate(params['data'], params['model'])

        # Check that methods were called
        mock_read_csv.assert_called_once_with('mock_data.csv')
        mock_pickle_load.assert_called_once()  # Ensure pickle.load was called
        mock_log_metric.assert_called_with('accuracy', 1.0)  # Accuracy will be mocked as 1.0 (based on predictions)
        mock_open.assert_called_once_with('mock_model.pkl', 'rb')  # Ensure open was called with the mock model path

if __name__ == '__main__':
    unittest.main()
