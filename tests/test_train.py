import unittest
from unittest.mock import patch
import pandas as pd
from src.train import train  # Make sure to import the correct function

class TestRandomForestModel(unittest.TestCase):

    @patch('pandas.read_csv')
    def test_train_function(self, mock_read_csv):
        # Mock dataset
        mock_read_csv.return_value = pd.DataFrame({
            'Pregnancies': [6, 1, 8, 1, 0],
            'Glucose': [148, 85, 183, 89, 137],
            'BloodPressure': [72, 66, 64, 66, 40],
            'SkinThickness': [35, 29, 0, 23, 35],
            'Insulin': [0, 0, 0, 94, 168],
            'BMI': [33.6, 26.6, 23.3, 28.1, 43.1],
            'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288],
            'Age': [50, 31, 32, 21, 33],
            'Outcome': [1, 0, 1, 0, 1]
        })

        # Run train function
        train("data/raw/data.csv")

        # Check if the `read_csv` was called with the correct path
        mock_read_csv.assert_called_once_with("data/raw/data.csv")

if __name__ == '__main__':
    unittest.main()
