import unittest
from app import app  # Assuming your Flask app is in a file named app.py
import json

class TextPredictionTest(unittest.TestCase):
    
    def setUp(self):
        """Set up the Flask test client."""
        # Create a test client for our Flask app
        self.client = app.test_client()
        self.client.testing = True
    
    def test_predict_common_cold(self):
        """Test prediction for symptoms of the common cold."""
        data = {
            'symptoms': 'cough sore throat runny nose'  # Symptoms for Common Cold
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)
        
        # Check if the response includes the predicted disease (Common Cold)
        self.assertIn(b'Predicted Disease: Common Cold', response.data)

    def test_predict_pneumonia(self):
        """Test prediction for symptoms of Pneumonia."""
        data = {
            'symptoms': 'chest pain shortness of breath'  # Symptoms for Pneumonia
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)
        
        # Check if the response includes the predicted disease (Pneumonia)
        self.assertIn(b'Predicted Disease: Pneumonia', response.data)

    def test_predict_influenza(self):
        """Test prediction for symptoms of Influenza."""
        data = {
            'symptoms': 'fever, running nose, sore throat, yellow tonsillar spots'  # Symptoms for Bacterial Tonsillitis (but can check other logic here)
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)
        
        # Check if the response includes the predicted disease
        self.assertIn(b'Predicted Disease: Influenza', response.data)

    def test_invalid_symptoms(self):
        """Test prediction when symptoms input is empty."""
        data = {
            'symptoms': ''  # Empty input
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)
        
        # Check if the response asks for symptom input
        self.assertIn(b'Enter Symptoms', response.data)

    def test_multiple_symptoms_prediction(self):
        """Test prediction with a mix of symptoms."""
        data = {
            'symptoms': 'fever, blocked nose, red eye, headache'  # Symptoms for Influenza
        }
        response = self.client.post('/', data=data)
        self.assertEqual(response.status_code, 200)
        
        # Check if the response includes the predicted disease (Influenza)
        self.assertIn(b'Predicted Disease: Influenza', response.data)
        
        # Verify that prediction confidence is shown
        self.assertIn(b'Prediction Confidence:', response.data)

if __name__ == "__main__":
    unittest.main()
