import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Function to extract symptoms from input text
def extract_symptoms(text):
    """
    Extract symptoms from the input text.
    Args:
    text (str): The input text containing symptoms.
    Returns:
    list: A list of extracted symptoms.
    """
    common_symptoms = ["cough", "fever", "headache", "fatigue", "sore throat", "runny nose", "muscle aches"]
    symptoms = [word for word in text.lower().split() if word in common_symptoms]
    return symptoms

# Function to load or train the disease prediction model
def load_or_train_model():
    """
    Loads the pre-trained model or trains a new model if not found.
    Returns:
    pipeline: A trained model pipeline.
    """
    try:
        # Try loading the existing model
        pipeline = joblib.load("disease_prediction_model.joblib")
    except FileNotFoundError:
        # If model is not found, train a new one
        train_data = [
            ("cough fever", "Flu"),
            ("cough sore throat runny nose", "Common Cold"),
            ("chest pain shortness of breath", "Pneumonia"),
            ("headache fatigue muscle aches", "Flu"),
            ("runny nose sneezing itchy eyes", "Allergies"),
            ("running nose, sore throat, temperature 37.6", "Common Cold"),
            ("sore throat, facial pain, fever 39, left cheek red", "Sinusitis"),
            ("fever 39.5, running nose, dry cough, dizziness, red eye", "Influenza"),
            ("fever, blocked nose, red eye, headache", "Influenza"),
            ("fever, running nose, sore throat, yellow tonsillar spots", "Bacterial Tonsillitis")
        ]
        
        # Convert the training data into a DataFrame
        df = pd.DataFrame(train_data, columns=["text", "label"])

        # Split into features (X) and target (y)
        X_train = df["text"]
        y_train = df["label"]

        # Create a pipeline for training the model
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),  # Convert text to TF-IDF features
            ('clf', MultinomialNB()),  # Use Naive Bayes for classification
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Save the trained model for future use
        joblib.dump(pipeline, "disease_prediction_model.joblib")
    
    return pipeline

# Function to predict the disease based on symptoms
def predict_disease(symptoms):
    """
    Predict the disease based on the symptoms.
    Args:
    symptoms (list): A list of symptoms.
    Returns:
    dict: A log of the prediction process.
    """
    pipeline = load_or_train_model()  # Load or train the model
    text = " ".join(symptoms)  # Convert symptoms to a single string

    # Log the received symptoms
    log = {"step": "Received Symptoms", "symptoms": symptoms}

    # Step 1: Transform the symptoms using TF-IDF
    vectorized_text = pipeline.named_steps['tfidf'].transform([text])
    log["step_2"] = "Vectorizing Text"

    try:
        # Step 2: Make the prediction
        prediction = pipeline.predict([text])[0]
        log["step_3"] = "Predicted Disease"
        log["predicted_class"] = prediction

        return log  # Return the log with prediction details

    except ValueError:
        log["error"] = "Could not make a prediction based on the provided symptoms."
        return log
