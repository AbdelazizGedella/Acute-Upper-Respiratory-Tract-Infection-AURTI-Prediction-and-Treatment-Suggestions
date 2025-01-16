import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

def load_or_train_model():
    try:
        pipeline = joblib.load("disease_prediction_model.joblib")
    except FileNotFoundError:
        # Sample training data (replace with your actual data later)
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
        df = pd.DataFrame(train_data, columns=["text", "label"])

        X_train = df["text"]
        y_train = df["label"]

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB()),
        ])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, "disease_prediction_model.joblib")
    return pipeline

def predict_disease(symptoms):
    pipeline = load_or_train_model()  # Load the model inside the function
    text = " ".join(symptoms)

    # Step 1: Log symptom input
    log = {"step": "Received Symptoms", "symptoms": symptoms}

    # Step 2: Transform the symptoms using TF-IDF
    vectorized_text = pipeline.named_steps['tfidf'].transform([text])
    log["step_2"] = "Vectorizing Text"

    try:
        # Step 3: Predict the disease
        prediction = pipeline.predict([text])[0]
        log["step_3"] = "Predicted Disease"
        log["predicted_class"] = prediction

        return log  # Return the log with intermediate steps

    except ValueError:
        log["error"] = "Could not make a prediction based on the provided symptoms."
        return log
