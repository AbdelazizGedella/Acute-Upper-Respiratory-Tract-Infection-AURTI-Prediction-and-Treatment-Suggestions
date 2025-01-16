from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def get_treatments_for_disease(disease):
    """
    Fetch treatments for the specified disease from the treatment database.
    """
    treatments = []

    try:
        # Load the treatment database
        treatment_data = pd.read_csv("treatment_database.csv")

        # Filter treatments for the given disease
        disease_treatments = treatment_data[treatment_data["Disease"].str.lower() == disease.lower()]

        if not disease_treatments.empty:
            treatments = disease_treatments["Treatment"].tolist()
        else:
            treatments = ["No treatment data available"]
    except FileNotFoundError:
        treatments = ["Treatment database not found"]

    return treatments

def load_or_train_model():
    try:
        # Load the pre-trained model if exists
        pipeline = joblib.load("disease_prediction_model.joblib")
    except FileNotFoundError:
        # Train a new model if none exists
        train_data = [
            ("cough sore throat runny nose", "Common Cold"),
            ("chest pain shortness of breath", "Pneumonia"),
            ("fever, running nose, sore throat, yellow tonsillar spots", "Bacterial Tonsillitis"),
            ("cough fever headache fatigue muscle aches", "Flu"),
            ("runny nose sneezing itchy eyes", "Allergies"),
            ("running nose, sore throat, temperature 37.6", "Common Cold"),
            ("sore throat, facial pain, fever 39, left cheek red", "Sinusitis"),
            ("fever 39.5, running nose, dry cough, dizziness, red eye", "Influenza"),
            ("fever, blocked nose, red eye, headache, dry cough", "Influenza"),
        ]
        df = pd.DataFrame(train_data, columns=["text", "label"])

        X = df["text"]
        y = df["label"]

        tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        smote = SMOTE(random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        X_resampled, y_resampled = smote.fit_resample(X_train_tfidf, y_train)

        param_grid = {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs'],
        }
        grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_resampled, y_resampled)
        model = grid_search.best_estimator_

        y_pred = model.predict(X_test_tfidf)
        print(classification_report(y_test, y_pred))

        pipeline = make_pipeline(tfidf, model)
        joblib.dump(pipeline, "disease_prediction_model.joblib")

    return pipeline

# Load the pipeline or train the model if not available
pipeline = load_or_train_model()

def predict_with_confidence(pipeline, text):
    label_probabilities = pipeline.predict_proba([text])[0]
    labels = pipeline.classes_
    predictions = sorted(zip(labels, label_probabilities), key=lambda x: x[1], reverse=True)
    return predictions

def generate_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm

def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(6,6))
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.title("Confusion Matrix")

    # Convert plot to PNG image and then to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return img_data

@app.route("/train", methods=["POST"])
def train():
    symptoms = request.form.get("symptoms")
    disease = request.form.get("disease")
    feedback = None

    if not symptoms or not disease:
        feedback = "Invalid input. Please provide both symptoms and disease name."
        return render_template("index.html", feedback=feedback)

    try:
        # Load existing training data if any
        existing_data = pd.read_csv("training_data.csv")
    except FileNotFoundError:
        existing_data = pd.DataFrame(columns=["text", "label"])

    # Add the new training data
    new_data = pd.DataFrame([[symptoms, disease]], columns=["text", "label"])
    updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    updated_data.to_csv("training_data.csv", index=False)

    # Retrain the model
    X = updated_data["text"]
    y = updated_data["label"]

    try:
        pipeline.fit(X, y)
        joblib.dump(pipeline, "disease_prediction_model.joblib")
        feedback = "Model retrained successfully with the new data!"
    except Exception as e:
        feedback = f"Error during model retraining: {str(e)}"

    return render_template("index.html", feedback=feedback)

@app.route("/add_treatment", methods=["POST"])
def add_treatment():
    disease = request.form.get("disease")
    treatment = request.form.get("treatment")

    if not disease or not treatment:
        feedback = "Invalid input. Please provide both disease and treatment."
        return render_template("index.html", feedback=feedback)

    try:
        # Load existing treatment data
        treatment_data = pd.read_csv("treatment_database.csv")
    except FileNotFoundError:
        treatment_data = pd.DataFrame(columns=["Disease", "Treatment"])

    # Add new treatment data
    new_entry = pd.DataFrame([[disease, treatment]], columns=["Disease", "Treatment"])
    updated_data = pd.concat([treatment_data, new_entry], ignore_index=True)
    updated_data.to_csv("treatment_database.csv", index=False)

    feedback = "Treatment added successfully!"
    return render_template("index.html", feedback=feedback)
@app.route("/", methods=["GET", "POST"])
def index():
    log = {}

    # Load training data dynamically
    try:
        existing_data = pd.read_csv("training_data.csv")
        symptom_disease_pairs = existing_data.to_dict(orient="records")  # Convert to list of dictionaries
    except FileNotFoundError:
        symptom_disease_pairs = []

    if request.method == "POST":
        symptoms = request.form["symptoms"]
        predictions = predict_with_confidence(pipeline, symptoms)

        label_probabilities = pipeline.predict_proba([symptoms])[0]
        labels = pipeline.classes_
        prediction_data = [{"label": label, "probability": round(prob * 100, 2)} for label, prob in zip(labels, label_probabilities)]

        log["symptoms"] = symptoms.split(",")
        log["predicted_class"] = predictions[0][0]
        log["confidence"] = round(predictions[0][1] * 100, 2)
        log["prediction_data"] = prediction_data
        log["all_predictions"] = predictions

        # Calculate max and min probability
        probabilities = [prediction["probability"] for prediction in prediction_data]
        log["max_probability"] = max(probabilities)
        log["min_probability"] = min(probabilities)

        # Fetch treatment recommendations for the highest probability disease
        highest_prob_disease = predictions[0][0]
        treatments = get_treatments_for_disease(highest_prob_disease)
        log["treatment_recommendations"] = treatments

        if len(predictions) > 1:
            y_true = [predictions[0][0], predictions[1][0]]
            y_pred = [predictions[0][0], predictions[1][0]]
            confusion_matrix_result = generate_confusion_matrix(y_true, y_pred, labels)
            log["confusion_matrix"] = confusion_matrix_result
            confusion_matrix_img = plot_confusion_matrix(confusion_matrix_result, labels)
            log["confusion_matrix_img"] = confusion_matrix_img
        else:
            log["confusion_matrix"] = "Not enough predictions to generate confusion matrix."

    return render_template("index.html", log=log, symptom_disease_pairs=symptom_disease_pairs)

if __name__ == "__main__":
    app.run(debug=True)
