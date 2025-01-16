**Acute Upper Respiratory Tract Infection (AURTI) Prediction and Treatment Suggestions**

**Project Definition**
This project aims to build a robust system for predicting Acute Upper Respiratory Tract Infections (AURTI) and providing evidence-based treatment options. 
By leveraging patient symptoms, medical history, and other clinical inputs, the system offers actionable insights for both diagnosis and management.

**Description**
The project focuses on creating a platform that simplifies the identification and treatment of AURTI. _It incorporates key features such as:_

- **Symptom-Based Disease Prediction:** Utilizing patient-reported symptoms to identify AURTI and its severity.
- **Treatment Option Suggestions:** Offering a tailored list of evidence-based treatment options aligned with clinical guidelines.
- **Explainability:** The system provides clear reasoning behind its predictions and recommendations to ensure transparency.
- **User-Centric Interface:** Designed with healthcare professionals in mind, the platform ensures ease of use and integration into clinical workflows.

**Python Libraries required:**
- pip install Flask
- pip install scikit-learn
- pip install pandas
- pip install medspacy
- pip install joblib
- pip install transformers

**Incase you would like to run code public for testing instead of local:**
_Use ngrok this token active_
- https://ngrok.com/
_Download the Agent then run these codes:_
- ngrok config add-authtoken 2pt6USy5bvzlggrmubU4hJdLAwG_6yYrevZsJc8K5LSsL1zA7
- ngrok.exe http 80
- ngrok http 5000

This open-source initiative invites contributions to improve prediction accuracy and expand the range of treatment options.

***

**Train Panel for Signs & Symptoms and Treatment Options**
_This section enables users to:_
- Train the model with specific signs and symptoms associated with different diseases.
- Input treatment options related to each disease for better recommendations.
- Utilize machine learning models to learn from the provided data, improving the accuracy of disease prediction and treatment suggestions.
<img width="1279" alt="Homepage" src="https://github.com/user-attachments/assets/91dc59b9-cdd2-41fe-9b63-6493374ac27e" />

***

**Enter the Encounter Description for the Patient**
_Steps:_
- Input the patient's encounter details in the provided text box.
- Include signs, symptoms, or other relevant observations.
- Press the Predict button to initiate the analysis.
The system processes the input and predicts the likely diseases along with their probabilities. The predictions are dynamically updated as additional data is added during the patient visit.
<img width="996" alt="1" src="https://github.com/user-attachments/assets/d887bd19-5843-4b5e-bccb-d53e43d310c3" />

***

**Class Probabilities for Proper Diagnosis**
_Key Features:_
- The system displays the probabilities of each potential disease based on the provided information.
- Highlights diseases with similar symptoms to guide further investigation.
- Allows users to refine predictions by adding extra details (e.g., lab results or additional symptoms).
- Ensures transparency by showing the rationale behind the predictions.
<img width="977" alt="2" src="https://github.com/user-attachments/assets/9506d134-ce26-41e1-9de5-be0c1c85e8ed" />

***

**Variables for Measuring Disease Probabilities**
_The system uses variables such as:_
- Patient’s age, gender, and medical history.
- Specific signs and symptoms input during the session.
- Lab test results or other diagnostic parameters.
**Note:** Proper training on the use of these variables is essential to ensure accurate disease probability estimation.
  
**Confusion Matrix for Enhanced Confidence Rates**
_The Confusion Matrix is a critical tool to:_
- Evaluate the performance of the predictive model.
- Identify overlapping symptoms between diseases and improve distinction through additional training.
- Visualize the model’s accuracy by showing true positives, false positives, true negatives, and false negatives.
- Iteratively enhance the model’s confidence rate, especially for diseases with common symptoms.
<img width="971" alt="3" src="https://github.com/user-attachments/assets/9e71c256-819c-4a2b-b500-120c338579e3" />

***

**Future Enhancements**
- **Data Augmentation:** Expand the dataset with diverse cases to improve model robustness.
- **User Feedback Loop:** Allow users to validate predictions and provide feedback for continuous learning.
- **Visualization Tools:** Introduce charts and graphs for better interpretability of predictions and class probabilities.
By following this structured approach, the homepage panel can effectively streamline the process of disease prediction and treatment suggestion, improving the overall user experience and clinical outcomes.
