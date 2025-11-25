import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("cancer_model.pkl", "rb"))

st.title("Cancer Stage Prediction App")
st.subheader("Fill the patient symptoms to predict the cancer level.")

# All features except Patient Id
features = {
    "Patient Id": st.number_input("Patient Id", min_value=1, max_value=120, step=1),
    "Age": st.number_input("Age", min_value=1, max_value=120, step=1),
    "Gender": st.selectbox("Gender (0=Female, 1=Male)", [0, 1]),
    "Air Pollution": st.selectbox("Air Pollution", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "Alcohol use": st.selectbox("Alcohol use", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "Dust Allergy": st.selectbox("Dust Allergy", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "OccuPational Hazards": st.selectbox("Occupational Hazards", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "Genetic Risk": st.selectbox("Genetic Risk", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "chronic Lung Disease": st.selectbox("Chronic Lung Disease", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "Balanced Diet": st.selectbox("Balanced Diet", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "Obesity": st.selectbox("Obesity", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "Smoking": st.selectbox("Smoking", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "Passive Smoker": st.selectbox("Passive Smoker", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "Chest Pain": st.selectbox("Chest Pain", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "Coughing of Blood": st.selectbox("Coughing of Blood", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "Fatigue": st.selectbox("Fatigue", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "Weight Loss": st.selectbox("Weight Loss", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "Shortness of Breath": st.selectbox("Shortness of Breath", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "Wheezing": st.selectbox("Wheezing", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "Swallowing Difficulty": st.selectbox("Swallowing Difficulty", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "Clubbing of Finger Nails": st.selectbox("Clubbing of Finger Nails", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "Frequent Cold": st.selectbox("Frequent Cold", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "Dry Cough": st.selectbox("Dry Cough", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    "Snoring": st.selectbox("Snoring", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
}

# Prepare input for prediction
input_data = np.array([list(features.values())])

if st.button("Predict Cancer Stage"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Cancer Level: {prediction}")


