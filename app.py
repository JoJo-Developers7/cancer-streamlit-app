import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("cancer_model.pkl", "rb"))

st.title("Cancer Stage Prediction App")
st.subheader("Fill the patient symptoms to predict the cancer level.")

# All features except Patient Id
features = {
    "Age": st.number_input("Age", min_value=1, max_value=120, step=1),
    "Gender": st.selectbox("Gender (0=Female, 1=Male)", [0, 1]),
    "Air Pollution": st.selectbox("Air Pollution", [0, 1, 2, 3, 4, 5]),
    "Alcohol use": st.selectbox("Alcohol use", [0, 1, 2, 3, 4, 5]),
    "Dust Allergy": st.selectbox("Dust Allergy", [0, 1, 2, 3, 4, 5]),
    "OccuPational Hazards": st.selectbox("Occupational Hazards", [0, 1, 2, 3, 4, 5]),
    "Genetic Risk": st.selectbox("Genetic Risk", [0, 1, 2, 3, 4, 5]),
    "chronic Lung Disease": st.selectbox("Chronic Lung Disease", [0, 1, 2, 3, 4, 5]),
    "Balanced Diet": st.selectbox("Balanced Diet", [0, 1, 2, 3, 4, 5]),
    "Obesity": st.selectbox("Obesity", [0, 1, 2, 3, 4, 5]),
    "Smoking": st.selectbox("Smoking", [0, 1, 2, 3, 4, 5]),
    "Passive Smoker": st.selectbox("Passive Smoker", [0, 1, 2, 3, 4, 5]),
    "Chest Pain": st.selectbox("Chest Pain", [0, 1, 2, 3, 4, 5]),
    "Coughing of Blood": st.selectbox("Coughing of Blood", [0, 1, 2, 3, 4, 5]),
    "Fatigue": st.selectbox("Fatigue", [0, 1, 2, 3, 4, 5]),
    "Weight Loss": st.selectbox("Weight Loss", [0, 1, 2, 3, 4, 5]),
    "Shortness of Breath": st.selectbox("Shortness of Breath", [0, 1, 2, 3, 4, 5]),
    "Wheezing": st.selectbox("Wheezing", [0, 1, 2, 3, 4, 5]),
    "Swallowing Difficulty": st.selectbox("Swallowing Difficulty", [0, 1, 2, 3, 4, 5]),
    "Clubbing of Finger Nails": st.selectbox("Clubbing of Finger Nails", [0, 1, 2, 3, 4, 5]),
    "Frequent Cold": st.selectbox("Frequent Cold", [0, 1, 2, 3, 4, 5]),
    "Dry Cough": st.selectbox("Dry Cough", [0, 1, 2, 3, 4, 5]),
    "Snoring": st.selectbox("Snoring", [0, 1, 2, 3, 4, 5])
}

# Prepare input for prediction
input_data = np.array([list(features.values())])

if st.button("Predict Cancer Stage"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Cancer Level: {prediction}")
