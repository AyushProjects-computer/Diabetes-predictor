import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('diabetes_model.pkl')

st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Early Stage Diabetes Risk Predictor")

st.markdown("Enter the following health metrics to check diabetes risk.")

# Form input fields
pregnancies = st.slider('Number of Pregnancies', 0, 20, 1)
glucose = st.slider('Glucose Level', 0, 200, 100)
blood_pressure = st.slider('Blood Pressure (mm Hg)', 0, 140, 70)
skin_thickness = st.slider('Skin Thickness (mm)', 0, 100, 20)
insulin = st.slider('Insulin Level', 0, 900, 80)
bmi = st.slider('BMI', 0.0, 70.0, 25.0)
dpf = st.slider('Diabetes Pedigree Function', 0.0, 3.0, 0.5)
age = st.slider('Age', 10, 100, 30)

# Predict button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes Detected!")
    else:
        st.success("✅ No Risk Detected. You're Safe!")

st.markdown("---")
st.caption("Created by Ayush Ranjan during Summer Internship @ Cognifyz Technologies")
