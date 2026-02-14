import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Adult Income Prediction System", page_icon="💼", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "features.pkl"))

st.title("💼 Adult Income Prediction System")
st.markdown("### Predict whether income exceeds $50K/year")
st.write("---")

st.sidebar.header("Enter Applicant Details")

age = st.sidebar.slider("Age", 18, 70, 30)
education_num = st.sidebar.slider("Education Level (Numeric)", 1, 16, 10)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)

input_dict = {
    'age': age,
    'education-num': education_num,
    'hours-per-week': hours_per_week
}

input_df = pd.DataFrame([input_dict])

for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_columns]

input_scaled = scaler.transform(input_df)

st.write("## Prediction Result")

if st.button("Predict Income Class"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0].max()

    if prediction == 1:
        st.success(f"Income > $50K (Confidence: {probability:.2f})")
    else:
        st.info(f"Income ≤ $50K (Confidence: {probability:.2f})")

st.write("---")
st.caption("Machine Learning Classification Assignment | Random Forest Deployment")
