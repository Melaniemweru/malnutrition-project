import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and preprocess data
data = pd.read_csv('simulated_child_malnutrition_data.csv')
data = data.dropna()
data = pd.get_dummies(data, columns=['gender','immunization_status'], drop_first=True)

x = data.drop('nutrition_status', axis=1)
y = data['nutrition_status']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

st.title("Child Malnutrition Prediction")

age_months = st.number_input("Age (months):", min_value=0, max_value=120, value=24)
weight_kg = st.number_input("Weight (kg):", min_value=0.0, max_value=50.0, value=10.0)
height_cm = st.number_input("Height (cm):", min_value=0.0, max_value=150.0, value=75.0)
muac_cm = st.number_input("MUAC (cm):", min_value=0.0, max_value=30.0, value=13.0)
recent_illness = st.selectbox("Recent illness?", options=[0, 1], format_func=lambda x: "Yes" if x else "No")
gender = st.selectbox("Gender:", options=["male", "female"])
immunization_status = st.selectbox("Immunization status:", options=["complete", "partial"])

gender_Male = 1 if gender == "male" else 0
immunization_status_Partial = 1 if immunization_status == "partial" else 0

user_data = pd.DataFrame({
    'age_months': [age_months],
    'weight_kg': [weight_kg],
    'height_cm': [height_cm],
    'muac_cm': [muac_cm],
    'recent_illness': [recent_illness],
    'gender_Male': [gender_Male],
    'immunization_status_Partial': [immunization_status_Partial]
})

if st.button("Predict"):
    prediction = model.predict(user_data)
    st.success(f"Predicted nutrition status: {prediction[0]}")