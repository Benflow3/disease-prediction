import streamlit as st
import numpy as np
from modelhandler import load_model_pipeline,process_and_predict_disease,generate_lime_explanation

st.set_page_config(
    page_title = "Disease Prediction System",
    page_icon = "this",
    layout = 'wide'
)

st.title("Disease Prediction System")
st.markdown("Enter the patient's health parameters to predict the likelihood of heart disease with AI explanation.")

@st.cache_resource
def load_model():
    return load_model_pipeline
model = load_model()

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 20, 80, 50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x:"Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Serum Cholesterol", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x:"No" if x == 0 else "Yes")
    restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x])
with col2:
    thalach = st.slider("Maximum Heart Rate Achieved", 70, 200, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x:"No" if x == 0 else "Yes")
    oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [0, 1, 2], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x])

raw_input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
if st.button("Analyze Patient Data",use_container_width=True):
    prediction,probability,input_scale = process_and_predict_disease(model,input_data=raw_input_data)
    st.markdown("___")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"The model predicts that the patient is likely to have heart disease with probabilty of {probability[1]:.2%}")
        else:
            st.success("The model predicts that the patient is unlikely to have heart disease with probabilty of {probability[0]:.2%}")
    with col2:
        st.subheader("AI Explanation")
        with st.spinner('Generating LIME explanation...'):
            lime_html = generate_lime_explanation(model,input_scale,input_data,featuers)
            components.html(lime_html,height=400,scrolling=True)