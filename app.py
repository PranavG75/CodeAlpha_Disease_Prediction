import streamlit as st
import joblib

# Load Models
heart_model = joblib.load("models/heart_model.pkl")
diabetes_model = joblib.load("models/diabetes_model.pkl")
breast_model = joblib.load("models/breast_cancer_model.pkl")

# Title
st.title("🏥 Multi Disease Prediction System")

disease = st.radio(
    "Choose Disease",
    [
        "Heart Disease",
        "Diabetes",
        "Breast Cancer"
    ]
)

st.write("Selected Disease:", disease)
# =====================================
# HEART DISEASE
# =====================================

if disease == "Heart Disease":

    st.header("❤️ Heart Disease Prediction")

    age = st.number_input("Age")
    sex = st.number_input("Sex (0 = Female, 1 = Male)")
    cp = st.number_input("Chest Pain Type")
    trestbps = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Cholesterol")
    fbs = st.number_input("Fasting Blood Sugar")
    restecg = st.number_input("Rest ECG")
    thalach = st.number_input("Maximum Heart Rate Achieved")
    exang = st.number_input("Exercise Induced Angina")
    oldpeak = st.number_input("Oldpeak")
    slope = st.number_input("Slope")
    ca = st.number_input("Number of Major Vessels")
    thal = st.number_input("Thal")

    if st.button("Predict Heart Disease"):

        data = [[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]]

        prediction = heart_model.predict(data)

        if prediction[0] == 1:
            st.error("⚠️ Heart Disease Detected")
        else:
            st.success("✅ No Heart Disease Detected")

# =====================================
# DIABETES
# =====================================

elif disease == "Diabetes":

    st.header("🩸 Diabetes Prediction")

    preg = st.number_input("Pregnancies")
    glucose = st.number_input("Glucose")
    bp = st.number_input("Blood Pressure")
    skin = st.number_input("Skin Thickness")
    insulin = st.number_input("Insulin")
    bmi = st.number_input("BMI")
    dpf = st.number_input("Diabetes Pedigree Function")
    age = st.number_input("Age")

    if st.button("Predict Diabetes"):

        data = [[
            preg,
            glucose,
            bp,
            skin,
            insulin,
            bmi,
            dpf,
            age
        ]]

        prediction = diabetes_model.predict(data)

        if prediction[0] == 1:
            st.error("⚠️ Diabetes Detected")
        else:
            st.success("✅ No Diabetes Detected")

# =====================================
# BREAST CANCER
# =====================================

else:

    st.header("🎗️ Breast Cancer Prediction")

    st.warning(
        "Your breast cancer model was trained on 30 features. "
        "To use it properly, you'll need a form with all 30 inputs "
        "or retrain a simplified model."
    )

    st.info(
        "Heart Disease and Diabetes predictions are fully functional."
    )