import joblib

heart_model = joblib.load("models/heart_model.pkl")
diabetes_model = joblib.load("models/diabetes_model.pkl")
breast_model = joblib.load("models/breast_cancer_model.pkl")

def predict_heart(data):
    return heart_model.predict([data])

def predict_diabetes(data):
    return diabetes_model.predict([data])

def predict_breast(data):
    return breast_model.predict([data])