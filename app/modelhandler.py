import joblib
import numpy as np

def _load_model():
    model = joblib.load('disease-prediction\models\heart_disease_model.joblib')
    return model

def predict_disease(input_data):
    model = _load_model()
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]