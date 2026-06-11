import joblib
import pandas as pd
import numpy as np
import lime
import dill



def load_model_pipeline():
    model = joblib.load(r'models\heart_disease_model (1).joblib')
    return model

def _scaler_and_classifier(model):
    scaler = model.named_steps['preproc']
    classifier = model.named_steps['clf']
    return scaler, classifier


def process_and_predict_disease(model,input_data):
    # Convert 1D list to 2D array for the model
    input_array = np.array(input_data).reshape(1, -1)
    df = pd.DataFrame(input_array, columns=model.feature_names_in_)
    
    # Predict using the full pipeline (handles scaling + classification)
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]
    
    # LIME needs a function that accepts a 2D numpy array and returns probabilities.
    # We wrap the pipeline's predict_proba to ensure it receives a DataFrame with names.
    predict_fn = lambda x: model.predict_proba(pd.DataFrame(x, columns=model.feature_names_in_))
    
    # Return raw input_array (aliased to input_scale for compatibility with app.py)
    return prediction, probability, input_array, predict_fn


def generate_lime_explanation(model,Input_scale,prob_fn):
    
    with open(r"models\lime_explainer.pkl","rb") as f:
        explainer = dill.load(f)

    exp = explainer.explain_instance(
        data_row = Input_scale[0],
        predict_fn = prob_fn
    )

    return exp.as_html()
