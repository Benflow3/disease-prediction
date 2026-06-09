import joblib

import numpy as np



def load_model_pipeline():

    model = joblib.load('disease-prediction\models\heart_disease_model.joblib')
    return model

def _scaler_and_classifier(model):
    scaler = model.named_steps['scaler']

    classifier = model.named_steps['classifier']
    return scaler, classifier


def process_and_predict_disease(model,input_data):
    scaler, classifier = _scaler_and_classifier(model)
    input_scale = scaler.transform(input_data)
    prediction = classifier.predict(input_scale)[0]
    probabilty = classifier.predict_proba(input_scale)[0]
    return prediction,probabilty,input_scale


def generate_lime_explanation(model,Input_scale,x_train,features_name):

    explainer = lime.lime_tabular.LimeTabularExplainers(
        training_data=x_train,
        feature_names=features_name,
        class_names=['Unlikely','Likely'],
        mode="classification"

    )
    scaler ,classifier = _scaler_and_classifier(model)
    exp = explainer.explain_instance(
        data_row = Input_scale[0],
        predict_fn = classifier.predict_proba

    )

    return exp.as_html()