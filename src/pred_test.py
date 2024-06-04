from data_preprocessing import preprocess_data_predict
import keras
from keras.models import load_model
import shap
import lime
import numpy as np

model = load_model('./models/civil_war_model_non_pca.h5')

def predict(country):
    """
    Function to predict the probability of civil war for a given country
    """
    X, columns = preprocess_data_predict(country, pca=False)
    if X is None:
        return "Insufficient Data"
    prob = model.predict(X)
    return prob[0][0]

def get_shap_plot(country):
    X, columns = preprocess_data_predict(country, pca=False)
    if X is None:
        return "Insufficient Data"
    explainer = lime.lime_tabular.LimeTabularExplainer(X, mode='regression')
    print(columns)
    
    
    
    
print(predict('MEX'))
get_shap_plot('MEX')