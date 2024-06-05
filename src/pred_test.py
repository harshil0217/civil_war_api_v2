from data_preprocessing import preprocess_data_predict
import keras
from keras.models import load_model
import shap
import lime
import numpy as np
import pandas as pd
import sqlite3

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
    X_train = pd.read_csv('./data/X_train_non_pca.csv')
    conn = sqlite3.connect('./data/2023_data.db')
    cur = conn.cursor()
    cur.execute("SELECT * FROM indicators_2023")
    data_2023 = cur.fetchall()
    columns = [description[0] for description in cur.description]
    data_2023 = np.array(data_2023)
    data_2023 = data_2023[:, 2:]
    data_2023 = data_2023.astype(float)
    columns = columns[2:]
    explainer = shap.DeepExplainer(model, X_train[0:100])
    shap_values = explainer.shap_values(data_2023)
    shap.initjs()
    shap.force_plot(explainer.expected_value[0].numpy(), shap_values[0][0],
                    feature_names=columns, matplotlib=True, show=True)

    
    
    
    
print(predict('MEX'))
get_shap_plot('MEX')