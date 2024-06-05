from data_preprocessing import preprocess_data_predict
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
import sqlite3
import xgboost as xgb

model = xgb.XGBClassifier()
model.load_model('./models/xgb_model.json') 

#prediction function
def predict(country):
    """
    Function to predict the probability of civil war for a given country
    """
    X, columns = preprocess_data_predict(country, pca=False)
    if X is None:
        return "Insufficient Data"
    prob = model.predict(X)
    return prob[0]

def get_lime_plot(country):
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
    def prob(data):
        return np.array(list(zip(1-model.predict(data), model.predict(data))))
    explainer = LimeTabularExplainer(X_train.astype(float).values,
                                     mode='classification', feature_names=columns, 
                                     training_labels=['No Civil War', 'Civil War'])
    explanation = explainer.explain_instance(data_2023[0], prob, num_features=5)
    preds = model.predict_proba(data_2023)
    return explanation, preds

print(predict('PAK'))
print(get_lime_plot('MEX'))


    