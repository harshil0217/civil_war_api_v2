from data_preprocessing import preprocess_data_predict
from keras.models import load_model
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


model = load_model('./models/civil_war_model_non_pca.h5')
X_train = pd.read_csv('./data/X_train_non_pca.csv')
y_train = pd.read_csv('./data/y_train_non_pca.csv')
X, columns = preprocess_data_predict('AFG', pca=False)
explainer = LimeTabularExplainer(X_train.to_numpy(), mode = 'classification',
                                     training_labels= y_train.to_numpy().ravel(), 
                                     feature_names=columns)
'''
conn = sqlite3.connect('./data/2023_data.db')
cur = conn.cursor()
cur.execute("SELECT * FROM indicators_2023")
data_2023 = cur.fetchall()
columns = [description[0] for description in cur.description]
data_2023 = np.array(data_2023)
data_2023 = pd.DataFrame(data_2023, columns=columns)
data_2023[data_2023.columns[2:]] = data_2023[data_2023.columns[2:]].astype(float)
print(columns)
'''


def predict(country):
    """
    Function to predict the probability of civil war for a given country
    """
    X, columns = preprocess_data_predict(country, pca=False)
    if X is None:
        return "Insufficient Data"
    prob = model.predict(X)
    return prob[0][0]
    

#create probability function
def prob(data):
    p1, p2 = (np.ones(len(data))[0] - model.predict(data)), model.predict(data)
    prediction = [[x,y] for x,y in zip(p1,p2)]
    return np.array(prediction).reshape(len(data),2)

#function to get lime plot
def lime_exp_as_plot(exp, flag):
    exp = exp.as_list(label=1)
    exp = sorted(exp, key=lambda x: abs(x[1]))
    
    exp = exp[-10:]
    
    vals = [x[1] for x in exp]
    names = [x[0] for x in exp]
    
    colors = ['red' if x < 0 else 'green' for x in vals]
    pos = np.arange(len(exp)) + .5
    
    plt.barh(pos, vals, color=colors)
    plt.yticks(pos, names)
    plt.title('LIME Plot for Civil War Prediction')
    plt.xlabel('Effect on Prediction')
    plt.ylabel('Feature')
    plt.show()
    
    
def get_lime(country):
    X, columns = preprocess_data_predict(country, pca=False)
    X = X[0]
    positive_lime = explainer.explain_instance(X,prob, num_features=283)  
    
    lime_exp_as_plot(positive_lime, 'positive')
    
print(predict('MMR'))
get_lime('MMR')

