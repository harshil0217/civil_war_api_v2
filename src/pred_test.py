from data_preprocessing import preprocess_data_predict
import keras
from keras.models import load_model

model = load_model('./models/civil_war_model.h5')

def predict(country):
    X = preprocess_data_predict(country)
    if X is None:
        return "Insufficient Data"
    prob = model.predict(X)
    return prob[0][0]


print(predict('RUS'))