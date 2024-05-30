from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from data_preprocessing import preprocess_data_predict
import keras


#loading in model
model = keras.models.load_model('../models/civil_war_model.h5')

#creating fastpi app
app = FastAPI()

# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'PENIS'}

# Defining path operation for /name endpoint
@app.get('/{name}')
def hello_name(name : str): 
    # Defining a function that takes only string as input and output the
    # following message. 
    return {'message': f'PENIS, {name}'}

# Defining request format
class request_body(BaseModel):
    country: str

# Defining endpoint
@app.post('/predict')
def predict(data: request_body):
    X = preprocess_data_predict(data.country)
    if X is None:
        raise HTTPException(status_code=400, detail ="No Data Available for Country in 2023")
    prob = model.predict(X)
    return ({"prob of civil war": prob})

