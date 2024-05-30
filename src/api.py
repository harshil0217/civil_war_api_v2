from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from data_preprocessing import preprocess_data_predict
import keras

#creating fastpi app
app = FastAPI()

@app.on_event("startup")
async def load_model():
    global model
    model = keras.models.load_model('../models/civil_war_model.h5')


# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'HOME'}

# Defining request format
class request_body(BaseModel):
    country: str

# Defining endpoint
@app.post('/predict')
async def predict(data: request_body):
    X = preprocess_data_predict(data.country)
    if X is None:
        raise HTTPException(status_code=400, detail ="No Data Available for Country in 2023")
    prob = model.predict(X)
    print(prob[0][0])
    return ({"prob of civil war": prob[0][0].item()})

