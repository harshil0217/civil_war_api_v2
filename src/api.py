from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

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
    

