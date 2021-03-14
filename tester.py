from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict
from enum import Enum



class ModelName(str, Enum):
    dt = "DecisionTree"
    svm = "SVM"
    lr = "LogisticRegression"

class PredictRequest(BaseModel):
    features: Dict[str, List[float]]


class TrainRequest(BaseModel):
    model: ModelName
    save: bool    

app = FastAPI()

@app.get("/")
def home():
    return {"Hello": "World"}

@app.get("/test")
def show():
    request = PredictRequest
    print(request.features)
    return {"hi":"wut"}

@app.post("/receive")
def receive(request: PredictRequest):
    print(request)
    return request.features

@app.post("/train")
def receive(request: TrainRequest):
    return {"model name" : request.model, "save:": request.save}

@app.get("/train/{modelName}")
async def train(modelName:ModelName, save=False):
    return {"model name" : modelName, "save": save}
