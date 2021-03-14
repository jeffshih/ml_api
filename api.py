from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from starlette.routing import request_response
from starlette.types import Message
from model import ModelFactory
from typing import Optional, List, Dict
from enum import Enum
import numpy as np



class FeatureException(Exception):
    def __init__(self, name:str):
        self.name=name

class ModelName(str, Enum):
    dt = "DecisionTree"
    svm = "SVM"
    lr = "LogisticRegression"


class PredictRequest(BaseModel):
    features: Dict[str, List[float]]


class ModelResponse(BaseModel):
    predictions: Optional[List[Dict[str, float]]]
    error: Optional[str]


class TrainRequest(BaseModel):
    model: ModelName
    save: bool

class TrainResponse(BaseModel):
    message: str
    error: Optional[str]



MF = ModelFactory()
app = FastAPI()



@app.get("/", response_model=ModelResponse)
async def root() -> ModelResponse:
    return ModelResponse(error="This is a test endpoint.")


@app.get("/predict", response_model=ModelResponse)
async def explain_api() -> ModelResponse:
    return ModelResponse(
        error="Send a POST request to this endpoint with 'features' data."
    )


@app.post("/predict")
async def get_model_predictions(request: PredictRequest) -> ModelResponse:
    #TODO  YOUR CODE HERE
    pass

@app.post("/train")
async def trainModel(request: TrainRequest):
    modelName = request.model
    MF.setModel(modelName.value)
    if request.save:
        return "saving {}".format(modelName)
    else:
        return "model {} is doing good".format(modelName)
    

    '''
    MF.genDataSet(modelName)
    MF.setModel() 
    m = MF.genModel()
    '''

@app.get("/trainSave")
async def saveModel()-> FileResponse:
    MF.genDataSet()
    MF.setModel()
    m = MF.genModel()
    T = MF.getModelTestRes()
    V = MF.getModetValRes()
    msg = "Test score :{}\n Validation score:{}".format(T,V)
    modelPkl = open("./temp/model.pkl",mode='rb')
    return FileResponse(filename="./temp/model.pkl")


@app.get("/testModel", response_model=ModelResponse)
async def testMF() -> ModelResponse:
    X = np.array([[63,1,103,1,35,0,179000,0.9,136,1,1,270]])
    #try:
    P = MF.predict(X)
    #except:
    #    raise ValueError
    return ModelResponse(predictions=[{"Prediction":P}])

@app.get("/testResponse", response_model=ModelResponse)
async def testMF() -> ModelResponse:
    return ModelResponse(predictions=[{"a":1.0},{"b":0.3,"c":2.1}])

@app.get("/testSaveModel", response_model=TrainResponse)
async def saveModel() ->TrainResponse:

    return ModelResponse()