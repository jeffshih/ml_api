from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from starlette.responses import StreamingResponse
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
    split: Optional[List[float]]
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
    feature=request.features
    tmp = []
    try:
        for k, v in feature.items():
            if len(v) != 12:
                raise IndexError
            tmp.append(v)
        inputVector = np.asarray(tmp)
    except IndexError:
        return ModelResponse(error="Invalid input data shape")
#    print(inputVector)
    output = MF.predict(inputVector)
#    print(output)
    res = {}
    for idx, k in enumerate(feature.keys()):
        res[k]= output[idx]
#    print(res)
    return ModelResponse(predictions=[res])
    

@app.post("/train")
async def setData(request:TrainRequest):
    modelName = request.model
    try:
        splitRatio = np.array(request.split)
        if len(splitRatio)!=3 or len([*filter(lambda x: x>=0, splitRatio)]) <3 or np.sum(splitRatio)!=1:
            raise ValueError
    except ValueError:
        return "Invalid input format for split ratio"
    MF.genDataSet(*splitRatio)
    MF.setModel(modelName.value)
    T = MF.getModelTestRes()
    V = MF.getModetValRes()
    if request.save == True:
        modelPkl = open("./temp/{}.pkl".format(modelName), "rb")
        #return StreamingResponse(modelPkl)
        return FileResponse(path="./temp/{}.pkl".format(modelName), filename="{}.pkl".format(modelName))
    else:
        return "Training finished" 


@app.get("/train/{modelName}/")
async def trainModel(modelName: ModelName, save: bool = False):
    MF.genDataSet()
    MF.setModel(modelName.value)
    T = MF.getModelTestRes()
    V = MF.getModetValRes()
    if save == True:
        modelPkl = open("./temp/{}.pkl".format(modelName), "rb")
        #return StreamingResponse(modelPkl)
        return FileResponse(path="./temp/{}.pkl".format(modelName), filename="{}.pkl".format(modelName))
    else:
        return "Training finished"

@app.get("/trainSave")
async def saveModel()-> FileResponse:
    MF.genDataSet()
    MF.setModel()
    m = MF.genModel()
    T = MF.getModelTestRes()
    V = MF.getModetValRes()
    msg = "{} {}".format(T,V)
    modelPkl = open("./temp/model.pkl",mode='rb')
    return FileResponse(filename="./temp/model.pkl")


@app.get("/testModel", response_model=ModelResponse)
async def testMF() -> ModelResponse:
    X = np.array([[63,1,103,1,35,0,179000,0.9,136,1,1,270]])
    P = MF.predict(X)
    return ModelResponse(predictions=[{"Prediction":P}])

@app.get("/getModelScore")
async def getScore():
    T = MF.getModelTestRes()
    V = MF.getModetValRes()
    return "{},{}".format(T,V)
