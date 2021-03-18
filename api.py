from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from starlette.routing import request_response
import pandas as pd
from model import ModelFactory
from typing import Optional, List, Dict
from enum import Enum
import numpy as np
from config import *



class ModelName(str, Enum):
    dt = "DecisionTree"
    svm = "SVM"
    lr = "LogisticRegression"
    rf = "RandomForest"


class PredictRequest(BaseModel):
    features: Dict[str, float]


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
    try:
        for k, v in feature.items():
            if k not in featureKeys:
                raise KeyError
            if type(v) != int and type(v) != float:
                raise ValueError
    except KeyError:
        return ModelResponse(error = "Invalid input feature key")
    except ValueError:
        return ModelResponse(error = "Invalid input feature value")
    d = pd.DataFrame.from_dict([feature])
    f = d.drop(columns=['DEATH_EVENT'])
    output = MF.predict(f)
    res = zip([str(i) for i in range(d.shape[0])], output)
    return ModelResponse(predictions=[res])
    


@app.get("/train")
async def trainModel(save : bool=False):
    MF.genDataSet()
    MF.getModel()
    if save:
        return FileResponse(path="./temp/model.joblib", filename="model.joblib")
    else:
        return TrainResponse(message="Training Success") 

@app.get("/getModel")
async def getModel()->FileResponse:
    return FileResponse(path="./temp/model.joblib", filename="model.joblib") 


@app.get("/getModelScore")
async def getScore():
    T = MF.getModelTestRes()
    V = MF.getModetValRes()
    return "{},{}".format(T,V)
