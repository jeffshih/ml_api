from joblib import dump, load
import numpy as np
import argparse
import json
import pandas as pd
from config import *
import requests
import datetime
from io import BytesIO

BaseUrl = 'http://0.0.0.0:8000/'

def processFeature(inputFile):
    f = open(inputFile, 'rb')
    feature = json.load(f)['features']
    try:
        for k, v in feature.items():
            if k not in featureKeys:
                raise KeyError
            if type(v) != int and type(v) != float:
                raise ValueError
    except KeyError:
        return "Invalid input feature key"
    except ValueError:
        return "Invalid input feature value"
    d = pd.DataFrame.from_dict([feature])
    f = d.drop(columns=['DEATH_EVENT'])[selectedFeatures]
    return f

def getTrain():
    url = BaseUrl+"train"
    res = requests.get(url)
    time = datetime.datetime.now().strftime("%H%M%S")
    print(time)
    print(res.status_code)
    print(res.text)

def getTrainSaved():
    url = BaseUrl+"train"
    payload = {"save":True}
    res = requests.get(url,params=payload)
    time = datetime.datetime.now().strftime("%H%M%S")
    print(time)
    print(res.status_code)
    with open("./model.joblib", 'wb') as f:
        f.write(res.content)

def getTrainResult():
    url = BaseUrl+"getModelScore"
    res = requests.get(url)
    time = datetime.datetime.now().strftime("%H%M%S")
    print(time)
    print(res.status_code)
    print(res.text)

def postPredict():
    url = BaseUrl+"predict"
    headers = {'Content-type': 'application/json'}
    f = open("./predict.json", "rb")
    jsonFile = json.load(f)
    res = requests.post(url, headers=headers, json=jsonFile)
    print(res.status_code)
    print(res.text)
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--prediction",help="flag to turn off loading/predicting stuff", default = False)
    parser.add_argument("-m","--model", help="Path to the model", default="temp/model.joblib")
    parser.add_argument("-f","--feature", help="Assign the path to the feature to predict", default="predict.json")
    args = parser.parse_args()

    if args.prediction:
        model = open(args.model, 'rb')
        clf = load(model)
        model.close
   
        input = processFeature(args.feature)
        print(clf.predict(input))
    else:
        getTrain()
        getTrainSaved()
        getTrainResult()
        postPredict()