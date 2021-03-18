from enum import Enum
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
import numpy as np
import argparse
import json
import pandas as pd
from config import *


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model", help="Path to the model", default="temp/model.joblib")
    parser.add_argument("-f","--feature", help="Assign the path to the feature to predict", default="predict.json")
    args = parser.parse_args()

    model = open(args.model, 'rb')
    clf = load(model)
    model.close
   
    input = processFeature(args.feature)
    print(clf.predict(input))
