from enum import Enum
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
import numpy as np
import argparse
import json



class ModelName(str, Enum):
    dt = "DecisionTree"
    svm = "SVM"
    lr = "LogisticRegression"


def processFeature(inputFile):
    f = open(inputFile, 'rb')
    feature = json.load(f)['features']
    tmp = []
    for k, v in feature.items():
        tmp.append(v)
    inputVector = np.asarray(tmp)
    f.close()
    return inputVector

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model", help="Path to the model", default="temp/LogisticRegression.joblib")
    parser.add_argument("-f","--feature", help="Assign the path to the feature to predict", default="feature.json")
    args = parser.parse_args()

    model = open(args.model, 'rb')
    clf = load(model)
    model.close
   
    input = processFeature(args.feature)
    print(clf.predict(input))
