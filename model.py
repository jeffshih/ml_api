from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import pickle




class ModelFactory(object):

    def __init__(self):
        
        self.modelType = None
        
        self.model = None
        self.modelResult = None


        self.dataset = pd.read_csv("./heart_failure_clinical_records_dataset.csv")
        self.X = self.dataset.drop(columns=['DEATH_EVENT'])
        self.y = self.dataset['DEATH_EVENT']
        self.X_val = None
        self.y_val = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.t_test = None
                


    def setModel(self, modelName = "DecisionTree"):
        self.modelType = modelName
        if(modelName == "DecisionTree"):
            self.__genDecisionTree()
        elif(modelName == "LogisticRegression"):
            self.__genLR()
        else:
            self.__genSVM()
        path = "./temp/{}.pkl".format(modelName)
        modelDump = open(path, "wb")
        pickle.dump(self.model, modelDump)
        modelDump.close()

    def printValidationSet(self):
        print(self.X_val)
        print(self.y_val)
    
    def genDataSet(self, train=0.75, test=0.15, val=0.1):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=1-train, stratify=self.y)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_test, self.y_test, test_size=test/(test + val), stratify=self.y_test)


    def __genDecisionTree(self):
        rng = np.random.RandomState(42)
        params_dt = [{'max_depth':np.arange(1,21),
                      'min_samples_leaf':[1, 5, 10, 20, 50, 100, 200]}]
        self.model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=30), n_estimators=300, random_state=rng)
        self.model.fit(self.X_train, self.y_train)

        
    def __genLR(self):
        self.model = LogisticRegression(solver='liblinear', max_iter=1000)
        self.model.fit(self.X_train, self.y_train)
        

    def __genSVM(self):
        self.model = SVR()
        self.model.fit(self.X_train, self.y_train)



    def getModelTestRes(self):
        return('test score with {} model: {}'.format(self.modelType, self.model.score(self.X_test, self.y_test)))

    def getModetValRes(self):
        return('val score with {} model: {}'.format(self.modelType, self.model.score(self.X_val, self.y_val))) 

    def predict(self, feature):
        return self.model.predict(feature)


if __name__ == "__main__":

    MF = ModelFactory()
    MF.genDataSet()
    MF.setModel()
    model = MF.genModel()
    print(MF.getModelTestRes())
    #MF.printValidationSet()
    print(MF.getModetValRes())

    X = np.array([[63,1,103,1,35,0,179000,0.9,136,1,1,270]])
    print(model.predict(X))
    
