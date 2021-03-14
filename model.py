from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
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
            self.model = 
        #todo choose model

    def printValidationSet(self):
        print(self.X_val)
        print(self.y_val)
    
    def genDataSet(self, train=0.75, val=0.15, test=0.1):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=1-train, stratify=self.y)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_test, self.y_test, test_size=test/(test + val), stratify=self.y_test)


    def __genDecisionTree(self):
        rng = np.random.RandomState(42)
        params_dt = [{'max_depth':np.arange(1,21),
                      'min_samples_leaf':[1, 5, 10, 20, 50, 100, 200]}]
        self.model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators=300, random_state=rng)
        

    def genModel(self):
        dt = DecisionTreeRegressor()
        rng = np.random.RandomState(1)
        #print(dt.get_params().keys())
        params_dt = [{'max_depth':np.arange(1,21),
                      'min_samples_leaf':[1, 5, 10, 20, 50, 100, 200]}]
        ab = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators=300,random_state=rng)
        dt_gs = GridSearchCV(dt, param_grid=params_dt, cv=5)
        dt_gs.fit(self.X_train, self.y_train)
        ab.fit(self.X_train, self.y_train)
        
        #if ensemble means using single algorithms, using adaboost to construct the model otherwise prepare the api
        #and ready to use GSCV to tune hyperparam
        
        #print('adaboost: {}'.format(ab.score(self.X_test, self.y_test)))
        dt_best = dt_gs.best_estimator_
        #print('dt_gs: {}'.format(dt_best.score(self.X_test, self.y_test)))
        self.model = ab
        modelDump = open("./temp/model.pkl","wb")
        pickle.dump(self.model, modelDump)
        modelDump.close()
        self.modelResult = dt_best
        return self.model

    def getModelTestRes(self):
#        print('dt: {}'.format(self.modelResult.score(self.X_test, self.y_test)))
        return('ab: {}'.format(self.model.score(self.X_test, self.y_test)))

    def getModetValRes(self):
#        print('dt: {}'.format(self.modelResult.score(self.X_val, self.y_val))) 
        return('ab: {}'.format(self.model.score(self.X_val, self.y_val))) 

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
    
