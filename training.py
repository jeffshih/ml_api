from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np





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
                


    def setModel(self, modelType = "DecisionTree"):
        self.modelType = modelType
        #todo choose model

    def printValidationSet(self):
        print(self.X_val)
        print(self.y_val)
    
    def genDataSet(self, train=0.75, val=0.15, test=0.1):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=1-train, stratify=self.y)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_test, self.y_test, test_size=test/(test + val), stratify=self.y_test)


    def genModel(self):
        dt = DecisionTreeRegressor()
        rng = np.random.RandomState(1)
        #print(dt.get_params().keys())
        params_dt = [{'max_depth':np.arange(1,21),
                      'min_samples_leaf':[1, 5, 10, 20, 50, 100, 200]}]
        ab = AdaBoostRegressor(DecisionTreeRegressor(max_depth=20), n_estimators=300,random_state=rng)
        dt_gs = GridSearchCV(dt, param_grid=params_dt, cv=5)
        dt_gs.fit(self.X_train, self.y_train)
        ab.fit(self.X_train, self.y_train)
        print('adaboost: {}'.format(ab.score(self.X_test, self.y_test)))
        dt_best = dt_gs.best_estimator_
        print('dt_gs: {}'.format(dt_best.score(self.X_test, self.y_test)))
        self.model = ab
        self.modelResult = dt_best

    def getModelTestRes(self):
        print('dt: {}'.format(self.modelResult.score(self.X_test, self.y_test)))
        print('ab: {}'.format(self.model.score(self.X_test, self.y_test)))

    def getModetValRes(self):
        print('dt: {}'.format(self.modelResult.score(self.X_val, self.y_val))) 
        print('ab: {}'.format(self.model.score(self.X_val, self.y_val))) 



if __name__ == "__main__":

    MF = ModelFactory()
    MF.genDataSet()
    MF.setModel()
    MF.genModel()
    MF.getModelTestRes()
    #MF.printValidationSet()
    MF.getModetValRes()
