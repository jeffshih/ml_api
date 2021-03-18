from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import metrics
import pandas as pd
import numpy as np
from joblib import dump
from config import *


class ModelFactory(object):

    def __init__(self):
               
        self.model = None


        self.dataset = pd.read_csv("./heart_failure_clinical_records_dataset.csv")
        self.X_ori = self.dataset.drop(columns=['DEATH_EVENT'])[selectedFeatures]
        self.y = self.dataset['DEATH_EVENT']
        col_names = list(self.X_ori.columns)
        self.stdScaler = preprocessing.StandardScaler()
        self.stdScaler.fit(self.X_ori)
        self.X = self.stdScaler.transform(self.X_ori)
        self.X = pd.DataFrame(self.X, columns=col_names)
        self.X_val = None
        self.y_val = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.t_test = None
                


    def getModel(self):
        self.__genDecisionTree()
        self.__genBoostTree()
        self.__genLR()
        self.__genSVM()
        self.__genKNN()
        self.__genRF()
        #estimators=[('KNN', self.KNN), ('SVC', self.SVM)]
        self.model = StackingClassifier(estimators=[('LR', self.LR), ('KNN', self.KNN)],final_estimator=self.SVM)
        self.model.fit(self.X_train, self.y_train)
        #self.model = make_pipeline(self.stdScaler, self.vote)
        path = "./temp/model.joblib"
        modelDump = open(path, "wb")
        dump(self.model, modelDump)
        modelDump.close()

    def printValidationSet(self):
        print(self.X_val)
        print(self.y_val)
    
    def genDataSet(self, train=0.8, test=0.1, val=0.1):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=1-train,random_state=2)#, stratify=self.y)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_test, self.y_test, test_size=test/(test + val),random_state=2)#stratify=self.y_test)


    def __genBoostTree(self):
        rng = np.random.RandomState(42)
        self.BT = AdaBoostClassifier(DecisionTreeClassifier(max_depth=None), n_estimators=10, random_state=rng)
        #self.BT.fit(self.X_train, self.y_train)

    def __genDecisionTree(self):
        self.DT = DecisionTreeClassifier()
        #self.DT.fit(self.X_train, self.y_train)
        
    def __genKNN(self):
        self.KNN = KNeighborsClassifier(n_neighbors=18)
        #self.KNN.fit(self.X_train, self.y_train)

    def __genRF(self):
        self.RF = RandomForestClassifier(n_estimators=10)
        #self.RF.fit(self.X_train, self.y_train)

    def __genGNB(self):
        self.GNB = GaussianNB()
        #self.GNB.fit(self.X_train, self.y_train)

    def __genLR(self):
        self.LR = LogisticRegression(solver='liblinear', max_iter=1000, penalty='l1', C=0.01)
       # self.LR.fit(self.X_train, self.y_train)
        

    def __genSVM(self):
        self.SVM = SVC(kernel='linear',C=1e2,gamma=1e-04, probability=True)
        #self.SVM.fit(self.X_train, self.y_train)
        


    def getModelTestRes(self):
        y_pred = self.model.predict(self.X_test)
        acc = "Accuracy:",metrics.accuracy_score(self.y_test, y_pred)
        return('Test score: {}'.format( acc))

    def getModetValRes(self):
        y_pred = self.model.predict(self.X_val)
        acc = "Accuracy:",metrics.accuracy_score(self.y_val, y_pred)
        return('Validation score: {}'.format(acc)) 

    def predict(self, feature):
        input = feature[selectedFeatures]
        return self.model.predict(input)


if __name__ == "__main__":

    MF = ModelFactory()
    MF.genDataSet()
    MF.getModel()
    print(MF.getModelTestRes())
    print(MF.getModetValRes())

    X = np.array([[63,1,103,1,35,0,179000,0.9,136,1,1,270]])
    print(MF.predict(X))
    
