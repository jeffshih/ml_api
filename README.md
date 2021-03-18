# ML_API

## Run server
### Using docker
```shell
$ docker built -t ml_api ./
$ docker run -d --name api -p 8000:80 ml_api:latest
```

### Using virtualenv
```shell
$ python3 -m venv mlapi
$ source mlapi/bin/activate
$ pip install -r requirements.txt
$ uvicorn api:app
```

## Usage
### Train model
```shell
#Train the model without saving
curl -X GET "http://127.0.0.1:8000/train" 

#Train the modle and save to disk
curl -X GET "http://0.0.0.0:8000/train?save=true" --output model.joblib

#Save the model after training
curl -X GET "http://0.0.0.0:8000/getModel" --output model.joblib
```

### Interaction
```shell
#Get the current score of model on testing set and validation set
curl -X GET "http://0.0.0.0:8000/getModelScore" 


#Do prediction with current model with sample json file
curl -X POST "http://0.0.0.0:8000/predict" -H "accept: application/json" -H "Content-Type: application/json" -d @predict.json 

#Perform basic testing
python tester.py

#Perform loading models and test the inpput

python tester.py -p true -m {path/to/model} -f {path/to/predict.json}

```



#### Sample json file

Feature must be a dict with a 13 features str:float, with basic input checking.
```json
{
"features":
    {
        "age":75,"anaemia":0,"creatinine_phosphokinase":582,"diabetes":0,"ejection_fraction":20,"high_blood_pressure":1,"platelets":265000,"serum_creatinine":1.9,"serum_sodium":130,"sex":1,"smoking":0,"time":4,"DEATH_EVENT":1
    }
}
```

### Model design

Looking through the data (In Visualization and Modeling notebook), this is a relative small dataset.

With some mixed binary/numerical data, the numerical data has obvious outlier.

By visualize the Pearson correlation coefficient, we can clear see some data, especially the binary columns like gender has very little correlation with the mortality.

Due to this small dataset, I choose to do some feature engineering, neglect some data and use relativly simple model to reduce the variance, and use stacking method to ensemble the model.

Logistic Regression and SVM perform really well on this scale of data, decision tree related are easily overfit, and I choose KNN because by looking through this dataset, it's pretty balanced so I think KNN might be a good one algorithm. 



### Reference

Ensemble
https://towardsdatascience.com/ensemble-learning-using-scikit-learn-85c4531ff86a

GridSearch for optimization and pipeline for basic feature engineering
https://scikit-learn.org/stable/modules/grid_search.html#composite-grid-search

