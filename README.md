#ML_API

##Run server
###Using docker
```shell
$ docker built -t ml_api ./
$ docker run -d --name api -p 8000:80 ml_api:latest
```

###Using virtualenv
```shell
$ python3 -m venv mlapi
$ source mlapi/bin/activate
$ pip install -r requirements.txt
$ uvicorn api:app
```

##Usage
###Train model
Currently support 3 kind of models, DecisionTree, LogisticRegression and SVM
save controls to save the model to local disk or not
split is for splitting the dataset, each represent the ratio of training, testing, validation
```json
#trainrequest.json

{
    "model":"LogisticRegression",  
    "save":false,
    "split":[0.6, 0.2, 0.2]
}
```
```shell
#Train the model without saving
curl -X POST "http://0.0.0.0:8000/train" -H "accept: application/json" -H "Content-Type: application/json" -d @trainrequest.json

#Train the modle and save to disk
curl -X POST "http://0.0.0.0:8000/train" -H "accept: application/json" -H "Content-Type: application/json" -d @trainrequest.json --output LR.pkl

#Save the model after training
curl -X GET "http://0.0.0.0:8000/train/LogisticRegression" --output LR.pkl 
```

###Interaction
```shell
#Get the current score of model on testing set and validation set
curl -X GET "http://0.0.0.0:8000/getModelScore" 


#Do prediction with current model with sample json file
curl -X POST "http://0.0.0.0:8000/predict" -H "accept: application/json" -H "Content-Type: application/json" -d @feature.json 
```

####Sample json file

Feature must be a string key follow with a list of 12 number, with basic input checking.
```json
{
    "features":{
        "1":[50,0,115,0,45,1,184000,0.9,134,1,1,118],
        "2":[45,1,981,0,30,0,136000,1.1,137,1,0,11]
        }
}
```
