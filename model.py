import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

prediction = None

Model = RandomForestClassifier()

def split_data(test_size):
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test

def get_score():
    global x_train, x_test, y_train, y_test, Model
    return Model.score(x_test, y_test)    

def run_model(test_size):
    global x_train, x_test, y_train, y_test, Model
    x_train, x_test, y_train, y_test = split_data(test_size)
    Model.fit(x_train, y_train)
    print("Model trained successfully!")
    
    score = Model.score(x_test, y_test)
    print("Model score: "+str(score))
    # return Model.score(x_test, y_test)

def predict(sepal_length, sepal_width, petal_length, petal_width):
    global Model
    prediction = Model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    return prediction
