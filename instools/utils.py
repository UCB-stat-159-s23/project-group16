import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def model_accuracy(model,df,predictors, prediction):
    """takes in the classification model, dataframe, predictors, prediction to return the testing and training accuracy
    Inputs:
    model- classification model: either KNeighborsClassifier(), SVC(), DecisionTreeClassifier(), or RandomForestClassifier() with extra variables.
    df- the data frame that the predictors and prediction will take from.
    predictors- list of columns from df that will be used to classify y, must be categorical features.
    prediction- ONE column name from df, must be categorical features.
    Output:
    printed training and testing set accuracy
    """
    y = df[prediction]
    X = df[predictors]
    xtrain, xtest, ytrain, ytest = train_test_split(X,y,random_state = 4, test_size = 0.2)
    model.fit(xtrain,ytrain)
    ypred = model.predict(xtest)
    train_accuracy = model.score(xtrain,ytrain)
    test_accuracy = model.score(xtest, ytest)
    print(str(model)[:-2], 'Accuracy estimating', prediction)
    print(f'Training Accuracy: {train_accuracy}\nTesting Accuracy: {test_accuracy}')
    return model