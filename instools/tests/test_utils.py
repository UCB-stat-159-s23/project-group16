from instools import utils as ut
import pytest
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

ins = pd.read_csv('data/insurance.csv')

def test_label_encoder():
    df = ut.label_encoder(ins, ['region'])
    
    assert len(df) == 1338
    assert df['region'].dtype != str

def test_model_accuracy():
    