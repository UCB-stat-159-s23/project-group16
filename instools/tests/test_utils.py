from instools import utils as ut
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

ins = pd.read_csv('data/insurance.csv')

ins_mod = ins.copy()
obj_col = ins_mod.select_dtypes('object').columns
oe = OrdinalEncoder()
ins_mod[obj_col] = ins_mod[obj_col].astype(str)
ins_mod[obj_col] = oe.fit_transform(ins_mod[obj_col])

x= ['age', 'bmi', 'children','smoker']
y= ins['sex']

def test_label_encoder():
    df = ut.label_encoder(ins, ['region'])
    
    assert len(df) == 1338
    assert df['region'].dtype != str

def test_model_accuracy():
    knn= ut.model_accuracy(KNeighborsClassifier(), ins_mod,x, 'sex')
    
    assert type(knn) != [float, int, object]
    assert type(x) is list
    assert type(y) != float