from instools import utils as ut
import pytest
import numpy as np
import pandas as pd

ins = pd.read_csv('data/insurance.csv')

def test_data():
    assert len(ins) == 