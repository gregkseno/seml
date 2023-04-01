import os
import sys
import pytest

sys.path.insert(0, os.path.abspath('src/market-regression'))

import marketreg
import numpy as np
from sklearn.exceptions import NotFittedError
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split



def test_preprocessing():
    """
    Tests for various assertion cheks written in the preprocessing function
    """
    #=================================
    # TEST SUITE
    #=================================
    # Test type of raw data
    try:
        msg = marketreg.preprocess([0, 1, 2, 3])
    except AssertionError as msg:
        assert isinstance(msg, AssertionError)
        assert msg.args[0] == "Raw data type must be pd.DataFrame"

    # Test the columns
    try:
        msg = marketreg.preprocess(pd.DataFrame())
    except AssertionError as msg:
        assert isinstance(msg, AssertionError)
        assert msg.args[0] == "Wrong columns"

def test_predict():
    """
    Tests for various assertion cheks written in the predict function
    """
    #=================================
    # TEST SUITE
    #=================================
    # Test model type
    try:
        msg = marketreg.predict(xgb.XGBClassifier(), np.ones(shape=(1, 298)))
    except AssertionError as msg:
        assert msg.args[0] == "Model type must be xgb.sklearn.XGBRegressor"

    # Test that model is not fitted
    try:
        msg = marketreg.predict(xgb.XGBRegressor(), np.ones(shape=(1, 298)))
    except NotFittedError as msg:
        assert msg.args[0] == "This XGBRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."

    test_X = np.ones(shape=(1, 292))
    test_y = np.ones(shape=(1,)) 
    # Test data type
    try:
        msg = marketreg.predict(xgb.XGBRegressor().fit(test_X, test_y), [0, 1, 2, 3])
    except AssertionError as msg:
        assert msg.args[0] == "Input data type must be np.ndarray"

    # Test data length
    try:
        msg = marketreg.predict(xgb.XGBRegressor().fit(test_X, test_y), np.ones(shape=(1, 292)))
    except AssertionError as msg:
        assert msg.args[0] == "Wrong features length"

    
@pytest.mark.parametrize('X,X_val,y,y_val', [marketreg.get_data('test/test_data.csv')])
def test_train(X, X_val, y, y_val):
    """
    Tests for various assertion cheks written in the train function
    """
    # Remove NaN labels
    ids = ~np.isnan(y)
    y = y[ids]
    X = X[ids]
    ids = ~np.isnan(y_val)
    y_val = y_val[ids]
    X_val = X_val[ids]
    #=================================
    # ASSERTATION TEST SUITE
    #=================================
    # Test model type
    try:
        msg = marketreg.train(xgb.XGBClassifier(), X, y, X_val, y_val)
    except AssertionError as msg:
        assert msg.args[0] == "Model type must be xgb.sklearn.XGBRegressor"

    # Test train data type
    try:
        msg = marketreg.train(xgb.XGBRegressor(), [1], y, X_val, y_val)
    except AssertionError as msg:
        assert msg.args[0] == "Train data type must be np.ndarray"

    # Test train label type
    try:
        msg = marketreg.train(xgb.XGBRegressor(), X, [1], X_val, y_val)
    except AssertionError as msg:
        assert msg.args[0] == "Train label type must be np.ndarray"

    # Test validation data type
    try:
        msg = marketreg.train(xgb.XGBRegressor(), X, y, [1], y_val)
    except AssertionError as msg:
        assert msg.args[0] == "Validation data type must be np.ndarray"

    # Test validation label type
    try:
        msg = marketreg.train(xgb.XGBRegressor(), X, y, X_val, [1])
    except AssertionError as msg:
        assert msg.args[0] == "Validation label type must be np.ndarray"

    # Test validation label type
    try:
        msg = marketreg.train(xgb.XGBRegressor(), X[:, :2], y, X_val, y_val)
    except AssertionError as msg:
        assert msg.args[0] == "Train data wrong shape"

    # Test validation label type
    try:
        msg = marketreg.train(xgb.XGBRegressor(), X, y, X_val[:, :2], y_val)
    except AssertionError as msg:
        assert msg.args[0] == "Validation data wrong shape"

    #=================================
    # MODEL TEST SUITE
    #=================================
    # Test decrease of train metric
    _, metrics = marketreg.train(marketreg.build_model(), X, y, X_val, y_val)
    print(metrics)
    for i, metric in enumerate(metrics[1:]):
        assert metric < metrics[i]
    


def test_get_data():
    """
    Tests for various assertion cheks written in the get_data function
    """
    #=================================
    # TEST SUITE
    #=================================
    # Test model type
    try:
        msg = marketreg.get_data(1)
    except AssertionError as msg:
        assert msg.args[0] == "Data path must be string"