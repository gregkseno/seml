import marketreg
import numpy as np
from sklearn.exceptions import NotFittedError
import xgboost as xgb
import pandas as pd

def test_preprocessing():
    """
    Tests for various assertion cheks written in the preprocessing function
    """
    #=================================
    # TEST SUITE
    #=================================
    # Test type of raw data
    msg = marketreg.preprocess([0, 1, 2, 3])
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "Raw data type must be pd.DataFrame"

    # Test the length of features
    msg = marketreg.preprocess(pd.DataFrame())
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "Wrong features length"

    # Test that raw data has 'timestamp' column
    test_columns = ['timestamp'] + list(range(297))
    msg = marketreg.preprocess(pd.DataFrame(columns=test_columns))
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "Raw data doesn't consist timestamp"
    

def test_predict():
    """
    Tests for various assertion cheks written in the predict function
    """
    #=================================
    # TEST SUITE
    #=================================
    # Test model type
    msg = marketreg.predict(xgb.XGBClassifier())
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "Model type must be xgb.sklearn.XGBRegressor"

    # Test that model is not fitted
    msg = marketreg.predict(xgb.XGBRegressor())
    assert isinstance(msg, NotFittedError)

    # Test data type
    msg = marketreg.preprocess([0, 1, 2, 3])
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "Input data type must be np.ndarray"

    # Test data length
    msg = marketreg.preprocess(np.ones(shape=(1, 298)))
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "Input data type must be np.ndarray"
