import os
import sys

sys.path.insert(0, os.path.abspath('src/market-regression'))

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

    # Test the columns
    msg = marketreg.preprocess(pd.DataFrame())

def test_predict():
    """
    Tests for various assertion cheks written in the predict function
    """
    #=================================
    # TEST SUITE
    #=================================
    # Test model type
    msg = marketreg.predict(xgb.XGBClassifier(), np.ones(shape=(1, 298)))

    # Test that model is not fitted
    msg = marketreg.predict(xgb.XGBRegressor(), np.ones(shape=(1, 298)))


    test_X = np.ones(shape=(1, 292))
    test_y = np.ones(shape=(1,)) 
    # Test data type
    msg = marketreg.predict(xgb.XGBRegressor().fit(test_X, test_y), [0, 1, 2, 3])

    # Test data length
    msg = marketreg.predict(xgb.XGBRegressor().fit(test_X, test_y), np.ones(shape=(1, 292)))

    

def test_train():
    """
    Tests for various assertion cheks written in the train function
    """
    #=================================
    # ASSERTATION TEST SUITE
    #=================================
    # Test model type
    msg = marketreg.train(xgb.XGBClassifier(), 
                            np.ones(shape=(1, 298)),
                            np.ones(shape=(1, )),
                            np.ones(shape=(1, 298)),
                            np.ones(shape=(1, )),)

    # Test train data type
    msg = marketreg.train(xgb.XGBRegressor(), 
                            [1],
                            np.ones(shape=(1, )),
                            np.ones(shape=(1, 298)),
                            np.ones(shape=(1, ))
                            )

    # Test train label type
    msg = marketreg.train(xgb.XGBRegressor(), 
                            np.ones(shape=(1, 298)),
                            [1],
                            np.ones(shape=(1, 298)),
                            np.ones(shape=(1, ))
                            )

    # Test validation data type
    msg = marketreg.train(xgb.XGBRegressor(), 
                            np.ones(shape=(1, 298)),
                            np.ones(shape=(1, )),
                            [1],
                            np.ones(shape=(1, ))
                            )

    # Test validation label type
    msg = marketreg.train(xgb.XGBRegressor(), 
                            np.ones(shape=(1, 298)),
                            np.ones(shape=(1, )),
                            np.ones(shape=(1, 298)),
                            [1]
                            )

    # Test validation label type
    msg = marketreg.train(xgb.XGBRegressor(), 
                            np.ones(shape=(1, 2)),
                            np.ones(shape=(1, )),
                            np.ones(shape=(1, 298)),
                            np.ones(shape=(1, )),
                            )

    # Test validation label type
    msg = marketreg.train(xgb.XGBRegressor(), 
                            np.ones(shape=(1, 298)),
                            np.ones(shape=(1, )),
                            np.ones(shape=(1, 2)),
                            np.ones(shape=(1, )),
                            )

    #=================================
    # MODEL TEST SUITE
    #=================================
    # Test R^2
    


def test_get_data():
    """
    Tests for various assertion cheks written in the get_data function
    """
    #=================================
    # TEST SUITE
    #=================================
    # Test model type
    msg = marketreg.get_data(1)