import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn import preprocessing

def build_model() -> XGBRegressor:
    """Builds XGBosst model

    This method implements XGBoost Regressor model with prediscrbed parameters from XGBoost library 

    Returns
    -------
    xgb.XGBRegressor
        Model with prediscribed parameters
    """
    params = {
        "objective": "reg:squarederror",
        "n_estimators":1000,
        "max_depth": 8,
        'eta': 0.01,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_lambda": 0.3,
        "random_state": 42,
        "early_stopping_rounds": 20,
    }
    return XGBRegressor(**params)

def predict(model: xgb.sklearn.XGBRegressor, X: np.ndarray) -> np.ndarray:
    """Predicts values using trained XGBoost Regressor model

    This method implements XGBoost Regressor model with prediscribed parameters from XGBoost library 

    Parameters
    ----------
    model : xgb.sklearn.XGBRegressor
        Trained XGBosst Regressor model
    X : np.ndarray
        Array of features to be used input data

    Returns
    -------
    np.ndarray
        Predicted values
    """
    return model.predict(X)

