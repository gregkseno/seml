import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
import os
import sys

import marketreg


def train(model: xgb.sklearn.XGBRegressor,
          X: np.ndarray,
          y: np.ndarray,
          X_val: np.ndarray,
          y_val: np.ndarray,
        ) -> xgb.sklearn.XGBRegressor:
    """Trains XGBosst Regressor model

    This method implements training of XGBoost Regressor model with prediscribed parameters

    Parameters
    ----------
    model : xgb.sklearn.XGBRegressor
        Initialized XGBoost Regressor model
    X : np.ndarray
        Array of training features
    y : np.ndarray
        Array of training targets
    X_val : np.ndarray
        Array of validation features
    y_val : np.ndarray
        Array of validation targets

    Returns
    -------
    xgb.sklearn.XGBRegressor
        Trained XGBoost Regressor model
    """
    model.fit(X, y,
              eval_set=[(X, y) ,(X_val, y_val)],
              )


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) <= 1:
        save_path = os.path.join(args[0], 'regressor.json')
        X, y = get_data()
        model = marketreg.build_model()
        model = train(model, X, y)
        model.save_model(save_path)
