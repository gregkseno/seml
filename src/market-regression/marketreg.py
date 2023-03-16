import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing


def build_model() -> xgb.sklearn.XGBRegressor:
    """Builds XGBosst model

    This function implements XGBoost Regressor model with prediscribed parameters from sklearn library 

    Returns
    -------
    xgb.sklearn.XGBRegressor
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
    return xgb.XGBRegressor(**params)

def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses input data

    This method adds new data features, replaces categorical values with numerical 

    Parameters
    ----------
    data : pd.DataFrame
        Raw data

    Returns
    -------
    pd.DataFrame
        Preprocessed data
    """
    # Add new date features and drop timestamp
    data["yearmonth"] = data["timestamp"].dt.year*100 + data["timestamp"].dt.month
    data["yearweek"] = data["timestamp"].dt.year*100 + data["timestamp"].dt.weekofyear
    data["year"] = data["timestamp"].dt.year
    data["month_of_year"] = data["timestamp"].dt.month
    data["week_of_year"] = data["timestamp"].dt.weekofyear
    data["day_of_week"] = data["timestamp"].dt.weekday
    data = data.drop(["id", "timestamp"], axis=1)

    # Replace categorical values with numerical
    for f in data.columns:
        if data[f].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(data[f].values)) 
            data[f] = lbl.transform(list(data[f].values))

    return data

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

