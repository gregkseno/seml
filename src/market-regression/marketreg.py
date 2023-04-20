"""Market Regression module.

This module implemets functions to build and use the regression model to predict houshold prices.

Example
-------
::

    import marketreg
    import pandas

    # Load train data
    X, X_val, y, y_val = marketreg.get_data(data_path)

    # Build and train model
    model = marketreg.build_model()
    model, _ = marketreg.train(model, X, y, X_val, y_val)

    # Load and preprocess test data
    data = pd.read_csv(data_path, parse_dates=['timestamp'])
    ids = data["id"]
    data = marketreg.preprocess(data)

    # Get predictions
    preds = marketreg.predict(model, data.to_numpy())

    
More information about the functions are shown below.
    
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split


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
    assert type(data) == pd.DataFrame, "Raw data type must be pd.DataFrame"
    assert list(data.columns) == open('src/market-regression/columns.txt').readline().split(sep=',') or \
    list(data.columns) == open('src/market-regression/columns.txt').readline().split(sep=',')[:-1], "Wrong columns"
    
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

def get_data(data_path: str) -> tuple:
    """Trains XGBosst Regressor model

    This method imports dataset and makes basic and specific preprocessing of data
    
    Parameters
    ----------
    data_path : str
        Path to dataset

    Returns
    -------
    np.ndarray
        Training features
    np.ndarray
        Validation features
    np.ndarray
        Training targets
    np.ndarray
        Validation targets
    """
    assert type(data_path) == str, "Data path must be string"

    
    train_df = pd.read_csv(data_path, parse_dates=['timestamp'])
    # Fix max_floor
    for index, row in train_df.iterrows():    
        if row['floor'] > row['max_floor']:
            train_df.loc[index, 'max_floor'] = row['floor']
    print(train_df['price_doc'].isna())

    # Fix full_sq    
    for index, row in train_df.iterrows():    
        if row['life_sq'] > row['full_sq']:
            train_df.loc[index, 'full_sq'] = row['life_sq']
    print(train_df['price_doc'].isna())

    # Fix odd build_year
    train_df.loc[15223, 'build_year'] = 2007 
    train_df.loc[10092, 'build_year'] = 2007
    print(train_df['price_doc'].isna())
    # Fix NaN build_year
    train_df.loc[13120, 'build_year'] = 1970
    print(train_df['price_doc'].isna())
    # Fix kitch_sq
    for index, row in train_df.iterrows():    
        if row['kitch_sq'] > row['full_sq']:
            train_df.loc[index, 'kitch_sq'] = row['full_sq'] - row['life_sq']
    print(train_df['price_doc'].isna())
    # Fix NaN kitch_sq and life_sq
    for index, row in train_df.iterrows():
        if np.isnan(row['full_sq']):
            continue
        if np.isnan(row['kitch_sq']):
            if np.isnan(row['life_sq']):
                train_df.loc[index, 'life_sq'] = row['full_sq'] * 0.8
                train_df.loc[index, 'kitch_sq'] = row['full_sq'] * 0.2
            else:
                train_df.loc[index, 'kitch_sq'] = row['full_sq'] - row['life_sq']
    print(train_df['price_doc'].isna())
    train_df = preprocess(train_df)
    
    X = train_df.drop(["price_doc"], axis=1).to_numpy()
    y = train_df['price_doc'].to_numpy()

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

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
    assert type(model) == xgb.sklearn.XGBRegressor, "Model type must be xgb.sklearn.XGBRegressor"

    assert type(X) == np.ndarray, "Train data type must be np.ndarray"
    assert type(y) == np.ndarray, "Train label type must be np.ndarray"
    assert type(X_val) == np.ndarray, "Validation data type must be np.ndarray"
    assert type(y_val) == np.ndarray, "Validation label type must be np.ndarray"
    assert X.shape[1:] == (295, ), "Train data wrong shape"
    assert X_val.shape[1:] == (295, ), "Validation data wrong shape"
    
    model.fit(X, y,
        eval_set=[(X, y) ,(X_val, y_val)],
        verbose=False,
        )
    train_metric = model.evals_result()['validation_0']['rmse']
    return model, train_metric

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
    assert type(model) == xgb.sklearn.XGBRegressor, "Model type must be xgb.sklearn.XGBRegressor"
    check_is_fitted(model)
    assert type(X) == np.ndarray, "Input data type must be np.ndarray"
    assert X.shape[1:] == (295, ), "Wrong features length"
    
    return model.predict(X)