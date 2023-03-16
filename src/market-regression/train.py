import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
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
    return model.fit(X, y,
              eval_set=[(X, y) ,(X_val, y_val)],
              )
    
def get_data(data_path: str) -> tuple:
    """Trains XGBosst Regressor model

    This method imports dataset and makes basic preprocessing of data
    
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
    train_df = pd.read_csv(data_path, parse_dates=['timestamp'])
    # Fix max_floor
    for index, row in train_df.iterrows():    
        if row['floor'] > row['max_floor']:
            train_df.loc[index, 'max_floor'] = row['floor']

    # Fix full_sq    
    for index, row in train_df.iterrows():    
        if row['life_sq'] > row['full_sq']:
            train_df.loc[index, 'full_sq'] = row['life_sq']

    # Fix odd build_year
    train_df.loc[15223, 'build_year'] = 2007 
    train_df.loc[10092, 'build_year'] = 2007

    # Fix NaN build_year
    train_df.loc[13120, 'build_year'] = 1970

    # Fix kitch_sq
    for index, row in train_df.iterrows():    
        if row['kitch_sq'] > row['full_sq']:
            train_df.loc[index, 'kitch_sq'] = row['full_sq'] - row['life_sq']

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
    
    # Add new date features and drop timestamp
    train_df["yearmonth"] = train_df["timestamp"].dt.year*100 + train_df["timestamp"].dt.month
    train_df["yearweek"] = train_df["timestamp"].dt.year*100 + train_df["timestamp"].dt.weekofyear
    train_df["year"] = train_df["timestamp"].dt.year
    train_df["month_of_year"] = train_df["timestamp"].dt.month
    train_df["week_of_year"] = train_df["timestamp"].dt.weekofyear
    train_df["day_of_week"] = train_df["timestamp"].dt.weekday

    train_df.drop("timestamp", axis=1 , inplace=True)

    # Replace categorical values with numerical
    for f in train_df.columns:
        if train_df[f].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values)) 
            train_df[f] = lbl.transform(list(train_df[f].values))
    
    X = train_df.drop(["id", "price_doc"], axis=1).to_numpy()
    y = train_df['price_doc'].to_numpy()

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val



if __name__ == "__main__":
    if len(sys.argv) <= 3:
        data_path, save_path = sys.argv[1:]
        X, X_val, y, y_val = get_data(data_path)
        model = marketreg.build_model()
        model = train(model, X, y, X_val, y_val)
        model.save_model(os.path.join(save_path, 'model.json'))
