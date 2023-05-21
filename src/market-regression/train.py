"""Train script.

This script implements training of regression model.

Example
-------
::

    [~]$ ls <save_path>
    [~]$ python train.py <data_location> <save_path>
    [~]$ ls <save_path>
    model.json

"""

import os
import argparse
import marketreg


parser = argparse.ArgumentParser(description='Train model to predict houshold prices')
parser.add_argument('data_path', type=str, help='Path of train data')
parser.add_argument('save_path', type=str, help='Path to save trained model')
parser.add_argument('--objective', type=str, default='reg:squarederror', help='Train objective')
parser.add_argument('--n_estimators', type=int, default=1000, help='Number of gradient boosted trees')
parser.add_argument('--max_depth', type=int, default=8, help='Maximum tree depth for base learners')
parser.add_argument('--eta', type=float, default=0.01, help='Boosting learning rate')
parser.add_argument('--subsample', type=float, default=0.7, help='Subsample ratio of the training instance')
parser.add_argument('--colsample_bytree', type=float, default=0.7, help='Subsample ratio of columns when constructing each tree')
parser.add_argument('--reg_lambda', type=float, default=0.3, help='L2 regularization term on weights')
parser.add_argument('--random_state', type=int, default=42, help='Random number seed')
parser.add_argument('--early_stopping_rounds', type=int, default=20, help='Activates early stopping')
    

if __name__ == "__main__":
    args = parser.parse_args()
    X, X_val, y, y_val = marketreg.get_data(args.data_path)
    params = vars(args)
    save_path = params.pop('save_path')
    params.pop('data_path')
    model = marketreg.build_model(params)
    model, _ = marketreg.train(model, X, y, X_val, y_val)
    marketreg.save_model(model, save_path)
