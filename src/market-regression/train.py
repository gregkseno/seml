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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model to predict houshold prices')
    parser.add_argument('data_path', type=str, help='Path of train data')
    parser.add_argument('save_path', type=str, help='Path to save trained model')
    args = parser.parse_args()
    X, X_val, y, y_val = marketreg.get_data(args.data_path)
    model = marketreg.build_model()
    model, _ = marketreg.train(model, X, y, X_val, y_val)
    marketreg.save_model(model, os.path.join(args.save_path, 'model.json'))
