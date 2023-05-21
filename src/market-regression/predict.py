"""Predict script.

This script implements prediction of houshold prices using pretrained model.

Example
-------
::

    [~]$ python predict.py <data_path> <model_path> <save_path>
    [~]$ ls <save_path>
    predictions.npy

"""


import os
import numpy as np
import pandas as pd
import argparse

import marketreg

parser = argparse.ArgumentParser(description='Predict houshold prices')
parser.add_argument('data_path', type=str, help='Path of data to predict')
parser.add_argument('model_path', type=str, help='Path of trained model')
parser.add_argument('save_path', type=str, help='Path of trained model')


if __name__ == "__main__":
    args = parser.parse_args()

    model = marketreg.load_model(os.path.join(args.model_path))

    data = pd.read_csv(args.data_path, parse_dates=['timestamp'])

    data = marketreg.preprocess(data)

    preds = marketreg.predict(model, data.to_numpy()).astype(int)
    np.save(os.path.join(args.save_path, 'predictions.npy'), preds)
    