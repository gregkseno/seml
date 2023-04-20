"""Predict script.

This script implements prediction of houshold prices using pretrained model.

Example
-------
::

    [~]$ python predict.py <data_location> <model_location>
    +----------------+-------------------+
    | Transaction ID | Approximate Price |
    +----------------+-------------------+
    |     30474      |      5709956      |
    |     30475      |      8662689      |
    |     30476      |      5963731      |
    |     30477      |      6130314      |
    |     30478      |      5387550      |
    |     30479      |      9045907      |
    |     38135      |      9206236      |
    +----------------+-------------------+


Notes
-----
    The predicted price could be saved in file using ``<command> > result.txt``.

"""


import os
import sys
import pandas as pd
from xgboost import XGBRegressor
from prettytable import PrettyTable

import marketreg

if __name__ == "__main__":
    if len(sys.argv) <= 3:
        data_path, model_path = sys.argv[1:]

        model = XGBRegressor()
        model.load_model(os.path.join(model_path))

        data = pd.read_csv(data_path, parse_dates=['timestamp'])
        ids = data["id"]

        data = marketreg.preprocess(data)

        preds = marketreg.predict(model, data.to_numpy()).astype(int)
        
        pt = PrettyTable()
        pt.field_names = ["Transaction ID", "Approximate Price"]
        pt.add_rows(zip(ids, preds))
        print(pt)
        