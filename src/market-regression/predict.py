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
        