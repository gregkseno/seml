import os
import sys
import pandas as pd
from xgboost import XGBRegressor
from sklearn import preprocessing
from prettytable import PrettyTable

import marketreg

if __name__ == "__main__":
    if len(sys.argv) <= 3:
        data_path, model_path = sys.argv[1:]

        model = XGBRegressor()
        model.load_model(os.path.join(model_path))

        data = pd.read_csv(data_path, parse_dates=['timestamp'])
        ids = data["id"]

        # Add new date features and drop timestamp
        data["yearmonth"] = data["timestamp"].dt.year*100 + data["timestamp"].dt.month
        data["yearweek"] = data["timestamp"].dt.year*100 + data["timestamp"].dt.weekofyear
        data["year"] = data["timestamp"].dt.year
        data["month_of_year"] = data["timestamp"].dt.month
        data["week_of_year"] = data["timestamp"].dt.weekofyear
        data["day_of_week"] = data["timestamp"].dt.weekday
        data.drop(["id", "timestamp"], axis=1 , inplace=True)

        # Replace categorical values with numerical
        for f in data.columns:
            if data[f].dtype=='object':
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(data[f].values)) 
                data[f] = lbl.transform(list(data[f].values))

        preds = marketreg.predict(model, data.to_numpy()).astype(int)
        
        pt = PrettyTable()
        pt.field_names = ["Transaction ID", "Approximate Price"]
        pt.add_rows(zip(ids, preds))
        print(pt)
        