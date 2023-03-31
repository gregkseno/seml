import os
import sys
import marketreg


if __name__ == "__main__":
    if len(sys.argv) <= 3:
        data_path, save_path = sys.argv[1:]
        X, X_val, y, y_val = marketreg.get_data(data_path)
        model = marketreg.build_model()
        model = marketreg.train(model, X, y, X_val, y_val)
        model.save_model(os.path.join(save_path, 'model.json'))
