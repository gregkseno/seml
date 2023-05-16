from flask import Flask
import redis
import marketreg

app = Flask(__name__)


@app.route('/models')
def models():
    return 'Models section!'

@app.route('/predict')
def predict():
    return 'Predict section!'

@app.route('/dataset')
def dataset():
    return 'Dataset section!'
