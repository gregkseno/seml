"""Market Regression service.

This module implemets RESTful API service using Flask and Redis as DB.
    
More information about the endpoints are shown below.
    
"""

from flask import Flask, request, jsonify, Response
from datetime import datetime
import numpy as np
import redis
import hashlib
import os

app = Flask(__name__)

def decode_none(s):
    """Decodes data

    This method decodes data from Redis from bytes to string 

    Parameters
    ----------
    s : dict, list, bytes, NoneType
        Data to decode

    Returns
    -------
    dict, list, str, NoneType
        Decoded data
    """
    if s is None:
        return None
    if isinstance(s, dict):
        return {decode_none(k): decode_none(v) for k, v in s.items()}
    if isinstance(s, list):
        return [decode_none(k) for k in s]
    if isinstance(s, bytes):
        return s.decode(encoding='utf-8')
    return str(s)


def encode_none(s):
    """Encodes data

    This method encodes data to Redis from string to bytes 

    Parameters
    ----------
    s : dict, list, str, NoneType
        Data to encode

    Returns
    -------
    dict, list, bytes, NoneType
        Encode data
    """
    if s is None:
        return None
    if isinstance(s, dict):
        return {encode_none(k): decode_none(v) for k, v in s.items()}
    if isinstance(s, list):
        return [encode_none(k) for k in s]

    return bytes(str(s), encoding='utf-8')

def hashed_key():
    """Generates key of the model

    Hashes datetime to generate key of the model

    Returns
    -------
    str
        Hashed key
    """
    string = datetime.now().isoformat()
    return hashlib.shake_128(str.encode(string)).hexdigest(4)

def to_db(model_id, save_path):
    """Converts data to add into DB

    Encodes data to add into DB

    Parameters
    ----------
    model_id : str
        Hashed model id
    save_path : str
        Save path of the model

    Returns
    -------
    dict
        Dict with encoded data
    """
    return dict(
        model_id=encode_none(model_id),
        save_path=encode_none(save_path),
    )

def from_db(model_id, save_path, created):
    """Converts data to get from DB

    Decodes data to get from DB

    Parameters
    ----------
    model_id : str
        Hashed model id
    save_path : str
        Save path of the model
    created : str
        Created timestamp

    Returns
    -------
    dict
        Dict with decoded data
    """
    return dict(
        model_id=decode_none(model_id),
        save_path=decode_none(save_path),
        created=decode_none(created),
    )


@app.route("/models", methods=["POST"])
def add_model():
    """POST method of /models

    Parameters
    ----------
    All arguments of train.py script

    Trains, saves and addes model to DB
    """
    args = dict(request.args)
    save_path = args.pop('save_path')
    if save_path is None: return Response("Argument save_path must be given", status=409)
    if os.path.exists(save_path): return Response("Path already exists", status=409)

    r = redis.Redis()
    hashed = hashed_key()

    with r.pipeline() as pipe:
        try:
            pipe.watch(f"/models/{hashed}")
            res = pipe.hsetnx(f"/models/{hashed}", 'created', str(datetime.now()))
            if res == 1:
                pipe.hset(f"/models/{hashed}", mapping=to_db(
                    model_id=hashed,
                    save_path=save_path
                ))
                pipe.sadd('/models', hashed)
                
                params = " ".join([f'--{key} {arg}' for key, arg in args.items()])

                os.system(f'python src/market-regression/train.py {params} data/train.csv {save_path}')
                response = Response(f"Model was created and saved at /models/{f'/models/{hashed}'}", status=201, mimetype='application/json')
                response.headers['Location'] = f"/models/{hashed}"
                return response

            return Response("Field already exists in the hash and no operation was performed", status=409)

        except redis.WatchError:
            return Response("Redis is not launched", status=409)


@app.route("/models/<model_id>")
def get_model(model_id):
    """GET method of /models/<model_id>

    Gets data of trained model by id
    """
    r = redis.Redis()
    res = r.hgetall(f'/models/{model_id}')

    if len(res) > 0:
        res = from_db(**decode_none(res))
        response = jsonify(res)
        response.status_code = 200
        return response

    return Response("Model not found", status=404)

@app.route("/models/all")
def get_all_model():
    """GET method of /models/all

    Responses with list of all trained models hashes
    """
    r = redis.Redis()
    res = r.scan(_type='hash')
    res = list(filter(lambda x: x.startswith(b'/models/'), res[1]))
    if len(res) > 0:
        res = decode_none(res)
        response = jsonify(res)
        response.status_code = 200
        return response

    return Response("There are no models", status=404)

@app.route("/predict")
def predict_model():
    """GET method of /predict

    Predicts test data by given model
    """
    model_id = request.args.get('m')
    if model_id is None: return Response("Argument m (model_id) must be given", status=409)
    r = redis.Redis()
    res = r.hgetall(f'/models/{model_id}')
    if len(res) > 0:
        res = from_db(**decode_none(res))
        save_path = res['save_path']
        # run the prediction
        os.system(f'python src/market-regression/predict.py data/test.csv {save_path} {os.path.dirname(save_path)}')
        response = jsonify(np.load(f"{os.path.dirname(save_path)}/predictions.npy").tolist())
        response.status_code = 200
        return response

    return Response("Model not found", status=404)

app.run()