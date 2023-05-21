from flask import Flask, request, jsonify, Response
from datetime import datetime
import numpy as np
import redis
import hashlib
import os

app = Flask(__name__)

def decode_none(s):
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
    if s is None:
        return None
    if isinstance(s, dict):
        return {encode_none(k): decode_none(v) for k, v in s.items()}
    if isinstance(s, list):
        return [encode_none(k) for k in s]

    return bytes(str(s), encoding='utf-8')


def model_hash(model_id):
    return f"/models/{model_id}"

def model_key():
    string = datetime.now().isoformat()
    return hashlib.shake_128(str.encode(string)).hexdigest(4)

def to_db(model_id, save_path):
    return dict(
        model_id=encode_none(model_id),
        save_path=encode_none(save_path),
    )

def from_db(model_id, save_path, created):
    return dict(
        model_id=decode_none(model_id),
        save_path=decode_none(save_path),
        created=decode_none(created),
    )


@app.route("/models", methods=["POST"])
def add_model():
    args = dict(request.args)
    save_path = args.pop('save_path')
    if save_path is None: return Response("Argument save_path must be given", status=409)

    r = redis.Redis()
    key = model_key()
    hashed = model_hash(key)
    with r.pipeline() as pipe:
        try:
            pipe.watch(hashed)
            res = pipe.hsetnx(hashed, 'created', str(datetime.now()))
            if res == 1:
                pipe.hset(hashed, mapping=to_db(
                    model_id=key,
                    save_path=save_path
                ))
                pipe.sadd('/models', key)
                
                params = " ".join([f'--{key} {arg}' for key, arg in args.items()])

                os.system(f'python src/market-regression/train.py {params} data/train.csv {save_path}')
                response = Response(f"Model was created and saved at /models/{hashed}", status=201, mimetype='application/json')
                response.headers['Location'] = hashed
                return response

            return Response("Field already exists in the hash and no operation was performed", status=409)

        except redis.WatchError:
            return Response("Redis is not launched", status=409)


@app.route("/models/<model_id>")
def get_model(model_id):
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
    r = redis.Redis()
    res = r.scan(_type='hash')
    res = list(filter(lambda x: x.startswith(b'/models/'), res[1]))
    if len(res) > 0:
        res = decode_none(res)
        response = jsonify(res)
        response.status_code = 200
        return response

    return Response("There are no models", status=404)

@app.route("/models/<model_id>/predict")
def predict_model(model_id):
    r = redis.Redis()
    res = r.hgetall(f'/models/{model_id}')
    if len(res) > 0:
        res = from_db(**decode_none(res))
        save_path = res['save_path']
        # run the prediction
        os.system(f'python src/market-regression/predict.py data/test.csv {save_path}/model.json {save_path}')
        response = jsonify(np.load(f"{save_path}/predictions.npy").tolist())
        response.status_code = 200
        return response

    return Response("Model not found", status=404)


app.run()