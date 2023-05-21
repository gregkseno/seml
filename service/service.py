from flask import Flask, request, jsonify
from datetime import datetime
import redis
import marketreg
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

def state_function(func):
    class custom_str(str):
        def __call__(self):
            return func.__name__, str(datetime.now())

    return custom_str(func.__name__)

class ModelModel():
    """
    Models are stored as hash objects in Redis KV

    # states
    created

    # attributes
    model_id: str
    date: str
    ckpt_path: str

    """
    @staticmethod
    def model_hash(model_id):
        return f"/models/{model_id}"

    @staticmethod
    def model_key():
        string = datetime.now().isoformat()
        return hashlib.shake_128(str.encode(string)).hexdigest(4)

    @staticmethod
    @state_function
    def created():
        ...

    @staticmethod
    def models():
        return '/models'

    @staticmethod
    def to_db(**kwargs):
        return dict(
            model_id=encode_none(kwargs["model_id"]),
            ckpt_path=encode_none(kwargs["ckpt_path"]),
        )

    @staticmethod
    def from_db(**kwargs):
        return dict(
            model_id=decode_none(kwargs["model_id"]),
            ckpt_path=decode_none(kwargs["ckpt_path"]),
            created=decode_none(kwargs["created"]),
        )


class AddModel():
    ckpt_path: str


class GetModelOut():
    model_id: str
    ckpt_path: str
    created: str


@app.route("/models", methods=["POST"])
def add_model():
    model = AddModel(**request.json)
    r = redis.Redis()
    model_key = ModelModel.model_key()
    model_hash = ModelModel.model_hash(model_key)

    with r.pipeline() as pipe:
        try:
            pipe.watch(model_hash)

            res = pipe.hsetnx(model_hash, *ModelModel.created())
            if res == 1:
                pipe.hset(model_hash, mapping=ModelModel.to_db(
                    model_id=model_key,
                    ckpt_path=model.ckpt_path
                ))

                pipe.sadd(ModelModel.models(), model_key)

                response = jsonify()
                response.status_code = 201
                response.headers['Location'] = model_hash
                # run the training
                os.system(
                    f'python ames_model/train.py --save-folder={model.ckpt_path}')
                return response

            response = jsonify()
            response.status_code = 409
            return response

        except redis.WatchError:
            response = jsonify()
            response.status_code = 409
            return response


@app.route("/models/<model_id>")
def get_model(model_id):
    r = redis.Redis()
    res = r.hgetall(f'/models/{model_id}')
    if len(res) > 0:
        res = ModelModel.from_db(**decode_none(res))
        response = jsonify(GetModelOut(**res))
        response.status_code = 200
        return response

    response = jsonify()
    response.status_code = 404
    return response


@app.route("/models/<model_id>/predict")
def predict_model(model_id):
    r = redis.Redis()
    res = r.hgetall(f'/models/{model_id}')
    if len(res) > 0:
        res = ModelModel.from_db(**decode_none(res))
        ckpt_path = res['ckpt_path']
        # run the prediction
        os.system(f'python src/market-regression/predict.py --model-folder={ckpt_path} --save-path={ckpt_path}')
        response = jsonify(GetModelOut(**res))
        response.status_code = 200
        return response

    response = jsonify()
    response.status_code = 404
    return response