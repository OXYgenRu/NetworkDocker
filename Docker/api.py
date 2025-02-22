import datetime
from pprint import pprint

from flask import Flask, jsonify, request, g
from usecase import UseCase, MODEL_CREATED
import threading

from database import DB, Container
from usecase import UseCase

app = Flask(__name__)
db = DB()
use_case = UseCase(db)


@app.route('/models', methods=["POST"])
def create_model():
    data = request.json

    if not data:
        return jsonify({"error": "Empty request body"}), 400

    model = use_case.create_model(data["sequential"], data["code"])
    if model is None:
        return jsonify({"error": "incorrect data transmitted"}), 400
    return jsonify({"message": f"{MODEL_CREATED}", "model": model.to_dict()}), 201


@app.teardown_appcontext
def close_db(error):
    db_conn = g.pop("db_conn", None)
    if db_conn is not None:
        db_conn.close()


if __name__ == '__main__':
    app.run()
