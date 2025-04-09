import multiprocessing
import traceback

import torch
from flask import Flask, jsonify, request, g

from Runner.env.config import Config
from Runner.env.env_builder import EnvBuilder
from usecase import Runner

app = Flask(__name__)
config = Config()
config.parse("config.json")
env_builder = EnvBuilder(config)
runner = Runner(config, env_builder)


@app.route("/runner/containers/<int:container_id>/execute", methods=["POST"])
def execute_container(container_id):
    data = request.json

    if not data:
        return jsonify({"error": "Empty request body"}), 400
    try:
        runner.execute_container(container_id, data.get("epochs"), data.get("reset_progress"))
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 400
    return jsonify({"message": "ok"}), 201


@app.route("/runner/containers/<int:container_id>/test", methods=["POST"])
def test_container(container_id):
    try:
        avg_loss = runner.test_container(container_id)
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 400
    return jsonify({"message": "ok", "avg_loss": avg_loss}), 201


@app.route("/runner/containers/<int:container_id>/predict", methods=["POST"])
def predict_container(container_id):
    data = request.json

    if not data:
        return jsonify({"error": "Empty request body"}), 400
    try:
        result = runner.predict(container_id, data.get("tensor"))
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 400
    return jsonify({"message": "ok", "prediction": result}), 201


@app.route("/runner/processes/<int:container_id>", methods=["DELETE"])
def delete_process(container_id):
    try:
        runner.delete_process(container_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"message": "process_deleted"}), 201


@app.route("/runner/processes/<int:container_id>", methods=["GET"])
def read_process(container_id):
    try:
        status = runner.read_process_status(container_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"message": "ok", "is_alive": status}), 201


@app.route("/runner/processes/<int:container_id>/stop", methods=["POST"])
def stop_process(container_id):
    try:
        runner.stop_process(container_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"message": "signal to stop process sent"}), 201


def start():
    app.run(host=config.runner_host, port=config.runner_port)
