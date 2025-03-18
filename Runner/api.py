import multiprocessing
import traceback

import torch
from flask import Flask, jsonify, request, g
from usecase import Runner

app = Flask(__name__)
runner = Runner()


@app.route("/runner/containers/<int:container_id>/load", methods=["POST"])
def load_container(container_id):
    data = request.json

    if not data:
        return jsonify({"error": "Empty request body"}), 400
    try:
        runner.load_container(container_id, data.get("local_path"))
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 400
    return jsonify({"message": "ok"}), 201


@app.route("/runner/containers/<int:container_id>/shutdown", methods=["POST"])
def shutdown_container(container_id):
    try:
        runner.shutdown_container(container_id)
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 400
    return jsonify({"message": "ok"}), 201


@app.route("/runner/statuses", methods=["GET"])
def read_statuses():
    try:
        containers = runner.read_statuses()
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 400
    return jsonify({"message": containers}), 201


@app.route("/runner/statuses/<int:container_id>", methods=["PUT"])
def update_statuses(container_id):
    data = request.json

    if not data:
        return jsonify({"error": "Empty request body"}), 400
    try:
        container = runner.update_statuses(container_id, data.get("status"))
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 400
    return jsonify({"message": container}), 201


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


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    app.run(host='127.0.0.1', port=8080)
