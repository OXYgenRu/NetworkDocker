import multiprocessing
import traceback

import torch
from flask import Flask, jsonify, request, g
from usecase import Runner

app = Flask(__name__)
runner = Runner()


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


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    app.run(host='127.0.0.1', port=8080)
