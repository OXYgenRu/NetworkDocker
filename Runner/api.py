import traceback

from flask import Flask, jsonify, request, g
from usecase import Runner

app = Flask(__name__)
runner = Runner()


@app.route("/runner/containers/<int:container_id>/load")
def load_container(container_id):
    data = request.json

    if not data:
        return jsonify({"error": "Empty request body"}), 400
    try:
        runner.load_container(container_id, data.get("local_path"))
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 400
    return jsonify({"message": "ok"}), 201


@app.route("/runner/containers/<int:container_id>/shutdown")
def shutdown_container(container_id):
    try:
        runner.shutdown_container(container_id)
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 400
    return jsonify({"message": "ok"}), 201


@app.route("/runner/containers")
def read_containers():
    try:
        containers = runner.read_containers()
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 400
    return jsonify({"message": containers}), 201


app.run(host='127.0.0.1', port=8080)
