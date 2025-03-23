import datetime
from pprint import pprint

import torch.nn
from flask import Flask, jsonify, request, g
from usecase import UseCase, MODEL_CREATED, FILE_CREATED, OPTIMIZER_CREATED, CONTAINER_CREATED, SESSION_CREATED, \
    HISTORY_CREATED, MODEL_COPIED, OPTIMIZER_COPIED, CONTAINER_COPIED, MODEL_UPDATED, OPTIMIZER_UPDATED, \
    CONTAINER_UPDATED, FILE_UPDATED, SESSION_UPDATED, CONTAINER_READ, FILE_READ, MODEL_READ, OPTIMIZER_READ, \
    HISTORY_READ, SESSION_READ, CONTAINERS_READ, MODELS_READ, OPTIMIZERS_READ, FILES_READ, HISTORIES_READ, SESSIONS_READ
import threading

from database import DB
from usecase import UseCase

app = Flask(__name__)
db = DB()
use_case = UseCase(db)


@app.route("/tables", methods=["POST"])
def create_tables():
    result = use_case.create_tables()
    if result is None:
        return jsonify({"error": "tables creating error"}), 400
    return jsonify({"message": f"tables created"})


@app.route('/models', methods=["POST"])
def create_model():
    data = request.json

    if not data:
        return jsonify({"error": "Empty request body"}), 400

    model = use_case.create_model(data.get("sequential"), data.get("code"))
    if model is None:
        return jsonify({"error": "model creating error"}), 400
    return jsonify({"message": f"{MODEL_CREATED}", "model": model.to_dict()}), 201


@app.route("/files", methods=["POST"])
def create_file():
    data = request.json

    if not data:
        return jsonify({"error": "Empty request body"}), 400

    file = use_case.create_file(data.get("file_type"), data.get("comment"))
    if file is None:
        return jsonify({"error": "file creating error"}), 400
    return jsonify({"message": f"{FILE_CREATED}", "file": file.to_dict()}), 201


@app.route("/optimizers", methods=["POST"])
def create_optimizer():
    data = request.json

    if not data:
        return jsonify({"error": "Empty request body"}), 400

    optimizer = use_case.create_optimizer(data.get("code"))
    if optimizer is None:
        return jsonify({"error": "optimizer creating error"}), 400
    return jsonify({"message": f"{OPTIMIZER_CREATED}", "optimizer": optimizer.to_dict()}), 201


@app.route("/containers", methods=["POST"])
def create_container():
    data = request.json

    if not data:
        return jsonify({"error": "Empty request body"}), 400
    container = use_case.create_container(data.get("dataset_id"), data.get("model_id"), data.get("optimizer_id"),
                                          data.get("normalise_dataset"), data.get("criterion_code"),
                                          data.get("online_training"), data.get("name"), data.get("comment"))
    if container is None:
        return jsonify({"error": "container creating error"}), 400
    return jsonify({"message": f"{CONTAINER_CREATED}", "container": container.to_dict()}), 201


@app.route("/histories", methods=["POST"])
def create_history():
    data = request.json

    if not data:
        return jsonify({"error": "Empty request body"}), 400

    history = use_case.create_history(data.get("container_id"), data.get("model_id"), data.get("optimizer_id"),
                                      data.get("history_type"), data.get("comment"), data.get("file_id"))

    if history is None:
        return jsonify({"error": "history creating error"}), 400
    return jsonify({"message": f"{HISTORY_CREATED}", "history": history.to_dict()}), 201


@app.route("/sessions", methods=["POST"])
def create_session():
    data = request.json

    if not data:
        return jsonify({"error": "Empty request body"}), 400
    session = use_case.create_session(data.get("container_id"), data.get("status"), data.get("file_id"),
                                      data.get("epochs"),
                                      data.get("reset_progress"))
    if session is None:
        return jsonify({"error": "session creating error"}), 400
    return jsonify({"message": f"{SESSION_CREATED}", "session": session.to_dict()}), 201


@app.route("/models/<int:model_id>/clone", methods=["POST"])
def clone_model(model_id):
    data = request.json

    if not data:
        return jsonify({"error": "Empty request body"}), 400
    model = use_case.copy_model(model_id, data.get("copy_progress"))
    if model is None:
        return jsonify({"error": "model copying error"}), 400
    return jsonify({"message": f"{MODEL_COPIED}", "session": model.to_dict()}), 201


@app.route("/optimizers/<int:optimizer_id>/clone", methods=["POST"])
def clone_optimizer(optimizer_id):
    data = request.json

    if not data:
        return jsonify({"error": "Empty request body"}), 400
    optimizer = use_case.copy_optimizer(optimizer_id, data.get("copy_progress"))
    if optimizer is None:
        return jsonify({"error": "optimizer copying error"}), 400
    return jsonify({"message": f"{OPTIMIZER_COPIED}", "session": optimizer.to_dict()}), 201


@app.route("/containers/<int:container_id>/clone", methods=["POST"])
def clone_container(container_id):
    data = request.json

    if not data:
        return jsonify({"error": "Empty request body"}), 400
    container = use_case.copy_container(container_id, data.get("copy_progress"))
    if container is None:
        return jsonify({"error": "container copying error"}), 400
    return jsonify({"message": f"{CONTAINER_COPIED}", "session": container.to_dict()}), 201


@app.route("/models/<int:model_id>", methods=["PUT"])
def update_model(model_id):
    data = request.json

    if not data:
        return jsonify({"error": "Empty request body"}), 400
    model = use_case.update_model(model_id, data.get("file_id"), data.get("was_trained"), data.get("sequential"),
                                  data.get("code"))
    if model is None:
        return jsonify({"error": "model updating error"}), 400
    return jsonify({"message": f"{MODEL_UPDATED}", "model": model.to_dict()}), 201


@app.route("/optimizers/<int:optimizer_id>", methods=["PUT"])
def update_optimizer(optimizer_id):
    data = request.json

    if not data:
        return jsonify({"error": "Empty request body"}), 400

    optimizer = use_case.update_optimizer(optimizer_id, data.get("file_id"), data.get("was_trained"), data.get("code"))

    if optimizer is None:
        return jsonify({"error": "optimizer updating error"}), 400
    return jsonify({"message": f"{OPTIMIZER_UPDATED}", "optimizer": optimizer.to_dict()}), 201


@app.route("/containers/<int:container_id>", methods=["PUT"])
def update_container(container_id):
    data = request.json

    if not data:
        return jsonify({"error": "Empty request body"}), 400

    container = use_case.update_container(container_id, data.get("dataset_id"), data.get("model_id"),
                                          data.get("optimizer_id"), data.get("normalise_dataset"),
                                          data.get("criterion_code"), data.get("name"),
                                          data.get("online_training"), data.get("comment"))
    if container is None:
        return jsonify({"error": "container updating error"}), 400
    return jsonify({"message": f"{CONTAINER_UPDATED}", "container": container.to_dict()}), 201


@app.route("/files/<int:file_id>", methods=["PUT"])
def update_file(file_id):
    data = request.json

    if not data:
        return jsonify({"error": "Empty request body"}), 400
    file = use_case.update_file(file_id, data.get("file_type"), data.get("comment"), data.get("path"))
    if file is None:
        return jsonify({"error": "file updating error"}), 400
    return jsonify({"message": f"{FILE_UPDATED}", "file": file.to_dict()}), 201


@app.route("/sessions/<int:session_id>", methods=["PUT"])
def update_session(session_id):
    data = request.json

    if not data:
        return jsonify({"error": "Empty request body"}), 400
    session = use_case.update_session(session_id, data.get("status"), data.get("file_id"), data.get("epochs"),
                                      data.get("reset_progress"), data.get("container_id"))

    if session is None:
        return jsonify({"error": "session updating error"}), 400
    return jsonify({"message": f"{SESSION_UPDATED}", "session": session.to_dict()}), 201


@app.route("/containers/<int:container_id>", methods=["GET"])
def read_container(container_id):
    container = use_case.read_container(container_id)

    if container is None:
        return jsonify({"error": "container reading error"}), 400
    return jsonify({"message": f"{CONTAINER_READ}", "container": container.to_dict()}), 201


@app.route("/files/<int:file_id>", methods=["GET"])
def read_file(file_id):
    file = use_case.read_file(file_id)

    if file is None:
        return jsonify({"error": "file reading error"}), 400
    return jsonify({"message": f"{FILE_READ}", "file": file.to_dict()}), 201


@app.route("/models/<int:model_id>", methods=["GET"])
def read_model(model_id):
    model = use_case.read_model(model_id)

    if model is None:
        return jsonify({"error": "model reading error"}), 400

    return jsonify({"message": f"{MODEL_READ}", "model": model.to_dict()}), 201


@app.route("/optimizers/<int:optimizer_id>", methods=["GET"])
def read_optimizer(optimizer_id):
    optimizer = use_case.read_optimizer(optimizer_id)

    if optimizer is None:
        return jsonify({"error": "optimizer reading error"}), 400
    return jsonify({"message": f"{OPTIMIZER_READ}", "optimizer": optimizer.to_dict()}), 201


@app.route("/histories/<int:history_id>", methods=["GET"])
def read_history(history_id):
    history = use_case.read_history(history_id)

    if history is None:
        return jsonify({"error": "history reading error"}), 400
    return jsonify({"message": f"{HISTORY_READ}", "history": history.to_dict()}), 201


@app.route("/sessions/<int:session_id>", methods=["GET"])
def read_session(session_id):
    session = use_case.read_session(session_id)

    if session is None:
        return jsonify({"error": "session reading error"}), 400
    return jsonify({"message": f"{SESSION_READ}", "session": session.to_dict()}), 201


@app.route("/containers", methods=["GET"])
def read_containers():
    containers = use_case.read_containers()

    if containers is None:
        return jsonify({"error": "containers reading error"}), 400

    return jsonify({"message": f"{CONTAINERS_READ}", "containers": [container.to_dict() for container in containers]})


@app.route("/models", methods=["GET"])
def read_models():
    models = use_case.read_models()

    if models is None:
        return jsonify({"error": "models reading error"}), 400
    return jsonify({"message": f"{MODELS_READ}", "models": [item.to_dict() for item in models]})


@app.route("/optimizers", methods=["GET"])
def read_optimizers():
    optimizers = use_case.read_optimizers()

    if optimizers is None:
        return jsonify({"error": "optimizers reading error"}), 400
    return jsonify({"message": f"{OPTIMIZERS_READ}", "optimizers": [item.to_dict() for item in optimizers]})


@app.route("/files", methods=["GET"])
def read_files():
    files = use_case.read_files()

    if files is None:
        return jsonify({"error": "files reading error"}), 400
    return jsonify({"message": f"{FILES_READ}", "files": [item.to_dict() for item in files]})


@app.route("/histories", methods=["GET"])
def read_histories():
    histories = use_case.read_histories()

    if histories is None:
        return jsonify({"error": "histories reading error"}), 400
    return jsonify({"message": f"{HISTORIES_READ}", "histories": [item.to_dict() for item in histories]})


@app.route("/sessions", methods=["GET"])
def read_sessions():
    sessions = use_case.read_sessions()

    if sessions is None:
        return jsonify({"error": "sessions reading error"}), 400
    return jsonify({"message": f"{SESSIONS_READ}", "sessions": [item.to_dict() for item in sessions]})


@app.route("/files/<int:file_id>/content", methods=["GET"])
def read_file_content(file_id):
    file_content = use_case.read_file_content(file_id)

    if file_content is None:
        return jsonify({"error": "file content reading error"}), 400
    return jsonify({"message": f"file content read", "file_content": file_content})


@app.teardown_appcontext
def close_db(error):
    db_conn = g.pop("db_conn", None)
    if db_conn is not None:
        db_conn.close()


if __name__ == '__main__':
    app.run()
