import csv
import datetime

import traceback
from pprint import pprint

import numpy as np
from torch import nn
from torch.utils.data import random_split, DataLoader
from requests.exceptions import HTTPError

from Runner.env.config import Config
from Runner.env.env_builder import EnvBuilder
from Runner.structs import Container, Model, Optimizer, Session, History, File
import requests
import torch
import os
import matplotlib
from matplotlib import pyplot as plt

import multiprocessing

from ignite.engine import Events
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, MeanSquaredError

OFFLINE = "offline"
BLUEPRINT = "blueprint"
ASSEMBLED = "assembled"
TRAINING = "training"

CONTAINER_LOADING_ERROR = "container_loading_error"
CONTAINER_EXECUTING_ERROR = "container_executing_error"
CONTAINER_ASSEMBLING_ERROR = "container_assembling_error"
CONTAINER_DISASSEMBLING_ERROR = "container_disassembling_error"
CONTAINER_PREDICTION_ERROR = "container_prediction_error"

CONTAINER_LOADING = "container_loading"
CONTAINER_ASSEMBLING = "container_assembling"
CONTAINER_DISASSEMBLING = "container_disassembling"
CONTAINER_TRAINING = "container_training"
CONTAINER_EXECUTING = "container_executing"
CONTAINER_PREDICTION = "container_prediction"


def collate_fn(batch):
    inputs, targets = zip(*batch)

    targets = torch.tensor(targets).float()

    inputs = torch.stack(inputs).float()

    return inputs, targets.unsqueeze(-1)


def log_history(container_id, history_type, comment, config):
    headers = {"Content-Type": "application/json"}
    data = {"container_id": container_id, "history_type": history_type,
            "comment": comment}
    requests.post(f"{config.get_docker_ip()}/histories", json=data, headers=headers)


class ExecutableContainer:
    def __init__(self, container_struct, model_struct, optimizer_struct, dataset_file, model_file, optimizer_file,
                 env_builder: EnvBuilder, config: Config):
        self.valid_dataloader = None
        self.train_dataloader = None
        self.valid_set = None
        self.train_set = None
        self.container_struct = container_struct
        self.model_struct = model_struct
        self.optimizer_struct = optimizer_struct
        self.dataset_file = dataset_file
        self.model_file = model_file
        self.optimizer_file = optimizer_file
        self.status = BLUEPRINT
        self.env_builder = env_builder
        self.config = config
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.dataset = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def assemble_container(self, reset_progress, mode):
        print("assemble")
        self.status = ASSEMBLED
        log_history(self.container_struct.container_id, CONTAINER_ASSEMBLING, f"Container assembling started",
                    self.config)
        try:
            self.load_model(reset_progress)
        except Exception:
            error_message = traceback.format_exc()
            log_history(self.container_struct.container_id, CONTAINER_ASSEMBLING_ERROR,
                        f"Model assembling error\n{error_message}", self.config)
            raise

        try:
            self.load_optimizer(reset_progress)
        except Exception:
            error_message = traceback.format_exc()
            log_history(self.container_struct.container_id, CONTAINER_ASSEMBLING_ERROR,
                        f"Optimizer assembling error\n{error_message}", self.config)
            raise

        try:
            self.load_criterion()
        except Exception:
            error_message = traceback.format_exc()
            log_history(self.container_struct.container_id, CONTAINER_ASSEMBLING_ERROR,
                        f"Criterion assembling error\n{error_message}", self.config)
            raise

        try:
            self.load_dataset()
        except Exception:
            error_message = traceback.format_exc()
            log_history(self.container_struct.container_id, CONTAINER_ASSEMBLING_ERROR,
                        f"Dataset assembling error\n{error_message}", self.config)
            raise

        if mode == "train":
            self.train_set, self.valid_set = random_split(self.dataset, (0.90, 0.10))
        else:
            self.train_set, self.valid_set = random_split(self.dataset, (0.10, 0.90))
        self.train_dataloader = DataLoader(self.train_set, batch_size=64, shuffle=True,
                                           collate_fn=collate_fn)
        self.valid_dataloader = DataLoader(self.valid_set, batch_size=64, shuffle=False,
                                           collate_fn=collate_fn)

        log_history(self.container_struct.container_id, CONTAINER_ASSEMBLING, f"Container assembling completed",
                    self.config)

    def disassemble_container(self):
        print("desassemble")
        self.status = BLUEPRINT

        log_history(self.container_struct.container_id, CONTAINER_DISASSEMBLING, f"Container disassembling started",
                    self.config)

        try:
            torch.save(self.model.state_dict(), self.env_builder.get_storage_path(self.model_file.path))
            self.model = None
        except Exception:
            error_message = traceback.format_exc()
            log_history(self.container_struct.container_id, CONTAINER_DISASSEMBLING_ERROR,
                        f"Model saving error\n{error_message}", self.config)
            raise
        try:
            torch.save(self.optimizer.state_dict(), self.env_builder.get_storage_path(self.optimizer_file.path))
            self.optimizer = None
        except Exception:
            error_message = traceback.format_exc()
            log_history(self.container_struct.container_id, CONTAINER_DISASSEMBLING_ERROR,
                        f"Optimizer saving error\n{error_message}", self.config)
            raise
        self.model_struct.was_trained = True
        self.optimizer_struct.was_trained = True

        requests.put(f"{self.config.get_docker_ip()}/models/{self.model_struct.model_id}",
                     json={"was_trained": 1}).raise_for_status()
        requests.put(f"{self.config.get_docker_ip()}/optimizers/{self.optimizer_struct.optimizer_id}",
                     json={"was_trained": 1}).raise_for_status()

        log_history(self.container_struct.container_id, CONTAINER_DISASSEMBLING, f"Container disassembling completed",
                    self.config)

    def load_model(self, reset_progress):
        if self.model_struct.sequential == 0:
            namespace = {"nn": torch.nn, "torch": torch}
            exec(self.model_struct.code, namespace)
            self.model = namespace["MyModel"]()
        else:
            code = f"nn.Sequential({self.model_struct.code})"
            self.model = eval(code, {"torch": torch})
        if self.model_struct.was_trained != 0 and reset_progress == 0:
            self.model.load_state_dict(torch.load(self.env_builder.get_storage_path(self.model_file.path)))
        self.model = self.model.to(self.device)

    def load_optimizer(self, reset_progress):
        self.optimizer = eval(self.optimizer_struct.code, {"torch": torch, "model": self.model})
        if self.optimizer_struct.was_trained != 0 and reset_progress == 0:
            self.optimizer.load_state_dict(torch.load(self.env_builder.get_storage_path(self.optimizer_file.path)))

    def load_criterion(self):
        self.criterion = eval(self.container_struct.criterion_code, {"torch": torch})

    def load_dataset(self):
        self.dataset = torch.load(self.env_builder.get_storage_path(self.dataset_file.path))


class Runner:
    def __init__(self, config: Config, env_builder: EnvBuilder):
        self.processes = {}
        self.config = config
        self.env_builder = env_builder

    def create_process(self, container_id, process, stop_event):
        self.processes[container_id] = {"id": container_id, "process": process, "stop_event": stop_event}

    def delete_process(self, container_id):
        if container_id not in self.processes:
            raise KeyError("Process not found")
        del self.processes[container_id]

    def read_process_status(self, container_id):
        if container_id not in self.processes:
            raise KeyError("Process not found")
        return self.processes[container_id]["process"].is_alive()

    def stop_process(self, container_id):
        if container_id not in self.processes:
            raise KeyError("Process not found")
        if self.processes[container_id]["process"].is_alive():
            self.processes[container_id]["stop_event"].set()

    def load_container(self, container_id: int):
        log_history(container_id, CONTAINER_LOADING, f"Container loading started", self.config)
        container_response = requests.get(f"{self.config.get_docker_ip()}/containers/{container_id}")
        if container_response.status_code != 201:
            log_history(container_id, CONTAINER_LOADING_ERROR,
                        f"Container loading response returned {container_response.status_code}", self.config)
            raise HTTPError("Container get error")
        container_struct = Container(**container_response.json()["container"])

        model_response = requests.get(f"{self.config.get_docker_ip()}/models/{container_struct.model_id}")
        if model_response.status_code != 201:
            log_history(container_id, CONTAINER_LOADING_ERROR,
                        f"Model loading response returned {model_response.status_code}", self.config)
            raise HTTPError("Model get error")
        model_struct = Model(**model_response.json()["model"])

        optimizer_response = requests.get(f"{self.config.get_docker_ip()}/optimizers/{container_struct.optimizer_id}")
        if optimizer_response.status_code != 201:
            log_history(container_id, CONTAINER_LOADING_ERROR,
                        f"Optimizer loading response returned {optimizer_response.status_code}", self.config)
            raise HTTPError("Optimizer get error")
        optimizer_struct = Optimizer(**optimizer_response.json()["optimizer"])

        dataset_response = requests.get(f"{self.config.get_docker_ip()}/files/{container_struct.dataset_id}")
        if dataset_response.status_code != 201:
            log_history(container_id, CONTAINER_LOADING_ERROR,
                        f"Dataset loading response returned {dataset_response.status_code}", self.config)
            raise HTTPError("Dataset get error")
        dataset_struct = File(**dataset_response.json()["file"])

        model_file_response = requests.get(f"{self.config.get_docker_ip()}/files/{model_struct.file_id}")
        if model_file_response.status_code != 201:
            log_history(container_id, CONTAINER_LOADING_ERROR,
                        f"Model file loading response returned {model_file_response.status_code}", self.config)
            raise HTTPError("Model file get error")
        model_file_struct = File(**model_file_response.json()["file"])

        optimizer_file_response = requests.get(f"{self.config.get_docker_ip()}/files/{optimizer_struct.file_id}")
        if optimizer_file_response.status_code != 201:
            log_history(container_id, CONTAINER_LOADING_ERROR,
                        f"Optimizer file loading response returned {optimizer_file_response.status_code}", self.config)
            raise HTTPError("Optimizer file get error")
        optimizer_file_struct = File(**optimizer_file_response.json()["file"])

        container = ExecutableContainer(container_struct, model_struct, optimizer_struct,
                                        dataset_struct, model_file_struct, optimizer_file_struct,
                                        self.env_builder, self.config)
        log_history(container_struct.container_id, CONTAINER_LOADING, "Container loaded", self.config)

        return container

    def execute_container(self, container_id: int, epochs: int, reset_progress: bool):
        log_history(container_id, CONTAINER_EXECUTING, f"Container executing prepare started", self.config)
        container = self.load_container(container_id)
        try:
            headers = {"Content-Type": "application/json"}
            data = {"file_type": "log", "comment": f"{container_id} training log file"}
            file_request = requests.post(f"{self.config.get_docker_ip()}/files", json=data, headers=headers)
            file_request.raise_for_status()

            file_struct = File(**file_request.json()["file"])
        except Exception:
            log_history(container_id, CONTAINER_EXECUTING_ERROR, f"Log file creating error\n {traceback.format_exc()}",
                        self.config)
            raise

        try:
            data = {"container_id": container_id, "status": "starting", "epochs": epochs,
                    "file_id": file_struct.file_id,
                    "reset_progress": reset_progress}
            session_request = requests.post(f"{self.config.get_docker_ip()}/sessions", json=data, headers=headers)
            session_request.raise_for_status()
            session = Session(**session_request.json()["session"])
        except Exception:
            log_history(container_id, CONTAINER_EXECUTING_ERROR, f"Session creating error\n {traceback.format_exc()}",
                        self.config)
            raise
        print(container.model_struct.was_trained)
        try:
            stop_event = torch.multiprocessing.Event()
            process = torch.multiprocessing.Process(target=self.train_model,
                                                    args=(container,
                                                          epochs,
                                                          file_struct.path, session.session_id, stop_event,
                                                          reset_progress))
            process.start()
            self.create_process(container_id, process, stop_event)

        except Exception:
            message = traceback.format_exc()
            log_history(container_id, CONTAINER_EXECUTING_ERROR, f"Process creating error \n{message}", self.config)

            data = {"status": "failed"}
            requests.put(f"{self.config.get_docker_ip()}/sessions/{session.session_id}", json=data, headers=headers)

            raise
        log_history(container_id, CONTAINER_EXECUTING, f"Container executing prepare completed", self.config)

    def test_container(self, container_id: int):

        container = self.load_container(container_id)
        container.assemble_container(0, "test")

        container.model.eval()
        test_loader = container.valid_dataloader
        total_loss = 0.0
        total = 0
        deviations = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(container.device), targets.to(container.device)

                outputs = container.model(inputs)

                deviation = (outputs - targets)
                for i in range(len(deviation)):
                    if len(deviations) <= 10:
                        deviations.append([outputs[i], targets[i], deviation[i]])

                loss = container.criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)

                total += targets.size(0)

        avg_loss = total_loss / total

        print("test_loader", total)
        pprint(deviations)
        container.disassemble_container()
        return avg_loss

    def predict(self, container_id: int, input_data):
        log_history(container_id, CONTAINER_PREDICTION, f"Container prediction started", self.config)

        container = self.load_container(container_id)
        container.assemble_container(0, "test")
        try:

            input_array = np.array(input_data, dtype=np.float32)
            if input_array.ndim < 2:
                input_array = np.expand_dims(input_array, axis=-1)

            input_data = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0)
            input_data = input_data.to(container.device)
            container.model.to(container.device)
            container.model.eval()
            with torch.no_grad():
                output = container.model(input_data)
        except Exception:
            log_history(container_id, CONTAINER_PREDICTION_ERROR,
                        f"Container prediction error\n{traceback.format_exc()}", self.config)
            raise
        return output.cpu().numpy().tolist()

    @staticmethod
    def train_model(container: ExecutableContainer, num_epochs,
                    log_file_path, session_id, event, reset_progress):
        log_history(container.container_struct.container_id, CONTAINER_EXECUTING, f"Container executing started",
                    container.config)

        headers = {"Content-Type": "application/json"}
        data = {"status": "training"}
        requests.put(f"{container.config.get_docker_ip()}/sessions/{session_id}", json=data, headers=headers)
        try:
            container.assemble_container(reset_progress, "train")
            container.model.to(container.device)

            train_losses, valid_losses = [], []

            with open(container.env_builder.get_storage_path(log_file_path), "w", newline="",
                      encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["time", "epoch", "train_loss", "valid_loss"])
                writer.writeheader()

            for epoch in range(num_epochs):

                if event.is_set():
                    headers = {"Content-Type": "application/json"}
                    data = {"status": "stopped"}
                    requests.put(f"{container.config.get_docker_ip()}/sessions/{session_id}", json=data,
                                 headers=headers)
                    return
                container.model.train()
                running_train_loss = 0.0

                for inputs, targets in container.train_dataloader:
                    if event.is_set():
                        headers = {"Content-Type": "application/json"}
                        data = {"status": "stopped"}
                        requests.put(f"{container.config.get_docker_ip()}/sessions/{session_id}", json=data,
                                     headers=headers)
                        return
                    inputs, targets = inputs.to(container.device), targets.to(container.device)

                    container.optimizer.zero_grad()

                    outputs = container.model(inputs)
                    loss = container.criterion(outputs, targets)
                    loss.backward()
                    container.optimizer.step()

                    running_train_loss += loss.item()

                avg_train_loss = running_train_loss / len(container.train_dataloader)
                train_losses.append(avg_train_loss)

                container.model.eval()
                running_valid_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in container.valid_dataloader:
                        inputs, targets = inputs.to(container.device), targets.to(container.device)
                        outputs = container.model(inputs)
                        loss = container.criterion(outputs, targets)
                        running_valid_loss += loss.item()

                avg_valid_loss = running_valid_loss / len(container.valid_dataloader)
                valid_losses.append(avg_valid_loss)

                with open(container.env_builder.get_storage_path(log_file_path), "a", newline="",
                          encoding="utf-8") as csvfile:
                    csv_writer = csv.DictWriter(csvfile, fieldnames=["time", "epoch", "train_loss", "valid_loss"])
                    csv_writer.writerow({"time": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "epoch": epoch,
                                         "train_loss": avg_train_loss, "valid_loss": avg_valid_loss})
            container.disassemble_container()
            log_history(container.container_struct.container_id, CONTAINER_EXECUTING, f"Container executing completed",
                        container.config)
            headers = {"Content-Type": "application/json"}
            data = {"status": "completed"}
            requests.put(f"{container.config.get_docker_ip()}/sessions/{session_id}", json=data, headers=headers)
        except Exception:
            log_history(container.container_struct.container_id, CONTAINER_EXECUTING_ERROR,
                        f"Container executing error\n{traceback.format_exc()}", container.config)
            headers = {"Content-Type": "application/json"}
            data = {"status": "failed"}
            requests.put(f"{container.config.get_docker_ip()}/sessions/{session_id}", json=data, headers=headers)
        requests.delete(
            f"{container.config.get_runner_ip()}/runner/processes/{container.container_struct.container_id}")
