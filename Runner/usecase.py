import csv
import datetime

import traceback
from torch import nn
from torch.utils.data import random_split, DataLoader
from requests.exceptions import HTTPError
from Docker.structs import Container, Model, Optimizer, Session, History, File
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

CONTAINER_LOADING = "container_loading"
CONTAINER_ASSEMBLING = "container_assembling"
CONTAINER_DISASSEMBLING = "container_disassembling"
CONTAINER_TRAINING = "container_training"
CONTAINER_EXECUTING = "container_executing"


def collate_fn(batch):
    inputs, targets = zip(*batch)

    targets = torch.tensor(targets).float()

    inputs = torch.stack(inputs).float()

    return inputs, targets.unsqueeze(-1)


class ExecutableContainer:
    def __init__(self, container_struct, model_struct, optimizer_struct, dataset_file, model_file, optimizer_file,
                 local_path: str):
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
        self.local_path = local_path
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.dataset = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def assemble_container(self, reset_progress):
        print("assemble")
        self.status = ASSEMBLED
        headers = {"Content-Type": "application/json"}
        data = {"container_id": self.container_struct.container_id, "history_type": CONTAINER_ASSEMBLING,
                "comment": f"Container assembling started"}
        requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)
        try:
            self.load_model(reset_progress)
        except Exception:
            error_message = traceback.format_exc()
            headers = {"Content-Type": "application/json"}
            data = {"container_id": self.container_struct.container_id, "history_type": CONTAINER_ASSEMBLING_ERROR,
                    "comment": f"Model assembling error\n{error_message}"}
            requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)
            raise

        try:
            self.load_optimizer(reset_progress)
        except Exception:
            error_message = traceback.format_exc()
            headers = {"Content-Type": "application/json"}
            data = {"container_id": self.container_struct.container_id, "history_type": CONTAINER_ASSEMBLING_ERROR,
                    "comment": f"Optimizer assembling error\n{error_message}"}
            requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)
            raise

        try:
            self.load_criterion()
        except Exception:
            error_message = traceback.format_exc()
            headers = {"Content-Type": "application/json"}
            data = {"container_id": self.container_struct.container_id, "history_type": CONTAINER_ASSEMBLING_ERROR,
                    "comment": f"Criterion assembling error\n{error_message}"}
            requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)
            raise

        try:
            self.load_dataset()
        except Exception:
            error_message = traceback.format_exc()
            headers = {"Content-Type": "application/json"}
            data = {"container_id": self.container_struct.container_id, "history_type": CONTAINER_ASSEMBLING_ERROR,
                    "comment": f"Dataset assembling error\n{error_message}"}
            requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)
            raise

        self.train_set, self.valid_set = random_split(self.dataset, (0.90, 0.10))

        self.train_dataloader = DataLoader(self.train_set, batch_size=64, shuffle=True,
                                           collate_fn=collate_fn)
        self.valid_dataloader = DataLoader(self.valid_set, batch_size=64, shuffle=False,
                                           collate_fn=collate_fn)
        headers = {"Content-Type": "application/json"}
        data = {"container_id": self.container_struct.container_id, "history_type": CONTAINER_ASSEMBLING,
                "comment": f"Container assembling completed"}
        requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)

    def disassemble_container(self):
        print("desassemble")
        self.status = BLUEPRINT
        headers = {"Content-Type": "application/json"}
        data = {"container_id": self.container_struct.container_id, "history_type": CONTAINER_DISASSEMBLING,
                "comment": f"Container disassembling started"}
        requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)
        try:
            torch.save(self.model.state_dict(), os.path.join(self.local_path, "storage", self.model_file.path))
            self.model = None
        except Exception:
            error_message = traceback.format_exc()
            headers = {"Content-Type": "application/json"}
            data = {"container_id": self.container_struct.container_id, "history_type": CONTAINER_DISASSEMBLING_ERROR,
                    "comment": f"Model saving error\n{error_message}"}
            requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)
            raise
        try:
            torch.save(self.optimizer.state_dict(), os.path.join(self.local_path, "storage", self.optimizer_file.path))
            self.optimizer = None
        except Exception:
            error_message = traceback.format_exc()
            headers = {"Content-Type": "application/json"}
            data = {"container_id": self.container_struct.container_id, "history_type": CONTAINER_DISASSEMBLING_ERROR,
                    "comment": f"Optimizer saving error\n{error_message}"}
            requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)
            raise
        self.model_struct.was_trained = True
        self.optimizer_struct.was_trained = True
        requests.put(f"http://127.0.0.1:5000/models/{self.model_struct.model_id}",
                     json={"was_trained": 1}).raise_for_status()
        requests.put(f"http://127.0.0.1:5000/optimizers/{self.optimizer_struct.optimizer_id}",
                     json={"was_trained": 1}).raise_for_status()
        headers = {"Content-Type": "application/json"}
        data = {"container_id": self.container_struct.container_id, "history_type": CONTAINER_DISASSEMBLING,
                "comment": f"Container disassembling completed"}
        requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)

    def load_model(self, reset_progress):
        if self.model_struct.sequential == 0:
            namespace = {"nn": torch.nn, "torch": torch}
            exec(self.model_struct.code, namespace)
            self.model = namespace["MyModel"]()
        else:
            code = f"nn.Sequential({self.model_struct.code})"
            self.model = eval(code, {"torch": torch})
        if self.model_struct.was_trained != 0 and reset_progress == 0:
            print("loading_model_from_file", reset_progress)
            self.model.load_state_dict(torch.load(os.path.join(self.local_path, "storage", self.model_file.path)))
        self.model = self.model.to(self.device)

    def load_optimizer(self, reset_progress):

        self.optimizer = eval(self.optimizer_struct.code, {"torch": torch, "model": self.model})
        if self.optimizer_struct.was_trained != 0 and reset_progress == 0:
            self.optimizer.load_state_dict(
                torch.load(os.path.join(self.local_path, "storage", self.optimizer_file.path)))

    def load_criterion(self):
        self.criterion = eval(self.container_struct.criterion_code, {"torch": torch})

    def load_dataset(self):
        self.dataset = torch.load(os.path.join(self.local_path, "storage", self.dataset_file.path))


class Runner:
    def __init__(self, local_path="C:\\Users\\EgorRychkov\\PycharmProjects\\NetworkDocker\\Docker\\local"):
        self.processes = {}
        self.local_path = local_path

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

    def load_container(self, container_id: int, local_path: str):
        headers = {"Content-Type": "application/json"}
        data = {"container_id": container_id, "history_type": CONTAINER_LOADING,
                "comment": f"Container loading started"}
        requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)
        container_response = requests.get(f"http://127.0.0.1:5000/containers/{container_id}")
        if container_response.status_code != 201:
            headers = {"Content-Type": "application/json"}
            data = {"container_id": container_id, "history_type": CONTAINER_LOADING_ERROR,
                    "comment": f"Container loading response returned {container_response.status_code}"}
            requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)
            raise HTTPError("Container get error")
        container_struct = Container(**container_response.json()["container"])

        model_response = requests.get(f"http://127.0.0.1:5000/models/{container_struct.model_id}")
        if model_response.status_code != 201:
            headers = {"Content-Type": "application/json"}
            data = {"container_id": container_id, "history_type": CONTAINER_LOADING_ERROR,
                    "comment": f"Model loading response returned {model_response.status_code}"}
            requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)
            raise HTTPError("Model get error")
        model_struct = Model(**model_response.json()["model"])

        optimizer_response = requests.get(f"http://127.0.0.1:5000/optimizers/{container_struct.optimizer_id}")
        if optimizer_response.status_code != 201:
            headers = {"Content-Type": "application/json"}
            data = {"container_id": container_id, "history_type": CONTAINER_LOADING_ERROR,
                    "comment": f"Optimizer loading response returned {optimizer_response.status_code}"}
            requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)
            raise HTTPError("Optimizer get error")
        optimizer_struct = Optimizer(**optimizer_response.json()["optimizer"])

        dataset_response = requests.get(f"http://127.0.0.1:5000/files/{container_struct.dataset_id}")
        if dataset_response.status_code != 201:
            headers = {"Content-Type": "application/json"}
            data = {"container_id": container_id, "history_type": CONTAINER_LOADING_ERROR,
                    "comment": f"Dataset loading response returned {dataset_response.status_code}"}
            requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)
            raise HTTPError("Dataset get error")
        dataset_struct = File(**dataset_response.json()["file"])

        model_file_response = requests.get(f"http://127.0.0.1:5000/files/{model_struct.file_id}")
        if model_file_response.status_code != 201:
            headers = {"Content-Type": "application/json"}
            data = {"container_id": container_id, "history_type": CONTAINER_LOADING_ERROR,
                    "comment": f"Model file loading response returned {model_file_response.status_code}"}
            requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)
            raise HTTPError("Model file get error")
        model_file_struct = File(**model_file_response.json()["file"])

        optimizer_file_response = requests.get(f"http://127.0.0.1:5000/files/{optimizer_struct.file_id}")
        if optimizer_file_response.status_code != 201:
            headers = {"Content-Type": "application/json"}
            data = {"container_id": container_id, "history_type": CONTAINER_LOADING_ERROR,
                    "comment": f"Optimizer file loading response returned {optimizer_file_response.status_code}"}
            requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)
            raise HTTPError("Optimizer file get error")
        optimizer_file_struct = File(**optimizer_file_response.json()["file"])

        container = ExecutableContainer(container_struct, model_struct, optimizer_struct,
                                        dataset_struct, model_file_struct, optimizer_file_struct,
                                        local_path)
        headers = {"Content-Type": "application/json"}
        data = {"container_id": container_struct.container_id, "history_type": CONTAINER_LOADING,
                "comment": "Container loaded"}
        requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)

        return container

    def execute_container(self, container_id: int, epochs: int, reset_progress: bool):
        headers = {"Content-Type": "application/json"}
        data = {"container_id": container_id, "history_type": CONTAINER_EXECUTING,
                "comment": f"Container executing prepare started"}
        requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)

        container = self.load_container(container_id, self.local_path)
        try:
            headers = {"Content-Type": "application/json"}
            data = {"file_type": "log", "comment": f"{container_id} training log file"}
            file_request = requests.post(f"http://127.0.0.1:5000/files", json=data, headers=headers)
            file_request.raise_for_status()

            file_struct = File(**file_request.json()["file"])
        except Exception:
            headers = {"Content-Type": "application/json"}
            data = {"container_id": container_id, "history_type": CONTAINER_EXECUTING_ERROR,
                    "comment": f"Log file creating error\n {traceback.format_exc()}"}
            requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)
            raise

        try:
            data = {"container_id": container_id, "status": "starting", "epochs": epochs,
                    "file_id": file_struct.file_id,
                    "reset_progress": reset_progress}
            session_request = requests.post(f"http://127.0.0.1:5000/sessions", json=data, headers=headers)
            session_request.raise_for_status()
            session = Session(**session_request.json()["session"])
        except Exception:
            headers = {"Content-Type": "application/json"}
            data = {"container_id": container_id, "history_type": CONTAINER_EXECUTING_ERROR,
                    "comment": f"Session creating error\n {traceback.format_exc()}"}
            requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)
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
            headers = {"Content-Type": "application/json"}
            data = {"container_id": container_id, "history_type": CONTAINER_EXECUTING_ERROR,
                    "comment": f"Process creating error \n{message}"}
            requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)

            data = {"status": "failed"}
            requests.put(f"http://127.0.0.1:5000/sessions/{session.session_id}", json=data, headers=headers)

            raise

        headers = {"Content-Type": "application/json"}
        data = {"container_id": container_id, "history_type": CONTAINER_EXECUTING,
                "comment": f"Container executing prepare completed"}
        requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)

    def test_container(self, container_id: int):

        container = self.load_container(container_id, self.local_path)
        container.assemble_container(0)

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
                for i in deviation:
                    deviations.append(i[0].cpu().numpy())

                loss = container.criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)

                total += targets.size(0)

        avg_loss = total_loss / total

        container.disassemble_container()
        return avg_loss

    @staticmethod
    def train_model(container: ExecutableContainer, num_epochs,
                    log_file_path, session_id, event, reset_progress):
        headers = {"Content-Type": "application/json"}
        data = {"container_id": container.container_struct.container_id, "history_type": CONTAINER_EXECUTING,
                "comment": f"Container executing started"}
        requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)

        headers = {"Content-Type": "application/json"}
        data = {"status": "training"}
        requests.put(f"http://127.0.0.1:5000/sessions/{session_id}", json=data, headers=headers)
        try:
            container.assemble_container(reset_progress)
            container.model.to(container.device)

            train_losses, valid_losses = [], []

            with open(os.path.join(container.local_path, "storage", log_file_path), "w", newline="",
                      encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["time", "epoch", "train_loss", "valid_loss"])
                writer.writeheader()

            for epoch in range(num_epochs):
                if event.is_set():
                    headers = {"Content-Type": "application/json"}
                    data = {"status": "stopped"}
                    requests.put(f"http://127.0.0.1:5000/sessions/{session_id}", json=data, headers=headers)
                    return
                container.model.train()
                running_train_loss = 0.0

                for inputs, targets in container.train_dataloader:
                    if event.is_set():
                        headers = {"Content-Type": "application/json"}
                        data = {"status": "stopped"}
                        requests.put(f"http://127.0.0.1:5000/sessions/{session_id}", json=data, headers=headers)
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

                with open(os.path.join(container.local_path, "storage", log_file_path), "a", newline="",
                          encoding="utf-8") as csvfile:
                    csv_writer = csv.DictWriter(csvfile, fieldnames=["time", "epoch", "train_loss", "valid_loss"])
                    csv_writer.writerow({"time": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "epoch": epoch,
                                         "train_loss": avg_train_loss, "valid_loss": avg_valid_loss})
            container.disassemble_container()
            headers = {"Content-Type": "application/json"}
            data = {"container_id": container.container_struct.container_id, "history_type": CONTAINER_EXECUTING,
                    "comment": f"Container executing completed"}
            requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)

            headers = {"Content-Type": "application/json"}
            data = {"status": "completed"}
            requests.put(f"http://127.0.0.1:5000/sessions/{session_id}", json=data, headers=headers)
        except Exception:
            headers = {"Content-Type": "application/json"}
            data = {"container_id": container.container_struct.container_id, "history_type": CONTAINER_EXECUTING_ERROR,
                    "comment": f"Container executing error\n{traceback.format_exc()}"}
            requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)

            headers = {"Content-Type": "application/json"}
            data = {"status": "failed"}
            requests.put(f"http://127.0.0.1:5000/sessions/{session_id}", json=data, headers=headers)
        requests.delete(f"http://127.0.0.1:8080/runner/processes/{container.container_struct.container_id}")
