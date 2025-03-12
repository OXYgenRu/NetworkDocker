from torch import nn
from torch.utils.data import random_split, DataLoader
from requests.exceptions import HTTPError
from Docker.structs import Container, Model, Optimizer, Session, History, File
import requests
import torch
import os
import traceback

OFFLINE = "offline"
ONLINE = "online"
TRAINING = "training"

CONTAINER_LOADING_ERROR = "container_loading_error"


def collate_fn(batch):
    inputs, targets = zip(*batch)

    targets = torch.tensor(targets).float()

    inputs = torch.stack(inputs).float()

    return inputs, targets.unsqueeze(-1)


class ExecutableContainer:
    def __init__(self, container_struct, model_struct, optimizer_struct, dataset_file, model_file, optimizer_file,
                 local_path: str):
        self.container_struct = container_struct
        self.model_struct = model_struct
        self.optimizer_struct = optimizer_struct
        self.dataset_file = dataset_file
        self.model_file = model_file
        self.optimizer_file = optimizer_file
        self.status = ONLINE
        self.local_path = local_path
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.dataset = None

        try:
            self.load_model()
        except Exception:
            error_message = traceback.format_exc()
            headers = {"Content-Type": "application/json"}
            data = {"container_id": container_struct.container_id, "history_type": CONTAINER_LOADING_ERROR,
                    "comment": f"Model execution error\n{error_message}"}
            requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)
            raise

        try:
            self.load_optimizer()
        except Exception:
            error_message = traceback.format_exc()
            headers = {"Content-Type": "application/json"}
            data = {"container_id": container_struct.container_id, "history_type": CONTAINER_LOADING_ERROR,
                    "comment": f"Optimizer execution error\n{error_message}"}
            requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)
            raise

        try:
            self.load_criterion()
        except Exception:
            error_message = traceback.format_exc()
            headers = {"Content-Type": "application/json"}
            data = {"container_id": container_struct.container_id, "history_type": CONTAINER_LOADING_ERROR,
                    "comment": f"Criterion execution error\n{error_message}"}
            requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)
            raise

        try:
            self.load_dataset()
        except Exception as e:
            error_message = traceback.format_exc()
            headers = {"Content-Type": "application/json"}
            data = {"container_id": container_struct.container_id, "history_type": CONTAINER_LOADING_ERROR,
                    "comment": f"Dataset execution error\n{error_message}"}
            requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)
            raise

        self.train_set, self.valid_set = random_split(self.dataset, (0.95, 0.05))

        self.train_dataloader = DataLoader(self.train_set, batch_size=64, shuffle=True,
                                           collate_fn=collate_fn)
        self.valid_dataloader = DataLoader(self.valid_set, batch_size=64, shuffle=False,
                                           collate_fn=collate_fn)
        
    def load_model(self):
        if self.model_struct.was_trained == 0:
            if self.model_struct.sequential == 0:
                namespace = {"nn": torch.nn, "torch": torch}
                exec(self.model_struct.code, namespace)
                self.model = namespace["MyModel"]()
            else:
                code = f"nn.Sequential({self.model_struct.code})"
                self.model = eval(code, {"torch": torch})
        else:
            self.model = torch.load(os.path.join(self.local_path, "storage", self.model_file.path))

    def load_optimizer(self):
        if self.optimizer_struct.was_trained == 0:
            self.optimizer = eval(self.optimizer_struct.code, {"torch": torch, "model": self.model})
        else:
            self.optimizer = torch.load(os.path.join(self.local_path, "storage", self.optimizer_file.path))

    def load_criterion(self):
        self.criterion = eval(self.container_struct.criterion_code, {"torch": torch})

    def load_dataset(self):
        self.dataset = torch.load(os.path.join(self.local_path, "storage", self.dataset_file.path))


class Runner:
    def __init__(self):
        self.containers = {}

    def load_container(self, container_id: int, local_path: str):
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

        self.containers[container_id] = ExecutableContainer(container_struct, model_struct, optimizer_struct,
                                                            dataset_struct, model_file_struct, optimizer_file_struct,
                                                            local_path)
        headers = {"Content-Type": "application/json"}
        data = {"container_id": container_struct.container_id, "history_type": "Container loaded",
                "comment": "Container loaded"}
        requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)

    def shutdown_container(self, container_id: int):
        if container_id in self.containers:
            del self.containers[container_id]
        headers = {"Content-Type": "application/json"}
        data = {"container_id": container_id, "history_type": "Container turned off",
                "comment": "Container loaded"}
        requests.post(f"http://127.0.0.1:5000/histories", json=data, headers=headers)

    def read_containers(self) -> dict[dict]:
        containers = {}
        for key, val in self.containers.items():
            containers[key] = {"id": key, "status": val.status}
        return containers
