import datetime

from database import DB, Model, File, Optimizer, Container, History
import shutil
import os


def generate_path(file_id, file_type, created_at):
    return f"{file_id}.{created_at}.{file_type}.pth"


MODEL_CREATED = "model_created"
OPTIMIZER_CREATED = "optimizer_created"
CONTAINER_CREATED = "container_created"
MODEL_COPIED = "model_copied"
OPTIMIZER_COPIED = "optimizer_copied"
CONTAINER_COPIED = "container_copied"


class UseCase:
    def __init__(self, db_repository: DB):
        self.db_repository = db_repository

    def create_model(self, sequential: bool = None, code: str = None) -> int:
        now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        model_file = File(created_at=now_time, updated_at=now_time, file_type="model", comment="", path="empty")
        model_file_id = self.db_repository.create_file(model_file)
        model_file.file_id = model_file_id
        model_file.path = generate_path(model_file_id, "model", now_time)
        self.db_repository.update_file(model_file)

        with open(os.path.join("local", "storage", model_file.path), "w") as f:
            pass

        model: Model = Model(created_at=now_time, updated_at=now_time, file_id=model_file_id, was_trained=False,
                             sequential=sequential, code=code)

        model_id = self.db_repository.create_model(model)

        self.create_history(model_id=model_id, history_type=MODEL_CREATED)

        return model_id

    def create_file(self, file_type: str = None, comment: str = None) -> int:
        now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        file: File = File(created_at=now_time, updated_at=now_time, file_type=file_type, comment=comment, path="empty")
        file_id = self.db_repository.create_file(file)
        file.file_id = file_id
        file.path = generate_path(file_id, file_type, now_time)
        self.db_repository.update_file(file)

        return file_id

    def create_optimizer(self, code: str = None) -> int:
        now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        optimizer_file: File = File(created_at=now_time, updated_at=now_time, file_type="optimizer", comment="",
                                    path="empty")
        optimizer_file_id = self.db_repository.create_file(optimizer_file)
        optimizer_file.file_id = optimizer_file_id
        optimizer_file.path = generate_path(optimizer_file_id, "optimizer", now_time)
        self.db_repository.update_file(optimizer_file)
        with open(os.path.join("local", "storage", optimizer_file.path), "w") as f:
            pass

        optimizer: Optimizer = Optimizer(created_at=now_time, updated_at=now_time, file_id=optimizer_file_id,
                                         was_trained=False, code=code)

        optimizer_id = self.db_repository.create_optimizer(optimizer)

        self.create_history(optimizer_id=optimizer_id, history_type=OPTIMIZER_CREATED)

        return optimizer_id

    def create_container(self, dataset_id: int = None, model_id: int = None, optimizer_id: int = None,
                         normalise_dataset: bool = None,
                         name: str = None,
                         comment: str = None) -> int:
        now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if self.db_repository.check_model_id(model_id):
            raise ValueError("model_id already used by another container")
        if self.db_repository.check_optimizer_id(optimizer_id):
            raise ValueError("optimizer_id already used by another container")

        container: Container = Container(created_at=now_time, updated_at=now_time, dataset_id=dataset_id,
                                         model_id=model_id, optimizer_id=optimizer_id,
                                         normalise_dataset=normalise_dataset, name=name, comment=comment)
        container_id = self.db_repository.create_container(container)

        self.create_history(container_id=container_id, history_type=CONTAINER_CREATED)

        return container_id

    def create_history(self, container_id=None, model_id=None, optimizer_id=None, history_type=None, comment=None,
                       file_id=None) -> int:
        now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        history: History = History(created_at=now_time, updated_at=now_time, container_id=container_id,
                                   model_id=model_id, optimizer_id=optimizer_id, history_type=history_type,
                                   comment=comment, file_id=file_id)
        history.history_id = self.db_repository.create_history(history)

        return history.history_id

    def copy_model(self, source_model_id: int, copy_progress: bool) -> int:
        now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        model_file = File(created_at=now_time, updated_at=now_time, file_type="model", comment="", path="empty")
        model_file_id = self.db_repository.create_file(model_file)
        model_file.file_id = model_file_id
        model_file.path = generate_path(model_file_id, "model", now_time)
        self.db_repository.update_file(model_file)
        with open(os.path.join("local", "storage", model_file.path), "w") as f:
            pass

        source_model = self.db_repository.read_model(source_model_id)

        model: Model = Model(created_at=now_time, updated_at=now_time, sequential=source_model.sequential,
                             code=source_model.code, file_id=model_file_id, was_trained=False)
        if copy_progress:
            model.was_trained = source_model.was_trained
            source_model_file = self.db_repository.read_file(source_model.file_id)

            shutil.copy(os.path.join("local", "storage", source_model_file.path),
                        os.path.join("local", "storage", model_file.path))
        model_id = self.db_repository.create_model(model)

        self.create_history(model_id=model_id, history_type=MODEL_COPIED, comment=f"from {source_model_id}")

        return model_id

    def copy_optimizer(self, source_optimizer_id: int, copy_progress: bool) -> int:
        now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        optimizer_file: File = File(created_at=now_time, updated_at=now_time, file_type="optimizer", comment="",
                                    path="empty")
        optimizer_file_id = self.db_repository.create_file(optimizer_file)
        optimizer_file.file_id = optimizer_file_id
        optimizer_file.path = generate_path(optimizer_file_id, "optimizer", now_time)
        self.db_repository.update_file(optimizer_file)
        with open(os.path.join("local", "storage", optimizer_file.path), "w") as f:
            pass

        source_optimizer = self.db_repository.read_optimizer(source_optimizer_id)

        optimizer: Optimizer = Optimizer(created_at=now_time, updated_at=now_time, file_id=optimizer_file_id,
                                         was_trained=False, code=source_optimizer.code)

        if copy_progress:
            optimizer.was_trained = source_optimizer.was_trained
            source_optimizer_file = self.db_repository.read_file(source_optimizer.file_id)

            shutil.copy(os.path.join("local", "storage", source_optimizer_file.path),
                        os.path.join("local", "storage", optimizer_file.path))
        optimizer_id = self.db_repository.create_optimizer(optimizer)

        self.create_history(optimizer_id=optimizer_id, history_type=OPTIMIZER_COPIED,
                            comment=f"from {source_optimizer_id}")

        return optimizer_id

    def copy_container(self, source_container_id: int, copy_progress: bool) -> int:
        now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        source_container = self.db_repository.read_container(source_container_id)
