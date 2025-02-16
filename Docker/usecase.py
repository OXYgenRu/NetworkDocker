import datetime
from typing import Optional

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

CONTAINER_UPDATED = "container_updated"
ERROR = "error"


class UseCase:
    def __init__(self, db_repository: DB):
        self.db_repository = db_repository

    def create_model(self, sequential: bool = None, code: str = None) -> Optional[Model]:
        try:
            self.db_repository.begin_transaction()

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

            model.model_id = self.db_repository.create_model(model)

            history: History = History(created_at=now_time, updated_at=now_time, model_id=model.model_id,
                                       history_type=MODEL_CREATED)
            history.history_id = self.db_repository.create_history(history)

            self.db_repository.commit_transaction()
        except Exception as e:
            self.db_repository.rollback_transaction()

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction()
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Model Creating Error: {e}"))
            self.db_repository.commit_transaction()
            print(f"Model Creating Error: {e}")
            return None
        return model

    def create_file(self, file_type: str = None, comment: str = None) -> Optional[File]:
        try:
            self.db_repository.begin_transaction()
            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            file: File = File(created_at=now_time, updated_at=now_time, file_type=file_type, comment=comment,
                              path="empty")
            file_id = self.db_repository.create_file(file)
            file.file_id = file_id
            file.path = generate_path(file_id, file_type, now_time)
            self.db_repository.update_file(file)
            self.db_repository.commit_transaction()
        except Exception as e:
            self.db_repository.rollback_transaction()

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction()
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"File creating Error: {e}"))
            self.db_repository.commit_transaction()
            print(f"File creating Error: {e}")
            return None

        return file

    def create_optimizer(self, code: str = None) -> Optional[Optimizer]:
        try:
            self.db_repository.begin_transaction()
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

            optimizer.optimizer_id = self.db_repository.create_optimizer(optimizer)

            history: History = History(created_at=now_time, updated_at=now_time, optimizer_id=optimizer.optimizer_id,
                                       history_type=OPTIMIZER_CREATED)
            history.history_id = self.db_repository.create_history(history)

            self.db_repository.commit_transaction()
        except Exception as e:
            self.db_repository.rollback_transaction()

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction()
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Optimizer creating Error: {e}"))
            self.db_repository.commit_transaction()
            print(f"Optimizer creating Error: {e}")
            return None

        return optimizer

    def create_container(self, dataset_id: int = None, model_id: int = None, optimizer_id: int = None,
                         normalise_dataset: bool = None,
                         name: str = None,
                         comment: str = None) -> Optional[Container]:
        try:
            self.db_repository.begin_transaction()
            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if self.db_repository.check_model_id(model_id):
                raise ValueError("model_id already used by another container")
            if self.db_repository.check_optimizer_id(optimizer_id):
                raise ValueError("optimizer_id already used by another container")

            container: Container = Container(created_at=now_time, updated_at=now_time, dataset_id=dataset_id,
                                             model_id=model_id, optimizer_id=optimizer_id,
                                             normalise_dataset=normalise_dataset, name=name, comment=comment)
            container.container_id = self.db_repository.create_container(container)

            history: History = History(created_at=now_time, updated_at=now_time, container_id=container.container_id,
                                       history_type=CONTAINER_CREATED)
            history.history_id = self.db_repository.create_history(history)

            self.db_repository.commit_transaction()
        except Exception as e:
            self.db_repository.rollback_transaction()

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction()
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Container creating Error: {e}"))
            self.db_repository.commit_transaction()
            print(f"Container creating Error: {e}")
            return None
        return container

    def create_history(self, container_id=None, model_id=None, optimizer_id=None, history_type=None, comment=None,
                       file_id=None) -> Optional[History]:
        try:
            self.db_repository.begin_transaction()
            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            history: History = History(created_at=now_time, updated_at=now_time, container_id=container_id,
                                       model_id=model_id, optimizer_id=optimizer_id, history_type=history_type,
                                       comment=comment, file_id=file_id)
            history.history_id = self.db_repository.create_history(history)
            self.db_repository.commit_transaction()
        except Exception as e:
            self.db_repository.rollback_transaction()

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction()
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"History creating Error: {e}"))
            self.db_repository.commit_transaction()
            print(f"History creating Error: {e}")
            return None
        return history

    def copy_model(self, source_model_id: int, copy_progress: bool) -> Optional[Model]:
        try:
            self.db_repository.begin_transaction()
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
            model.model_id = self.db_repository.create_model(model)

            history: History = History(created_at=now_time, updated_at=now_time, model_id=model.model_id,
                                       history_type=MODEL_COPIED, comment=f"from {source_model_id}")
            history.history_id = self.db_repository.create_history(history)

            self.db_repository.commit_transaction()
        except Exception as e:
            self.db_repository.rollback_transaction()

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction()
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Model copying Error: {e}"))
            self.db_repository.commit_transaction()
            print(f"Model copying Error: {e}")
            return None
        return model

    def copy_optimizer(self, source_optimizer_id: int, copy_progress: bool) -> Optional[Optimizer]:
        try:
            self.db_repository.begin_transaction()
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
            optimizer.optimizer_id = self.db_repository.create_optimizer(optimizer)

            history: History = History(created_at=now_time, updated_at=now_time, optimizer_id=optimizer.optimizer_id,
                                       history_type=OPTIMIZER_COPIED,
                                       comment=f"from {source_optimizer_id}")
            history.history_id = self.db_repository.create_history(history)

            self.db_repository.commit_transaction()
        except Exception as e:
            self.db_repository.rollback_transaction()

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction()
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Optimizer copying Error: {e}"))
            self.db_repository.commit_transaction()
            print(f"Optimizer copying Error: {e}")
            return None
        return optimizer

    def copy_container(self, source_container_id: int, copy_progress: bool) -> Optional[Container]:
        try:
            self.db_repository.begin_transaction()
            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            source_container = self.db_repository.read_container(source_container_id)

            # copy model

            model_file = File(created_at=now_time, updated_at=now_time, file_type="model", comment="", path="empty")
            model_file_id = self.db_repository.create_file(model_file)
            model_file.file_id = model_file_id
            model_file.path = generate_path(model_file_id, "model", now_time)
            self.db_repository.update_file(model_file)
            with open(os.path.join("local", "storage", model_file.path), "w") as f:
                pass

            source_model = self.db_repository.read_model(source_container.model_id)

            model: Model = Model(created_at=now_time, updated_at=now_time, sequential=source_model.sequential,
                                 code=source_model.code, file_id=model_file_id, was_trained=False)
            if copy_progress:
                model.was_trained = source_model.was_trained
                source_model_file = self.db_repository.read_file(source_model.file_id)

                shutil.copy(os.path.join("local", "storage", source_model_file.path),
                            os.path.join("local", "storage", model_file.path))
            model.model_id = self.db_repository.create_model(model)

            history: History = History(created_at=now_time, updated_at=now_time, model_id=model.model_id,
                                       history_type=MODEL_COPIED, comment=f"from {source_container.model_id}")
            history.history_id = self.db_repository.create_history(history)

            # end
            # copy optimizer

            optimizer_file: File = File(created_at=now_time, updated_at=now_time, file_type="optimizer", comment="",
                                        path="empty")
            optimizer_file_id = self.db_repository.create_file(optimizer_file)
            optimizer_file.file_id = optimizer_file_id
            optimizer_file.path = generate_path(optimizer_file_id, "optimizer", now_time)
            self.db_repository.update_file(optimizer_file)
            with open(os.path.join("local", "storage", optimizer_file.path), "w") as f:
                pass

            source_optimizer = self.db_repository.read_optimizer(source_container.optimizer_id)

            optimizer: Optimizer = Optimizer(created_at=now_time, updated_at=now_time, file_id=optimizer_file_id,
                                             was_trained=False, code=source_optimizer.code)

            if copy_progress:
                optimizer.was_trained = source_optimizer.was_trained
                source_optimizer_file = self.db_repository.read_file(source_optimizer.file_id)

                shutil.copy(os.path.join("local", "storage", source_optimizer_file.path),
                            os.path.join("local", "storage", optimizer_file.path))
            optimizer.optimizer_id = self.db_repository.create_optimizer(optimizer)

            history: History = History(created_at=now_time, updated_at=now_time, optimizer_id=optimizer.optimizer_id,
                                       history_type=OPTIMIZER_COPIED,
                                       comment=f"from {source_container.optimizer_id}")
            history.history_id = self.db_repository.create_history(history)

            # end

            container: Container = Container(created_at=now_time, updated_at=now_time,
                                             dataset_id=source_container.dataset_id, model_id=model.model_id,
                                             optimizer_id=optimizer.optimizer_id,
                                             normalise_dataset=source_container.normalise_dataset)
            container.name = source_container.name + "_"

            container.container_id = self.db_repository.create_container(container)

            history: History = History(created_at=now_time, updated_at=now_time, container_id=container.container_id,
                                       history_type=CONTAINER_COPIED, comment=f"from {source_container.container_id}")
            history.history_id = self.db_repository.create_history(history)

            self.db_repository.commit_transaction()
        except Exception as e:
            self.db_repository.rollback_transaction()

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction()
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Container copying Error: {e}"))
            self.db_repository.commit_transaction()
            print(f"Container copying Error: {e}")
            return None
        return container

    def update_container(self, container_id: int, dataset_id: int, model_id: int, optimizer_id: int,
                         normalise_dataset: bool, name: str, comment: str):
        try:
            self.db_repository.begin_transaction()
            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            container = self.db_repository.read_container(container_id)

            old_container = container.__str__()

            container.updated_at = now_time
            container.dataset_id = dataset_id
            container.model_id = model_id
            container.optimizer_id = optimizer_id
            container.normalise_dataset = normalise_dataset
            container.name = name
            container.comment = comment

            self.db_repository.update_container(container)

            comment = f"{old_container}\n{container.__str__()}"

            self.db_repository.create_history(
                History(created_at=now_time, updated_at=now_time, container_id=container_id,
                        history_type=CONTAINER_UPDATED, comment=comment))
            self.db_repository.commit_transaction()
        except Exception as e:
            self.db_repository.rollback_transaction()

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction()
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Container updating Error: {e}"))
            self.db_repository.commit_transaction()
            print(f"Container updating Error: {e}")
            return None
    # def run_container(self, container_id: int):
