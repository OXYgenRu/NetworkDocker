import datetime
from typing import Optional

from database import DB
from structs import Container, Model, Optimizer, History, File, Session
import shutil
import os


def generate_path(file_id, file_type, created_at):
    return f"{file_id}.{created_at}.{file_type}.pth"


MODEL_CREATED = "model_created"
OPTIMIZER_CREATED = "optimizer_created"
CONTAINER_CREATED = "container_created"
FILE_CREATED = "file_created"
SESSION_CREATED = "session_created"
HISTORY_CREATED = "history_created"

MODEL_COPIED = "model_copied"
OPTIMIZER_COPIED = "optimizer_copied"
CONTAINER_COPIED = "container_copied"

CONTAINER_UPDATED = "container_updated"
MODEL_UPDATED = "model_updated"
OPTIMIZER_UPDATED = "optimizer_updated"
SESSION_UPDATED = "session_updated"
FILE_UPDATED = "file_updated"

CONTAINER_READ = "container_read"
FILE_READ = "file_read"
MODEL_READ = "model_read"
OPTIMIZER_READ = "optimizer_read"
HISTORY_READ = "history_read"
SESSION_READ = "session_read"
CONTAINERS_READ = "containers_read"
MODELS_READ = "models_read"
OPTIMIZERS_READ = "optimizers_read"
FILES_READ = "files_read"
HISTORIES_READ = "histories_read"
SESSIONS_READ = "sessions_read"

ERROR = "error"


class UseCase:
    def __init__(self, db_repository: DB):
        self.db_repository = db_repository

    def create_tables(self):
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            self.db_repository.create_tables(conn)
            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Tables Creating Error: {e}"), conn)
            self.db_repository.commit_transaction(conn)
            print(f"Tables Creating Error: {e}")
            return None
        return True

    def create_model(self, sequential: bool = None, code: str = None) -> Optional[Model]:
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            model_file = File(created_at=now_time, updated_at=now_time, file_type="model", comment="", path="empty")
            model_file_id = self.db_repository.create_file(model_file, conn)
            model_file.file_id = model_file_id
            model_file.path = generate_path(model_file_id, "model", now_time)
            self.db_repository.update_file(model_file, conn)

            with open(os.path.join("local", "storage", model_file.path), "w") as f:
                pass

            model: Model = Model(created_at=now_time, updated_at=now_time, file_id=model_file_id, was_trained=False,
                                 sequential=sequential, code=code)

            model.model_id = self.db_repository.create_model(model, conn)

            history: History = History(created_at=now_time, updated_at=now_time, model_id=model.model_id,
                                       history_type=MODEL_CREATED)
            history.history_id = self.db_repository.create_history(history, conn)

            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Model Creating Error: {e}"), conn)
            self.db_repository.commit_transaction(conn)
            print(f"Model Creating Error: {e}")
            return None
        return model

    def create_file(self, file_type: str = None, comment: str = None) -> Optional[File]:
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            file: File = File(created_at=now_time, updated_at=now_time, file_type=file_type, comment=comment,
                              path="empty")
            file_id = self.db_repository.create_file(file, conn)
            file.file_id = file_id
            file.path = generate_path(file_id, file_type, now_time)
            self.db_repository.update_file(file, conn)
            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"File creating Error: {e}"), conn)
            self.db_repository.commit_transaction(conn)
            print(f"File creating Error: {e}")
            return None

        return file

    def create_optimizer(self, code: str = None) -> Optional[Optimizer]:
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            optimizer_file: File = File(created_at=now_time, updated_at=now_time, file_type="optimizer", comment="",
                                        path="empty")
            optimizer_file_id = self.db_repository.create_file(optimizer_file, conn)
            optimizer_file.file_id = optimizer_file_id
            optimizer_file.path = generate_path(optimizer_file_id, "optimizer", now_time)
            self.db_repository.update_file(optimizer_file, conn)
            with open(os.path.join("local", "storage", optimizer_file.path), "w") as f:
                pass

            optimizer: Optimizer = Optimizer(created_at=now_time, updated_at=now_time, file_id=optimizer_file_id,
                                             was_trained=False, code=code)

            optimizer.optimizer_id = self.db_repository.create_optimizer(optimizer, conn)

            history: History = History(created_at=now_time, updated_at=now_time, optimizer_id=optimizer.optimizer_id,
                                       history_type=OPTIMIZER_CREATED)
            history.history_id = self.db_repository.create_history(history, conn)

            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Optimizer creating Error: {e}"), conn)
            self.db_repository.commit_transaction(conn)
            print(f"Optimizer creating Error: {e}")
            return None

        return optimizer

    def create_container(self, dataset_id: int = None, model_id: int = None, optimizer_id: int = None,
                         normalise_dataset: bool = None, criterion_code: str = None, online_training: bool = None,
                         name: str = None,
                         comment: str = None) -> Optional[Container]:
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if self.db_repository.check_model_id(model_id, conn):
                raise ValueError("model_id already used by another container")
            if self.db_repository.check_optimizer_id(optimizer_id, conn):
                raise ValueError("optimizer_id already used by another container")

            container: Container = Container(created_at=now_time, updated_at=now_time, dataset_id=dataset_id,
                                             model_id=model_id, optimizer_id=optimizer_id,
                                             normalise_dataset=normalise_dataset, criterion_code=criterion_code,
                                             online_training=online_training, name=name, comment=comment)
            container.container_id = self.db_repository.create_container(container, conn)

            history: History = History(created_at=now_time, updated_at=now_time, container_id=container.container_id,
                                       history_type=CONTAINER_CREATED)
            history.history_id = self.db_repository.create_history(history, conn)

            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Container creating Error: {e}"), conn)
            self.db_repository.commit_transaction(conn)
            print(f"Container creating Error: {e}")
            return None
        return container

    def create_history(self, container_id=None, model_id=None, optimizer_id=None, history_type=None, comment=None,
                       file_id=None) -> Optional[History]:
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            history: History = History(created_at=now_time, updated_at=now_time, container_id=container_id,
                                       model_id=model_id, optimizer_id=optimizer_id, history_type=history_type,
                                       comment=comment, file_id=file_id)
            history.history_id = self.db_repository.create_history(history, conn)
            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"History creating Error: {e}"), conn)
            self.db_repository.commit_transaction(conn)
            print(f"History creating Error: {e}")
            return None
        return history

    def create_session(self, container_id=None, status=None, file_id=None, epochs=None, reset_progress=None) -> \
            Optional[Session]:
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            session: Session = Session(created_at=now_time, updated_at=now_time, status=status, file_id=file_id,
                                       epochs=epochs, reset_progress=reset_progress, container_id=container_id)
            session.session_id = self.db_repository.create_session(session, conn)
            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.db_repository.begin_transaction(conn)

            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Session creating Error: {e}"), conn)
            self.db_repository.commit_transaction(conn)
            print(f"Session creating Error: {e}")
            return None
        return session

    def copy_model(self, source_model_id: int, copy_progress: bool) -> Optional[Model]:
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            model_file = File(created_at=now_time, updated_at=now_time, file_type="model", comment="", path="empty")
            model_file_id = self.db_repository.create_file(model_file, conn)
            model_file.file_id = model_file_id
            model_file.path = generate_path(model_file_id, "model", now_time)
            self.db_repository.update_file(model_file, conn)
            with open(os.path.join("local", "storage", model_file.path), "w") as f:
                pass

            source_model = self.db_repository.read_model(source_model_id, conn)

            model: Model = Model(created_at=now_time, updated_at=now_time, sequential=source_model.sequential,
                                 code=source_model.code, file_id=model_file_id, was_trained=False)
            if copy_progress:
                model.was_trained = source_model.was_trained
                source_model_file = self.db_repository.read_file(source_model.file_id, conn)

                shutil.copy(os.path.join("local", "storage", source_model_file.path),
                            os.path.join("local", "storage", model_file.path))
            model.model_id = self.db_repository.create_model(model, conn)

            history: History = History(created_at=now_time, updated_at=now_time, model_id=model.model_id,
                                       history_type=MODEL_COPIED, comment=f"from {source_model_id}")
            history.history_id = self.db_repository.create_history(history, conn)

            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Model copying Error: {e}", model_id=source_model_id),
                                              conn)
            self.db_repository.commit_transaction(conn)
            print(f"Model copying Error: {e}")
            return None
        return model

    def copy_optimizer(self, source_optimizer_id: int, copy_progress: bool) -> Optional[Optimizer]:
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            optimizer_file: File = File(created_at=now_time, updated_at=now_time, file_type="optimizer", comment="",
                                        path="empty")
            optimizer_file_id = self.db_repository.create_file(optimizer_file, conn)
            optimizer_file.file_id = optimizer_file_id
            optimizer_file.path = generate_path(optimizer_file_id, "optimizer", now_time)
            self.db_repository.update_file(optimizer_file, conn)
            with open(os.path.join("local", "storage", optimizer_file.path), "w") as f:
                pass

            source_optimizer = self.db_repository.read_optimizer(source_optimizer_id, conn)

            optimizer: Optimizer = Optimizer(created_at=now_time, updated_at=now_time, file_id=optimizer_file_id,
                                             was_trained=False, code=source_optimizer.code)

            if copy_progress:
                optimizer.was_trained = source_optimizer.was_trained
                source_optimizer_file = self.db_repository.read_file(source_optimizer.file_id, conn)

                shutil.copy(os.path.join("local", "storage", source_optimizer_file.path),
                            os.path.join("local", "storage", optimizer_file.path))
            optimizer.optimizer_id = self.db_repository.create_optimizer(optimizer, conn)

            history: History = History(created_at=now_time, updated_at=now_time, optimizer_id=optimizer.optimizer_id,
                                       history_type=OPTIMIZER_COPIED,
                                       comment=f"from {source_optimizer_id}")
            history.history_id = self.db_repository.create_history(history, conn)

            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Optimizer copying Error: {e}",
                                                      optimizer_id=source_optimizer_id), conn)
            self.db_repository.commit_transaction(conn)
            print(f"Optimizer copying Error: {e}")
            return None
        return optimizer

    def copy_container(self, source_container_id: int, copy_progress: bool) -> Optional[Container]:
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            source_container = self.db_repository.read_container(source_container_id, conn)

            # copy model

            model_file = File(created_at=now_time, updated_at=now_time, file_type="model", comment="", path="empty")
            model_file_id = self.db_repository.create_file(model_file, conn)
            model_file.file_id = model_file_id
            model_file.path = generate_path(model_file_id, "model", now_time)
            self.db_repository.update_file(model_file, conn)
            with open(os.path.join("local", "storage", model_file.path), "w") as f:
                pass

            source_model = self.db_repository.read_model(source_container.model_id, conn)

            model: Model = Model(created_at=now_time, updated_at=now_time, sequential=source_model.sequential,
                                 code=source_model.code, file_id=model_file_id, was_trained=False)
            if copy_progress:
                model.was_trained = source_model.was_trained
                source_model_file = self.db_repository.read_file(source_model.file_id, conn)

                shutil.copy(os.path.join("local", "storage", source_model_file.path),
                            os.path.join("local", "storage", model_file.path))
            model.model_id = self.db_repository.create_model(model, conn)

            history: History = History(created_at=now_time, updated_at=now_time, model_id=model.model_id,
                                       history_type=MODEL_COPIED, comment=f"from {source_container.model_id}")
            history.history_id = self.db_repository.create_history(history, conn)

            # end
            # copy optimizer

            optimizer_file: File = File(created_at=now_time, updated_at=now_time, file_type="optimizer", comment="",
                                        path="empty")
            optimizer_file_id = self.db_repository.create_file(optimizer_file, conn)
            optimizer_file.file_id = optimizer_file_id
            optimizer_file.path = generate_path(optimizer_file_id, "optimizer", now_time)
            self.db_repository.update_file(optimizer_file, conn)
            with open(os.path.join("local", "storage", optimizer_file.path), "w") as f:
                pass

            source_optimizer = self.db_repository.read_optimizer(source_container.optimizer_id, conn)

            optimizer: Optimizer = Optimizer(created_at=now_time, updated_at=now_time, file_id=optimizer_file_id,
                                             was_trained=False, code=source_optimizer.code)

            if copy_progress:
                optimizer.was_trained = source_optimizer.was_trained
                source_optimizer_file = self.db_repository.read_file(source_optimizer.file_id, conn)

                shutil.copy(os.path.join("local", "storage", source_optimizer_file.path),
                            os.path.join("local", "storage", optimizer_file.path))
            optimizer.optimizer_id = self.db_repository.create_optimizer(optimizer, conn)

            history: History = History(created_at=now_time, updated_at=now_time, optimizer_id=optimizer.optimizer_id,
                                       history_type=OPTIMIZER_COPIED,
                                       comment=f"from {source_container.optimizer_id}")
            history.history_id = self.db_repository.create_history(history, conn)

            # end

            container: Container = Container(created_at=now_time, updated_at=now_time,
                                             dataset_id=source_container.dataset_id, model_id=model.model_id,
                                             optimizer_id=optimizer.optimizer_id,
                                             normalise_dataset=source_container.normalise_dataset,
                                             criterion_code=source_container.criterion_code,
                                             online_training=source_container.online_training)
            container.name = source_container.name + "_"

            container.container_id = self.db_repository.create_container(container, conn)

            history: History = History(created_at=now_time, updated_at=now_time, container_id=container.container_id,
                                       history_type=CONTAINER_COPIED, comment=f"from {source_container.container_id}")
            history.history_id = self.db_repository.create_history(history, conn)

            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Container copying Error: {e}",
                                                      container_id=source_container_id), conn)
            self.db_repository.commit_transaction(conn)
            print(f"Container copying Error: {e}")
            return None
        return container

    def update_model(self, model_id: int, file_id: int = None, was_trained: bool = None, sequential: bool = None,
                     code: str = None):
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            model = self.db_repository.read_model(model_id, conn)

            old_model_str = model.__str__()

            model.update_if_provided(now_time, file_id, was_trained, sequential, code)

            self.db_repository.update_model(model, conn)

            comment = f"{old_model_str}\n{model.__str__()}"

            self.db_repository.create_history(
                History(created_at=now_time, updated_at=now_time, model_id=model_id, history_type=MODEL_UPDATED,
                        comment=comment), conn)

            self.db_repository.commit_transaction(conn)

        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Model updating Error: {e}", model_id=model_id), conn)
            self.db_repository.commit_transaction(conn)
            print(f"Model updating Error: {e}")
            return None
        return model

    def update_optimizer(self, optimizer_id: int, file_id: int = None, was_trained: bool = None, code: str = None):
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            optimizer = self.db_repository.read_optimizer(optimizer_id, conn)

            old_optimizer_str = optimizer.__str__()

            optimizer.update_if_provided(now_time, file_id, was_trained, code)

            self.db_repository.update_optimizer(optimizer, conn)

            comment = f"{old_optimizer_str}\n{optimizer.__str__()}"

            self.db_repository.create_history(
                History(created_at=now_time, updated_at=now_time, optimizer_id=optimizer_id,
                        history_type=OPTIMIZER_UPDATED,
                        comment=comment), conn)

            self.db_repository.commit_transaction(conn)

        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Optimizer updating Error: {e}",
                                                      optimizer_id=optimizer_id), conn)
            self.db_repository.commit_transaction(conn)
            print(f"Optimizer updating Error: {e}")
            return None
        return optimizer

    def update_container(self, container_id: int, dataset_id: int = None, model_id: int = None,
                         optimizer_id: int = None,
                         normalise_dataset: bool = None, criterion_code: str = None, online_training: bool = None,
                         name: str = None, comment: str = None):
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            container = self.db_repository.read_container(container_id, conn)

            old_container_str = container.__str__()

            container.update_if_provided(now_time, dataset_id, model_id, optimizer_id, normalise_dataset,
                                         criterion_code, online_training, name, comment)

            self.db_repository.update_container(container, conn)

            comment = f"{old_container_str}\n{container.__str__()}"

            self.db_repository.create_history(
                History(created_at=now_time, updated_at=now_time, container_id=container_id,
                        history_type=CONTAINER_UPDATED, comment=comment), conn)
            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Container updating Error: {e}",
                                                      container_id=container_id), conn)
            self.db_repository.commit_transaction(conn)
            print(f"Container updating Error: {e}")
            return None
        return container

    def update_file(self, file_id: int, file_type: str, comment: str, path: str):
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            file = self.db_repository.read_file(file_id, conn)

            old_file_str = file.__str__()

            file.update_if_provided(updated_at=now_time, file_type=file_type, comment=comment, path=path)

            self.db_repository.update_file(file, conn)

            comment = f"{old_file_str}\n{file.__str__()}"

            self.db_repository.create_history(
                History(created_at=now_time, updated_at=now_time,
                        history_type=FILE_UPDATED, comment=comment), conn)
            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"File updating Error: {e}"), conn)
            self.db_repository.commit_transaction(conn)
            print(f"File updating Error: {e}")
            return None
        return file

    def update_session(self, session_id: int, status: str, file_id: int, epochs: int, reset_progress: bool,
                       container_id: int):
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            session = self.db_repository.read_session(session_id, conn)

            old_session_str = session.__str__()
            # print(session, "aa")
            session.update_if_provided(container_id=container_id, updated_at=now_time, status=status, file_id=file_id,
                                       epochs=epochs,
                                       reset_progress=reset_progress)
            # print(session, "bb")
            self.db_repository.update_session(session, conn)

            comment = f"{old_session_str}\n{session.__str__()}"

            self.db_repository.create_history(
                History(created_at=now_time, updated_at=now_time,
                        history_type=SESSION_UPDATED, comment=comment), conn)
            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Session updating Error: {e}"), conn)
            self.db_repository.commit_transaction(conn)
            print(f"Session updating Error: {e}")
            return None
        return session

    def read_container(self, container_id: int) -> Optional[Container]:
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            container = self.db_repository.read_container(container_id, conn)
            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Container reading Error: {e}",
                                                      container_id=container_id), conn)
            self.db_repository.commit_transaction(conn)
            print(f"Container reading Error: {e}")
            return None
        return container

    def read_file(self, file_id: int) -> Optional[File]:
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            file = self.db_repository.read_file(file_id, conn)
            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"File reading Error: {e}"), conn)
            self.db_repository.commit_transaction(conn)
            print(f"File reading Error: {e}")
            return None
        return file

    def read_model(self, model_id: int) -> Optional[Model]:
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            model = self.db_repository.read_model(model_id, conn)
            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Model reading Error: {e}", model_id=model_id), conn)
            self.db_repository.commit_transaction(conn)
            print(f"Model reading Error: {e}")
            return None
        return model

    def read_optimizer(self, optimizer_id: int) -> Optional[Optimizer]:
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            optimizer = self.db_repository.read_optimizer(optimizer_id, conn)
            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Optimizer reading Error: {e}",
                                                      optimizer_id=optimizer_id), conn)
            self.db_repository.commit_transaction(conn)
            print(f"Optimizer reading Error: {e}")
            return None
        return optimizer

    def read_history(self, history_id: int) -> Optional[History]:
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            history = self.db_repository.read_history(history_id, conn)
            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"History reading Error: {e}", ), conn)
            self.db_repository.commit_transaction(conn)
            print(f"History reading Error: {e}")
            return None
        return history

    def read_session(self, session_id: int) -> Optional[Session]:
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            session = self.db_repository.read_session(session_id, conn)
            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Session reading Error: {e}", ), conn)
            self.db_repository.commit_transaction(conn)
            print(f"Session reading Error: {e}")
            return None
        return session

    def read_containers(self):
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            containers = self.db_repository.read_containers(conn)
            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Containers reading Error: {e}"), conn)
            self.db_repository.commit_transaction(conn)
            print(f"Containers reading Error: {e}")
            return None
        return containers

    def read_models(self):
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            models = self.db_repository.read_models(conn)
            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Models reading Error: {e}"), conn)
            self.db_repository.commit_transaction(conn)
            print(f"Models reading Error: {e}")
            return None
        return models

    def read_optimizers(self):
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            optimizers = self.db_repository.read_optimizers(conn)
            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Optimizers reading Error: {e}"), conn)
            self.db_repository.commit_transaction(conn)
            print(f"Optimizers reading Error: {e}")
            return None
        return optimizers

    def read_files(self):
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            files = self.db_repository.read_files(conn)
            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Files reading Error: {e}"), conn)
            self.db_repository.commit_transaction(conn)
            print(f"Files reading Error: {e}")
            return None
        return files

    def read_histories(self):
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            histories = self.db_repository.read_histories(conn)
            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Histories reading Error: {e}"), conn)
            self.db_repository.commit_transaction(conn)
            print(f"Histories reading Error: {e}")
            return None
        return histories

    def read_sessions(self):
        conn = self.db_repository.get_connection()
        try:
            self.db_repository.begin_transaction(conn)
            sessions = self.db_repository.read_sessions(conn)
            self.db_repository.commit_transaction(conn)
        except Exception as e:
            self.db_repository.rollback_transaction(conn)

            now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.db_repository.begin_transaction(conn)
            self.db_repository.create_history(History(created_at=now_time, updated_at=now_time, history_type=ERROR,
                                                      comment=f"Sessions reading Error: {e}"), conn)
            self.db_repository.commit_transaction(conn)
            print(f"Sessions reading Error: {e}")
            return None
        return sessions
# def run_container(self, container_id: int):
