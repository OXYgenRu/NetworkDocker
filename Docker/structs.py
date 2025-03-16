class Container:
    def __init__(self, container_id=None, created_at=None, updated_at=None, dataset_id=None, model_id=None,
                 optimizer_id=None, normalise_dataset=None, criterion_code=None, online_training=None, name=None,
                 comment=None):
        self.container_id = container_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.dataset_id = dataset_id
        self.model_id = model_id
        self.optimizer_id = optimizer_id
        self.normalise_dataset = normalise_dataset
        self.criterion_code = criterion_code
        self.online_training = online_training
        self.name = name
        self.comment = comment

    def __str__(self):
        return (
            f"container_id {self.container_id} | updated_at {self.updated_at} | dataset_id {self.dataset_id} | model_id {self.model_id}"
            f" | optimizer_id {self.optimizer_id} | normalise_dataset {self.normalise_dataset} | criterion_code {self.criterion_code} | online_training {self.online_training} | name {self.name} | comment {self.comment}")

    def update_if_provided(self, updated_at=None, dataset_id=None, model_id=None, optimizer_id=None,
                           normalise_dataset=None, criterion_code=None, online_training=None, name=None, comment=None):
        if updated_at is not None:
            self.updated_at = updated_at
        if dataset_id is not None:
            self.dataset_id = dataset_id
        if model_id is not None:
            self.model_id = model_id
        if optimizer_id is not None:
            self.optimizer_id = optimizer_id
        if normalise_dataset is not None:
            self.normalise_dataset = normalise_dataset
        if criterion_code is not None:
            self.criterion_code = criterion_code
        if online_training is not None:
            self.online_training = online_training
        if name is not None:
            self.name = name
        if comment is not None:
            self.comment = comment

    def to_dict(self):
        return self.__dict__


class File:
    def __init__(self, file_id=None, created_at=None, updated_at=None, file_type=None, comment=None, path=None):
        self.file_id = file_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.file_type = file_type
        self.comment = comment
        self.path = path

    def __str__(self):
        return (
            f"file_id {self.file_id} | updated_at {self.updated_at} | file_type {self.file_type} | comment {self.comment}"
            f" | path {self.path}")

    def update_if_provided(self, updated_at=None, file_type=None, comment=None, path=None):
        if updated_at is not None:
            self.updated_at = updated_at
        if file_type is not None:
            self.file_type = file_type
        if comment is not None:
            self.comment = comment
        if path is not None:
            self.path = path

    def to_dict(self):
        return self.__dict__


class Model:
    def __init__(self, model_id=None, created_at=None, updated_at=None, file_id=None, was_trained=None, sequential=None,
                 code=None):
        self.model_id = model_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.file_id = file_id
        self.was_trained = was_trained
        self.sequential = sequential
        self.code = code

    def __str__(self):
        return (
            f"model_id {self.model_id} | updated_at {self.updated_at} | file_id {self.file_id} | was_trained {self.was_trained}"
            f" | sequential {self.sequential} |  code {self.code}")

    def update_if_provided(self, updated_at=None, file_id=None, was_trained=None, sequential=None, code=None):
        if updated_at is not None:
            self.updated_at = updated_at
        if file_id is not None:
            self.file_id = file_id
        if was_trained is not None:
            self.was_trained = was_trained
        if sequential is not None:
            self.sequential = sequential
        if code is not None:
            self.code = code

    def to_dict(self):
        return self.__dict__


class Optimizer:
    def __init__(self, optimizer_id=None, created_at=None, updated_at=None, file_id=None, was_trained=None, code=None):
        self.optimizer_id = optimizer_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.file_id = file_id
        self.was_trained = was_trained
        self.code = code

    def __str__(self):
        return (
            f"optimizer_id {self.optimizer_id} | updated_at {self.updated_at} | file_id {self.file_id} | was_trained {self.was_trained}"
            f" | code {self.code}")

    def update_if_provided(self, updated_at=None, file_id=None, was_trained=None, code=None):
        if updated_at is not None:
            self.updated_at = updated_at
        if file_id is not None:
            self.file_id = file_id
        if was_trained is not None:
            self.was_trained = was_trained
        if code is not None:
            self.code = code

    def to_dict(self):
        return self.__dict__


class History:
    def __init__(self, history_id=None, created_at=None, updated_at=None, container_id=None, model_id=None,
                 optimizer_id=None, history_type=None,
                 comment=None, file_id=None):
        self.history_id = history_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.container_id = container_id
        self.model_id = model_id
        self.optimizer_id = optimizer_id
        self.history_type = history_type
        self.comment = comment
        self.file_id = file_id

    def update_if_provided(self, updated_at=None, container_id=None, model_id=None,
                           optimizer_id=None, history_type=None,
                           comment=None, file_id=None):
        if updated_at is not None:
            self.updated_at = updated_at
        if container_id is not None:
            self.container_id = container_id
        if model_id is not None:
            self.model_id = model_id
        if optimizer_id is not None:
            self.optimizer_id = optimizer_id
        if history_type is not None:
            self.history_type = history_type
        if comment is not None:
            self.comment = comment
        if file_id is not None:
            self.file_id = file_id

    def to_dict(self):
        return self.__dict__


class Session:
    def __init__(self, session_id=None, created_at=None, updated_at=None, container_id=None, status=None, file_id=None,
                 epochs=None,
                 reset_progress=None):
        self.session_id = session_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.container_id = container_id
        self.status = status
        self.file_id = file_id
        self.epochs = epochs
        self.reset_progress = reset_progress

    def __str__(self):
        return (
            f"session_id {self.session_id} | updated_at {self.updated_at} | container_id {self.container_id} | status {self.status} | file_id {self.file_id}"
            f" | epochs {self.epochs} | reset_progress {self.reset_progress}")

    def update_if_provided(self, updated_at=None, container_id=None, status=None, file_id=None, epochs=None,
                           reset_progress=None):

        if updated_at is not None:
            self.updated_at = updated_at
        if container_id is not None:
            self.container_id = container_id
        if status is not None:
            self.status = status
        if file_id is not None:
            self.file_id = file_id
        if epochs is not None:
            self.epochs = epochs
        if reset_progress is not None:
            self.reset_progress = reset_progress

    def to_dict(self):
        return self.__dict__
