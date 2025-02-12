import datetime
import sqlite3

migration_query = """
    CREATE TABLE models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at Text DEFAULT CURRENT_TIMESTAMP,
        updated_at Text DEFAULT CURRENT_TIMESTAMP,
        file_id INTEGER DEFAULT NULL,
        was_trained BLOB DEFAULT FALSE,
        sequential BLOB DEFAULT FALSE,
        code TEXT DEFAULT '',
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE 
    );
    CREATE TABLE optimizers(  
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at Text DEFAULT CURRENT_TIMESTAMP,
        updated_at Text DEFAULT CURRENT_TIMESTAMP,
        file_id INTEGER DEFAULT NULL,
        was_trained BLOB DEFAULT FALSE,
        code TEXT DEFAULT '',
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE 
    );
    CREATE TABLE containers(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at Text DEFAULT CURRENT_TIMESTAMP,
        updated_at Text DEFAULT CURRENT_TIMESTAMP,
        dataset_id INTEGER  DEFAULT NULL,
        model_id INTEGER  DEFAULT NULL,
        optimizers_id INTEGER  DEFAULT NULL,
        normalise_dataset BLOB DEFAULT FALSE,
        name TEXT,
        comment TEXT DEFAULT '',
        FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE,
        FOREIGN KEY (optimizers_id) REFERENCES optimizers(id) ON DELETE CASCADE,
        FOREIGN KEY (dataset_id) REFERENCES files(id) ON DELETE CASCADE
    );
    CREATE TABLE history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        container_id INTEGER DEFAULT NULL,
        type TEXT,
        comment TEXT DEFAULT '',
        file_id INTEGER DEFAULT NULL,
        FOREIGN KEY (container_id) REFERENCES containers(id) ON DELETE CASCADE,
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE 
    );
    CREATE TABLE files(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at Text DEFAULT CURRENT_TIMESTAMP,
        updated_at Text DEFAULT CURRENT_TIMESTAMP,
        type TEXT DEFAULT '',
        comment TEXT DEFAULT '',
        path TEXT DEFAULT ''
    )
"""

create_container_query = """
    INSERT INTO containers (created_at, updated_at, dataset_id, model_id, optimizers_id, normalise_dataset, name, comment)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

create_file_query = """
    INSERT INTO files (created_at, updated_at, type, comment, path) VALUES (?, ?, ?, ?, ?)
"""
create_model_query = """
    INSERT INTO models (created_at, updated_at, file_id, was_trained, sequential, code) VALUES (?, ?, ?, ?, ?, ?)
"""
create_optimizer_query = """
    INSERT INTO optimizers (created_at, updated_at, file_id, was_trained, code) VALUES (?, ?, ?, ?, ?)
"""

create_history_query = """
    INSERT INTO history (created_at, updated_at, container_id, type, comment, file_id) VALUES (?, ?, ?, ?, ?, ?)
"""

update_container_query = """
    UPDATE containers SET updated_at=?,dataset_id=?,model_id=?,optimizers_id=?,normalise_dataset=?,
    name=?,comment=? WHERE id=?
"""

update_file_query = """
    UPDATE files SET updated_at=?,type=?,comment=?,path=? WHERE id=?
"""

update_model_query = """
    UPDATE models SET updated_at=?,file_id=?,was_trained=?,sequential=?,code=? WHERE id=?
"""

update_optimizer_query = """
    UPDATE optimizers SET updated_at=?,file_id=?,was_trained=?,code=? WHERE id = ?
"""

update_history_query = """
    UPDATE history SET updated_at=?,container_id=?,type=?,comment=?,file_id=? WHERE id=?
"""


class Container:
    def __init__(self, container_id=None, created_at=None, updated_at=None, dataset_id=None, model_id=None,
                 optimizers_id=None, normalise_dataset=None, name=None, comment=None):
        self.container_id = container_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.dataset_id = dataset_id
        self.model_id = model_id
        self.optimizers_id = optimizers_id
        self.normalise_dataset = normalise_dataset
        self.name = name
        self.comment = comment


class File:
    def __init__(self, file_id=None, created_at=None, updated_at=None, file_type=None, comment=None, path=None):
        self.file_id = file_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.file_type = file_type
        self.comment = comment
        self.path = path


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


class Optimizer:
    def __init__(self, optimizer_id=None, created_at=None, updated_at=None, file_id=None, was_trained=None, code=None):
        self.optimizer_id = optimizer_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.file_id = file_id
        self.was_trained = was_trained
        self.code = code


class History:
    def __init__(self, history_id=None, created_at=None, updated_at=None, container_id=None, history_type=None,
                 comment=None, file_id=None):
        self.history_id = history_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.container_id = container_id
        self.history_type = history_type
        self.comment = comment
        self.file_id = file_id


class DB:
    def __init__(self):
        self.conn = sqlite3.connect("local/database.db")
        self.cursor = self.conn.cursor()

    def create_tables(self):
        self.cursor.executescript(migration_query)
        self.conn.commit()

    def create_container(self, container: Container):
        # now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.cursor.execute(create_container_query,
                            (container.created_at, container.updated_at, container.dataset_id, container.model_id,
                             container.optimizers_id,
                             container.normalise_dataset, container.name, container.comment))
        new_id = self.cursor.lastrowid

        self.conn.commit()

        return new_id

    def create_file(self, file: File) -> int:
        # now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # path = f"{now_time}.{file.file_type}.txt"

        self.cursor.execute(create_file_query,
                            (file.created_at, file.updated_at, file.file_type, file.comment, file.path))
        new_id = self.cursor.lastrowid

        self.conn.commit()

        return new_id

    def create_model(self, model: Model) -> int:
        # now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.cursor.execute(create_model_query,
                            (model.created_at, model.updated_at, model.file_id, model.was_trained, model.sequential,
                             model.code))
        new_id = self.cursor.lastrowid

        self.conn.commit()

        return new_id

    def create_optimizer(self, optimizer: Optimizer) -> int:
        self.cursor.execute(create_optimizer_query,
                            (optimizer.created_at, optimizer.updated_at, optimizer.file_id, optimizer.was_trained,
                             optimizer.code))
        new_id = self.cursor.lastrowid

        self.conn.commit()

        return new_id

    def create_history(self, history: History) -> int:
        self.cursor.execute(create_history_query,
                            (history.created_at, history.updated_at, history.container_id, history.history_type,
                             history.comment,
                             history.file_id))
        new_id = self.cursor.lastrowid

        self.conn.commit()

        return new_id

    # def update
