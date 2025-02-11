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


class DB:
    def __init__(self):
        self.conn = sqlite3.connect("local/database.db")
        self.cursor = self.conn.cursor()

    def create_tables(self):
        self.cursor.executescript(migration_query)
        self.conn.commit()

    def create_container(self, dataset_id: int, model_id: int, optimizers_id: int,
                         normalise_dataset: bool, name: str, comment: str):
        now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.cursor.execute(create_container_query,
                            (now_time, now_time, dataset_id, model_id, optimizers_id, normalise_dataset, name, comment))
        new_id = self.cursor.lastrowid

        self.conn.commit()

        return new_id

    def create_file(self, file_type: str, comment: str) -> int:
        now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = f"{now_time}.{file_type}.txt"

        self.cursor.execute(create_file_query, (now_time, now_time, file_type, comment, path))
        new_id = self.cursor.lastrowid

        self.conn.commit()

        return new_id

    def create_model(self, file_id: int, was_trained: bool, sequential: bool, code: str) -> int:
        now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.cursor.execute(create_model_query, (now_time, now_time, file_id, was_trained, sequential, code))
        new_id = self.cursor.lastrowid

        self.conn.commit()

        return new_id

    def create_optimizer(self, file_id: int, was_trained: bool, code: str) -> int:
        now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.cursor.execute(create_optimizer_query, (now_time, now_time, file_id, was_trained, code))
        new_id = self.cursor.lastrowid

        self.conn.commit()

        return new_id

    def create_history(self, container_id: int, history_type: str, comment: str, file_id: int) -> int:
        now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.cursor.execute(create_history_query, (now_time, now_time, container_id, history_type, comment, file_id))
        new_id = self.cursor.lastrowid

        self.conn.commit()

        return new_id

    def update
