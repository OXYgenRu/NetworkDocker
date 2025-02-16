import datetime
import sqlite3

migration_query = """
    CREATE TABLE models (
        container_id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at Text DEFAULT CURRENT_TIMESTAMP,
        updated_at Text DEFAULT CURRENT_TIMESTAMP,
        file_id INTEGER DEFAULT NULL,
        was_trained BLOB DEFAULT FALSE,
        sequential BLOB DEFAULT FALSE,
        code TEXT DEFAULT '',
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE 
    );
    CREATE TABLE optimizers(  
        optimizer_id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at Text DEFAULT CURRENT_TIMESTAMP,
        updated_at Text DEFAULT CURRENT_TIMESTAMP,
        file_id INTEGER DEFAULT NULL,
        was_trained BLOB DEFAULT FALSE,
        code TEXT DEFAULT '',
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE 
    );
    CREATE TABLE containers(
        container_id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at Text DEFAULT CURRENT_TIMESTAMP,
        updated_at Text DEFAULT CURRENT_TIMESTAMP,
        dataset_id INTEGER  DEFAULT NULL,
        model_id INTEGER  DEFAULT NULL,
        optimizer_id INTEGER  DEFAULT NULL,
        normalise_dataset BLOB DEFAULT FALSE,
        name TEXT,
        comment TEXT DEFAULT '',
        FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE,
        FOREIGN KEY (optimizer_id) REFERENCES optimizers(id) ON DELETE CASCADE,
        FOREIGN KEY (dataset_id) REFERENCES files(id) ON DELETE CASCADE
    );
    CREATE TABLE history(
        history_id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        container_id INTEGER DEFAULT NULL,
        model_id INTEGER DEFAULT NULL,
        optimizer_id INTEGER DEFAULT NULL,
        history_type TEXT,
        comment TEXT DEFAULT '',
        file_id INTEGER DEFAULT NULL,
        FOREIGN KEY (container_id) REFERENCES containers(id) ON DELETE CASCADE,
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE 
    );
    CREATE TABLE files(
        file_id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at Text DEFAULT CURRENT_TIMESTAMP,
        updated_at Text DEFAULT CURRENT_TIMESTAMP,
        file_type TEXT DEFAULT '',
        comment TEXT DEFAULT '',
        path TEXT DEFAULT ''
    )
"""

create_container_query = """
    INSERT INTO containers (created_at, updated_at, dataset_id, model_id, optimizer_id, normalise_dataset, name, comment)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

create_file_query = """
    INSERT INTO files (created_at, updated_at, file_type, comment, path) VALUES (?, ?, ?, ?, ?)
"""
create_model_query = """
    INSERT INTO models (created_at, updated_at, file_id, was_trained, sequential, code) VALUES (?, ?, ?, ?, ?, ?)
"""
create_optimizer_query = """
    INSERT INTO optimizers (created_at, updated_at, file_id, was_trained, code) VALUES (?, ?, ?, ?, ?)
"""

create_history_query = """
    INSERT INTO history (created_at, updated_at, container_id,model_id,optimizer_id, history_type, comment, file_id) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

update_container_query = """
    UPDATE containers SET updated_at=?,dataset_id=?,model_id=?,optimizer_id=?,normalise_dataset=?,
    name=?,comment=? WHERE container_id=?
"""

update_file_query = """
    UPDATE files SET updated_at=?,file_type=?,comment=?,path=? WHERE file_id=?
"""

update_model_query = """
    UPDATE models SET updated_at=?,file_id=?,was_trained=?,sequential=?,code=? WHERE model_id=?
"""

update_optimizer_query = """
    UPDATE optimizers SET updated_at=?,file_id=?,was_trained=?,code=? WHERE optimizer_id = ?
"""

update_history_query = """
    UPDATE history SET updated_at=?,container_id=?,model_id=?,optimizer_id=?,type=?,comment=?,file_id=? WHERE history_id=?
"""

read_container_query = """
    SELECT * FROM containers WHERE container_id=?
"""

read_container_query = """
    SELECT * FROM files WHERE file_id=?
"""

read_container_query = """
    SELECT * FROM models WHERE model_id=?
"""

read_container_query = """
    SELECT * FROM optimizers WHERE optimizer_id=?
"""

read_container_query = """
    SELECT * FROM history WHERE history_id=?
"""

check_model_id_query = """
    SELECT * FROM containers WHERE model_id=?
"""

check_optimizer_id_query = """
    SELECT * FROM containers WHERE optimizer_id=?
"""


class Container:
    def __init__(self, container_id=None, created_at=None, updated_at=None, dataset_id=None, model_id=None,
                 optimizer_id=None, normalise_dataset=None, name=None, comment=None):
        self.container_id = container_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.dataset_id = dataset_id
        self.model_id = model_id
        self.optimizer_id = optimizer_id
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
                             container.optimizer_id,
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
                            (history.created_at, history.updated_at, history.container_id, history.model_id,
                             history.optimizer_id, history.history_type,
                             history.comment,
                             history.file_id))
        new_id = self.cursor.lastrowid

        self.conn.commit()

        return new_id

    def update_container(self, container: Container):
        self.cursor.execute(update_container_query, (container.updated_at, container.dataset_id, container.model_id,
                                                     container.optimizer_id, container.normalise_dataset,
                                                     container.name, container.comment))
        self.conn.commit()

    def update_file(self, file: File):
        self.cursor.execute(update_file_query,
                            (file.updated_at, file.file_type, file.comment, file.path, file.file_id))

        self.conn.commit()

    def update_model(self, model: Model):
        self.cursor.execute(update_model_query,
                            (model.updated_at, model.file_id, model.was_trained, model.sequential, model.code,
                             model.model_id))
        self.conn.commit()

    def update_optimizer(self, optimizer: Optimizer):
        self.cursor.execute(update_optimizer_query,
                            (optimizer.updated_at, optimizer.file_id, optimizer.was_trained, optimizer.code,
                             optimizer.optimizer_id))
        self.conn.commit()

    def update_history(self, history: History):
        self.cursor.execute(update_history_query, (
            history.updated_at, history.container_id, history.model_id, history.optimizer_id, history.history_type,
            history.comment, history.file_id,
            history.history_id))

    def read_container(self, container_id) -> Container:
        self.cursor.execute(read_container_query, (container_id,))
        query_result = self.cursor.fetchone()

        container = Container(**query_result)

        return container

    def read_file(self, file_id) -> File:
        self.cursor.execute(read_container_query, (file_id,))
        query_result = self.cursor.fetchone()

        file = File(**query_result)

        return file

    def read_model(self, model_id) -> Model:
        self.cursor.execute(read_container_query, (model_id,))
        query_result = self.cursor.fetchone()

        model = Model(**query_result)

        return model

    def read_optimizer(self, optimizer_id) -> Optimizer:
        self.cursor.execute(read_container_query, (optimizer_id,))
        query_result = self.cursor.fetchone()

        optimizer = Optimizer(**query_result)

        return optimizer

    def read_history(self, history_id) -> History:
        self.cursor.execute(read_container_query, (history_id,))
        query_result = self.cursor.fetchone()

        history = History(**query_result)

        return history

    def check_model_id(self, model_id):
        self.cursor.execute(check_model_id_query, (model_id,))
        query_result = self.cursor.fetchall()

        return len(query_result) > 0

    def check_optimizer_id(self, optimizer_id):
        self.cursor.execute(check_optimizer_id_query, (optimizer_id,))
        query_result = self.cursor.fetchall()

        return len(query_result) > 0
