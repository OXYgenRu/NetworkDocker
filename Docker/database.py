import datetime
import sqlite3
from flask import g
from structs import Container, Model, Optimizer, History, File, Session

migration_query = """
    CREATE TABLE models (
        model_id INTEGER PRIMARY KEY AUTOINCREMENT,
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
        criterion_code TEXT,
        online_training BLOB DEFAULT FALSE,
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
    );
    CREATE TABLE sessions(
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        container_id INTEGER DEFAULT NULL,
        status TEXT DEFAULT '',
        file_id INTEGER DEFAULT NULL,
        epochs INTEGER DEFAULT NULL,
        reset_progress BLOB DEFAULT FALSE,
        FOREIGN KEY (container_id) REFERENCES containers(id) ON DELETE NO ACTION
    )
"""

create_container_query = """
    INSERT INTO containers (created_at, updated_at, dataset_id, model_id, optimizer_id, normalise_dataset,criterion_code,online_training, name, comment)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
create_session_query = """
    INSERT INTO sessions (created_at,updated_at,container_id,status,file_id,epochs,reset_progress) 
    VALUES (?,?,?,?,?,?,?)
"""

update_container_query = """
    UPDATE containers SET updated_at=?,dataset_id=?,model_id=?,optimizer_id=?,normalise_dataset=?,criterion_code=?,online_training=?,
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

update_sessions_query = """
    UPDATE sessions SET updated_at=?,container_id=?,status=?,file_id=?,epochs=?,reset_progress=? WHERE session_id=?
"""

read_container_query = """
    SELECT * FROM containers WHERE container_id=?
"""

read_file_query = """
    SELECT * FROM files WHERE file_id=?
"""

read_model_query = """
    SELECT * FROM models WHERE model_id=?
"""

read_optimizer_query = """
    SELECT * FROM optimizers WHERE optimizer_id=?
"""

read_history_query = """
    SELECT * FROM history WHERE history_id=?
"""
read_session_query = """
    SELECT * FROM sessions WHERE session_id=?
"""

check_model_id_query = """
    SELECT * FROM containers WHERE model_id=?
"""

check_optimizer_id_query = """
    SELECT * FROM containers WHERE optimizer_id=?
"""

read_models_query = """
    SELECT * FROM models
"""

read_optimizers_query = """
    SELECT * FROM optimizers
"""

read_containers_query = """
    SELECT * FROM containers
"""

read_files_query = """
    SELECT * FROM files
"""

read_histories_query = """
    SELECT * FROM history
"""

read_sessions_query = """
    SELECT * FROM sessions
"""


class DB:
    def __init__(self):
        self.conn = sqlite3.connect("local/database.db")

    def get_connection(self):
        if 'db_conn' not in g:
            g.db_conn = sqlite3.connect("local/database.db")
        return g.db_conn

    def close_connection(self):
        conn = g.pop('db_conn', None)
        if conn:
            conn.close()

    def create_tables(self, conn):
        cursor = conn.cursor()
        cursor.executescript(migration_query)

    def begin_transaction(self, conn):
        conn.execute("BEGIN;")

    def commit_transaction(self, conn):
        conn.commit()

    def rollback_transaction(self, conn):
        conn.rollback()

    def create_container(self, container: Container, conn):

        cursor = conn.cursor()
        cursor.execute(create_container_query,
                       (container.created_at, container.updated_at, container.dataset_id, container.model_id,
                        container.optimizer_id,
                        container.normalise_dataset, container.criterion_code, container.online_training,
                        container.name, container.comment))
        new_id = cursor.lastrowid

        return new_id

    def create_file(self, file: File, conn) -> int:
        cursor = conn.cursor()
        # now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # path = f"{now_time}.{file.file_type}.txt"

        cursor.execute(create_file_query,
                       (file.created_at, file.updated_at, file.file_type, file.comment, file.path))
        new_id = cursor.lastrowid

        return new_id

    def create_model(self, model: Model, conn) -> int:
        cursor = conn.cursor()
        # now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        cursor.execute(create_model_query,
                       (model.created_at, model.updated_at, model.file_id, model.was_trained, model.sequential,
                        model.code))
        new_id = cursor.lastrowid

        return new_id

    def create_optimizer(self, optimizer: Optimizer, conn) -> int:
        cursor = conn.cursor()
        cursor.execute(create_optimizer_query,
                       (optimizer.created_at, optimizer.updated_at, optimizer.file_id, optimizer.was_trained,
                        optimizer.code))
        new_id = cursor.lastrowid

        return new_id

    def create_history(self, history: History, conn) -> int:
        cursor = conn.cursor()
        cursor.execute(create_history_query,
                       (history.created_at, history.updated_at, history.container_id, history.model_id,
                        history.optimizer_id, history.history_type,
                        history.comment,
                        history.file_id))
        new_id = cursor.lastrowid

        return new_id

    def create_session(self, session: Session, conn) -> int:
        # print(session.to_dict())
        cursor = conn.cursor()
        cursor.execute(create_session_query,
                       (session.created_at, session.updated_at, session.container_id, session.status, session.file_id,
                        session.epochs, session.reset_progress))
        new_id = cursor.lastrowid
        return new_id

    def update_container(self, container: Container, conn):
        cursor = conn.cursor()
        cursor.execute(update_container_query, (container.updated_at, container.dataset_id, container.model_id,
                                                container.optimizer_id, container.normalise_dataset,
                                                container.criterion_code, container.online_training,
                                                container.name, container.comment, container.container_id))

    def update_file(self, file: File, conn):
        cursor = conn.cursor()
        cursor.execute(update_file_query,
                       (file.updated_at, file.file_type, file.comment, file.path, file.file_id))

    def update_model(self, model: Model, conn):
        cursor = conn.cursor()
        cursor.execute(update_model_query,
                       (model.updated_at, model.file_id, model.was_trained, model.sequential, model.code,
                        model.model_id))

    def update_optimizer(self, optimizer: Optimizer, conn):
        cursor = conn.cursor()
        cursor.execute(update_optimizer_query,
                       (optimizer.updated_at, optimizer.file_id, optimizer.was_trained, optimizer.code,
                        optimizer.optimizer_id))

    def update_history(self, history: History, conn):
        cursor = conn.cursor()
        cursor.execute(update_history_query, (
            history.updated_at, history.container_id, history.model_id, history.optimizer_id, history.history_type,
            history.comment, history.file_id,
            history.history_id))

    def update_session(self, session: Session, conn):
        cursor = conn.cursor()
        cursor.execute(update_sessions_query,
                       (session.updated_at, session.container_id, session.status, session.file_id, session.epochs,
                        session.reset_progress,
                        session.session_id))

    def read_container(self, container_id, conn) -> Container:
        cursor = conn.cursor()
        cursor.execute(read_container_query, (container_id,))
        query_result = cursor.fetchone()
        container = Container(container_id=query_result[0], created_at=query_result[1], updated_at=query_result[2],
                              dataset_id=query_result[3], model_id=query_result[4], optimizer_id=query_result[5],
                              normalise_dataset=query_result[6], criterion_code=query_result[7],
                              online_training=query_result[8], name=query_result[9], comment=query_result[10])

        return container

    def read_file(self, file_id, conn) -> File:
        cursor = conn.cursor()
        cursor.execute(read_file_query, (file_id,))
        query_result = cursor.fetchone()

        file = File(file_id=query_result[0], created_at=query_result[1], updated_at=query_result[2],
                    file_type=query_result[3], comment=query_result[4], path=query_result[5])

        return file

    def read_model(self, model_id, conn) -> Model:
        cursor = conn.cursor()
        cursor.execute(read_model_query, (model_id,))
        query_result = cursor.fetchone()

        model = Model(model_id=query_result[0], created_at=query_result[1], updated_at=query_result[2],
                      file_id=query_result[3], was_trained=query_result[4], sequential=query_result[5],
                      code=query_result[6])

        return model

    def read_optimizer(self, optimizer_id, conn) -> Optimizer:
        cursor = conn.cursor()
        cursor.execute(read_optimizer_query, (optimizer_id,))
        query_result = cursor.fetchone()

        optimizer = Optimizer(optimizer_id=query_result[0], created_at=query_result[1], updated_at=query_result[2],
                              file_id=query_result[3], was_trained=query_result[4], code=query_result[5])

        return optimizer

    def read_history(self, history_id, conn) -> History:
        cursor = conn.cursor()
        cursor.execute(read_history_query, (history_id,))
        query_result = cursor.fetchone()

        history = History(history_id=query_result[0], created_at=query_result[1], updated_at=query_result[2],
                          container_id=query_result[3], model_id=query_result[4], optimizer_id=query_result[5],
                          history_type=query_result[6], comment=query_result[7], file_id=query_result[8])

        return history

    def read_session(self, session_id, conn) -> Session:
        cursor = conn.cursor()
        cursor.execute(read_session_query, (session_id,))
        query_result = cursor.fetchone()
        # print(query_result)
        session = Session(session_id=query_result[0], created_at=query_result[1], updated_at=query_result[2],
                          status=query_result[3],
                          file_id=query_result[4], epochs=query_result[5], reset_progress=query_result[6],
                          container_id=query_result[7])

        return session

    def check_model_id(self, model_id, conn):
        cursor = conn.cursor()
        cursor.execute(check_model_id_query, (model_id,))
        query_result = cursor.fetchall()

        return len(query_result) > 0

    def check_optimizer_id(self, optimizer_id, conn):
        cursor = conn.cursor()
        cursor.execute(check_optimizer_id_query, (optimizer_id,))
        query_result = cursor.fetchall()

        return len(query_result) > 0

    def read_containers(self, conn):
        cursor = conn.cursor()
        cursor.execute(read_containers_query)
        query_result = cursor.fetchall()

        result: list[Container] = []

        for row in query_result:
            result.append(
                Container(container_id=row[0], created_at=row[1], updated_at=row[2],
                          dataset_id=row[3], model_id=row[4], optimizer_id=row[5],
                          normalise_dataset=row[6], criterion_code=row[7],
                          online_training=row[8], name=row[9], comment=row[10]))
        return result

    def read_models(self, conn):
        cursor = conn.cursor()
        cursor.execute(read_models_query)
        query_result = cursor.fetchall()

        result: list[Model] = []

        for row in query_result:
            result.append(Model(model_id=row[0], created_at=row[1], updated_at=row[2],
                                file_id=row[3], was_trained=row[4], sequential=row[5],
                                code=row[6]))
        return result

    def read_optimizers(self, conn):
        cursor = conn.cursor()
        cursor.execute(read_optimizers_query)
        query_result = cursor.fetchall()

        result: list[Optimizer] = []

        for row in query_result:
            result.append(
                Optimizer(optimizer_id=row[0], created_at=row[1], updated_at=row[2],
                          file_id=row[3], was_trained=row[4], code=row[5]))
        return result

    def read_files(self, conn):
        cursor = conn.cursor()
        cursor.execute(read_files_query)
        query_result = cursor.fetchall()

        result: list[File] = []

        for row in query_result:
            result.append(
                File(file_id=row[0], created_at=row[1], updated_at=row[2],
                     file_type=row[3], comment=row[4], path=row[5]))
        return result

    def read_histories(self, conn):
        cursor = conn.cursor()
        cursor.execute(read_histories_query)
        query_result = cursor.fetchall()
        result: list[History] = []

        for row in query_result:
            result.append(
                History(history_id=row[0], created_at=row[1], updated_at=row[2],
                        container_id=row[3], model_id=row[4], optimizer_id=row[5],
                        history_type=row[6], comment=row[7], file_id=row[8]))
        return result

    def read_sessions(self, conn):
        cursor = conn.cursor()
        cursor.execute(read_sessions_query)
        query_result = cursor.fetchall()
        result: list[Session] = []

        for row in query_result:
            # print(row)
            result.append(Session(session_id=row[0], created_at=row[1], updated_at=row[2], status=row[3],
                                  file_id=row[4], epochs=row[5], reset_progress=row[6],
                                  container_id=row[7]))
        return result
