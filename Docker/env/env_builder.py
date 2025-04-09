import os.path

from Docker.env.config import Config


class EnvBuilder:
    def __init__(self, config: Config):
        self.docker_dir: str = os.path.dirname(os.path.dirname(__file__))
        self.config: Config = config

    def build(self):
        if not self.config.local_path.endswith("local"):
            raise ValueError("Path must end with 'local'")
        local_folder_path: str = os.path.join(self.config.local_path)
        if not os.path.isdir(local_folder_path):
            os.mkdir(local_folder_path)
        storage_folder_path: str = os.path.join(local_folder_path, "storage")
        if not os.path.isdir(storage_folder_path):
            os.mkdir(local_folder_path)

    def get_storage_path(self, file):
        return os.path.join(self.config.local_path, "storage", file)

    def get_local_path(self, file):
        return os.path.join(self.config.local_path, file)
