import os.path

from Docker.env.config import Config


class EnvBuilder:
    def __init__(self, config: Config):
        self.config: Config = config

    def build(self):
        if not self.config.local_path.endswith("local"):
            raise ValueError("Path must end with 'local'")

    def get_storage_path(self, file):
        return os.path.join(self.config.local_path, "storage", file)

    def get_local_path(self, file):
        return os.path.join(self.config.local_path, file)
