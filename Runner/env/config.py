import json
import os.path


class Config:
    def __init__(self):
        self.docker_ip = None
        self.runner_ip = None
        self.runner_host = None
        self.runner_port = None
        self.docker_host = None
        self.docker_port = None
        self.local_path = None

    def parse(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config was not found: Path doesnt exists: {config_path}")
        with open(config_path, 'r') as file:
            config = json.load(file)
            self.runner_host = config["runner_host"]
            self.runner_port = config["runner_port"]
            self.docker_host = config["docker_host"]
            self.docker_port = config["docker_port"]
            self.local_path = config["local_path"]
            self.runner_ip = f"http://{self.runner_host}:{self.runner_port}"
            self.docker_ip = f"http://{self.docker_host}:{self.docker_port}"

    def get_runner_ip(self):
        return self.runner_ip

    def get_docker_ip(self):
        return self.docker_ip
