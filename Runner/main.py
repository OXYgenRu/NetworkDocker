import torch

from api import start

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    start()
