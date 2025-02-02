import os
from pathlib import Path


def get_base_dir_path() -> Path:
    return Path(__file__).parent.parent


def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)
