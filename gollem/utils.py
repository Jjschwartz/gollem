from pathlib import Path


def get_base_dir_path() -> Path:
    return Path(__file__).parent.parent
