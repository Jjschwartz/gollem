import importlib

import tiktoken

from gollem.data.config import DataConfig


dataset_registry = {
    "tiny_shakespeare": ("tinyshakespeare", "load_data"),
}


def load_dataset(
    name: str,
    encoder: tiktoken.Encoding,
) -> DataConfig:
    assert name in dataset_registry
    dataset_module_name, load_fn_name = dataset_registry[name]
    dataset_module = importlib.import_module(f"gollem.data.{dataset_module_name}")
    load_fn = getattr(dataset_module, load_fn_name)
    return load_fn(encoder)
