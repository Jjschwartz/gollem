import importlib

from gollem.data.config import DataConfig
from gollem.tokenizer import BaseTokenizer


dataset_registry = {
    "tinyshakespeare": ("tinyshakespeare", "load_data", None),
    "tinystories": ("tinystories", "load_data", None),
    "fineweb_edu_100B": ("fineweb", "load_data", {"version": "edu", "size": "100B"}),
    "fineweb_edu_10B": ("fineweb", "load_data", {"version": "edu", "size": "10B"}),
    "fineweb_classic_100B": (
        "fineweb",
        "load_data",
        {"version": "classic", "size": "100B"},
    ),
    "fineweb_classic_10B": (
        "fineweb",
        "load_data",
        {"version": "classic", "size": "10B"},
    ),
}


def load_dataset(
    name: str,
    encoder: BaseTokenizer,
) -> DataConfig:
    assert name in dataset_registry
    dataset_module_name, load_fn_name, load_fn_kwargs = dataset_registry[name]
    dataset_module = importlib.import_module(f"gollem.data.{dataset_module_name}")
    load_fn = getattr(dataset_module, load_fn_name)
    if load_fn_kwargs is None:
        return load_fn(encoder)
    return load_fn(encoder, **load_fn_kwargs)
