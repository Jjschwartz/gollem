import importlib
from dataclasses import dataclass
from pathlib import Path

import tiktoken


# Dataset has a couple of levels:
# 1. Raw data (e.g. text files)
#    - these are the files that are downloaded from the internet
#    - identified by a name, e.g. tiny_shakespeare
# 2. Tokenized data (e.g. bin files)
#    - this is the data tokenized by a specific tokenizer (e.g. tiktoken('gpt2'))
#    - identified by (dataset_name, tokenizer_name)
# 3. Train, Val splits
#    - this is the tokenized data split into train and val sets
#    - identified by (dataset_name, tokenizer_name, split_name)
#
# For efficiency we may only store the raw data and train/val tokenized data.
# E.g. For the tiny_shakespeare dataset tokenized with tiktoken-gpt2, we'd get:
# - tiny_shakespeare.txt
# - tiny_shakespeare_tiktoken-gpt2_train.bin
# - tiny_shakespeare_tiktoken-gpt2_val.bin


@dataclass
class DataConfig:
    # Path to the training data.
    train_data: Path
    # Path to the validation data.
    val_data: Path | None


_registry = {
    "tiny_shakespeare": ("tinyshakespeare", "load_data"),
}


def load_dataset(
    name: str,
    encoder: tiktoken.Encoding,
) -> DataConfig:
    assert name in _registry
    dataset_module_name, load_fn_name = _registry[name]
    dataset_module = importlib.import_module(f"gollem.data.{dataset_module_name}")
    load_fn = getattr(dataset_module, load_fn_name)
    return load_fn(encoder)
