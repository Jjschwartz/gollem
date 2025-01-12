from dataclasses import dataclass
from pathlib import Path


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
