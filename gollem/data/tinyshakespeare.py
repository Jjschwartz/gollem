"""
Downloads and tokenizes the TinyShakespeare dataset.
- The download is from Github.
- The tokenization is GPT-2 tokenizer with tiktoken

The output is written to a newly created tinyshakespeare/ folder.
The script prints:

Saved 32768 tokens to tinyshakespeare/tiny_shakespeare_val.bin
Saved 305260 tokens to tinyshakespeare/tiny_shakespeare_train.bin

And runs in a few seconds depending on your internet
connection and computer. The .bin files are raw byte
streams of int32 numbers indicating the token ids.

Ref:
https://github.com/karpathy/llm.c/blob/master/dev/data/tinyshakespeare.py
"""

import os
from pathlib import Path

import tiktoken

from gollem.data.common import DATA_CACHE_DIR
from gollem.data.common import download_file
from gollem.data.common import write_datafile
from gollem.data.config import DataConfig


THIS_DATA_CACHE_DIR = DATA_CACHE_DIR / "tinyshakespeare"


def download():
    """Downloads the TinyShakespeare dataset to DATA_CACHE_DIR"""
    THIS_DATA_CACHE_DIR.mkdir(exist_ok=True)
    # download the TinyShakespeare dataset, unless it's already downloaded
    data_path = THIS_DATA_CACHE_DIR / "tiny_shakespeare.txt"
    if not data_path.exists():
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading {data_url} to {data_path}...")
        download_file(data_url, data_path)
    else:
        print(f"{data_path} already exists, skipping download...")


def tokenize(encoder: tiktoken.Encoding) -> tuple[Path, Path | None]:
    data_filename = THIS_DATA_CACHE_DIR / "tiny_shakespeare.txt"
    with open(data_filename, "r") as fin:
        text = fin.read()
    # let's treat every person's statement in the dialog as a separate document
    text = "<|endoftext|>" + text
    text = text.replace("\n\n", "\n\n<|endoftext|>")
    # encode the text
    tokens = encoder.encode(text, allowed_special={"<|endoftext|>"})
    # let's take the first 32,768 tokens as the validation split (~10%)
    val_tokens = tokens[:32768]
    train_tokens = tokens[32768:]
    # save to file
    encoder_data_dir = THIS_DATA_CACHE_DIR / encoder.name
    encoder_data_dir.mkdir(exist_ok=True)

    val_filename = encoder_data_dir / "tiny_shakespeare_val.bin"
    train_filename = encoder_data_dir / "tiny_shakespeare_train.bin"
    write_datafile(val_filename, val_tokens)
    write_datafile(train_filename, train_tokens)

    return train_filename, val_filename


def load_data(encoder: tiktoken.Encoding) -> DataConfig:
    download()
    train_filename, val_filename = tokenize(encoder)
    return DataConfig(
        train_data=train_filename,
        val_data=val_filename,
    )


if __name__ == "__main__":
    download()
    enc = tiktoken.get_encoding("gpt2")
    tokenize(enc)
