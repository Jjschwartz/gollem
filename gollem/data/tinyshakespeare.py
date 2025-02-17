"""
Downloads and tokenizes the TinyShakespeare dataset.
- The download is from Github.

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

import argparse
from pathlib import Path

from gollem.data.common import DATA_CACHE_DIR
from gollem.data.common import download_file
from gollem.data.common import write_datafile
from gollem.data.config import DataConfig
from gollem.tokenizer import BaseTokenizer
from gollem.tokenizer import get_tokenizer
from gollem.utils import print0


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


def tokenize(tokenizer: BaseTokenizer) -> tuple[Path, Path]:
    encoder_data_dir = THIS_DATA_CACHE_DIR / tokenizer.name
    encoder_data_dir.mkdir(exist_ok=True)
    val_filename = encoder_data_dir / "tiny_shakespeare_val.bin"
    train_filename = encoder_data_dir / "tiny_shakespeare_train.bin"
    if val_filename.exists() and train_filename.exists():
        print0("Tokenized data already exists, skipping tokenization...")
        return val_filename, train_filename

    data_filename = THIS_DATA_CACHE_DIR / "tiny_shakespeare.txt"
    with open(data_filename, "r") as fin:
        text = fin.read()

    # let's treat every individual chunk of text as a separate "document"
    sections = text.split("\n\n")
    tokens = []
    for s in sections:
        tokens.extend(tokenizer.encode(s, add_eot=True))

    # let's take the first 32,768 tokens as the validation split (~10%)
    val_tokens = tokens[:32768]
    train_tokens = tokens[32768:]
    # save to file

    write_datafile(val_filename, val_tokens, tokenizer.n_vocab)
    write_datafile(train_filename, train_tokens, tokenizer.n_vocab)

    return train_filename, val_filename


def load_data(tokenizer: BaseTokenizer) -> DataConfig:
    download()
    train_filename, val_filename = tokenize(tokenizer)
    return DataConfig(
        name="tinyshakespeare",
        train_data=[train_filename],
        val_data=[val_filename] if val_filename is not None else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tiny Shakespeare dataset preprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model_desc",
        type=str,
        default="gpt-2",
        choices=["gpt-2", "llama-3"],
        help="Model type (determines the tokenizer)",
    )
    args = parser.parse_args()
    model_tokenizer = get_tokenizer(args.model_desc)
    load_data(model_tokenizer)
