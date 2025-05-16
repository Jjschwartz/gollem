"""
Downloads and tokenizes the TinyStories dataset.
- The download is from HuggingFace datasets.

The output is written to a newly created tinystories/ folder.
The script prints:

For GPT-2:
Number of shards: 50
Tokenizing val split...
writing 19,043,638 tokens to tinystories/TinyStories_val.bin
Tokenizing train split...
writing 925,653,391 tokens to tinystories/TinyStories_train.bin

For LLaMA 3:
Number of shards: 50
Tokenizing val split...
writing 18,660,516 tokens to tinystories/TinyStories_val.bin
Tokenizing train split...
writing 907,021,844 tokens to tinystories/TinyStories_train.bin

And runs in few minutes two depending on your internet
connection and computer. The .bin files are raw byte
streams of uint16 (gpt-2) or uint32 (llama) numbers indicating the token ids.

Ref:
https://github.com/karpathy/llm.c/blob/master/dev/data/tinystories.py
"""

import argparse
import glob
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path

from gollem.data.common import DATA_CACHE_DIR
from gollem.data.common import download_file
from gollem.data.common import write_datafile
from gollem.data.config import DataConfig
from gollem.tokenizer import BaseTokenizer
from gollem.tokenizer import get_tokenizer
from gollem.utils import print0


# -----------------------------------------------------------------------------
THIS_DATA_CACHE_DIR = DATA_CACHE_DIR / "tinystories"


def download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    THIS_DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = THIS_DATA_CACHE_DIR / "TinyStories_all_data.tar.gz"
    if not os.path.exists(data_filename):
        print0(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print0(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    data_dir = THIS_DATA_CACHE_DIR / "TinyStories_all_data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print0(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print0(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(str(data_dir / "*.json")))
    print0("Download done.")
    print0(f"Number of shards: {len(shard_filenames)}")
    # with open(shard_filenames[0], "r") as f:
    #     data = json.load(f)
    # print(f"Example story:\n{data[0]}")


def process_shard(
    shard_index: int, shard_filename: str, tokenizer: BaseTokenizer
) -> list[int]:
    with open(shard_filename, "r") as f:
        data = json.load(f)
    rng = random.Random(1337 + shard_index)
    rng.shuffle(data)
    all_tokens = []
    for example in data:
        text = example["story"]
        text = text.strip()  # get rid of leading/trailing whitespace
        tokens = tokenizer.encode(text, add_eot=True)
        all_tokens.extend(tokens)
    return all_tokens


def tokenize(tokenizer: BaseTokenizer) -> tuple[list[Path], list[Path]]:
    # check if the data already exists
    encoder_data_dir = THIS_DATA_CACHE_DIR / tokenizer.name
    encoder_data_dir.mkdir(exist_ok=True)
    val_filename = encoder_data_dir / "TinyStories_val.bin"
    train_filename = encoder_data_dir / "TinyStories_train.bin"
    if val_filename.exists() and train_filename.exists():
        print0("Tokenized data already exists, skipping tokenization...")
        return [val_filename], [train_filename]

    # shard 0 will be the val split, rest is train
    data_dir = THIS_DATA_CACHE_DIR / "TinyStories_all_data"
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    val_shards = [shard_filenames[0]]
    train_shards = shard_filenames[1:]
    for split_name, split_shards in [("val", val_shards), ("train", train_shards)]:
        print(f"Tokenizing {split_name} split...")
        all_tokens = []
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(process_shard, shard_index, shard_filename, tokenizer)
                for shard_index, shard_filename in enumerate(split_shards)
            ]
            for future in as_completed(futures):
                all_tokens.extend(future.result())

        split_filename = encoder_data_dir / f"TinyStories_{split_name}.bin"
        write_datafile(split_filename, all_tokens, tokenizer.n_vocab)

    return [val_filename], [train_filename]


def load_data(tokenizer: BaseTokenizer) -> DataConfig:
    download()
    val_filenames, train_filenames = tokenize(tokenizer)
    return DataConfig(
        name="tinystories",
        train_data=train_filenames,
        val_data=val_filenames,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tiny Stories dataset preprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model_desc",
        type=str,
        default="gpt2",
        choices=["gpt2", "llama3"],
        help="Model type (determines the tokenizer)",
    )
    args = parser.parse_args()
    model_tokenizer = get_tokenizer(args.model_desc)
    load_data(model_tokenizer)
