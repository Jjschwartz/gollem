"""
Downloads and tokenizes the FineWeb dataset.
- https://huggingface.co/datasets/HuggingFaceFW/fineweb

example doc to highlight the structure of the dataset:
{
  "text": "Posted by mattsmith on 20th April 2012\nStraight from...",
  "id": "<urn:uuid:d853d453-196e-4488-a411-efc2b26c40d2>",
  "dump": "CC-MAIN-2013-20",
  "url": "http://nleastchatter.com/philliesphandom/tag/freddy-galvis/",
  "date": "2013-05-18T07:24:47Z",
  "file_path": "s3://commoncrawl/long.../path.../file.gz",
  "language": "en",
  "language_score": 0.9185474514961243,
  "token_count": 594
}

Example of downloading the 100B dataset of FineWebEDU, from root directory:
python dev/data/fineweb.py -t edu -v 100B
100B runs for small few hours, depending on your internet and computer.

The tokenized dataset will be saved in the following directory:
{DATA_CACHE_DIR}/fineweb_{version]_{size}/{tokenizer_name}

Ref:
https://github.com/karpathy/llm.c/blob/master/dev/data/fineweb.py
"""

import argparse
import glob
import multiprocessing as mp
import os
from pathlib import Path
from typing import Literal

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from gollem.data.common import DATA_CACHE_DIR
from gollem.data.common import write_datafile
from gollem.data.config import DataConfig
from gollem.tokenizer import BaseTokenizer
from gollem.tokenizer import get_tokenizer
from gollem.utils import print0


# TODO test this:
# "For faster downloads, make sure to install pip install huggingface_hub[hf_transfer]
# and set the environment variable HF_HUB_ENABLE_HF_TRANSFER=1."

# -----------------------------------------------------------------------------

# FineWeb has a few possible versions available (edu|classic, 10B|100B)
# Here is a mapping from local directory to remote name
_DIRECTORIES = {
    ("classic", "10B"): ("fineweb_classic_10B", "sample-10BT"),
    ("classic", "100B"): ("fineweb_classic_100B", "sample-100BT"),
    ("edu", "10B"): ("fineweb_edu_10B", "sample-10BT"),
    ("edu", "100B"): ("fineweb_edu_100B", "sample-100BT"),
}


def download(
    version: Literal["classic", "edu"], size: Literal["10B", "100B"]
) -> tuple[dict, str, str]:
    local_dir_name, remote_name = _DIRECTORIES[(version, size)]
    print0(f"Downloading fineweb {version} {size} (remote name: {remote_name})...")
    # download the dataset
    if version == "classic":
        fw = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train")
        name = "fineweb"
    elif version == "edu":
        fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
        name = "fineweb_edu"

    print0("Download complete")
    return fw, name, local_dir_name


class TokenizeFn:
    """
    Tokenize a document into a numpy array of tokens.

    Needs to be a class so that we can pickle it for multiprocessing.
    """

    def __init__(self, tokenizer: BaseTokenizer):
        self.tokenizer = tokenizer
        if tokenizer.n_vocab <= 2**16:
            self.token_dtype = np.uint16
            self.max_token_val = 2**16 - 1
        elif tokenizer.n_vocab <= 2**32:
            self.token_dtype = np.uint32
            self.max_token_val = 2**32 - 1
        else:
            raise ValueError(
                f"vocab size {tokenizer.n_vocab} too large for uint16 or uint32"
            )

    def __call__(self, doc: dict) -> np.ndarray:
        text = doc["text"]
        tokens = self.tokenizer.encode(text, add_eot=True)
        tokens_np = np.array(tokens)
        assert (tokens_np >= 0).all() and (
            tokens_np <= self.max_token_val
        ).all(), f"token dictionary too large for {self.token_dtype}"
        tokens_np_uint = tokens_np.astype(self.token_dtype)
        return tokens_np_uint


def tokenize(
    tokenizer: BaseTokenizer,
    dataset_map: dict,
    dataset_name: str,
    local_dir_name: str,
    shard_size: int,
) -> tuple[list[Path], list[Path]]:
    # check if the data already exists
    encoder_data_dir = DATA_CACHE_DIR / local_dir_name / tokenizer.name
    encoder_data_dir.mkdir(exist_ok=True, parents=True)
    val_filename = encoder_data_dir / f"{dataset_name}_val_000000.bin"
    train_filenames = glob.glob(str(encoder_data_dir / f"{dataset_name}_train_*.bin"))
    if val_filename.exists() and train_filenames:
        print0("Tokenized data already exists, skipping tokenization...")
        return [val_filename], [Path(f) for f in train_filenames]

    tokenize_fn = TokenizeFn(tokenizer)

    os_count = os.cpu_count() or 1
    nprocs = max(1, os_count - 2)  # don't hog the entire system
    val_filenames = []
    train_filenames = []
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=tokenize_fn.token_dtype)
        token_count = 0
        progress_bar = tqdm(
            total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
        )

        for tokens in pool.imap(tokenize_fn, dataset_map, chunksize=16):
            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                if shard_index == 0:
                    shard_filename = (
                        encoder_data_dir / f"{dataset_name}_val_{shard_index:06d}.bin"
                    )
                    val_filenames.append(shard_filename)
                else:
                    shard_filename = (
                        encoder_data_dir / f"{dataset_name}_train_{shard_index:06d}.bin"
                    )
                    train_filenames.append(shard_filename)

                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count : token_count + remainder] = tokens[
                    :remainder
                ]

                write_datafile(
                    shard_filename, all_tokens_np.tolist(), tokenizer.n_vocab
                )
                shard_index += 1
                progress_bar = tqdm(
                    total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
                )
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

    return val_filenames, train_filenames


def load_data(
    tokenizer: BaseTokenizer,
    version: Literal["classic", "edu"],
    size: Literal["10B", "100B"],
    shard_size: int = 100_000_000,
) -> DataConfig:
    dataset_map, dataset_name, local_dir_name = download(version, size)
    val_filenames, train_filenames = tokenize(
        tokenizer, dataset_map, dataset_name, local_dir_name, shard_size
    )
    return DataConfig(
        name=dataset_name,
        train_data=train_filenames,
        val_data=val_filenames,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tiny Stories dataset preprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="classic",
        help="Fineweb type, edu|classic",
        choices=["edu", "classic"],
    )
    parser.add_argument(
        "-s",
        "--size",
        type=str,
        default="10B",
        help="Fineweb data sample size, 10B|100B",
        choices=["10B", "100B"],
    )
    parser.add_argument(
        "-m",
        "--model_desc",
        type=str,
        default="gpt-2",
        choices=["gpt-2", "llama-3"],
        help="Model type (determines the tokenizer)",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=100_000_000,
        help="Size of each data shard in the output .bin files, in tokens",
    )
    args = parser.parse_args()
    model_tokenizer = get_tokenizer(args.model_desc)
    load_data(model_tokenizer, args.version, args.size, args.shard_size)
