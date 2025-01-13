"""
Common utilities for the datasets

Ref:
https://github.com/karpathy/llm.c/blob/master/dev/data/data_common.py
"""

from pathlib import Path
from typing import Sequence

import numpy as np
import requests
from tqdm import tqdm


DATA_CACHE_DIR = Path(__file__).parent / "datasets"
DATA_CACHE_DIR.mkdir(exist_ok=True)


def download_file(url: str, file_path: Path, chunk_size: int = 1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with (
        open(file_path, "wb") as file,
        tqdm(
            desc=str(file_path),
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def write_datafile(file_path: Path, toks: np.ndarray | Sequence[int], n_vocab: int):
    """Saves token data as a .bin file, for reading in C.

    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large"  # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)  # header is always 256 int32 values
    header[0] = 20240520  # magic
    header[1] = 1  # version
    # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    header[2] = len(toks)

    if n_vocab <= 2**16:
        token_dtype = np.uint16
    elif n_vocab <= 2**32:
        token_dtype = np.uint32
    else:
        raise ValueError(f"vocab size {n_vocab} too large for uint16 or uint32")
    toks_np = np.array(toks, dtype=token_dtype)

    num_bytes = (256 * 4) + (len(toks) * toks_np.itemsize)
    print(
        f"writing {len(toks):,} tokens to {file_path} ({num_bytes:,} bytes) in dtype={token_dtype}"
    )
    with open(file_path, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())
