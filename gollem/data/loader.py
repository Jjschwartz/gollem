import glob
from typing import Any
from typing import Iterator

import numpy as np
import torch

from gollem.utils import print0


def _peek_data_shard(filename: str) -> int:
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print0("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DataLoader:
    """Custom Dataloader for loading data from preprocessed files.

    Handles having the data spread across multiple files, aka "shards".
    """

    def __init__(
        self,
        filename_pattern: str,
        batch_size: int,
        seq_len: int,
        world_size: int,
        rank: int,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.world_size = world_size
        self.rank = rank

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert (
            len(self.files) > 0
        ), f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        self.ntok_per_shard = []
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= self.world_size * self.batch_size * self.seq_len + 1, (
                f"shard {fname} has only {shard_ntok} tokens, but requires >= "
                f"{self.world_size * self.batch_size * self.seq_len + 1} tokens"
            )
            ntok_total += shard_ntok
            self.ntok_per_shard.append(shard_ntok)
        self.ntok_total = ntok_total
        print0(
            f"DataLoader: total number of tokens: {ntok_total:,} "
            f"across {len(self.files)} files"
        )

        # kick things off
        self.current_shard = -1
        self.current_step_in_shard = 0
        # step/batch counter
        self.current_step = 0
        self.reset()

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_step_in_shard = 0
        self.current_step = 0

    def advance(self):
        """Advance to next data shard."""
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_step_in_shard = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])

    @property
    def current_position(self) -> int:
        return (
            (self.current_step_in_shard * self.world_size + self.rank)
            * self.batch_size
            * self.seq_len
        )

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = self.batch_size, self.seq_len
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the start pointer in current shard
        self.current_step_in_shard += 1
        # if loading the next batch would be out of bounds advance the shard
        # NOTE we recompute current position here with new current_step_in_shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.advance()
        self.current_step += 1
        return x, y

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.next_batch()

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        return self

    def state_dict(self) -> dict[str, Any]:
        return {
            "current_step": self.current_step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.current_step = state_dict["current_step"]
        if self.current_step <= 0:
            self.reset()
            return

        # compute current shard and step in shard from current step
        step_chunk_size = self.world_size * self.batch_size * self.seq_len
        # position in the overall dataset
        current_dataset_position = self.current_step * step_chunk_size
        # note we use chunks of size step_chunk_size, so we ignore the remainder
        dataset_size = self.ntok_total - self.ntok_total % step_chunk_size
        if current_dataset_position > dataset_size:
            # we've done >=1 full pass through the dataset,
            # so handle wrapping around to the beginning
            current_dataset_position = current_dataset_position % dataset_size
        ntok_so_far = 0
        for i, ntok in enumerate(self.ntok_per_shard):
            if current_dataset_position < ntok_so_far + ntok:
                self.current_shard = i
                break
            ntok_so_far += ntok
        current_position_in_shard = current_dataset_position - ntok_so_far
        self.current_step_in_shard = current_position_in_shard // step_chunk_size
        # load the tokens for the current shard
        self.tokens = _load_data_shard(self.files[self.current_shard])
