import os
import sys
from dataclasses import dataclass
from dataclasses import field
from pprint import pprint

import pyrallis
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group
from torch.distributed import init_process_group

from gollem.data import load_dataset
from gollem.models.gpt2.config import GPT2Config
from gollem.train.config import TrainConfig
from gollem.train.core import run


@dataclass
class RunConfig:
    # Name of the dataset to use (see `gollem.data.__init__.py`)
    dataset: str = field(default="tinyshakespeare")
    # GPT2Config
    model: GPT2Config = field(default_factory=GPT2Config)
    # Training configuration
    train: TrainConfig = field(default_factory=TrainConfig)


def main():
    try:
        print("Parsing config")
        cfg = pyrallis.parse(config_class=RunConfig)
        pprint(cfg)
    except Exception as e:
        print("Failed to parse config")
        print(e)
        sys.exit(1)

    model_cfg = cfg.model

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    try:
        # Handle process group initialization and dataset download before
        # launching training
        if ddp:
            world_size = int(os.environ["WORLD_SIZE"])
            print(f"Initializing DDP with {world_size} processes")
            # DDP atm demands CUDA, we set the device appropriately according to rank
            # TODO: we could use gloo to support CPU based DDP, but it's not really
            # worth it for now
            assert torch.cuda.is_available(), "We need CUDA for DDP"
            init_process_group(backend="nccl")
            # We only want one process per nodeto download the dataset
            # So we have rank 0 download it while the other processes wait
            ddp_local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if ddp_local_rank == 0:
                print("Downloading dataset on local rank 0")
                # initial call to load dataset will download it unless it already exists
                load_dataset(cfg.dataset, model_cfg.get_tokenizer())

            dist.barrier()

        dataset = load_dataset(cfg.dataset, model_cfg.get_tokenizer())

        # Run training
        run(
            dataset_config=dataset,
            model_config=model_cfg,
            train_config=cfg.train,
        )
    finally:
        if ddp:
            # clean up nicely
            print("Destroying process group")
            destroy_process_group()


if __name__ == "__main__":
    main()
