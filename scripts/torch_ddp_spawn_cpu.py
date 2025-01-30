import os

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed import destroy_process_group
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class MyTrainDataset(Dataset):
    def __init__(self, size: int):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index]


def main(
    rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # limit each process to a single thread
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    init_process_group(backend="gloo", rank=rank, world_size=world_size)


def main(
    rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int
):
    ddp_setup(rank, world_size)

    train_dataset = MyTrainDataset(max(2048, world_size * batch_size))
    train_data = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=False,
        sampler=DistributedSampler(train_dataset),
    )

    model = torch.nn.Linear(20, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # wrap the model with DDP
    model = DDP(model)

    for epoch in range(total_epochs):
        # batch size for this process
        batch_size = len(next(iter(train_data))[0])
        # number of steps within an epoch
        steps = len(train_data)
        print(f"[CPU{rank}] Epoch {epoch} | Batchsize: {batch_size} | Steps: {steps}")
        train_data.sampler.set_epoch(epoch)  # type: ignore
        for source, targets in train_data:
            optimizer.zero_grad()
            output = model(source)
            loss = F.cross_entropy(output, targets)
            loss.backward()
            optimizer.step()

        if rank == 0 and epoch % save_every == 0:
            # use model.module to save the state_dict, since model is wrapped in DDP
            PATH = "checkpoint.pt"
            # ckp = model.module.state_dict()
            # torch.save(ckp, PATH)
            print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="simple distributed training job")
    parser.add_argument(
        "total_epochs", type=int, help="Total epochs to train the model"
    )
    parser.add_argument("save_every", type=int, help="How often to save a snapshot")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Input batch size on each device (default: 32)",
    )
    args = parser.parse_args()

    n_cpus = os.cpu_count() or 4
    world_size = n_cpus
    print(f"Using {world_size} CPUs")
    mp.spawn(
        main,
        args=(world_size, args.save_every, args.total_epochs, args.batch_size),
        nprocs=world_size,
    )
