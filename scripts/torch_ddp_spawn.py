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
    # init process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # load dataset
    train_dataset = MyTrainDataset(2048)
    train_data = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,  # pin memory to GPU, ensu
        shuffle=False,  # don't shuffle, this is done in the DistributedSampler
        sampler=DistributedSampler(train_dataset),
    )

    # load model
    model = torch.nn.Linear(20, 1)
    model = model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # setup DDP
    model = DDP(model, device_ids=[rank])

    for epoch in range(total_epochs):
        batch_size = len(next(iter(train_data))[0])
        print(
            f"[GPU{rank}] Epoch {epoch} | Batchsize: {batch_size} | Steps: {len(train_data)}"
        )
        train_data.sampler.set_epoch(epoch)  # type: ignore
        for source, targets in train_data:
            source = source.to(rank)
            targets = targets.to(rank)

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

    world_size = torch.cuda.device_count()
    mp.spawn(
        main,
        args=(world_size, args.save_every, args.total_epochs, args.batch_size),
        nprocs=world_size,
    )
