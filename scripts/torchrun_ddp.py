import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.distributed import destroy_process_group
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


snapshot_path = Path(__file__).parent.parent / "results" / "torch_ddp_snapshot.pt"
snapshot_path.parent.mkdir(parents=True, exist_ok=True)


class MyTrainDataset(Dataset):
    def __init__(self, size: int):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index]


def main(
    save_every: int,
    total_epochs: int,
    batch_size: int,
):
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    print(f"Launching GPU{rank}|{local_rank} in process group {world_size}")

    print(f"[GPU{rank}] Initializing process group")
    init_process_group(backend="nccl")
    print(f"[GPU{rank}] Process group initialized")

    # load dataset
    dataset_size = max(2048, world_size * batch_size)
    print(f"[GPU{rank}] Loading dataset of size {dataset_size}")
    train_dataset = MyTrainDataset(dataset_size)
    train_data = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,  # pin memory to GPU
        shuffle=False,  # don't shuffle, this is done in the DistributedSampler
        sampler=DistributedSampler(train_dataset),
    )
    print(f"[GPU{rank}] Dataset loaded")

    # load model
    print(f"[GPU{rank}] Loading model")
    model = torch.nn.Linear(20, 1)
    model = model.cuda(local_rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    print(f"[GPU{rank}] Model loaded")

    # load snapshot if it exists
    epochs_run = 0
    if snapshot_path.exists():
        print(f"[GPU{rank}] Loading snapshot from {snapshot_path}")
        snapshot = torch.load(snapshot_path)
        epochs_run = snapshot["EPOCHS_RUN"]
        model.load_state_dict(snapshot["MODEL_STATE"])
        optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        print(f"[GPU{rank}] Snapshot loaded. Epochs run: {epochs_run}")

    # setup DDP
    print(f"[GPU{rank}] DDP setup")
    model = DDP(model, device_ids=[rank])
    print(f"[GPU{rank}] DDP setup complete")

    for epoch in range(epochs_run, total_epochs):
        batch_size = len(next(iter(train_data))[0])
        train_data.sampler.set_epoch(epoch)  # type: ignore
        loss = torch.tensor(0.0, device=local_rank)
        for source, targets in train_data:
            source = source.to(local_rank)
            targets = targets.to(local_rank)

            optimizer.zero_grad()
            output = model(source)
            loss = F.cross_entropy(output, targets)
            loss.backward()
            optimizer.step()

        print(
            f"[GPU{rank}|{local_rank}] Epoch {epoch} | Batchsize: {batch_size} | Steps: {len(train_data)} | Loss: {loss.item()}"
        )

        if rank == 0 and epoch % save_every == 0:
            # use model.module to save the state_dict, since model is wrapped in DDP
            snapshot = {
                "EPOCHS_RUN": epoch,
                "MODEL_STATE": model.module.state_dict(),
                "OPTIMIZER_STATE": optimizer.state_dict(),
            }
            torch.save(snapshot, snapshot_path)
            print(f"Epoch {epoch} | Training snapshot saved at {snapshot_path}")

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
    main(args.save_every, args.total_epochs, args.batch_size)
