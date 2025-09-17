import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, devices: list[torch.device]) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.devices = devices

        n_devices = len(self.devices)
        assert n_devices >= 1
        # to keep things simple
        assert self.d_mlp % n_devices == 0
        assert self.d_model % n_devices == 0

        # block size is same for both w1 (columnwise) and w2 (rowwise)
        # as we split on the d_mlp dimension for both
        w_block_size = self.d_mlp // n_devices

        w1_blocks = []
        w2_blocks = []
        for d in self.devices:
            w1 = nn.Linear(d_model, w_block_size, device=d)
            w2 = nn.Linear(w_block_size, d_model, device=d)
            w1_blocks.append(w1)
            w2_blocks.append(w2)

        self.w1_blocks = nn.ModuleList(w1_blocks)
        self.w2_blocks = nn.ModuleList(w2_blocks)

        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_device = x.device

        final_h = torch.zeros_like(x, device=input_device)
        for w1_block, w2_block, d in zip(self.w1_blocks, self.w2_blocks, self.devices):
            # broadcast input to device (scatter)
            h = x.to(d, copy=True)
            h = self.gelu(w1_block(h))
            h = w2_block(h)
            h = h.to(input_device)
            # accumulate (all-reduce sum)
            final_h = final_h + h

        return final_h


def main(n_steps: int, n_devices: int | None, debug: bool = False):
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_devices:
            assert n_gpus > n_devices
            n_gpus = n_devices
        print(f"Using CUDA with {n_gpus=}")
        devices = [f"cuda:{i}" for i in range(n_gpus)]
    else:
        # fake having multiple devices
        devices = ["cpu"] * n_devices if n_devices else ["cpu"] * 8
        print(f"Using CPUS with {len(devices)} devices")

    main_device = devices[0]

    batch_size = 8
    seq_len = 16
    d_model = 8
    d_mlp = 1 * d_model

    torch.manual_seed(35)

    x = torch.randn((batch_size, seq_len, d_model), device=main_device)
    y = torch.randn((batch_size, seq_len, d_model), device=main_device)

    model = MLP(d_model=d_model, d_mlp=d_mlp, devices=devices)
    model(x)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    missing = [
        n for n, p in model.named_parameters() if p.requires_grad and id(p) not in ids
    ]
    assert not missing, f"Params missing from optimizer: {missing}"

    for step in range(n_steps):
        optimizer.zero_grad()
        if debug:
            print(
                "-------------------------------- BEFORE BACKWARD --------------------------------"
            )
            for name, p in model.named_parameters():
                print(f"{name=} | {p.grad=}")
            print("--------------------------------")
        y_pred = model(x)

        loss = F.mse_loss(y_pred, y)
        loss.backward()
        if debug:
            print(f"{loss=} | {loss.grad=}")
        optimizer.step()

        if debug:
            print(
                "-------------------------------- AFTER BACKWARD --------------------------------"
            )
            for name, p in model.named_parameters():
                print(f"{name=} | {p.grad=}")
            print("--------------------------------")

            dead = [
                (n, p.device)
                for n, p in model.named_parameters()
                if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)
            ]
            print(f"{dead=}")
            # assert not dead, f"No/zero grads for: {dead}"

        print(f"{step=} | loss={loss.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int)
    parser.add_argument("--n_devices", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.n_steps, args.n_devices, debug=args.debug)
