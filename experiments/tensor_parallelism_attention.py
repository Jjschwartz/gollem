import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_head: int, device: torch.device
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.device = device

        self.qkv_proj = nn.Linear(d_model, 3 * n_heads * d_head, device=device)
        self.out_proj = nn.Linear(n_heads * d_head, d_model, device=device)
        # ensure we are still initializing the weights correctly
        # Linear layers are initialized to U(1/k, 1/k) where k is the input_dim
        # which is the n_heads*d_head by default, so we we need to manually set this
        # to use 1/d_model which is what is would be if we were on a single device
        nn.init.normal_(self.out_proj.weight, mean=1 / d_model, std=1 / d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()

        # Hack to make things work for single process, multi-gpu on cluster I have access too
        x = x.to("cpu")
        x = x.to(self.device, copy=True)

        # B, T, (3 * n_heads * d_head)
        qkv = self.qkv_proj(x)

        # (B, T, n_heads * d_head)
        q, k, v = qkv.split(self.n_heads * self.d_head, dim=2)

        # (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # (B, n_heads, T, d_head)
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # (B, T, n_heads * d_head)
        z = z.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
        # (B, T, d_model)
        out = self.out_proj(z)
        return out


class TPAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, devices: list[torch.device]) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.devices = devices

        n_devices = len(self.devices)
        assert n_devices >= 1
        # to keep things simple
        assert self.n_heads % n_devices == 0

        self.n_heads_per_device = self.n_heads // n_devices

        self.attn_slices = nn.ModuleList(
            [
                Attention(d_model, self.n_heads_per_device, self.d_head, d)
                for d in self.devices
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_device = x.device

        final_h = torch.zeros_like(x, device=input_device)
        for attn_slice, d in zip(self.attn_slices, self.devices):
            # broadcast input to device (scatter)
            h = attn_slice(x)
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
    d_model = 64
    n_heads = 8

    torch.manual_seed(35)

    x = torch.randn((batch_size, seq_len, d_model), device=main_device)
    y = torch.randn((batch_size, seq_len, d_model), device=main_device)

    model = TPAttention(d_model=d_model, n_heads=n_heads, devices=devices)
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
