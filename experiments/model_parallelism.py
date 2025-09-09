"""Playing around with model parallelism."""
import argparse
import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import scaled_dot_product_attention


# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False



D_MODEL = 768
N_HEADS = 8
N_LAYERS = 12
N_CTX = 256
N_VOCAB = 16


class MyAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.size()

        qkv = self.qkv_proj(x)
        assert qkv.shape == (B, N, 3 * self.d_model)

        q, k, v = qkv.split(self.d_model, dim=2)
        assert all(m.shape == (B, N, self.d_model) for m in [q, k, v])

        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        assert all(m.shape == (B, self.n_heads, N, self.d_head) for m in [q, k, v])

        z = scaled_dot_product_attention(q, k, v, is_causal=True)
        assert z.shape == (B, self.n_heads, N, self.d_head)

        z = z.transpose(1, 2).contiguous().view(B, N, self.d_model)
        assert z.shape == (B, N, self.d_model)

        o = self.out_proj(z)
        assert o.shape == (B, N, self.d_model)
        return o


class MLP(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.w2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.gelu(self.w1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        # self.ln_1 = nn.LayerNorm(d_model)
        # self.attn = MyAttention(d_model, n_heads)
        # self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x + self.attn(self.ln_1(x))
        # x = x + self.mlp(self.ln_2(x))
        x = x + self.mlp(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, n_vocab: int, n_layers: int, n_ctx: int, devices: list[torch.cuda.device]
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_vocab = n_vocab
        self.n_layers = n_layers
        self.n_ctx = n_ctx

        assert len(devices) == 8
        assert n_layers % (len(devices) - 2) == 0
        self.n_layers_per_device = n_layers // (len(devices) - 2)
        self.devices = devices

        self.embd = nn.Embedding(n_vocab, d_model, device=devices[0])
        # self.pos_embd = nn.Embedding(n_ctx, d_model, device=devices[0])
        print(f"token+pos - device {self.devices[0]}")

        # layers = []
        # for i in range(n_layers):
        #     dev_idx = (i // self.n_layers_per_device) + 1
        #     print(f"Layer {i} - device {self.devices[dev_idx]}")
        #     layers.append(TransformerBlock(d_model, n_heads).to(devices[dev_idx]))

        # self.layers = nn.ModuleList(layers)

        # self.ln_f = nn.LayerNorm(d_model).to(devices[-1])
        self.out_embd = nn.Linear(d_model, n_vocab, bias=False).to(devices[-1])
        print(f"out embd - device {self.devices[-1]}")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N = tokens.size()
        assert tokens.max().item() <= self.n_vocab - 1, tokens.max().item()
        assert 0 <= tokens.min().item(), tokens.min().item()
        
        tokens = tokens.to(self.devices[0])
        token_embd = self.embd(tokens)

        # pos = torch.arange(0, N, dtype=torch.long, device=self.devices[0])
        # pos_embd = self.pos_embd(pos)

        # x = token_embd + pos_embd
        x = token_embd

        # for i, layer in enumerate(self.layers):
        #     dev_idx = (i // self.n_layers_per_device) + 1
        #     x = x.to(self.devices[dev_idx]).contiguous()
        #     x = layer(x)
        
        print(f"{x=}")
        x = x.to("cpu")
        print(f"{x=}")
        x = x.to(self.devices[-1])
        print(f"{x=}")

        # x = self.ln_f(x)
        logits = self.out_embd(x)
        assert logits.shape == (B, N, self.n_vocab)
        return logits


def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad mean={param.grad.mean().item():.6f}, "
                  f"grad norm={param.grad.norm().item():.6f}")
        else:
            print(f"{name}: NO GRADIENT!")


def compare_per_layer(a_model, b_model, x0):
    a_model.eval(); b_model.eval()
    acts_a, acts_b, hooks = [], [], []
    def hook(bucket): return lambda m,i,o: bucket.append(o.detach().float().cpu())
    for i in range(len(a_model.layers)):
        hooks += [a_model.layers[i].register_forward_hook(hook(acts_a)),
                  b_model.layers[i].register_forward_hook(hook(acts_b))]
    with torch.no_grad():
        _ = a_model(x0.to(a_model.devices[0]))
        _ = b_model(x0.to(b_model.devices[0]))
    for h in hooks: h.remove()
    for i,(ua,ub) in enumerate(zip(acts_a,acts_b)):
        print(f"layer {i} max diff:", (ua-ub).abs().max().item())


def compare_models_pairwise_across_devices(x, devices):
    # for gpu_a_idx in range(len(devices)):
    for gpu_a_idx in [0]:
        model_a = Transformer(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_vocab=N_VOCAB,
            n_layers=N_LAYERS,
            n_ctx=N_CTX,
            devices=[f"cuda:{gpu_a_idx}" for _ in range(8)]
            # devices=[f"cpu" for _ in range(8)]
        )
        model_a.eval()

        # for gpu_b_idx in range(len(devices)):
        for gpu_b_idx in [1]:
            model_b = Transformer(
                d_model=D_MODEL,
                n_heads=N_HEADS,
                n_vocab=N_VOCAB,
                n_layers=N_LAYERS,
                n_ctx=N_CTX,
                devices=[f"cuda:{gpu_b_idx}" for _ in range(8)]
                # devices=[f"cpu" for _ in range(8)]
            )
            model_b.load_state_dict(model_a.state_dict(), strict=True)
            model_b.eval()
            # compare_per_layer(model_a, model_b, x)
            # return
            with torch.no_grad():
                a = model_a(x.to(devices[gpu_a_idx])).cpu()
                b = model_b(x.to(devices[gpu_b_idx])).cpu()
                all_close = torch.allclose(a, b)
                max_diff = (a-b).abs().max().item()
                print(f"cuda:{gpu_a_idx}  cuda:{gpu_b_idx} - {max_diff=} {all_close=}")



def main(n: int):

    assert torch.cuda.is_available()

    n_devices = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(n_devices)]
    # devices = [f"cuda:1" for i in range(n_devices)]
    devices = [f"cuda:0" for i in range(6)] + [f"cuda:1" for i in range(2)]
    # devices[-1] = devices[1]

    B = 8

    n_repeats = 1
    repeat_size = N_CTX // n_repeats
    repeated_sequence = torch.randint(0, N_VOCAB-1, (B, repeat_size), dtype=torch.long)
    # input_tokens = repeated_sequence.repeat((1, n_repeats))
    input_tokens = repeated_sequence
    assert input_tokens.shape == (B, N_CTX)
    # assert torch.all(
    #     input_tokens[:, :repeat_size]
    #     == input_tokens[:, repeat_size : 2 * (repeat_size)]
    # )
    
    x = input_tokens[:, :-1].to(devices[0])
    y = input_tokens[:, 1:].flatten().to(devices[-1])

    # compare_models_pairwise_across_devices(x, devices)

    model = Transformer(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_vocab=N_VOCAB,
        n_layers=N_LAYERS,
        n_ctx=N_CTX,
        devices=devices
    )
    model(x)

    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ids = {id(p) for g in optimizer.param_groups for p in g['params']}
    missing = [n for n,p in model.named_parameters() if p.requires_grad and id(p) not in ids]
    assert not missing, f"Params missing from optimizer: {missing}"
    
    for step in range(n):
        optimizer.zero_grad()
        print("-------------------------------- BEFORE BACKWARD --------------------------------")
        for n, p in model.named_parameters():
            print(f"{n=} | {p.grad=}")
        print("--------------------------------")
        logits = model(x)
        print(f"{logits=}")

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y, ignore_index=-1)
        loss.backward()
        print(f"{loss=} | {loss.grad=}")
        # check_gradients(model)
        optimizer.step()

        print("-------------------------------- AFTER BACKWARD --------------------------------")
        for n, p in model.named_parameters():
            print(f"{n=} | {p.grad=}")
        print("--------------------------------")

        dead = [
            (n, p.device) for n,p in model.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum()==0)
        ]
        print(f"{dead=}")
        # assert not dead, f"No/zero grads for: {dead}"

        print(f"{step=} | loss={loss.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    args = parser.parse_args()
    main(args.n)
