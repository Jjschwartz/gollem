"""Playing around with data parallelism."""

import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import scaled_dot_product_attention


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
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MyAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_vocab: int,
        n_layers: int,
        n_ctx: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_vocab = n_vocab
        self.n_layers = n_layers
        self.n_ctx = n_ctx

        self.embd = nn.Embedding(n_vocab, d_model)
        self.pos_embd = nn.Embedding(n_ctx, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.out_embd = nn.Linear(d_model, n_vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        device = tokens.device
        B, N = tokens.size()

        print(f"  Transformer Forward - {device=} {tokens.shape=}")

        token_embd = self.embd(tokens)

        pos = torch.arange(0, N, dtype=torch.long, device=device)
        pos_embd = self.pos_embd(pos)

        x = token_embd + pos_embd

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.out_embd(x)
        assert logits.shape == (B, N, self.n_vocab)
        return logits


def main(n: int):
    assert torch.cuda.is_available()

    n_devices = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(n_devices)]
    print(f"Num GPUs: {n_devices}")

    # Batch size will be 8 inputs per GPU
    B = 8 * n_devices

    n_repeats = 8
    repeat_size = N_CTX // n_repeats
    repeated_sequence = torch.randint(
        0, N_VOCAB - 1, (B, repeat_size), dtype=torch.long
    )
    input_tokens = repeated_sequence.repeat((1, n_repeats))
    assert input_tokens.shape == (B, N_CTX)
    assert torch.all(
        input_tokens[:, :repeat_size]
        == input_tokens[:, repeat_size : 2 * (repeat_size)]
    )

    x = input_tokens[:, :-1].to(devices[0])
    y = input_tokens[:, 1:].flatten().to(devices[0])

    model = Transformer(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_vocab=N_VOCAB,
        n_layers=N_LAYERS,
        n_ctx=N_CTX,
    )
    model = nn.DataParallel(
        model,
        device_ids=devices,
        output_device=devices[0],
    )
    model.to(devices[0])

    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(n):
        optimizer.zero_grad()
        print(f"Step {step} - Input device: {x.device}")
        logits = model(x)
        print(f"Step {step} - Logits device: {logits.device}")

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y, ignore_index=-1)
        loss.backward()
        optimizer.step()

        print(logits)
        print(loss)

        print(f"Step {step} | loss={loss.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    args = parser.parse_args()
    main(args.n)
