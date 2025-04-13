"""Utility model config stuff."""

import math
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import Tuple

import torch

from gollem.models.config import ModelActivations
from gollem.models.config import ModelConfig
from gollem.models.config import ModelFlops
from gollem.models.config import ModelParams
from gollem.models.llama3.model import Llama3
from gollem.models.model import BaseLLM
from gollem.tokenizer import BaseTokenizer
from gollem.tokenizer import get_tokenizer
from gollem.utils import print0


# Some notes:
# - no bias in the MLP
# - RMSNorm with no bias
# - no dropout
# - group query attention
# - embeddings not shared
# - RoPE positional embeddings


@dataclass
class Llama3Config(ModelConfig):
    # Name of the model
    model_name: str = field(default="llama3")
    # Context length
    # NOTE: Llama3 uses up to 128k context size
    # setting this to 1024 for now for early experiments
    n_ctx: int = field(default=1024)
    # Number of layers
    n_layer: int = field(default=12)
    # Number of attention heads
    n_head: int = field(default=12)
    # Number of key value heads
    n_kv_head: int = field(default=12)
    # Model dimension
    d_model: int = field(default=4096)
    # MLP intermediate dimension
    intermediate_size: int = field(default=14336)
    # Vocabulary size
    vocab_size: int = field(default=128000)
    # Learning rate
    learning_rate: float = field(default=3e-4)
    # Learning rate warmup iterations
    warmup_iters: int = field(default=0)
    # Learning rate decay fraction (final learning rate = learning_rate * learning_rate_decay_frac)
    learning_rate_decay_frac: float = field(default=0.001)
    # Rope embedding base frequency
    # NOTE: Llama3 uses 500k for 128k context size
    # here we use 10k for 4096 context size
    rope_theta: float = field(default=10000)
    # RMSNorm epsilon (A small value added to the denominator for numerical stability.)
    rmsnorm_eps: float = field(default=1e-6)
    # Weight decay
    weight_decay: float = field(default=0.1)
    # Max gradient magnitude.
    grad_clip: float = field(default=1.0)
    # adamw beta params
    betas: Tuple[float, float] = field(default=(0.9, 0.95))
    # Use fused version of AdamW optimizer.
    fused_adamw: bool = field(default=True)
    # Use ZeroRedundancyOptimizer.
    zero_optimizer: bool = field(default=True)
    # Use flash attention.
    flash: bool = field(default=True)
    # Use activation checkpointing
    activation_checkpointing: bool = field(default=False)
    # Torch.compile the model.
    compile: bool = field(default=True)
    # Load from pretrained weights
    from_pretrained: bool = field(default=False)
    # Maximum batch size for sampling
    max_sample_batch_size: int = field(default=1)
    # whether to use KV caching for sampling or not
    use_kv_caching: bool = field(default=False)

    def __post_init__(self):
        assert self.n_head % self.n_kv_head == 0
        assert self.d_model % self.n_head == 0

    def get_tokenizer(self) -> BaseTokenizer:
        return get_tokenizer("llama-3")

    def get_model_and_optimizer(
        self, device: str | torch.device
    ) -> Tuple[BaseLLM, torch.optim.Optimizer]:
        if isinstance(device, str):
            device_type = "cuda" if "cuda" in device else "cpu"
        else:
            device_type = device.type
        model = Llama3.from_pretrained(self) if self.from_pretrained else Llama3(self)
        model.to(device)
        optimizer = model.configure_optimizers(device_type=device_type)

        if self.compile:
            print0("compiling the model...")
            model = torch.compile(model)  # type: ignore

        return model, optimizer  # type: ignore

    def get_lr_scheduler(self, num_iterations: int) -> Callable[[int], float]:
        def get_lr(it: int) -> float:
            # cosine (with warmup)
            min_lr = self.learning_rate * self.learning_rate_decay_frac
            # 1) linear warmup for warmup_iters steps
            # increasing from 0 to learning rate over warm-up iters
            if it < self.warmup_iters:
                return self.learning_rate * (it + 1) / self.warmup_iters
            # 2) if it > lr_decay_iters, return min learning rate
            if it > num_iterations:
                return min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (it - self.warmup_iters) / (
                num_iterations - self.warmup_iters
            )
            assert 0 <= decay_ratio <= 1
            # coeff starts at 1 and goes to 0
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (self.learning_rate - min_lr)

        return get_lr

    def get_params(self) -> ModelParams:
        out = OrderedDict()

        # token embeddings
        out["embedding/token"] = self.vocab_size * self.d_model
        out["embedding"] = out["embedding/token"]

        # attention blocks
        out["attention/norm"] = self.d_model
        d_head = self.d_model // self.n_head
        out["attention/wq"] = self.d_model * self.d_model
        out["attention/wk"] = self.d_model * self.n_kv_head * d_head
        out["attention/wv"] = self.d_model * self.n_kv_head * d_head
        out["attention/wo"] = self.d_model * self.d_model
        out["attention"] = (
            out["attention/norm"]
            + out["attention/wq"]
            + out["attention/wk"]
            + out["attention/wv"]
            + out["attention/wo"]
        )
        out["attention_total"] = self.n_layer * out["attention"]

        # MLP blocks
        out["mlp/norm"] = self.d_model
        out["mlp/w1"] = self.d_model * self.intermediate_size
        out["mlp/w2"] = self.intermediate_size * self.d_model
        out["mlp/w3"] = self.d_model * self.intermediate_size
        out["mlp"] = out["mlp/norm"] + out["mlp/w1"] + out["mlp/w2"] + out["mlp/w3"]
        out["mlp_total"] = self.n_layer * out["mlp"]

        # the transformer and the rest of it
        out["block"] = out["attention"] + out["mlp"]
        out["transformer"] = self.n_layer * out["block"]
        out["norm_final"] = self.d_model
        out["out_embedding"] = self.d_model * self.vocab_size

        # total
        out["total"] = (
            out["embedding"]
            + out["transformer"]
            + out["norm_final"]
            + out["out_embedding"]
        )
        return ModelParams(total=out["total"], per_component=out)

    def compute_activations(self, batch_size: int, dtype: str) -> ModelActivations:
        # Total: $8B + 16T + L \times (34TBH + 5AT^2B) + 4TBH$
        out = OrderedDict()
        B = batch_size
        TBH = self.n_ctx * B * self.d_model
        bytes_per_activation = 2 if dtype in ["bfloat16", "float16"] else 4
        bytes_per_long = 8

        # token and position embeddings
        out["embedding"] = self.n_ctx * B * bytes_per_long

        # attention blocks
        out["attention/norm"] = TBH * bytes_per_activation
        out["attention/kqv"] = TBH * bytes_per_activation

        # flash attention requires K, Q, V as well as two vectors l, m of length T (size BT)
        out["attention/attention_over_v"] = (
            TBH * 3 + 2 * self.n_ctx * B
        ) * bytes_per_activation

        out["attention/proj"] = TBH * bytes_per_activation
        out["attention"] = (
            out["attention/norm"]
            + out["attention/kqv"]
            + out["attention/attention_over_v"]
            + out["attention/proj"]
        )

        # MLP blocks
        out["mlp/norm"] = TBH * bytes_per_activation
        out["mlp/w1"] = TBH * bytes_per_activation
        out["mlp/w3"] = TBH * bytes_per_activation
        out["mlp/silu"] = self.n_ctx * B * self.intermediate_size * bytes_per_activation
        out["mlp/w2"] = self.n_ctx * B * self.intermediate_size * bytes_per_activation
        out["mlp"] = (
            out["mlp/norm"]
            + out["mlp/w1"]
            + out["mlp/w2"]
            + out["mlp/w3"]
            + out["mlp/silu"]
        )

        # the transformer and the rest of it
        out["block"] = out["attention"] + out["mlp"]
        out["transformer"] = self.n_layer * out["block"]

        # final layernorm and output projection
        out["norm_final"] = TBH * bytes_per_activation
        out["out_embedding"] = TBH * bytes_per_activation

        # total
        out["total"] = (
            out["embedding"]
            + out["transformer"]
            + out["norm_final"]
            + out["out_embedding"]
        )
        return ModelActivations(total=out["total"], per_component=out)

    def compute_flops(self) -> ModelFlops:
        # we only count Weight FLOPs,
        # FLOPS for all other layers (LayerNorm, Softmax, etc) and bias vector additian are effectively irrelevant
        # we count actual FLOPs, not MACs. Hence 2* all over the place
        # basically for any matrix multiply A (BxC) @ B (CxD) -> (BxD) flops are 2*B*C*D
        out = OrderedDict()
        d_head = self.d_model // self.n_head

        # attention blocks
        # 1) the projection to key, query, values
        out["attention/wq"] = 2 * self.n_ctx * self.d_model * self.d_model
        out["attention/wk"] = 2 * self.n_ctx * self.d_model * self.n_kv_head * d_head
        out["attention/wv"] = 2 * self.n_ctx * self.d_model * self.n_kv_head * d_head
        out["attention/wo"] = 2 * self.n_ctx * self.d_model * self.d_model
        # 2) calculating the attention scores
        out["attention/scores"] = 2 * self.n_ctx * self.n_ctx * self.d_model
        # 3) the reduction of the values (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out["attention/reduce"] = 2 * self.n_head * (self.n_ctx * self.n_ctx * d_head)
        out["attention"] = (
            out["attention/wq"]
            + out["attention/wk"]
            + out["attention/wv"]
            + out["attention/wo"]
            + out["attention/scores"]
            + out["attention/reduce"]
        )
        # MLP blocks
        out["mlp/w1"] = 2 * self.n_ctx * (self.d_model * self.intermediate_size)
        out["mlp/w2"] = 2 * self.n_ctx * (self.intermediate_size * self.d_model)
        out["mlp/w3"] = 2 * self.n_ctx * (self.d_model * self.intermediate_size)
        out["mlp"] = out["mlp/w1"] + out["mlp/w2"] + out["mlp/w3"]

        # the transformer and the rest of it
        out["block"] = out["attention"] + out["mlp"]
        out["transformer"] = self.n_layer * out["block"]
        out["out_embedding"] = 2 * self.n_ctx * (self.d_model * self.vocab_size)

        # forward,backward,total
        out["forward_total"] = out["transformer"] + out["out_embedding"]
        out["backward_total"] = (
            2 * out["forward_total"]
        )  # use common estimate of bwd = 2*fwd
        out["total"] = out["forward_total"] + out["backward_total"]

        return ModelFlops(
            total=out["total"],
            forward_total=out["forward_total"],
            backward_total=out["backward_total"],
            per_component=out,
        )


# 33M params
# 99% of params ar in the input and output embeddings
LLAMA3_33M_CONFIG = Llama3Config(
    model_name="llama3-33M",
    n_layer=2,
    n_head=2,
    n_kv_head=2,
    d_model=128,
    intermediate_size=512,
    learning_rate=6e-4,
)
# 272M params
# Similar to GPT-2 124M but with larger vocab size so more parameters in the embeddings
LLAMA3_272M_CONFIG = Llama3Config(
    model_name="llama3-272M",
    n_layer=8,
    n_head=8,
    n_kv_head=8,
    d_model=768,
    intermediate_size=3072,  # 4 * d_model
    learning_rate=6e-4,
)
# 1B params
# Llama3 scaled down to 1B params (1.4B to be exact)
LLAMA3_1B_CONFIG = Llama3Config(
    model_name="llama3-1B",
    n_layer=16,
    n_head=16,
    n_kv_head=8,
    d_model=2048,  # d_head = 128
    intermediate_size=7168,  # 3.5 * d_model
    learning_rate=3e-4,
)
# 2B params
# Similar architecture to GPT-2 1.5B in terms of n_layers, n_heads, d_model
# But with larger vocab size and using GQA
LLAMA3_2B_CONFIG = Llama3Config(
    model_name="llama3-2B",
    n_layer=48,
    n_head=24,
    n_kv_head=8,
    d_model=1536,  # n_head * 64
    intermediate_size=6144,  # 4 * d_model to nearest multiple of 2048
    learning_rate=3e-4,
)
# 8B params
LLAMA3_8B_CONFIG = Llama3Config(
    model_name="llama3-8B",
    n_layer=32,
    n_head=32,
    n_kv_head=8,
    d_model=4096,  # d_head = 128
    intermediate_size=14336,  # 3.5 * d_model
    learning_rate=3e-4,
)
# 70B params
LLAMA3_70B_CONFIG = Llama3Config(
    model_name="llama3-70B",
    n_layer=80,
    n_head=64,
    n_kv_head=8,
    d_model=8192,  # d_head = 128
    intermediate_size=22016,  # 3.5 * d_model
    learning_rate=1.5e-4,
)

# 405B params
LLAMA3_405B_CONFIG = Llama3Config(
    model_name="llama3-405B",
    n_layer=126,
    n_head=128,
    n_kv_head=8,
    d_model=16384,  # d_head = 128
    intermediate_size=53248,  # 3.25 * d_model
    learning_rate=8e-5,
)

LLAMA3_CONFIGS = {
    cfg.model_name: cfg
    for cfg in [
        LLAMA3_33M_CONFIG,
        LLAMA3_272M_CONFIG,
        LLAMA3_1B_CONFIG,
        LLAMA3_2B_CONFIG,
        LLAMA3_8B_CONFIG,
        LLAMA3_70B_CONFIG,
        LLAMA3_405B_CONFIG,
    ]
}


def get_llama3_model_config(name: str) -> Llama3Config:
    if name not in LLAMA3_CONFIGS:
        raise ValueError(
            f"No config found for name='{name}'. "
            f"Available configs: {', '.join(LLAMA3_CONFIGS.keys())}"
        )
    return LLAMA3_CONFIGS[name]
