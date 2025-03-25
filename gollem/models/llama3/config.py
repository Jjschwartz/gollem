"""Utility model config stuff."""

import math
from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import Tuple

import torch

from gollem.models.config import ModelConfig
from gollem.models.gpt2.model import GPT
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
    n_ctx: int = field(default=4096)
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
    rope_theta: float = field(default=500000)
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

    def get_tokenizer(self) -> BaseTokenizer:
        return get_tokenizer("llama-3")

    def get_model_and_optimizer(
        self, device: str | torch.device
    ) -> Tuple[BaseLLM, torch.optim.Optimizer]:
        if isinstance(device, str):
            device_type = "cuda" if "cuda" in device else "cpu"
        else:
            device_type = device.type
        model = GPT.from_pretrained(self) if self.from_pretrained else GPT(self)
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


# 150m params
# TODO actually calculate the number of params
LLAMA3_150M_CONFIG = Llama3Config(
    model_name="llama-3.1-150M",
    n_layer=8,
    n_attn_heads=8,
    n_kv_heads=8,
    d_model=1024,
    intermediate_size=4096,  # 4 * d_model
    learning_rate=6e-4,
)
# 3B params
# TODO actually calculate the number of params
LLAMA3_3B_CONFIG = Llama3Config(
    model_name="llama-3.1-3B",
    n_layer=16,
    n_attn_heads=16,
    n_kv_heads=8,
    d_model=2048,
    intermediate_size=7168,  # 3.5 * d_model
    learning_rate=3e-4,
)
# 8B params
LLAMA3_8B_CONFIG = Llama3Config(
    model_name="llama-3.1-8B",
    n_layer=32,
    n_attn_heads=32,
    n_kv_heads=8,
    d_model=4096,
    intermediate_size=14336,  # 3.5 * d_model
    learning_rate=3e-4,
)
# 70B params
LLAMA3_70B_CONFIG = Llama3Config(
    model_name="llama-3.1-70B",
    n_layer=80,
    n_attn_heads=64,
    n_kv_heads=8,
    d_model=8192,
    intermediate_size=22016,  # 3.5 * d_model
    learning_rate=1.5e-4,
)

# 405B params
LLAMA3_405B_CONFIG = Llama3Config(
    model_name="llama-3.1-405B",
    n_layer=126,
    n_attn_heads=128,
    n_kv_heads=8,
    d_model=16384,
    intermediate_size=53248,  # 3.25 * d_model
    learning_rate=8e-5,
)


def get_llama3_model_config(name: str) -> Llama3Config:
    for cfg in [
        LLAMA3_150M_CONFIG,
        LLAMA3_3B_CONFIG,
        LLAMA3_8B_CONFIG,
        LLAMA3_70B_CONFIG,
        LLAMA3_405B_CONFIG,
    ]:
        if cfg.model_name == name:
            return cfg
    raise ValueError(f"No config found for name='{name}'")
