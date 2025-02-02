"""Utility model config stuff."""

import math
from dataclasses import dataclass
from typing import Callable
from typing import Tuple

import torch

from gollem.models.config import ModelConfig
from gollem.models.gpt2.model import GPT
from gollem.models.model import BaseLLM
from gollem.tokenizer import BaseTokenizer
from gollem.tokenizer import get_tokenizer


@dataclass
class GPT2Config(ModelConfig):
    # Name of the model
    model_name: str = "gpt2"
    # Context length
    n_ctx: int = 1024
    # Number of layers
    n_layer: int = 12
    # Number of attention heads
    n_head: int = 12
    # Model dimension
    d_model: int = 768
    # MLP dimension
    d_mlp: int = 4 * 768
    # Vocabulary size
    vocab_size: int = 50257
    # Whether to use layer normalization bias
    ln_bias: bool = True
    # Whether to use MLP bias
    mlp_bias: bool = True
    # Whether to share embedding parameters
    share_embd_params: bool = True
    # Learning rate.
    learning_rate: float = 1e-4
    # Learning rate warmup iterations.
    warmup_iters: int = 0
    # Learning rate decay fraction (final learning rate = learning_rate * learning_rate_decay_frac).
    learning_rate_decay_frac: float = 1.0
    # Weight decay.
    weight_decay: float = 0.0
    # Max gradient magnitude.
    grad_clip: float = 1.0
    # adamw beta params
    betas: Tuple[float, float] = (0.9, 0.95)
    # Use fused version of AdamW optimizer.
    fused_adamw: bool = True
    # Use ZeroRedundancyOptimizer.
    zero_optimizer: bool = True
    # Use flash attention.
    flash: bool = True
    # Torch.compile the model.
    compile: bool = True
    # Load from pretrained weights
    from_pretrained: bool = False

    def get_tokenizer(self) -> BaseTokenizer:
        return get_tokenizer("gpt2")

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
            print("compiling the model...")
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


# 124M params
GPT2_CONFIG = GPT2Config(
    model_name="gpt2",
    n_layer=12,
    n_head=12,
    d_model=768,
    d_mlp=4 * 768,
    learning_rate=0.0006,
    warmup_iters=700,
    learning_rate_decay_frac=0.0,
    weight_decay=0.1,
)
# 350M params
GPT2_MEDIUM_CONFIG = GPT2Config(
    model_name="gpt2-medium",
    n_layer=24,
    n_head=16,
    d_model=1024,
    d_mlp=4 * 1024,
)

# 774M params
GPT2_LARGE_CONFIG = GPT2Config(
    model_name="gpt2-large",
    n_layer=36,
    n_head=20,
    d_model=1280,
    d_mlp=4 * 1280,
)

# 1558M params
GPT2_XL_CONFIG = GPT2Config(
    model_name="gpt2-xl",
    n_layer=48,
    n_head=25,
    d_model=1600,
    d_mlp=4 * 1600,
)
