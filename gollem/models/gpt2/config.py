"""Utility model config stuff."""

import math
from dataclasses import asdict
from dataclasses import dataclass
from typing import Callable
from typing import Tuple

import tiktoken
import torch

from gollem.models.config import ModelConfig
from gollem.models.gpt2.model import GPT
from gollem.models.model import BaseLLM


@dataclass
class GPT2Config(ModelConfig):
    model_name: str = "gpt2"
    n_ctx: int = 1024
    vocab_size: int = 50257
    ln_bias: bool = True
    mlp_bias: bool = True
    share_embd_params: bool = True

    # Optimizer hyper params
    # Learning rate.
    learning_rate: float = 1e-4
    # Learning rate warmup iterations.
    warmup_iters: int = 0
    # Learning rate decay fraction.
    learning_rate_decay_frac: float = 1.0
    # Weight decay.
    weight_decay: float = 0.0
    # Max gradient magnitude.
    grad_clip: float = 1.0
    # adamw beta params
    betas: Tuple[float, float] = (0.9, 0.95)
    # Use fused version of AdamW optimizer.
    fused_adamw: bool = False

    # Model performance options
    # Use flash attention.
    flash: bool = False
    # Torch.compile the model.
    compile: bool = False

    # Load from pretrained weights
    from_pretrained: bool = False

    def get_tokenizer(self) -> tiktoken.Encoding:
        return tiktoken.get_encoding("gpt2")

    def get_model_and_optimizer(
        self, device: str
    ) -> Tuple[BaseLLM, torch.optim.Optimizer]:
        device_type = "cuda" if "cuda" in device else "cpu"
        model = GPT.from_pretrained(self) if self.from_pretrained else GPT(self)
        model.to(device)
        optimizer = model.configure_optimizers(device_type=device_type)

        if self.compile:
            print("compiling the model...")
            torch.compile(model)

        return model, optimizer

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


MODEL_CONFIGS = {
    # 124M params
    "gpt2": GPT2Config(
        n_layer=12,
        n_head=12,
        d_model=768,
        d_mlp=4 * 768,
    ),
    # 350M params
    "gpt2-medium": GPT2Config(
        n_layer=24,
        n_head=16,
        d_model=1024,
        d_mlp=4 * 1024,
    ),
    # 774M params
    "gpt2-large": GPT2Config(
        n_layer=36,
        n_head=20,
        d_model=1280,
        d_mlp=4 * 1280,
    ),
    # 1558M params
    "gpt2-xl": GPT2Config(
        n_layer=48,
        n_head=25,
        d_model=1600,
        d_mlp=4 * 1600,
    ),
}


def get_model_config(name: str, **kwargs) -> ModelConfig:
    """Get named model config.

    Any defaults will be overwridden by values in kwargs.
    """
    assert name in MODEL_CONFIGS
    if not kwargs:
        return MODEL_CONFIGS[name]

    base_cfg_kwargs = asdict(MODEL_CONFIGS[name])
    base_cfg_kwargs.update(kwargs)
    return ModelConfig(**base_cfg_kwargs)
