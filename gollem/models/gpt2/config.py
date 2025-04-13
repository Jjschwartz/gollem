"""Utility model config stuff."""

import math
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import field
from typing import Callable

import torch

from gollem.models.config import ModelActivations
from gollem.models.config import ModelConfig
from gollem.models.config import ModelFlops
from gollem.models.config import ModelParams
from gollem.models.gpt2.model import GPT
from gollem.models.model import BaseLLM
from gollem.tokenizer import BaseTokenizer
from gollem.tokenizer import get_tokenizer
from gollem.utils import print0


@dataclass
class GPT2Config(ModelConfig):
    # Name of the model
    model_name: str = field(default="gpt2")
    # Context length
    n_ctx: int = field(default=1024)
    # Number of layers
    n_layer: int = field(default=12)
    # Number of attention heads
    n_head: int = field(default=12)
    # Model dimension
    d_model: int = field(default=768)
    # MLP dimension
    d_mlp: int = field(default=4 * 768)
    # Vocabulary size
    vocab_size: int = field(default=50257)
    # Whether to use layer normalization bias
    ln_bias: bool = field(default=True)
    # Whether to use MLP bias
    mlp_bias: bool = field(default=True)
    # Whether to share embedding parameters
    share_embd_params: bool = field(default=True)
    # Learning rate
    learning_rate: float = field(default=1e-4)
    # Learning rate warmup iterations
    warmup_iters: int = field(default=0)
    # Learning rate decay fraction (final learning rate = learning_rate * learning_rate_decay_frac)
    learning_rate_decay_frac: float = field(default=1.0)
    # Weight decay
    weight_decay: float = field(default=0.0)
    # Max gradient magnitude.
    grad_clip: float = field(default=1.0)
    # adamw beta params
    betas: tuple[float, float] = field(default=(0.9, 0.95))
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
        return get_tokenizer("gpt2")

    def get_model_and_optimizer(
        self, device: str | torch.device
    ) -> tuple[BaseLLM, torch.optim.Optimizer]:
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

    def get_params(self) -> ModelParams:
        out = OrderedDict()

        # token and position embeddings
        out["embedding/position"] = self.n_ctx * self.d_model
        out["embedding/token"] = self.vocab_size * self.d_model
        out["embedding"] = out["embedding/position"] + out["embedding/token"]

        # attention blocks
        out["attention/ln"] = self.d_model + int(self.ln_bias) * self.d_model
        out["attention/kqv"] = self.d_model * 3 * self.d_model
        out["attention/proj"] = self.d_model**2
        out["attention"] = (
            out["attention/ln"] + out["attention/kqv"] + out["attention/proj"]
        )

        # MLP blocks
        out["mlp/ln"] = self.d_model + int(self.ln_bias) * self.d_model
        out["mlp/ffw"] = self.d_model * self.d_mlp + int(self.ln_bias) * self.d_mlp
        out["mlp/proj"] = self.d_mlp * self.d_model + int(self.ln_bias) * self.d_model
        out["mlp"] = out["mlp/ln"] + out["mlp/ffw"] + out["mlp/proj"]

        # the transformer and the rest of it
        out["block"] = out["attention"] + out["mlp"]
        out["transformer"] = self.n_layer * out["block"]
        out["ln_f"] = self.d_model + int(self.ln_bias) * self.d_model  # final layernorm
        if self.share_embd_params:
            # 0 because of parameter sharing. This layer uses the weights from the embedding layer
            out["out_embedding"] = 0
        else:
            out["out_embedding"] = self.d_model * self.vocab_size

        # total
        out["total"] = (
            out["embedding"] + out["transformer"] + out["ln_f"] + out["out_embedding"]
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
        out["embedding/position"] = self.n_ctx * bytes_per_long
        out["embedding/token"] = self.n_ctx * B * bytes_per_long
        out["embedding"] = out["embedding/position"] + out["embedding/token"]

        # attention blocks
        out["attention/ln"] = TBH * bytes_per_activation
        out["attention/kqv"] = TBH * bytes_per_activation
        if self.flash:
            # when using flash attention a bunch of optimizations are done
            out["attention/qk_matmul"] = 0
            out["attention/softmax"] = 0
            # flash attention requires K, Q, V as well as two vectors l, m of length T (size BT)
            out["attention/attention_over_v"] = (
                TBH * 3 + 2 * self.n_ctx * B
            ) * bytes_per_activation
        else:
            out["attention/qk_matmul"] = TBH * 2 * bytes_per_activation
            out["attention/softmax"] = (
                self.n_head * self.n_ctx**2 * B * bytes_per_activation
            )
            out["attention/attention_over_v"] = (
                TBH + self.n_head * self.n_ctx**2 * B
            ) * bytes_per_activation
        out["attention/proj"] = TBH * bytes_per_activation
        out["attention"] = (
            out["attention/ln"]
            + out["attention/kqv"]
            + out["attention/qk_matmul"]
            + out["attention/softmax"]
            + out["attention/attention_over_v"]
            + out["attention/proj"]
        )

        # MLP blocks
        out["mlp/ln"] = TBH * bytes_per_activation
        out["mlp/ffw"] = TBH * bytes_per_activation
        out["mlp/ffw_activation"] = self.n_ctx * B * self.d_mlp * bytes_per_activation
        out["mlp/proj"] = self.n_ctx * B * self.d_mlp * bytes_per_activation
        out["mlp"] = (
            out["mlp/ln"] + out["mlp/ffw"] + out["mlp/ffw_activation"] + out["mlp/proj"]
        )

        # the transformer and the rest of it
        out["block"] = out["attention"] + out["mlp"]
        out["transformer"] = self.n_layer * out["block"]

        # final layernorm and output projection
        out["ln_f"] = TBH * bytes_per_activation
        out["out_embedding"] = TBH * bytes_per_activation

        # total
        out["total"] = (
            out["embedding"] + out["transformer"] + out["ln_f"] + out["out_embedding"]
        )
        return ModelActivations(total=out["total"], per_component=out)

    def compute_flops(self) -> ModelFlops:
        # we only count Weight FLOPs,
        # FLOPS for all other layers (LayerNorm, Softmax, etc) and bias vector additian are effectively irrelevant
        # we count actual FLOPs, not MACs. Hence 2* all over the place
        # basically for any matrix multiply A (BxC) @ B (CxD) -> (BxD) flops are 2*B*C*D

        out = OrderedDict()
        head_size = self.d_model // self.n_head

        # attention blocks
        # 1) the projection to key, query, values
        out["attention/kqv"] = 2 * self.n_ctx * (self.d_model * 3 * self.d_model)
        # 2) calculating the attention scores
        out["attention/scores"] = 2 * self.n_ctx * self.n_ctx * self.d_model
        # 3) the reduction of the values (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out["attention/reduce"] = (
            2 * self.n_head * (self.n_ctx * self.n_ctx * head_size)
        )
        # 4) the final linear projection
        out["attention/proj"] = 2 * self.n_ctx * (self.d_model * self.d_model)
        out["attention"] = sum(
            out["attention/" + k] for k in ["kqv", "scores", "reduce", "proj"]
        )

        # MLP blocks
        out["mlp/ffw1"] = 2 * self.n_ctx * (self.d_model * self.d_mlp)
        out["mlp/ffw2"] = 2 * self.n_ctx * (self.d_mlp * self.d_model)
        out["mlp"] = out["mlp/ffw1"] + out["mlp/ffw2"]

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
    learning_rate=0.0006,
    warmup_iters=700,
    learning_rate_decay_frac=0.0,
    weight_decay=0.1,
)

# 774M params
GPT2_LARGE_CONFIG = GPT2Config(
    model_name="gpt2-large",
    n_layer=36,
    n_head=20,
    d_model=1280,
    d_mlp=4 * 1280,
    learning_rate=0.0006,
    warmup_iters=700,
    learning_rate_decay_frac=0.0,
    weight_decay=0.1,
)

# 1558M params
GPT2_XL_CONFIG = GPT2Config(
    model_name="gpt2-xl",
    n_layer=48,
    n_head=25,
    d_model=1600,
    d_mlp=4 * 1600,
    learning_rate=0.0006,
    warmup_iters=700,
    learning_rate_decay_frac=0.0,
    weight_decay=0.1,
)


GPT2_CONFIGS = {
    cfg.model_name: cfg
    for cfg in [GPT2_CONFIG, GPT2_MEDIUM_CONFIG, GPT2_LARGE_CONFIG, GPT2_XL_CONFIG]
}


def get_gpt2_model_config(name: str) -> GPT2Config:
    if name not in GPT2_CONFIGS:
        raise ValueError(
            f"No config found for name='{name}'. "
            f"Available configs: {', '.join(GPT2_CONFIGS.keys())}"
        )
    return GPT2_CONFIGS[name]
