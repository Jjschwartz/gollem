from dataclasses import dataclass
from dataclasses import field
from typing import Literal


@dataclass
class TrainConfig:
    # Output dir where logs and checkpoints will be saved.
    output_dir: str = ""
    # RNG seed.
    seed: int = 42

    # Batch size per individual training forward pass in #sequences.
    batch_size: int = 4
    # Max sequence length in batch.
    seq_len: int = 64
    # Total desired batch size in #tokens per model update.
    total_batch_size: int = 256
    # Number of training iterations (i.e. batches of `total_batch_size`).
    num_iterations: int = 10
    # Number of gradient accumulation steps (i.e. minibatches).
    # This will be calculated automatically (whether set or not).
    grad_accum_steps: int = field(default=-1)
    # Every how many steps to evaluate val loss.
    val_loss_every: int = 0
    # How many batches of val to average.
    val_max_steps: int = 20
    # How often to sample from the model.
    sample_every: int = 0
    # Whether to save the model checkpoint.
    save_every: int = 0

    # Device and memory management and optimization options
    # Device to use (autodetect by default, if empty or set to "auto").
    device: str = "auto"
    # Dtype to use for model
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    # Use GPU tensorcores.
    tensorcores: bool = False

    # Logging
    # Whether to use wandb.
    use_wandb: bool = False

    def __post_init__(self):
        # calculate the number of gradient accumulation steps from the desired total batch
        # size, minibatch size, and sequence length
        # Having multiple steps allows us to update model with larger batch size than can
        # be handled by the hardware in a single batch
        B, T = self.batch_size, self.seq_len
        tokens_per_fwdbwd = B * T
        assert self.total_batch_size % tokens_per_fwdbwd == 0
        self.grad_accum_steps = self.total_batch_size // tokens_per_fwdbwd
