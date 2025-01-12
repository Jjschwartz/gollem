from dataclasses import dataclass
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

    # Evaluation
    # Every how many steps to evaluate val loss.
    val_loss_every: int = 0
    # How many batches of val to average.
    val_max_steps: int = 20
    # How often to sample from the model.
    sample_every: int = 0

    # Device and memory management and optimization options
    # Device to use (autodetect by default).
    device: str = ""
    # Dtype to use for model
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    # Use GPU tensorcores.
    tensorcores: int = 0
