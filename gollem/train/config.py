from dataclasses import dataclass


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
    # Every how many steps to evaluate val loss.
    val_loss_every: int = 0
    # How many batches of val to average.
    val_max_steps: int = 20
    # How often to sample from the model.
    sample_every: int = 0
    # How often to save the model weights (0 = never).
    # This saves everything needed to load the model, but not the training state.
    # E.g. for running inference and evaluation on different model states.
    save_every: int = 0
    # How often to take a snapshot of the training state (0 = never).
    # Snapshotting is used for resuming training from a checkpoint.
    snapshot_every: int = 0

    # Device and memory management and optimization options
    # Device to use (autodetect by default, if empty or set to "auto").
    device: str = "auto"
    # Dtype to use for model (float32, float16, bfloat16).
    dtype: str = "float32"
    # Use GPU tensorcores.
    tensorcores: bool = False

    # Logging
    # Whether to use wandb.
    use_wandb: bool = False

    def __post_init__(self):
        assert self.dtype in ["float32", "float16", "bfloat16"]
