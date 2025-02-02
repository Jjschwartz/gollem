import pprint
from pathlib import Path
from warnings import warn


class RunLogger:
    """Logger that handles logging to wandb, stdout and local file."""

    def __init__(
        self,
        run_name: str,
        is_master_process: bool,
        output_dir: Path | None = None,
        use_wandb: bool = False,
    ):
        """Initialize the logger.

        Args:
            output_dir: Path to output directory. If None, won't log to file.
            use_wandb: Whether to log to wandb
        """
        self.run_name = run_name
        self.is_master_process = is_master_process
        self.output_dir = output_dir
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logfile = output_dir / "main.log"
            if self.logfile.exists():
                warn(f"Log file {self.logfile} already exists, overwriting")
            self.logfile.write_text("")
        else:
            self.logfile = None

        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb

                self.wandb = wandb
                self.wandb.init(
                    project="gollem",
                    name=run_name,
                )
            except ImportError:
                print("wandb not installed, disabling wandb logging")
                self.use_wandb = False

    def log(self, message: str, master_only: bool = True) -> None:
        """Log a message to all enabled outputs."""
        if master_only and not self.is_master_process:
            return
        print(message)
        if self.logfile:
            with open(self.logfile, "a") as f:
                f.write(message + "\n")

    def log_metrics(
        self, metrics: dict, step: int | None = None, master_only: bool = True
    ) -> None:
        """Log metrics to all enabled outputs.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number for the metrics
        """
        if master_only and not self.is_master_process:
            return

        # Log to stdout
        log_str = f"Step {step}: " if step is not None else ""
        log_str += ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        print(log_str)

        # Log to wandb
        if self.use_wandb:
            self.wandb.log(metrics, step=step)

        # Log to file
        if self.logfile:
            with open(self.logfile, "a") as f:
                f.write(log_str + "\n")

    def log_config(self, config: dict, master_only: bool = True) -> None:
        """Log configuration parameters.

        Args:
            config: Dictionary of config parameters
        """
        if master_only and not self.is_master_process:
            return

        if self.use_wandb:
            self.wandb.config.update(config)

        # Log to stdout
        print("\nConfig:")
        config_str = pprint.pformat(config)
        print(config_str)
        # for k, v in config.items():
        #     if isinstance(v, dict):
        #         v_str = pprint.pformat(v, indent=4)
        #         k_str =
        #     else:
        #         v_str = str(v)
        #     print(f"  {k}: {v_str}")

        # Log to file
        if self.logfile:
            with open(self.logfile, "a") as f:
                f.write(config_str)
