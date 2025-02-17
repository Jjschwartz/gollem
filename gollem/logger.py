import pprint
from pathlib import Path
from warnings import warn


class RunLogger:
    """Logger that handles logging to wandb, stdout and local file."""

    def __init__(
        self,
        run_id: str,
        run_name: str,
        is_master_process: bool,
        output_dir: Path | None = None,
        use_wandb: bool = False,
        resume_from: str | None = None,
    ):
        """Initialize the logger.

        Args:
            output_dir: Path to output directory. If None, won't log to file.
            use_wandb: Whether to log to wandb
            resume_from: wandb resume_from arg https://docs.wandb.ai/ref/python/init/
                Should be of the form "{run_id}?_step={step}"
        """
        self.run_id = run_id
        self.run_name = run_name
        self.is_master_process = is_master_process
        self.output_dir = output_dir
        if output_dir is not None and self.is_master_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logfile = output_dir / "main.log"
            if self.logfile.exists():
                if resume_from is None:
                    warn(f"Log file {self.logfile} already exists, overwriting")
                    self.logfile.write_text("")
                # else resuming from existing log file (this may be slightly messy
                # if new run is redoing steps from old run, but that's fine)
            else:
                self.logfile.write_text("")
        else:
            self.logfile = None

        self.use_wandb = use_wandb
        if use_wandb and self.is_master_process:
            try:
                import wandb

                self.wandb = wandb
                self.wandb.init(
                    project="gollem",
                    name=run_name,
                    id=run_id,
                    resume="auto",
                    # resume run if specified, and overwrite any overlapping steps
                    # resume_from=resume_from,
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
