"""Measure GPT-2 model efficiency with different batch sizes."""

import csv
import json
import os
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Union

import pyrallis
import torch
from gollem.data import load_dataset
from gollem.models.gpt2.config import GPT2Config
from gollem.models.gpt2.config import get_gpt2_model_config
from gollem.train.config import TrainConfig
from gollem.train.core import run


THIS_DIR = Path(__file__).parent

BASE_TRAIN_CONFIG = TrainConfig(
    output_dir="",
    batch_size=4,  # will be overriden
    seq_len=1024,
    total_batch_size=524288,  # will be overriden
    num_iterations=100,
    device="cuda",  # assuming cuda since we use bfloat16
    use_wandb=False,
    tensorcores=True,
    dtype="bfloat16",
)


def run_benchmark(model_config: GPT2Config, train_config: TrainConfig):
    debug = os.environ.get("GOLLEM_DEBUG", "0") == "1"

    results_dir = (
        THIS_DIR.parent
        / "results"
        / f"{model_config.model_name}_batch_size_benchmarking"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    results_file_path = results_dir / f"results_{time_str}.csv"
    # save model config
    if not debug:
        with open(results_file_path.with_suffix(".json"), "w") as f:
            json.dump(asdict(model_config), f)
    else:
        print("train config:")
        pprint(asdict(train_config))

    # get max mem available
    if torch.cuda.is_available():
        max_mem = torch.cuda.get_device_properties(0).total_memory
    else:
        max_mem = 0

    dataset_config = load_dataset(
        "tinystories",
        encoder=model_config.get_tokenizer(),
        include_val_set=False,
    )

    seq_len = train_config.seq_len
    # batch size = 2^batch_size_power * seq_len
    batch_size_power_range = [0, 12]
    total_num_runs = batch_size_power_range[1] - batch_size_power_range[0]
    print(f"Total number of runs: {total_num_runs}")

    result_headers = []
    for i, batch_pow in enumerate(range(*batch_size_power_range)):
        batch_size_num_seqs = 2**batch_pow
        batch_size_tokens = batch_size_num_seqs * seq_len
        total_batch_size = batch_size_tokens

        base_train_kwargs = asdict(train_config)
        base_train_kwargs.update(
            {
                "batch_size": batch_size_num_seqs,
                "total_batch_size": total_batch_size,
            }
        )
        train_config = TrainConfig(**base_train_kwargs)
        tokens_per_fwdbwd = batch_size_num_seqs * seq_len
        assert train_config.total_batch_size % tokens_per_fwdbwd == 0
        grad_accum_steps = train_config.total_batch_size // tokens_per_fwdbwd
        assert grad_accum_steps == 1

        run_name = f"batch_size={batch_size_tokens} ({batch_size_num_seqs}x{seq_len})"

        print("=" * 100)
        print(f"Run {i + 1}/{total_num_runs}: {run_name}")
        print("=" * 100)
        if debug:
            continue

        start_time = time.time()
        run_results = run(
            dataset_config=dataset_config,
            model_config=model_config,
            train_config=train_config,
        )
        end_time = time.time()
        time_taken = end_time - start_time

        print(f"Time={time_taken:.2f}s")
        pprint(run_results)

        if not result_headers:
            result_headers = [
                "run_name",
                "mini_batch_size",
                "total_batch_size",
                "total_time",
            ] + list(run_results.keys())
            with open(results_file_path, "w") as output_file:
                results_writer = csv.DictWriter(output_file, fieldnames=result_headers)
                results_writer.writeheader()

        with open(results_file_path, "a") as output_file:
            results_writer = csv.DictWriter(output_file, fieldnames=result_headers)
            results_writer.writerow(
                {
                    "run_name": run_name,
                    "mini_batch_size": batch_size_tokens,
                    "total_batch_size": total_batch_size,
                    "total_time": time_taken,
                    **run_results,
                }
            )

        if max_mem and "peak_mem_usage" in run_results:
            peak_mem = run_results["peak_mem_usage"]
            if peak_mem > 0.9 * max_mem:
                print(f"Peak mem: {peak_mem / 1e9:.2f}GB (90% of max mem). Stopping...")
                break


@dataclass
class RunConfig:
    # Model config to use if using pre-defined model configs
    # (choices:"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
    model_name: Union[str, None] = field(default=None)
    # GPT2Config
    model: GPT2Config = field(default_factory=GPT2Config)
    # TrainConfig
    train: TrainConfig = field(default_factory=TrainConfig)


def main():
    try:
        print("Parsing config")
        cfg = pyrallis.parse(config_class=RunConfig)
    except Exception as e:
        print("Failed to parse config")
        print(e)
        sys.exit(1)

    if cfg.model_name is None:
        model_cfg = cfg.model
    else:
        # Use the named model config and override any parameters that were explicitly
        # set (these are the ones that are different from the default config values)
        default_model_kwargs = asdict(GPT2Config())
        model_cfg_kwargs = asdict(cfg.model)
        changes = {}
        for k, v in model_cfg_kwargs.items():
            if default_model_kwargs[k] != v:
                changes[k] = v
        model_cfg = get_gpt2_model_config(cfg.model_name)
        model_cfg = GPT2Config.override(model_cfg, **changes)

    # Update the base train config with any changes supplied via the CLI
    train_cfg = cfg.train
    default_train_kwargs = asdict(TrainConfig())
    train_cfg_kwargs = asdict(train_cfg)
    changes = {}
    for k, v in train_cfg_kwargs.items():
        if default_train_kwargs[k] != v:
            changes[k] = v
    base_train_kwargs = asdict(BASE_TRAIN_CONFIG)
    base_train_kwargs.update(changes)
    train_cfg = TrainConfig(**base_train_kwargs)

    run_benchmark(model_cfg, train_cfg)


if __name__ == "__main__":
    main()
