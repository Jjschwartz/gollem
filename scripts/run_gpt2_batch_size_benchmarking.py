"""Measure GPT-2 model efficiency with different batch sizes."""

import csv
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from pprint import pprint

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


def run_benchmark(
    model_config: GPT2Config,
    use_activation_checkpointing: bool = False,
    debug: bool = False,
):
    results_dir = (
        THIS_DIR.parent
        / "results"
        / f"{model_config.model_name}_batch_size_benchmarking"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if use_activation_checkpointing:
        results_file_path = results_dir / f"results_{time_str}_ac.csv"
    else:
        results_file_path = results_dir / f"results_{time_str}.csv"

    if use_activation_checkpointing:
        model_config.activation_checkpointing = use_activation_checkpointing

    # save model config
    if not debug:
        with open(results_file_path.with_suffix(".json"), "w") as f:
            json.dump(asdict(model_config), f)

    # get max mem available
    if torch.cuda.is_available():
        max_mem = torch.cuda.get_device_properties(0).total_memory
    else:
        max_mem = 0

    dataset_config = load_dataset(
        "tinystories",
        encoder=model_config.get_tokenizer(),
    )
    # disable loading validation set since we don't need it
    dataset_config.val_data_pattern = None

    seq_len = BASE_TRAIN_CONFIG.seq_len
    # batch size = 2^batch_size_power * seq_len
    batch_size_power_range = [0, 12]
    total_num_runs = batch_size_power_range[1] - batch_size_power_range[0]
    print(f"Total number of runs: {total_num_runs}")

    result_headers = []
    for i, batch_pow in enumerate(range(*batch_size_power_range)):
        batch_size_num_seqs = 2**batch_pow
        batch_size_tokens = batch_size_num_seqs * seq_len
        total_batch_size = batch_size_tokens

        base_train_kwargs = asdict(BASE_TRAIN_CONFIG)
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        type=str,
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    )
    parser.add_argument("-a", "--use_activation_checkpointing", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    model_config = get_gpt2_model_config(args.model_name)
    run_benchmark(
        model_config,
        args.use_activation_checkpointing,
        args.debug,
    )
