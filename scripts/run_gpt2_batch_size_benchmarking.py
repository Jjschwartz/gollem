"""Measure GPT-2 model efficiency with different batch sizes."""

import csv
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from pprint import pprint

import torch
from gollem.data import load_dataset
from gollem.models.gpt2.config import GPT2Config
from gollem.train.config import TrainConfig
from gollem.train.core import run


BASE_TRAIN_CONFIG = TrainConfig(
    output_dir="",
    batch_size=4,
    seq_len=512,
    total_batch_size=524288,
    num_iterations=100,
    device="",
    use_wandb=False,
    tensorcores=True,
    dtype="float16",
)

MODEL_CONFIG = GPT2Config(
    model_name="gpt2",
    n_layer=12,
    n_head=12,
    d_model=768,
    d_mlp=4 * 768,
    learning_rate=0.0006,
    warmup_iters=700,
    learning_rate_decay_frac=0.0,
    weight_decay=0.1,
    flash=True,
    compile=True,
    fused_adamw=True,
)

DATASET_CONFIG = load_dataset(
    "tinystories",
    encoder=MODEL_CONFIG.get_tokenizer(),
)

THIS_DIR = Path(__file__).parent
RESULTS_DIR = THIS_DIR.parent / "results" / "gpt2_batch_size_benchmarking"


def run_benchmark(debug: bool = False):
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_file_path = RESULTS_DIR / f"results_{time_str}.csv"
    results_file_path.parent.mkdir(parents=True, exist_ok=True)

    # get max mem available
    if torch.cuda.is_available():
        max_mem = torch.cuda.get_device_properties(0).total_memory
    else:
        max_mem = 0

    # 512
    seq_len = BASE_TRAIN_CONFIG.seq_len
    # batch size = 2^batch_size_power * seq_len
    batch_size_power_range = [0, 10]
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
        base_train_kwargs.pop("grad_accum_steps")
        train_config = TrainConfig(**base_train_kwargs)
        assert train_config.grad_accum_steps == 1

        run_name = f"batch_size={batch_size_tokens}"

        print("=" * 100)
        print(f"Run {i + 1}/{total_num_runs}: {run_name}")
        print("=" * 100)
        if debug:
            continue

        start_time = time.time()
        run_results = run(DATASET_CONFIG, MODEL_CONFIG, train_config)
        end_time = time.time()
        time_taken = end_time - start_time

        print(f"Time={time_taken:.2f}s")
        pprint(run_results)

        if not result_headers:
            result_headers = ["run_name", "total_time"] + list(run_results.keys())
            with open(results_file_path, "w") as output_file:
                results_writer = csv.DictWriter(output_file, fieldnames=result_headers)
                results_writer.writeheader()

        with open(results_file_path, "a") as output_file:
            results_writer = csv.DictWriter(output_file, fieldnames=result_headers)
            results_writer.writerow(
                {
                    "run_name": run_name,
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
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run_benchmark(args.debug)
