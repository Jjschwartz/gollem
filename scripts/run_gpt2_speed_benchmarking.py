"""Measure tokens/second for GPT-2 models with different configurations."""

import csv
import itertools
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any

import torch
from gollem.data import load_dataset
from gollem.models.gpt2.config import GPT2_CONFIG
from gollem.models.gpt2.config import GPT2Config
from gollem.train.config import TrainConfig
from gollem.train.core import run
from gollem.train.utils import check_dtype_support
from gollem.train.utils import check_tensorcores_support


BASE_TRAIN_CONFIG = TrainConfig(
    output_dir="",
    batch_size=4,
    seq_len=64,
    total_batch_size=256,
    num_iterations=100,
    device="",
    use_wandb=False,
)

BASE_MODEL_CONFIG = GPT2_CONFIG

DATASET_CONFIG = load_dataset(
    "tinystories",
    encoder=BASE_MODEL_CONFIG.get_tokenizer(),
)

THIS_DIR = Path(__file__).parent
RESULTS_DIR = THIS_DIR.parent / "results" / "gpt2_speed_benchmarking"


def get_all_settings_combos(
    settings: list[tuple[str, list[Any]]],
) -> list[dict[str, Any]]:
    all_settings: list[dict[str, Any]] = []
    param_names: list[str] = [x[0] for x in settings]
    for param_value_combination in itertools.product(*[x[1] for x in settings]):
        all_settings.append(dict(zip(param_names, param_value_combination)))
    return all_settings


def get_train_settings_to_test() -> list[dict[str, Any]]:
    settings: list[tuple[str, list[Any]]] = [
        ("dtype", check_dtype_support()),
    ]
    if check_tensorcores_support():
        settings.append(("tensorcores", [True, False]))
    return get_all_settings_combos(settings)


def get_model_settings_to_test() -> list[dict[str, Any]]:
    settings = [
        ("compile", [True, False]),
        ("flash", [True, False]),
        ("fused_adamw", [True, False]),
    ]
    return get_all_settings_combos(settings)


def get_run_name(train_kwargs: dict[str, Any], model_kwargs: dict[str, Any]) -> str:
    setting_names = []
    for kwargs in [train_kwargs, model_kwargs]:
        for name, value in kwargs.items():
            if isinstance(value, bool):
                if value:
                    setting_names.append(name)
            else:
                setting_names.append(f"{name}={value}")
    return "_".join(setting_names)


def run_benchmark(debug: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_file_path = RESULTS_DIR / f"results_{time_str}.csv"
    results_file_path.parent.mkdir(parents=True, exist_ok=True)

    train_settings = get_train_settings_to_test()
    print("Train settings:")
    pprint(train_settings)
    model_settings = get_model_settings_to_test()
    print("Model settings:")
    pprint(model_settings)
    total_num_runs = len(train_settings) * len(model_settings)
    print(f"Total number of runs: {total_num_runs}")

    result_headers = ["run_name", "time", "mean_tps", "peak_mem_usage"]
    if not debug:
        with open(results_file_path, "w") as output_file:
            results_writer = csv.DictWriter(output_file, fieldnames=result_headers)
            results_writer.writeheader()

    for i, (train_kwargs, model_kwargs) in enumerate(
        itertools.product(train_settings, model_settings)
    ):
        base_train_kwargs = asdict(BASE_TRAIN_CONFIG)
        base_train_kwargs.update(train_kwargs)
        base_train_kwargs.pop("grad_accum_steps")

        train_config = TrainConfig(**base_train_kwargs)
        train_config = TrainConfig(**base_train_kwargs)
        base_model_kwargs = asdict(BASE_MODEL_CONFIG)
        base_model_kwargs.update(model_kwargs)
        model_config = GPT2Config(**base_model_kwargs)

        run_name = get_run_name(train_kwargs, model_kwargs)

        print("=" * 100)
        print(f"Run {i + 1}/{total_num_runs}: {run_name}")
        print("=" * 100)

        if not debug:
            start_time = time.time()
            run(DATASET_CONFIG, model_config, train_config)
            end_time = time.time()
            time_taken = end_time - start_time
        else:
            time_taken = 1000

        num_tokens = train_config.total_batch_size * train_config.num_iterations
        mean_tps = num_tokens / time_taken
        if device == "cuda":
            peak_mem_usage = torch.cuda.max_memory_allocated() // 1024 // 1024
        else:
            peak_mem_usage = 0

        print(
            f"Time={time_taken:.2f}s, TPS={mean_tps:.2f}, peak mem usage={peak_mem_usage}MB"
        )
        if not debug:
            with open(results_file_path, "a") as output_file:
                results_writer = csv.DictWriter(output_file, fieldnames=result_headers)
                results_writer.writerow(
                    {
                        "run_name": run_name,
                        "time": time_taken,
                        "mean_tps": mean_tps,
                        "peak_mem_usage": peak_mem_usage,
                    }
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run_benchmark(args.debug)
