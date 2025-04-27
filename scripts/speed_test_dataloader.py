import argparse
import time

import numpy as np
from gollem.data import load_dataset
from gollem.data.loader import DataLoader
from gollem.tokenizer import get_tokenizer
from matplotlib import pyplot as plt


def main(args):
    tokenizer = get_tokenizer("gpt2")
    dataset = load_dataset(args.dataset, tokenizer)

    data_loader = DataLoader(
        dataset.train_data_pattern,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        token_dtype=tokenizer.token_dtype,
        world_size=1,
        rank=0,
    )

    num_epochs = 3
    num_batches = 100000

    epoch_times = []
    epoch_batch_times = []
    for i in range(num_epochs + 1):
        data_loader.reset()
        epoch_start_time = time.monotonic()
        batch_times = []
        for j in range(num_batches):
            batch_start_time = time.monotonic()
            x, y = data_loader.next_batch()
            x.to(args.device)
            y.to(args.device)
            batch_times.append(time.monotonic() - batch_start_time)
        if i > 0:
            epoch_time = time.monotonic() - epoch_start_time
            final_shard = data_loader.current_shard
            batch_mean_time = np.mean(batch_times)
            print(
                f"Epoch {i}: {epoch_time:.4f}s, final shard: {final_shard}, batch mean time: {batch_mean_time:.4f}s"
            )
            # skip first epoch because it's loading the data
            epoch_times.append(epoch_time)
            epoch_batch_times.append(batch_times)

    fig, axs = plt.subplots(2, 1)
    # violin plot of epoch times
    axs[0].violinplot(epoch_times)
    axs[0].set_ylabel("Time (s)")
    axs[0].set_xlabel("Epoch")
    axs[0].set_title("Epoch Times")

    # line plot of batch times (by batch num)
    x = np.arange(len(epoch_batch_times[0]))
    for i, batch_times in enumerate(epoch_batch_times):
        axs[1].plot(x, batch_times, label=f"Epoch {i}", color="blue", alpha=0.5)
    # plot mean of batch times
    axs[1].plot(x, np.mean(epoch_batch_times, axis=0), label="Mean", color="red")

    axs[1].set_xlabel("Batch Number")
    axs[1].set_ylabel("Time (s)")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)
    args = parser.parse_args()

    main(args)
