import argparse
import time

import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--shape", type=int, nargs="+", required=True)
    parser.add_argument("-d", "--dtype", type=str, default="float32")
    args = parser.parse_args()

    print(f"Using shape: {args.shape}")
    dtype = getattr(torch, args.dtype)
    print(f"Using dtype: {dtype}")

    # get device (handle multiple devices
    available_devices = torch.cuda.device_count()
    device = torch.device("cuda:0") if available_devices > 1 else torch.device("cuda")
    print(f"Using device: {device}")

    mem_before = torch.cuda.memory_allocated(device=device)
    tensor = torch.ones(args.shape, device=device, dtype=dtype)
    time.sleep(1)
    mem_after = torch.cuda.memory_allocated(device=device)
    mem_used_bytes = mem_after - mem_before
    mem_used_gb = mem_used_bytes / 1024**3
    print(f"Memory usage: {mem_used_gb:.6f} GB")
