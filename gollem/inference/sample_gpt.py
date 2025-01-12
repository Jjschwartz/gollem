"""Script for sampling from pretrained GPT model.

Can be used to load pretrained GPT2 weights from huggingface or weights from a
checkpoint.
"""

import time
from contextlib import nullcontext

import numpy as np
import tiktoken
import torch

from gollem.models.gpt2.config import get_model_config
from gollem.models.gpt2 import GPT


if __name__ == "__main__":
    import argparse
    from pprint import pprint

    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # model
    parser.add_argument(
        "--model", type=str, default="gpt2", help="Model to train",
        choices=["gpt2", "gpt2-tiny", "gpt2-medium", "gpt2-large", "gpt2-xl"], 
    )
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint file to load from, leave black to load pretrained model weights from HF.")
    # training hyper params
    parser.add_argument("--num_seqs", type=int, default=1, help="Number of sequences to generate per prompt.")
    parser.add_argument("--max_seq_len", type=int, default=50, help="Max sequence length.")
    parser.add_argument("--top_k", type=int, default=None, help="Crop logits to only top k options.")
    parser.add_argument("--temparature", type=float, default=1.0, help="Softmax temperature.")
    # device and memory management and optimization options
    parser.add_argument("--device", type=str, default="", help="Device to use (autodetect by default).")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="Dtype to use for model")
    parser.add_argument("--flash", type=int, default=0, help="Use flash attention.")
    parser.add_argument("--compile", type=int, default=0, help="Torch.compile the model.")
    parser.add_argument("--tensorcores", type=int, default=0, help="Use GPU tensorcores.")
    args = parser.parse_args()
    # fmt: on
    assert args.max_seq_len >= 1
    assert args.num_seqs >= 1

    print(f"Running pytorch {torch.version.__version__}")
    print(f"Sampling from model: {args.model}")
    model_cfg = get_model_config(args.model, flash_attention=bool(args.flash))
    pprint(model_cfg)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    device_type = "cuda" if "cuda" in device else "cpu"

    # set up a context manager following the desired dtype and device
    # torch.autocast takes care of mixed-precision, basically setting the precision
    # based on the operation being performed
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    amp_ctx = (
        torch.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda"
        else nullcontext()
    )

    # rng / reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    if args.tensorcores:
        torch.set_float32_matmul_precision("high")

    # tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # load model
    if args.checkpoint:
        raise NotImplementedError
    else:
        model = GPT.from_pretrained(args.model)
    model.eval()
    model.to(device)

    # compile model
    if args.compile:
        print("compiling the model...")
        model = torch.compile(model)

    # reset CUDA memory stats to clear any stats from loading model
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # main loop
    timings = []
    try:
        while True:
            # get prompt from user
            prompt = input("\nPrompt: ")

            # encode prompt and repeat num_seqs times
            tokens = enc.encode(prompt)
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(args.num_seqs, 1)
            tokens = tokens.to(device)

            # get response from model
            t0 = time.time()
            response = model.generate(
                tokens,
                max_new_tokens=args.max_seq_len,
                temperature=args.temparature,
                top_k=args.top_k,
            )
            # wait on the CPU for all device work to end so we get accurate
            # per-iteration timings below
            if device == "cuda":
                torch.cuda.synchronize()
            timings.append(time.time() - t0)

            # print response
            print("Responses:")
            for i in range(args.num_seqs):
                tokens = response[i, : args.max_seq_len].tolist()
                decoded = enc.decode(tokens)
                print(">", decoded)

    except KeyboardInterrupt:
        print("\nShutting down")

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    if len(timings):
        mean_timing = np.mean(timings)
        mean_tps = args.num_seqs * args.max_seq_len / mean_timing
    else:
        mean_timing = 0.0
        mean_tps = 0.0
    mem_usage = torch.cuda.max_memory_allocated() // 1024 // 1024
    print(f"{len(timings)} iters avg: {mean_timing * 1000:.3f}ms {mean_tps:.0f} tok/s")
    print(f"peak mem usage: {mem_usage} MiB")
