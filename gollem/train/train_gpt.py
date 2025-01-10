"""Script for training GPT models.

Note, to allow for training models with larger batch sizes than can be handled on
the available hardware, the script includes functionality to split each model update
step into multiple smaller batches. The total number of tokens per update is specified
by `total_batch_size`, while the amount per smaller batch actually run thru the model
in a single pass is specified by `batch_size` and `seq_len`.

`num_iterations` is the number of gradient updates performed on the model using
`total_batch_size` tokens. It does not not correspond to directly to "epochs", where
an epoch is a full pass thru the training data.
"""

import math
import os
import time
from contextlib import nullcontext

import numpy as np
import tiktoken
import torch

from gollem.config.model import get_model_config
from gollem.data.loader import DataLoader
from gollem.models.gpt2 import GPT


if __name__ == "__main__":
    import argparse
    from pprint import pprint

    # fmt: off
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # model
    parser.add_argument(
        "--model", type=str, default="gpt2", help="Model to train",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], 
    )
    parser.add_argument("--from_pretrained", type=int, default=0, help="Load pretrained model weights.")
    # data and save dir
    parser.add_argument("--train_data", type=str, default="data/tinyshakespeare/tiny_shakespeare_train.bin", help="Preprocessed training data.")
    parser.add_argument("--val_data", type=str, default="", help="Preprocessed validation data.")
    parser.add_argument("--output_dir", type=str, default="", help="Output dir where logs and checkpoints will be saved.")
    # training hyper params
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per individual training forward pass in #sequences.")
    parser.add_argument("--seq_len", type=int, default=64, help="Max sequence length in batch.")
    parser.add_argument("--total_batch_size", type=int, default=256, help="Total desired batch size in #tokens per model update.")
    parser.add_argument("--num_iterations", type=int, default=10, help="Number of training iterations (i.e. batches of `total_batch_size`).")
    # optimizer hyper params
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Learning rate warmup iterations.")
    parser.add_argument("--learning_rate_decay_frac", type=float, default=1.0, help="Learning rate decay fraction.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max gradient magnitude.")
    # evaluation
    parser.add_argument("--val_loss_every", type=int, default=0, help="Every how many steps to evaluate val loss.")
    parser.add_argument("--val_max_steps", type=int, default=20, help="How many batches of val to average.")
    parser.add_argument("--sample_every", type=int, default=0, help="How often to sample from the model.")
    # debugging

    # device and memory management and optimization options
    parser.add_argument("--device", type=str, default="", help="Device to use (autodetect by default).")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="Dtype to use for model")
    parser.add_argument("--flash", type=int, default=0, help="Use flash attention.")
    parser.add_argument("--compile", type=int, default=0, help="Torch.compile the model.")
    parser.add_argument("--tensorcores", type=int, default=0, help="Use GPU tensorcores.")
    parser.add_argument("--fused_adamw", type=int, default=0, help="Use fused version of AdamW optimizer.")
    args = parser.parse_args()
    # fmt: on

    print(f"Running pytorch {torch.version.__version__}")
    print(f"Training model {args.model}")
    model_cfg = get_model_config(args.model, flash_attention=bool(args.flash))
    assert 1 <= args.seq_len <= model_cfg.n_ctx

    pprint(model_cfg)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    device_type = "cuda" if "cuda" in device else "cpu"

    # calculate the number of gradient accumulation steps from the desired total batch
    # size and the current run configuration
    # Having multiple steps allows us to update model with larger batch size than can
    # be handled by the hardware in a single batch
    B, T = args.batch_size, args.seq_len
    tokens_per_fwdbwd = B * T
    assert args.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
    print(f"Total desired batch size: {args.total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

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
    model = GPT.from_pretrained(args.model) if args.from_pretrained else GPT(model_cfg)
    model.train()
    model.to(device)

    # compile model
    if args.compile:
        print("compiling the model...")
        model = torch.compile(model)

    # setup dataloaders
    train_loader = DataLoader(args.train_data, B, T)
    val_loader = None
    if args.val_data:
        val_loader = DataLoader(args.val_data, B, T)

    # optimizer setup
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type=device_type,
        use_fused=bool(args.fused_adamw),
    )

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        # increasing from 0 to learning rate over warm-up iters
        if it < args.warmup_iters:
            return args.learning_rate * (it + 1) / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > args.num_iterations:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (
            args.num_iterations - args.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        # coeff starts at 1 and goes to 0
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (args.learning_rate - min_lr)

    # create the logging directory if it does not exist
    logfile = None
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        with open(logfile, "w") as f:
            pass

    # reset CUDA memory stats to clear any stats from loading model
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # main training loop
    timings = []
    for step in range(args.num_iterations + 1):
        final_step = step == args.num_iterations

        # once in a while evaluate the validation dataset
        if (
            args.val_loss_every > 0 and (step % args.val_loss_every == 0 or final_step)
        ) and (val_loader is not None):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(args.val_max_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y, return_logits=False)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps
            # log to console and to file
            print(f"val loss {val_loss}")
            if logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d tel:%f\n" % (step, val_loss))

        # once in a while perform model inference on the master process
        if args.sample_every > 0 and (step % args.sample_every == 0 or final_step):
            model.eval()
            # before we end, let's also do one round of inference
            # we'll kick off the generation with "<|endoftext|>", which designates the
            # start of a new sequence
            start_ids = [enc.eot_token]
            xg = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
            max_new_tokens = 32
            temperature = 1.0
            top_k = 40
            yg = model.generate(
                xg, max_new_tokens, temperature=temperature, top_k=top_k
            )
            print("---------------")
            print(enc.decode(yg[0].tolist()))
            print("---------------")

        # we run an extra step to run eval and sample after model training
        if final_step:
            break

        # Training section
        t0 = time.time()
        model.train()
        # micro-batch loop where we accumulate gradients for total batch size
        # mean loss over mini-batches
        lossf = 0.0
        for micro_step in range(grad_accum_steps):
            # fetch a batch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            with amp_ctx:
                _, loss = model(x, y, return_logits=False)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_steps
                # keep track of the mean loss
                lossf += loss.detach().item()

            loss.backward()

        # clip gradients
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # step the optimizer
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # end of training section

        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        if device == "cuda":
            torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        tokens_per_second = grad_accum_steps * B * T / (t1 - t0)
        print(
            f"step {step + 1:4d}/{args.num_iterations} "
            f"| train loss {lossf:.6f} "
            f"| norm {norm:.4f} "
            f"| lr {lr:.2e} "
            f"| ({(t1 - t0) * 1000:.2f} ms "
            f"| {tokens_per_second:.0f} tok/s)"
        )
        # log to logile
        if logfile is not None:
            with open(logfile, "a") as f:
                f.write("s:%d trl:%f\n" % (step, lossf))

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > args.num_iterations - 20:
            timings.append(t1 - t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    if len(timings):
        mean_timing = np.mean(timings)
        mean_tps = grad_accum_steps * B * T / mean_timing
    else:
        mean_timing = 0.0
        mean_tps = 0.0
    mem_usage = torch.cuda.max_memory_allocated() // 1024 // 1024
    print(
        f"final {len(timings)} iters avg: {mean_timing * 1000:.3f}ms {mean_tps:.0f} tok/s"
    )
    print(f"peak mem usage: {mem_usage} MiB")

    # log to logile
    if logfile is not None:
        with open(logfile, "a") as f:
            f.write(
                "end. timing:%f tps:%f mem:%d\n" % (mean_timing, mean_tps, mem_usage)
            )
