import time
from contextlib import nullcontext
from pprint import pprint

import numpy as np
import torch
import torch.version

from gollem.data.config import DataConfig
from gollem.data.loader import DataLoader
from gollem.models.config import ModelConfig
from gollem.train.config import TrainConfig


# General experiment workflow
# Inputs:
# 1. Select the model (name | config)
# 2. Select the dataset (name | config)
# 3. Select the hyperparams (config)
# Experiment flow:
# 1. Load the dataset
#  a. download data if not present
#  b. tokenize data
# 4. Run the experiment


def run(
    dataset_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
):
    print(f"Running pytorch {torch.version.__version__}")
    print("\nModel config:")
    pprint(model_config)
    print("\nTrain config:")
    pprint(train_config)
    print("\nDataset config:")
    pprint(dataset_config)

    if train_config.device:
        device = train_config.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    device_type = "cuda" if "cuda" in device else "cpu"

    # calculate the number of gradient accumulation steps from the desired total batch
    # size and the current run configuration
    # Having multiple steps allows us to update model with larger batch size than can
    # be handled by the hardware in a single batch
    B, T = train_config.batch_size, train_config.seq_len
    tokens_per_fwdbwd = B * T
    assert train_config.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = train_config.total_batch_size // tokens_per_fwdbwd
    print(f"Total desired batch size: {train_config.total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    # torch.autocast takes care of mixed-precision, basically setting the precision
    # based on the operation being performed
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[train_config.dtype]
    amp_ctx = (
        torch.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda"
        else nullcontext()
    )

    # rng / reproducibility
    torch.manual_seed(train_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(train_config.seed)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    if train_config.tensorcores:
        torch.set_float32_matmul_precision("high")

    # tokenizer
    enc = model_config.get_tokenizer()

    # load model
    model, optimizer = model_config.get_model_and_optimizer(device=device)

    # setup dataloaders
    train_loader = DataLoader(str(dataset_config.train_data), B, T)
    val_loader = None
    if dataset_config.val_data:
        val_loader = DataLoader(str(dataset_config.val_data), B, T)

    # learning rate decay scheduler (cosine with warmup)
    get_lr = model_config.get_lr_scheduler(train_config.num_iterations)

    # create the logging directory if it does not exist
    logfile = None
    if train_config.output_dir:
        train_config.output_dir.mkdir(parents=True, exist_ok=True)
        logfile = train_config.output_dir / "main.log"
        # create the log file "main.log" inside it, and wipe it clean
        logfile.write_text("")

    # reset CUDA memory stats to clear any stats from loading model
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # main training loop
    timings = []
    for step in range(train_config.num_iterations + 1):
        final_step = step == train_config.num_iterations

        # once in a while evaluate the validation dataset
        if (
            train_config.val_loss_every > 0
            and (step % train_config.val_loss_every == 0 or final_step)
        ) and (val_loader is not None):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(train_config.val_max_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y, return_logits=False)
                    val_loss += loss.item()
                val_loss /= train_config.val_max_steps
            # log to console and to file
            print(f"val loss {val_loss}")
            if logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d tel:%f\n" % (step, val_loss))

        # once in a while perform model inference on the master process
        if train_config.sample_every > 0 and (
            step % train_config.sample_every == 0 or final_step
        ):
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

        if hasattr(model_config, "grad_clip"):
            # clip gradients
            grad_clip = getattr(model_config, "grad_clip")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
            f"step {step + 1:4d}/{train_config.num_iterations} "
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
        if step > 0 and step > train_config.num_iterations - 20:
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
