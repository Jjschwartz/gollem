import os
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.version
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP

from gollem.data.config import DataConfig
from gollem.data.loader import DataLoader
from gollem.logger import RunLogger
from gollem.models.config import ModelConfig
from gollem.models.model import BaseLLM
from gollem.tokenizer import BaseTokenizer
from gollem.train.config import TrainConfig
from gollem.train.utils import get_snapshot_dir
from gollem.utils import print0


def get_snapshot_path() -> Path:
    snapshot_dir = get_snapshot_dir()
    snapshot_path = snapshot_dir / "snapshot.pt"
    return snapshot_path


def save_snapshot(
    run_id: str,
    model: BaseLLM,
    optimizer: torch.optim.Optimizer,
    train_config: TrainConfig,
    dataset_config: DataConfig,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    step: int,
) -> None:
    snapshot_path = get_snapshot_path()

    data = {
        "run_id": run_id,
        "model_state_dict": model.state_dict(),
        "model_config": model.cfg,
        "optimizer_state_dict": optimizer.state_dict(),
        "train_config": train_config,
        "dataset_config": dataset_config,
        "train_loader": train_loader.state_dict(),
        "val_loader": val_loader.state_dict() if val_loader is not None else None,
        "step": step,
    }
    torch.save(data, snapshot_path)


def load_snapshot(
    snapshot_path: Path,
    device: str | torch.device,
    ddp_world_size: int,
    ddp_rank: int,
) -> tuple[
    str,
    BaseLLM,
    torch.optim.Optimizer,
    TrainConfig,
    DataConfig,
    BaseTokenizer,
    DataLoader,
    DataLoader | None,
    int,
]:
    data = torch.load(snapshot_path, weights_only=False)
    run_id = data["run_id"]
    model_config = data["model_config"]
    train_config = data["train_config"]
    dataset_config = data["dataset_config"]

    model, optimizer = model_config.get_model_and_optimizer(device=device)
    model.load_state_dict(data["model_state_dict"])
    optimizer.load_state_dict(data["optimizer_state_dict"])

    if isinstance(optimizer, ZeroRedundancyOptimizer):
        # There is an annoying bug in the ZeroRedundancyOptimizer when using load_state_dict
        # where zero-dimensional tensors (like Adam "step") are forced on to the CPU
        # so we need to move them to the correct device
        # See: https://github.com/pytorch/pytorch/issues/124133
        # And also the code: https://github.com/pytorch/pytorch/blob/1cf62e86a47ee575b5fbb997fd00f60ef0163130/torch/distributed/optim/zero_redundancy_optimizer.py#L1177
        for param, values in optimizer.optim.state.items():
            for k, v in values.items():
                if torch.is_tensor(v):
                    optimizer.optim.state[param][k] = v.to(device)

    enc = model_config.get_tokenizer()
    train_loader = DataLoader(
        dataset_config.train_data_pattern,
        batch_size=train_config.batch_size,
        seq_len=train_config.seq_len,
        token_dtype=enc.token_dtype,
        world_size=ddp_world_size,
        rank=ddp_rank,
    )
    train_loader.load_state_dict(data["train_loader"])

    if data["val_loader"] is not None:
        val_loader = DataLoader(
            dataset_config.val_data_pattern,
            batch_size=train_config.batch_size,
            seq_len=train_config.seq_len,
            token_dtype=enc.token_dtype,
            world_size=ddp_world_size,
            rank=ddp_rank,
        )
        val_loader.load_state_dict(data["val_loader"])

    return (
        run_id,
        model,
        optimizer,
        train_config,
        dataset_config,
        enc,
        train_loader,
        val_loader,
        data["step"],
    )


def run(
    dataset_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> dict[str, Any]:
    """The main run function.

    If using DDP assumes `init_process_group` has already been called by the calling
    function, and similarly `destroy_process_group` should be called by the calling
    function.
    """
    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "We need CUDA for DDP"
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        # this process will do logging, checkpointing etc.
        is_master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        is_master_process = True
        # select the device
        if train_config.device and train_config.device != "auto":
            # provided explicitly by the user
            device = train_config.device
        else:
            # attempt to autodetect the device
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Apple Silicon
                device = "mps"

    # print device info etc for each process
    print(
        f"Launching GPU{ddp_rank}|{ddp_local_rank} in process group {ddp_world_size} "
        f"using device {device}"
    )
    device_type = "cuda" if "cuda" in device else "cpu"

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

    output_dir = (
        None if train_config.output_dir == "" else Path(train_config.output_dir)
    )

    # calculate the number of gradient accumulation steps from the desired total batch
    # size, minibatch size, and sequence length
    # Having multiple steps allows us to update model with larger batch size than can
    # be handled by the hardware in a single batch
    tokens_per_fwdbwd = train_config.batch_size * train_config.seq_len * ddp_world_size
    assert train_config.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = train_config.total_batch_size // tokens_per_fwdbwd
    print0(f"total desired batch size: {train_config.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    snapshot_path = get_snapshot_path()
    if snapshot_path.exists():
        print0("Found snapshot, resuming from snapshot")
        (
            run_id,
            model,
            optimizer,
            train_config,
            dataset_config,
            enc,
            train_loader,
            val_loader,
            snapshot_step,
        ) = load_snapshot(snapshot_path, device, ddp_world_size, ddp_rank)
        starting_step = snapshot_step + 1
        logger = RunLogger(
            run_id=run_id,
            run_name=f"{model_config.model_name}_{dataset_config.name}",
            is_master_process=is_master_process,
            output_dir=output_dir,
            use_wandb=train_config.use_wandb,
            resume_from=f"{run_id}?_step={starting_step}",
        )
        logger.log(f"Resuming {run_id} from step {starting_step}")
    else:
        run_id = uuid.uuid4().hex
        print0(f"Starting new run {run_id}")
        starting_step = 0
        logger = RunLogger(
            run_id=run_id,
            run_name=f"{model_config.model_name}_{dataset_config.name}",
            is_master_process=is_master_process,
            output_dir=output_dir,
            use_wandb=train_config.use_wandb,
            resume_from=None,
        )

        train_config_dict = asdict(train_config)
        train_config_dict["grad_accum_steps"] = grad_accum_steps
        logger.log_config(
            {
                "pytorch_version": torch.version.__version__,
                "device": device,
                "device_type": device_type,
                "train_config": train_config_dict,
                "model_config": asdict(model_config),
                "dataset_config": asdict(dataset_config),
            }
        )

        # load model
        model, optimizer = model_config.get_model_and_optimizer(device=device)

        # setup tokenizer and dataloaders
        enc = model_config.get_tokenizer()
        train_loader = DataLoader(
            dataset_config.train_data_pattern,
            batch_size=train_config.batch_size,
            seq_len=train_config.seq_len,
            token_dtype=enc.token_dtype,
            world_size=ddp_world_size,
            rank=ddp_rank,
        )
        val_loader = None
        if dataset_config.val_data_pattern is not None:
            val_loader = DataLoader(
                dataset_config.val_data_pattern,
                batch_size=train_config.batch_size,
                seq_len=train_config.seq_len,
                token_dtype=enc.token_dtype,
                world_size=ddp_world_size,
                rank=ddp_rank,
            )

    # wrap model in DDP if needed
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        raw_model = model.module
    else:
        raw_model = model

    # learning rate decay scheduler (cosine with warmup)
    get_lr = model_config.get_lr_scheduler(train_config.num_iterations)

    # reset CUDA memory stats to clear any stats from loading model
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # main training loop
    logger.log(f"Starting training for {train_config.num_iterations} iterations")
    timings = []
    for step in range(starting_step, train_config.num_iterations + 1):
        final_step = step == train_config.num_iterations

        # once in a while evaluate the validation dataset
        if (
            train_config.val_loss_every > 0
            and (step % train_config.val_loss_every == 0 or final_step)
        ) and (val_loader is not None):
            val_t0 = time.monotonic()
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
            val_t1 = time.monotonic()
            logger.log_metrics(
                {"val_loss": val_loss, "val_time": (val_t1 - val_t0) * 1000}, step=step
            )
        # once in a while perform m
        # odel inference on the master process
        if (
            train_config.sample_every > 0
            and (step % train_config.sample_every == 0 or final_step)
            and is_master_process
        ):
            # TODO model.generate doesn't exist
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
            sample_output = [
                "---------------",
                enc.decode(yg[0].tolist()),
                "---------------",
            ]
            logger.log("\n".join(sample_output))

        # we run an extra step to run eval and sample after model training
        if final_step:
            break

        # Training section
        t0 = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # micro-batch loop where we accumulate gradients for total batch size
        # mean loss over mini-batches
        lossf = torch.tensor(0.0, device=device)
        for micro_step in range(grad_accum_steps):
            # fetch a batch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            if ddp:
                assert isinstance(model, DDP)
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1

            with amp_ctx:
                _, loss = model(x, y, return_logits=False)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_steps
                # keep track of the mean loss
                lossf += loss.detach()

            loss.backward()

        if ddp:
            # get the average loss across all processes for logging
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)

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

        # end of training section

        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        # time and print
        t1 = time.monotonic()
        tokens_per_second = (
            grad_accum_steps
            * ddp_world_size
            * train_config.batch_size
            * train_config.seq_len
            / (t1 - t0)
        )
        logger.log_metrics(
            {
                "train_loss": lossf.item(),
                "norm": norm,
                "lr": lr,
                "tokens_per_second": tokens_per_second,
                "time_per_step": (t1 - t0) * 1000,
            },
            step=step,
        )

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > train_config.num_iterations - 20:
            timings.append(t1 - t0)

        if (
            output_dir is not None
            and is_master_process
            and train_config.save_every > 0
            and step > 0
            and step % train_config.save_every == 0
        ):
            save_start_time = time.monotonic()
            logger.log(f"saving model to {output_dir}/model_s{step}.pt")
            raw_model.save_model(f"{output_dir}/model_s{step}.pt")
            save_end_time = time.monotonic()
            logger.log(f"model saved in {save_end_time - save_start_time} seconds")

        if (
            train_config.snapshot_every > 0
            and step > 0
            and step % train_config.snapshot_every == 0
        ):
            logger.log(f"saving snapshot to {snapshot_path}")
            snapshot_start_time = time.monotonic()
            if ddp and isinstance(optimizer, ZeroRedundancyOptimizer):
                # consolidates the sharded optimizer state to the master process
                # needs to be called on all ranks prior to saving the optimizer state
                optimizer.consolidate_state_dict(to=0)

            if is_master_process:
                save_snapshot(
                    run_id,
                    raw_model,
                    optimizer,
                    train_config,
                    dataset_config,
                    train_loader,
                    val_loader,
                    step,
                )
                snapshot_end_time = time.monotonic()
                logger.log(
                    f"snapshot saved in {snapshot_end_time - snapshot_start_time} seconds"
                )

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    if len(timings):
        mean_timing = np.mean(timings)
        mean_tps = (
            grad_accum_steps
            * ddp_world_size
            * train_config.batch_size
            * train_config.seq_len
            / mean_timing
        )
    else:
        mean_timing = 0.0
        mean_tps = 0.0

    # TODO track memory usage each iteration (??)
    mem_usage = torch.cuda.max_memory_allocated() // 1024 // 1024
    logger.log(
        f"final {len(timings)} iters avg: {mean_timing * 1000:.3f}ms {mean_tps:.0f} tok/s"
    )
    logger.log(f"peak mem usage: {mem_usage} MiB")

    if output_dir is not None and train_config.save_every > 0 and is_master_process:
        logger.log(
            f"saving final model to {output_dir}/model_s{train_config.num_iterations}.pt"
        )
        raw_model.save_model(f"{output_dir}/model_s{train_config.num_iterations}.pt")

    if is_master_process and train_config.snapshot_every > 0:
        logger.log(f"saving final snapshot to {snapshot_path}")
        save_snapshot(
            run_id,
            raw_model,
            optimizer,
            train_config,
            dataset_config,
            train_loader,
            val_loader,
            train_config.num_iterations,
        )

    logger.log(f"Finished training {run_id}")

    if not is_master_process:
        return {}
    return {
        "mean_iter_time": mean_timing,
        "mean_tps": mean_tps,
        "peak_mem_usage": mem_usage,
    }
