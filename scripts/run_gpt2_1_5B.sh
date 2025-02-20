#!/bin/bash

# Train GPT2 1.5B model
# Ref for hyperparams: 
# https://github.com/karpathy/llm.c/blob/master/scripts/pyrun_gpt2_124M.sh
# https://github.com/karpathy/llm.c/blob/master/scripts/run_gpt2_1558M.sh

# May need to adjust batch size based on the GPU memory available
# So far GPT2 1.5B won't fit on 24GB GPU.
# On a H100 80GB GPU, the following works:
# - batch_size=16, seq_len=1024 (no activation checkpointing)
# - batch_size=64, seq_len=1024 (with activation checkpointing)

# Notes from the GPT2 paper:
# total_batch_size = 1048576 = 1024 * 1024 
# - i.e. the batch size and sequence length used in the paper
# num_iterations = 9500 = ~10B tokens / 1048576 
# - they train for 1 epoch on 10B tokens with total_batch_size=1048576
# - note technically 10B / 1048576 = 9,536 and 9500*1048576 = 9,961,472,000, so 
#   we are a bit off, but close enough. 9500 should hopefully account for truncation
#   of dataset shards.

# Additional notes:
# 1. Depending on your machine and whether it has NVLink setup or not you may need to
#    set `NCCL_P2P_DISABLE=1` to disable P2P communication between GPUs.
#    Basically if you try running this script and it hangs at the DDP call, then you
#    should first try setting this variable to 1 and see if that fixes it.
# 2. You may also want to change `activation_checkpointing` to `True` in the model
#    config depending on model size and GPU memory available

echo "Running GPT2 1.5B model"
# uv run python gollem/train_gpt2.py \
uv run torchrun --standalone --nproc_per_node=8 gollem/train_gpt2.py \
    --dataset fineweb_edu_10B \
    --model.model_name gpt2_1_5B \
    --model.n_ctx 1024 \
    --model.n_layer 48 \
    --model.n_head 25 \
    --model.d_model 1600 \
    --model.d_mlp 6400 \
    --model.vocab_size 50257 \
    --model.ln_bias True \
    --model.mlp_bias True \
    --model.share_embd_params True \
    --model.learning_rate 0.0006 \
    --model.warmup_iters 700 \
    --model.learning_rate_decay_frac 0.1 \
    --model.weight_decay 0.1 \
    --model.fused_adamw True \
    --model.flash True \
    --model.compile True \
    --model.zero_optimizer True \
    --model.from_pretrained False \
    --model.activation_checkpointing False \
    --train.output_dir results/gpt2_1_5B_fineweb_edu_10B \
    --train.seed 42 \
    --train.batch_size 16 \
    --train.seq_len 1024 \
    --train.total_batch_size 1048576 \
    --train.num_iterations 9500 \
    --train.val_loss_every 250 \
    --train.val_max_steps 20 \
    --train.sample_every 0 \
    --train.save_every 2000 \
    --train.snapshot_every 2000 \
    --train.device auto \
    --train.dtype bfloat16 \
    --train.tensorcores True \
    --train.use_wandb True

echo "Done"
