#!/bin/bash

# Train GPT2 124M model
# Ref for hyperparams: https://github.com/karpathy/llm.c/blob/master/scripts/pyrun_gpt2_124M.sh

# --batch_size and --seq_len can be adjusted till model fits on GPU
# smaller batches will take longer

# --fused_adamw may be slower or faster depending on your system, so recommended to
#     test on a small run first

# --tensorcores may have minimal effect, but at least doesn't seem to negatively 
#     affect speed if enables
python gollem/main.py \
    --model gpt2 \
    --dataset tinystories \
    --output_dir results/gpt2_tinystories \
    --batch_size 4 \
    --seq_len 512 \
    --total_batch_size 524288 \
    --num_iterations 18865 \
    --val_loss_every 250 \
    --val_max_steps 20 \
    --sample_every 0 \
    --dtype float16 \
    --tensorcores True \
    --use_wandb True

    
    
    --weight_decay 0.1 \
    --warmup_iters 700 \
    --learning_rate_decay_frac 0.0 \
    --flash 1 \
    --compile 1 \
    --fused_adamw 0 \
    --learning_rate 0.0006 \


    --write_tensors 0 \
    --model d12 \
    --batch_size 32 \
    --sequence_length 1024 \
    --total_batch_size 524288 \
    --dtype bfloat16 \
    --compile 1 \
    --tensorcores 1 \
    --flash 1 \
    --num_iterations 18865 \
    --weight_decay 0.1 \
    --zero_stage 1 \
    --learning_rate 0.0006 \
    --warmup_iters 700 \
    --learning_rate_decay_frac 0.0 \
    --overfit_single_batch 0