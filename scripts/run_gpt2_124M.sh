#!/bin/bash

# Train GPT2 124M model
# Ref for hyperparams: https://github.com/karpathy/llm.c/blob/master/scripts/pyrun_gpt2_124M.sh

# May need to adjust batch size based on the GPU memory available
# So far the following works:
# - 24GB GPU: batch_size=32, seq_len=1024

# Notes from the GPT2 paper:
# total_batch_size = 524288 = 512 * 1024 
# - i.e. the batch size and sequence length used in the paper
# num_iterations = 18865 = ~10B tokens / 524288 
# - they train for 1 epoch on 10B tokens with total_batch_size=524288
# - note technically 10B / 524288 = 19073 and 18865*524288 = 9,890,693,120, so 
#   we are a bit off, but close enough. 18865 is the number used by karpathy's code
#   which seems reasonable.

# Notes for tinystories:
# - 925,653,391 tokens
# - num_iterations = 925653391 / 524288 = 1766
# - changed warmup_iters to 70 (i.e. 10% of 700, since dataset is 10% of 10B)
echo "Running GPT2 124M model"
uv run torchrun --standalone --nproc_per_node=4 gollem/train_gpt2.py \
    --dataset tinystories \
    --model.model_name gpt2_124M \
    --model.n_ctx 1024 \
    --model.n_layer 12 \
    --model.n_head 12 \
    --model.d_model 768 \
    --model.d_mlp 3072 \
    --model.vocab_size 50257 \
    --model.ln_bias True \
    --model.mlp_bias True \
    --model.share_embd_params True \
    --model.learning_rate 0.0006 \
    --model.warmup_iters 700 \
    --model.learning_rate_decay_frac 0.0 \
    --model.weight_decay 0.1 \
    --model.fused_adamw True \
    --model.flash True \
    --model.compile True \
    --model.zero_optimizer True \
    --model.from_pretrained False \
    --train.output_dir results/gpt2_tinystories \
    --train.seed 42 \
    --train.batch_size 16 \
    --train.seq_len 1024 \
    --train.total_batch_size 524288 \
    --train.num_iterations 1766 \
    --train.val_loss_every 250 \
    --train.val_max_steps 20 \
    --train.sample_every 0 \
    --train.save_every 500 \
    --train.device auto \
    --train.dtype bfloat16 \
    --train.tensorcores True \
    --train.use_wandb True

echo "Done"
