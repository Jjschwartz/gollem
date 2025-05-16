#! /bin/bash

# Download the Llama 3 model
size=$1

if [ -z "$size" ]; then
    echo "Usage: $0 <size>"
    exit 1
fi

# Make sure weights directory exists
mkdir -p checkpoints

# Download the model
if [ "$size" == "8B" ]; then
    uv run huggingface-cli download meta-llama/Meta-Llama-3-8B --include "original/*" --local-dir checkpoints/Meta-Llama-3-8B
fi



