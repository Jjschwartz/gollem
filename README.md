# gollem

Go train your own LLM

## Install

If you have uv installed on your system, you can install a virtual environment with all the necessary packages by running the following commands:

```bash
uv sync --extra cpu
# or for torch with CUDA 12.1 support
uv sync --extra cu121
# or for torch with CUDA 12.4 support
uv sync --extra cu124
```

This will install the necessary packages for the gollem repository.