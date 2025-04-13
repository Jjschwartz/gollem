from dataclasses import dataclass

import torch


@dataclass
class GPUInfo:
    name: str
    memory_bytes: int
    fp32_flops: float
    fp16_flops: float
    tf32_flops: float | None = None
    bf16_flops: float | None = None
    int8_flops: float | None = None


def GiB_to_bytes(gb: int) -> int:
    return gb * 1024**3


def TFLOPS_to_FLOPS(tflops: float) -> float:
    return tflops * 1e12


# Reported FLOPS for different GPUs
GPU_INFO = {
    # https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
    "H100": GPUInfo(
        name="H100",
        memory_bytes=GiB_to_bytes(80),
        fp32_flops=TFLOPS_to_FLOPS(67),
        fp16_flops=TFLOPS_to_FLOPS(1979),
        tf32_flops=TFLOPS_to_FLOPS(989),
        bf16_flops=TFLOPS_to_FLOPS(1979),
        int8_flops=TFLOPS_to_FLOPS(3958),
    ),
    # https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf
    "A100": GPUInfo(
        name="A100",
        memory_bytes=GiB_to_bytes(80),
        fp32_flops=TFLOPS_to_FLOPS(19.5),
        fp16_flops=TFLOPS_to_FLOPS(312),
        tf32_flops=TFLOPS_to_FLOPS(156),
        bf16_flops=TFLOPS_to_FLOPS(312),
        int8_flops=TFLOPS_to_FLOPS(624),
    ),
    # https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889
    "RTX4090": GPUInfo(
        name="RTX4090",
        memory_bytes=GiB_to_bytes(24),
        fp32_flops=TFLOPS_to_FLOPS(83),
        fp16_flops=TFLOPS_to_FLOPS(83),
        tf32_flops=None,
        bf16_flops=None,
        int8_flops=None,
    ),
    # https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622
    "RTX3090": GPUInfo(
        name="RTX3090",
        memory_bytes=GiB_to_bytes(24),
        fp32_flops=TFLOPS_to_FLOPS(36),
        fp16_flops=TFLOPS_to_FLOPS(36),
        tf32_flops=None,
        bf16_flops=None,
        int8_flops=None,
    ),
}


def get_gpu_info(name: str) -> GPUInfo:
    return GPU_INFO[name]


_DTYPE_LABEL_MAPPING = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.int8: "int8",
    "tf32": "tf32",
    "fp32": "fp32",
    "fp16": "fp16",
    "bf16": "bf16",
    "int8": "int8",
    "float32": "fp32",
    "float16": "fp16",
    "bfloat16": "bf16",
}


def _get_dtype_label(dtype: torch.dtype | str) -> str:
    return _DTYPE_LABEL_MAPPING[dtype]


def get_gpu_flops(name: str, dtype: str | torch.dtype) -> float:
    gpu_info = get_gpu_info(name)
    dtype_label = _get_dtype_label(dtype)
    if dtype_label == "fp32":
        return gpu_info.fp32_flops
    elif dtype_label == "tf32":
        assert gpu_info.tf32_flops is not None, f"TF32 FLOPS not available for {name}"
        return gpu_info.tf32_flops
    elif dtype_label == "bf16":
        assert gpu_info.bf16_flops is not None, f"BF16 FLOPS not available for {name}"
        return gpu_info.bf16_flops
    elif dtype_label == "fp16":
        assert gpu_info.fp16_flops is not None, f"FP16 FLOPS not available for {name}"
        return gpu_info.fp16_flops
    elif dtype_label == "int8":
        assert gpu_info.int8_flops is not None, f"INT8 FLOPS not available for {name}"
        return gpu_info.int8_flops
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def get_gpu_flops_for_all_gpus(
    dtype: str, ignore_if_unavailable: bool = True
) -> dict[str, float]:
    result = {}
    for name in GPU_INFO:
        try:
            result[name] = get_gpu_flops(name, dtype)
        except ValueError:
            if ignore_if_unavailable:
                continue
            else:
                raise
    return result
