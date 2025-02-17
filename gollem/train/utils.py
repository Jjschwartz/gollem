import platform
from pathlib import Path

import psutil
import torch

from gollem.utils import get_base_dir_path


def get_snapshot_dir() -> Path:
    """Path to the directory where the snapshot for the current run is stored."""
    base_dir = get_base_dir_path()
    snapshot_dir = base_dir / "results" / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_dir


def clear_snapshot_dir() -> None:
    """Clear the snapshot directory."""
    snapshot_dir = get_snapshot_dir()
    if snapshot_dir.exists():
        for file in snapshot_dir.iterdir():
            file.unlink()


def check_dtype_support() -> list[str]:
    """Check which dtypes are supported by the current hardware.

    Returns:
        List of supported dtype strings from ["float32", "float16", "bfloat16"]
    """
    supported_dtypes = ["float32"]  # float32 is always supported
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        capabilities = torch.cuda.get_device_capability()
        if capabilities >= (7, 0):  # Volta and newer support tensor cores
            supported_dtypes.append("float16")
        if torch.cuda.is_bf16_supported():  # Ampere and newer support bfloat16
            supported_dtypes.append("bfloat16")

    print(f"Supported dtypes on {device}: {supported_dtypes}")
    return supported_dtypes


def check_tensorcores_support() -> bool:
    """Check if tensor cores are supported by the current hardware.

    Returns:
        True if tensor cores are supported, False otherwise
    """
    # volta and newer support tensor cores
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 0)


def get_hardware_info() -> dict:
    """Get detailed hardware information about the system.

    Returns:
        Dictionary containing hardware information including OS, CPU, RAM, and GPU details
    """
    info = {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "cpu": {
            "brand": platform.processor(),
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        },
    }

    # Add GPU information if available
    if torch.cuda.is_available():
        info["gpu"] = {
            "name": torch.cuda.get_device_name(),
            "count": torch.cuda.device_count(),
            "capability": torch.cuda.get_device_capability(),
            "memory_total_gb": round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            ),
            "memory_allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 2),
        }
    else:
        info["gpu"] = None

    return info


if __name__ == "__main__":
    from pprint import pprint

    pprint(get_hardware_info())
