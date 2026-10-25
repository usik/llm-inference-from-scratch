from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_STORAGE_BYTES = 8_056_404_654
RECOMMENDED_FREE_BYTES = 20 * 1024**3


def format_gib(value: int) -> str:
    return f"{value / 1024**3:.1f} GiB"


def physical_memory_bytes() -> int | None:
    if platform.system() != "Darwin":
        return None

    result = subprocess.run(
        ["sysctl", "-n", "hw.memsize"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return int(result.stdout.strip())


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def mps_status() -> str:
    try:
        import torch
    except ImportError:
        return "unknown (torch is not installed in this environment)"

    return f"built={torch.backends.mps.is_built()}, available={torch.backends.mps.is_available()}"


def main() -> int:
    free_bytes = shutil.disk_usage(Path.cwd()).free
    memory_bytes = physical_memory_bytes()
    disk_ready = free_bytes >= RECOMMENDED_FREE_BYTES

    print("Qwen3-4B PyTorch MPS preflight")
    print(f"platform: {platform.system()} {platform.machine()}")
    print(f"python: {platform.python_version()}")
    print(f"physical memory: {format_gib(memory_bytes) if memory_bytes else 'unknown'}")
    print(f"disk free: {format_gib(free_bytes)}")
    print(f"model repository size: {format_gib(MODEL_STORAGE_BYTES)}")
    print(f"recommended free space: {format_gib(RECOMMENDED_FREE_BYTES)}")
    print(f"torch: {package_version('torch')}")
    print(f"transformers: {package_version('transformers')}")
    print(f"mps: {mps_status()}")

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("status: blocked - this experiment targets Apple Silicon macOS")
        return 1
    if not disk_ready:
        print("status: blocked - free disk space before installing packages or downloading weights")
        return 1

    print("status: ready for uv sync and model download")
    return 0


if __name__ == "__main__":
    sys.exit(main())
