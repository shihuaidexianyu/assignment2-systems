from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def init_process_group(rank: int, world_size: int, backend: str, master_port: int) -> str:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA.")
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            raise RuntimeError("No CUDA devices found.")
        local_rank = rank % device_count
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return device


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def cleanup_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

