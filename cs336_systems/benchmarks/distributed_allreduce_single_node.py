from __future__ import annotations

import argparse
import multiprocessing as py_mp
import statistics
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .distributed_utils import cleanup_process_group, find_free_port, init_process_group, synchronize


@dataclass(frozen=True)
class AllReduceConfig:
    backend: str
    world_size: int
    tensor_mb: int
    warmup_steps: int
    measure_steps: int


def _to_markdown_table(rows: list[dict[str, str | int | float]]) -> str:
    headers = ["backend", "world_size", "tensor_mb", "mean_ms", "std_ms", "effective_gbps"]
    header_row = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join([header_row, sep, *body])


def _benchmark_allreduce(device: str, config: AllReduceConfig) -> list[float]:
    tensor_numel = config.tensor_mb * 1024 * 1024 // 4  # float32
    tensor = torch.randn(tensor_numel, dtype=torch.float32, device=device)

    latencies_ms: list[float] = []
    for step in range(config.warmup_steps + config.measure_steps):
        synchronize(device)
        t0 = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        synchronize(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if step >= config.warmup_steps:
            latencies_ms.append(elapsed_ms)
    return latencies_ms


def _worker(rank: int, world_size: int, config: AllReduceConfig, master_port: int, queue):
    device = init_process_group(rank=rank, world_size=world_size, backend=config.backend, master_port=master_port)
    try:
        local_latencies = _benchmark_allreduce(device, config)
        gathered: list[list[float]] = [None for _ in range(world_size)]  # type: ignore[list-item]
        dist.all_gather_object(gathered, local_latencies)
        if rank == 0:
            merged = [x for rank_latencies in gathered for x in rank_latencies]
            mean_ms = statistics.mean(merged)
            std_ms = statistics.stdev(merged) if len(merged) > 1 else 0.0
            bytes_per_allreduce = config.tensor_mb * 1024 * 1024
            effective_gbps = (bytes_per_allreduce / (mean_ms / 1000.0)) / 1e9
            queue.put(
                {
                    "backend": config.backend,
                    "world_size": config.world_size,
                    "tensor_mb": config.tensor_mb,
                    "mean_ms": f"{mean_ms:.3f}",
                    "std_ms": f"{std_ms:.3f}",
                    "effective_gbps": f"{effective_gbps:.3f}",
                }
            )
    finally:
        cleanup_process_group()


def _run_single_config(config: AllReduceConfig) -> dict[str, str | int | float]:
    queue = py_mp.get_context("spawn").SimpleQueue()
    port = find_free_port()
    mp.spawn(
        _worker,
        args=(config.world_size, config, port, queue),
        nprocs=config.world_size,
        join=True,
    )
    return queue.get()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-node distributed all-reduce benchmark.")
    parser.add_argument("--world-sizes", type=str, default="2,4,6", help="Comma-separated process counts.")
    parser.add_argument("--tensor-mb", type=str, default="1,10,100,1024", help="Comma-separated tensor sizes in MB.")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--output-markdown", type=str, default=None)
    parser.add_argument(
        "--backends",
        type=str,
        default="gloo,nccl",
        help="Comma-separated backends to run from {gloo,nccl}.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    world_sizes = [int(x) for x in args.world_sizes.split(",") if x.strip()]
    tensor_sizes = [int(x) for x in args.tensor_mb.split(",") if x.strip()]
    requested_backends = [x.strip() for x in args.backends.split(",") if x.strip()]

    rows: list[dict[str, str | int | float]] = []
    for backend in requested_backends:
        if backend == "nccl" and not torch.cuda.is_available():
            print("Skipping NCCL benchmark because CUDA is unavailable.")
            continue
        max_world_size = max(world_sizes)
        if backend == "nccl":
            max_available = torch.cuda.device_count()
            if max_available < 2:
                print("Skipping NCCL benchmark because at least 2 GPUs are required.")
                continue
            max_world_size = min(max_world_size, max_available)

        for world_size in world_sizes:
            if world_size > max_world_size:
                print(f"Skipping backend={backend}, world_size={world_size} (insufficient devices).")
                continue
            for tensor_mb in tensor_sizes:
                config = AllReduceConfig(
                    backend=backend,
                    world_size=world_size,
                    tensor_mb=tensor_mb,
                    warmup_steps=args.warmup_steps,
                    measure_steps=args.measure_steps,
                )
                print(f"Running backend={backend}, world_size={world_size}, tensor_mb={tensor_mb}")
                rows.append(_run_single_config(config))

    markdown = _to_markdown_table(rows)
    print("\nResults:\n")
    print(markdown)
    if args.output_markdown:
        with open(args.output_markdown, "w", encoding="utf-8") as f:
            f.write(markdown + "\n")
        print(f"\nSaved markdown table to: {args.output_markdown}")


if __name__ == "__main__":
    main()
