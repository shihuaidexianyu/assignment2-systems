from __future__ import annotations

import argparse
import multiprocessing as py_mp
import statistics
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.implementations import get_sharded_optimizer

from .distributed_utils import cleanup_process_group, find_free_port, init_process_group, synchronize


TABLE1_MODEL_SIZES = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


@dataclass(frozen=True)
class ShardingBenchmarkConfig:
    use_sharding: bool
    world_size: int
    model_size: str
    batch_size: int
    context_length: int
    vocab_size: int
    warmup_steps: int
    measure_steps: int
    lr: float
    backend: str
    precision: str


class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        for p in self.module.parameters():
            if dist.is_initialized():
                dist.broadcast(p.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        if self.world_size == 1:
            return
        for p in self.module.parameters():
            if p.grad is None:
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(self.world_size)


def _model_dtype(precision: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16}[precision]


def _build_model(config: ShardingBenchmarkConfig, device: str) -> NaiveDDP:
    m = TABLE1_MODEL_SIZES[config.model_size]
    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=m["d_model"],
        d_ff=m["d_ff"],
        num_layers=m["num_layers"],
        num_heads=m["num_heads"],
        rope_theta=10000.0,
    ).to(device=device, dtype=_model_dtype(config.precision))
    model.train()
    return NaiveDDP(model)


def _iter_unique_parameters(module: nn.Module):
    seen: set[int] = set()
    for p in module.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        yield p


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _parameter_bytes(module: nn.Module) -> int:
    return sum(_tensor_bytes(p) for p in _iter_unique_parameters(module))


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        if isinstance(state, dict):
            for v in state.values():
                if torch.is_tensor(v):
                    total += _tensor_bytes(v)
    return total


def _active_optimizer_for_state_bytes(optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    if hasattr(optimizer, "_local_optimizer") and getattr(optimizer, "_local_optimizer") is not None:
        return getattr(optimizer, "_local_optimizer")
    return optimizer


def _train_loop(device: str, config: ShardingBenchmarkConfig) -> dict[str, float]:
    if not device.startswith("cuda"):
        raise RuntimeError("sharded_optimizer_accounting benchmark requires CUDA devices.")
    ddp_model = _build_model(config, device=device)
    if config.use_sharding:
        optimizer = get_sharded_optimizer(ddp_model.parameters(), torch.optim.AdamW, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=config.lr)

    x = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device=device)
    y = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device=device)

    torch.cuda.reset_peak_memory_stats(device)
    init_alloc = torch.cuda.memory_allocated(device)
    param_bytes = _parameter_bytes(ddp_model)
    opt_state_bytes_init = _optimizer_state_bytes(_active_optimizer_for_state_bytes(optimizer))

    pre_step_samples: list[float] = []
    post_step_samples: list[float] = []
    step_ms_samples: list[float] = []

    for step in range(config.warmup_steps + config.measure_steps):
        optimizer.zero_grad(set_to_none=True)

        if config.precision == "bf16":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = ddp_model(x)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
        else:
            logits = ddp_model(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))

        loss.backward()
        ddp_model.finish_gradient_synchronization()
        synchronize(device)
        pre_step_alloc = torch.cuda.memory_allocated(device)

        t0 = time.perf_counter()
        optimizer.step()
        synchronize(device)
        step_ms = (time.perf_counter() - t0) * 1000.0
        post_step_alloc = torch.cuda.memory_allocated(device)

        if step >= config.warmup_steps:
            pre_step_samples.append(pre_step_alloc / (1024**2))
            post_step_samples.append(post_step_alloc / (1024**2))
            step_ms_samples.append(step_ms)

    opt_state_bytes = _optimizer_state_bytes(_active_optimizer_for_state_bytes(optimizer))
    peak_alloc = torch.cuda.max_memory_allocated(device)
    return {
        "init_alloc_mb": init_alloc / (1024**2),
        "pre_step_alloc_mb": statistics.mean(pre_step_samples),
        "post_step_alloc_mb": statistics.mean(post_step_samples),
        "peak_alloc_mb": peak_alloc / (1024**2),
        "param_mb": param_bytes / (1024**2),
        "optimizer_state_mb_init": opt_state_bytes_init / (1024**2),
        "optimizer_state_mb_post": opt_state_bytes / (1024**2),
        "other_mb_post": max((statistics.mean(post_step_samples) - (param_bytes + opt_state_bytes) / (1024**2)), 0.0),
        "step_ms": statistics.mean(step_ms_samples),
    }


def _worker(rank: int, config: ShardingBenchmarkConfig, master_port: int, queue):
    device = init_process_group(
        rank=rank,
        world_size=config.world_size,
        backend=config.backend,
        master_port=master_port,
    )
    try:
        local = _train_loop(device=device, config=config)
        gathered: list[dict[str, float]] = [None for _ in range(config.world_size)]  # type: ignore[list-item]
        dist.all_gather_object(gathered, local)
        if rank == 0:
            merged: dict[str, float] = {}
            for key in local.keys():
                merged[key] = statistics.mean([v[key] for v in gathered])
            queue.put(merged)
    finally:
        cleanup_process_group()


def _run(config: ShardingBenchmarkConfig) -> dict[str, float]:
    queue = py_mp.get_context("spawn").SimpleQueue()
    port = find_free_port()
    mp.spawn(_worker, args=(config, port, queue), nprocs=config.world_size, join=True)
    return queue.get()


def _to_markdown_table(rows: list[dict[str, str]]) -> str:
    headers = [
        "variant",
        "world_size",
        "model",
        "init_alloc_mb",
        "pre_step_alloc_mb",
        "post_step_alloc_mb",
        "peak_alloc_mb",
        "param_mb",
        "optimizer_state_mb_post",
        "other_mb_post",
        "step_ms",
    ]
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(row[h]) for h in headers) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark memory/time impact of optimizer state sharding.")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--model-size", type=str, default="xl", choices=list(TABLE1_MODEL_SIZES.keys()))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backend", type=str, default="nccl", choices=["gloo", "nccl"])
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--output-markdown", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL requires CUDA.")
        if torch.cuda.device_count() < args.world_size:
            raise RuntimeError(
                f"Requested world_size={args.world_size}, but only {torch.cuda.device_count()} GPUs are available."
            )

    rows: list[dict[str, str]] = []
    for use_sharding in [False, True]:
        cfg = ShardingBenchmarkConfig(
            use_sharding=use_sharding,
            world_size=args.world_size,
            model_size=args.model_size,
            batch_size=args.batch_size,
            context_length=args.context_length,
            vocab_size=args.vocab_size,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            lr=args.lr,
            backend=args.backend,
            precision=args.precision,
        )
        print(f"Running variant={'sharded' if use_sharding else 'baseline'}")
        result = _run(cfg)
        rows.append(
            {
                "variant": "sharded" if use_sharding else "baseline",
                "world_size": str(args.world_size),
                "model": args.model_size,
                "init_alloc_mb": f"{result['init_alloc_mb']:.3f}",
                "pre_step_alloc_mb": f"{result['pre_step_alloc_mb']:.3f}",
                "post_step_alloc_mb": f"{result['post_step_alloc_mb']:.3f}",
                "peak_alloc_mb": f"{result['peak_alloc_mb']:.3f}",
                "param_mb": f"{result['param_mb']:.3f}",
                "optimizer_state_mb_post": f"{result['optimizer_state_mb_post']:.3f}",
                "other_mb_post": f"{result['other_mb_post']:.3f}",
                "step_ms": f"{result['step_ms']:.3f}",
            }
        )

    table = _to_markdown_table(rows)
    print("\nResults:\n")
    print(table)
    if args.output_markdown:
        with open(args.output_markdown, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"\nSaved markdown table to: {args.output_markdown}")


if __name__ == "__main__":
    main()
