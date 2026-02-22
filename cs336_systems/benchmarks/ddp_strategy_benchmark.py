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
import torch.optim as optim

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.implementations.ddp_bucketed import DDPBucketed
from cs336_systems.implementations.ddp_individual import DDPIndividualParameters
from cs336_systems.implementations.common import broadcast_module_state

from .distributed_utils import cleanup_process_group, find_free_port, init_process_group, synchronize


MODEL_SIZES = {
    "tiny": {"d_model": 128, "d_ff": 512, "num_layers": 2, "num_heads": 4},
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
}


@dataclass(frozen=True)
class DDPMetaConfig:
    strategy: str
    model_size: str
    batch_size: int
    context_length: int
    vocab_size: int
    lr: float
    warmup_steps: int
    measure_steps: int
    bucket_size_mb: float
    backend: str


class NaiveDDPAllReduce(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.trainable_parameters: list[nn.Parameter] = []
        seen: set[int] = set()
        for p in self.module.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if p.requires_grad:
                self.trainable_parameters.append(p)
        broadcast_module_state(self.module)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def on_train_batch_start(self):
        return None

    def finish_gradient_synchronization(self):
        for p in self.trainable_parameters:
            if p.grad is None:
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(self.world_size)


class FlatDDPAllReduce(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.trainable_parameters: list[nn.Parameter] = []
        seen: set[int] = set()
        for p in self.module.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if p.requires_grad:
                self.trainable_parameters.append(p)
        broadcast_module_state(self.module)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def on_train_batch_start(self):
        return None

    def finish_gradient_synchronization(self):
        grads = [p.grad for p in self.trainable_parameters if p.grad is not None]
        if not grads:
            return
        flat = torch.cat([g.reshape(-1) for g in grads], dim=0)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(self.world_size)

        offset = 0
        for grad in grads:
            n = grad.numel()
            grad.copy_(flat[offset : offset + n].view_as(grad))
            offset += n


def _build_model(model_size: str, vocab_size: int, context_length: int) -> BasicsTransformerLM:
    size_cfg = MODEL_SIZES[model_size]
    return BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=size_cfg["d_model"],
        d_ff=size_cfg["d_ff"],
        num_layers=size_cfg["num_layers"],
        num_heads=size_cfg["num_heads"],
        rope_theta=10000.0,
    )


def _wrap_model(strategy: str, model: nn.Module, bucket_size_mb: float) -> nn.Module:
    if strategy == "naive":
        return NaiveDDPAllReduce(model)
    if strategy == "flat":
        return FlatDDPAllReduce(model)
    if strategy == "overlap_individual":
        return DDPIndividualParameters(model)
    if strategy == "overlap_bucketed":
        return DDPBucketed(model, bucket_size_mb=bucket_size_mb)
    raise ValueError(f"Unknown strategy: {strategy}")


def _run_train_loop(device: str, config: DDPMetaConfig) -> tuple[list[float], list[float]]:
    model = _build_model(config.model_size, vocab_size=config.vocab_size, context_length=config.context_length).to(device)
    ddp_model = _wrap_model(config.strategy, model=model, bucket_size_mb=config.bucket_size_mb)
    optimizer = optim.AdamW(ddp_model.parameters(), lr=config.lr)

    step_times_ms: list[float] = []
    comm_wait_ms: list[float] = []
    total_steps = config.warmup_steps + config.measure_steps

    for step in range(total_steps):
        if hasattr(ddp_model, "on_train_batch_start"):
            ddp_model.on_train_batch_start()

        optimizer.zero_grad(set_to_none=True)
        x = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device=device)
        y = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device=device)

        synchronize(device)
        t0 = time.perf_counter()
        logits = ddp_model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
        loss.backward()
        synchronize(device)

        comm_t0 = time.perf_counter()
        ddp_model.finish_gradient_synchronization()
        synchronize(device)
        comm_elapsed_ms = (time.perf_counter() - comm_t0) * 1000.0

        optimizer.step()
        synchronize(device)
        step_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if step >= config.warmup_steps:
            step_times_ms.append(step_elapsed_ms)
            comm_wait_ms.append(comm_elapsed_ms)

    return step_times_ms, comm_wait_ms


def _worker(rank: int, world_size: int, config: DDPMetaConfig, master_port: int, queue):
    device = init_process_group(rank=rank, world_size=world_size, backend=config.backend, master_port=master_port)
    try:
        local_step_ms, local_comm_ms = _run_train_loop(device, config)
        gathered_steps: list[list[float]] = [None for _ in range(world_size)]  # type: ignore[list-item]
        gathered_comm: list[list[float]] = [None for _ in range(world_size)]  # type: ignore[list-item]
        dist.all_gather_object(gathered_steps, local_step_ms)
        dist.all_gather_object(gathered_comm, local_comm_ms)
        if rank == 0:
            all_steps = [x for v in gathered_steps for x in v]
            all_comm = [x for v in gathered_comm for x in v]
            mean_step_ms = statistics.mean(all_steps)
            mean_comm_ms = statistics.mean(all_comm)
            comm_ratio = mean_comm_ms / max(mean_step_ms, 1e-9)
            queue.put(
                {
                    "strategy": config.strategy,
                    "model_size": config.model_size,
                    "mean_step_ms": f"{mean_step_ms:.3f}",
                    "mean_comm_wait_ms": f"{mean_comm_ms:.3f}",
                    "comm_wait_ratio": f"{comm_ratio:.4f}",
                    "bucket_size_mb": config.bucket_size_mb if config.strategy == "overlap_bucketed" else "-",
                }
            )
    finally:
        cleanup_process_group()


def _run_strategy(world_size: int, config: DDPMetaConfig) -> dict[str, str | float]:
    queue = py_mp.get_context("spawn").SimpleQueue()
    port = find_free_port()
    mp.spawn(_worker, args=(world_size, config, port, queue), nprocs=world_size, join=True)
    return queue.get()


def _to_markdown_table(rows: list[dict[str, str | float]]) -> str:
    headers = ["strategy", "model_size", "bucket_size_mb", "mean_step_ms", "mean_comm_wait_ms", "comm_wait_ratio"]
    header_row = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(row[h]) for h in headers) + " |" for row in rows]
    return "\n".join([header_row, sep, *body])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DDP strategies on a single node.")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--backend", type=str, default="gloo", choices=["gloo", "nccl"])
    parser.add_argument("--model-size", type=str, default="small", choices=sorted(MODEL_SIZES.keys()))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument(
        "--strategies",
        type=str,
        default="naive,flat,overlap_individual,overlap_bucketed",
        help="Comma-separated strategy names.",
    )
    parser.add_argument(
        "--bucket-sizes-mb",
        type=str,
        default="1,10,100,1000",
        help="Used only when overlap_bucketed is included. Comma-separated floats.",
    )
    parser.add_argument("--output-markdown", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    bucket_sizes = [float(x) for x in args.bucket_sizes_mb.split(",") if x.strip()]

    if args.backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requested but CUDA is unavailable.")
        if torch.cuda.device_count() < args.world_size:
            raise RuntimeError(
                f"Requested world_size={args.world_size} but only {torch.cuda.device_count()} GPUs are visible."
            )

    rows: list[dict[str, str | float]] = []
    for strategy in strategies:
        if strategy == "overlap_bucketed":
            for bucket_size_mb in bucket_sizes:
                cfg = DDPMetaConfig(
                    strategy=strategy,
                    model_size=args.model_size,
                    batch_size=args.batch_size,
                    context_length=args.context_length,
                    vocab_size=args.vocab_size,
                    lr=args.lr,
                    warmup_steps=args.warmup_steps,
                    measure_steps=args.measure_steps,
                    bucket_size_mb=bucket_size_mb,
                    backend=args.backend,
                )
                print(f"Running strategy={strategy}, bucket_size_mb={bucket_size_mb}")
                rows.append(_run_strategy(world_size=args.world_size, config=cfg))
        else:
            cfg = DDPMetaConfig(
                strategy=strategy,
                model_size=args.model_size,
                batch_size=args.batch_size,
                context_length=args.context_length,
                vocab_size=args.vocab_size,
                lr=args.lr,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
                bucket_size_mb=bucket_sizes[0] if bucket_sizes else 10.0,
                backend=args.backend,
            )
            print(f"Running strategy={strategy}")
            rows.append(_run_strategy(world_size=args.world_size, config=cfg))

    markdown = _to_markdown_table(rows)
    print("\nResults:\n")
    print(markdown)
    if args.output_markdown:
        with open(args.output_markdown, "w", encoding="utf-8") as f:
            f.write(markdown + "\n")
        print(f"\nSaved markdown table to: {args.output_markdown}")


if __name__ == "__main__":
    main()
