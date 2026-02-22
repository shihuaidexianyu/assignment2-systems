from __future__ import annotations

import argparse
import copy
import multiprocessing as py_mp

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from .distributed_utils import cleanup_process_group, find_free_port, init_process_group, synchronize


class ToyMLP(nn.Module):
    def __init__(self, d_in: int = 10, d_hidden: int = 32, d_out: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden, bias=False),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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

        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for p in self.trainable_parameters:
            if p.grad is None:
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(self.world_size)


def _run_worker(rank: int, world_size: int, steps: int, batch_size: int, master_port: int, queue):
    device = init_process_group(rank=rank, world_size=world_size, backend="gloo", master_port=master_port)
    try:
        torch.manual_seed(1234 + rank)
        non_parallel_model = ToyMLP().to(device)
        ddp_model = NaiveDDPAllReduce(copy.deepcopy(non_parallel_model)).to(device)

        non_parallel_optim = optim.SGD(non_parallel_model.parameters(), lr=0.1)
        ddp_optim = optim.SGD(ddp_model.parameters(), lr=0.1)
        loss_fn = nn.MSELoss()

        for step in range(steps):
            torch.manual_seed(999 + step)
            all_x = torch.randn(batch_size * world_size, 10, device=device)
            all_y = torch.randn(batch_size * world_size, 5, device=device)
            local_offset = rank * batch_size
            local_x = all_x[local_offset : local_offset + batch_size]
            local_y = all_y[local_offset : local_offset + batch_size]

            non_parallel_optim.zero_grad(set_to_none=True)
            ddp_optim.zero_grad(set_to_none=True)

            non_parallel_loss = loss_fn(non_parallel_model(all_x), all_y)
            ddp_loss = loss_fn(ddp_model(local_x), local_y)

            non_parallel_loss.backward()
            ddp_loss.backward()
            ddp_model.finish_gradient_synchronization()

            non_parallel_optim.step()
            ddp_optim.step()
            synchronize(device)

            if rank == 0:
                for p_ref, p_ddp in zip(non_parallel_model.parameters(), ddp_model.parameters()):
                    if not torch.allclose(p_ref, p_ddp, atol=1e-6, rtol=1e-5):
                        queue.put({"ok": False, "reason": f"Mismatch at step={step}"})
                        return

        if rank == 0:
            queue.put({"ok": True, "reason": f"Matched for {steps} steps"})
    finally:
        cleanup_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate naive DDP all-reduce against non-parallel training.")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--local-batch-size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    queue = py_mp.get_context("spawn").SimpleQueue()
    port = find_free_port()
    mp.spawn(
        _run_worker,
        args=(args.world_size, args.steps, args.local_batch_size, port, queue),
        nprocs=args.world_size,
        join=True,
    )
    result = queue.get()
    if not result["ok"]:
        raise RuntimeError(result["reason"])
    print(f"Naive DDP equivalence PASSED: {result['reason']}")


if __name__ == "__main__":
    main()

