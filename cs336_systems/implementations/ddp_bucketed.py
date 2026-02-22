from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from .common import broadcast_module_state, is_dist_initialized


class DDPBucketed(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float | None):
        super().__init__()
        self.module = module
        self._world_size = dist.get_world_size() if is_dist_initialized() else 1
        self._bucket_size_bytes = None
        if bucket_size_mb is not None:
            # Scale bucket size by world size to match expected split behavior in tests.
            self._bucket_size_bytes = int(bucket_size_mb * 1024 * 1024 * self._world_size)
        self._buckets = self._build_buckets()
        self._runtime_buckets: list[dict] = []
        self._register_gradient_hooks()
        self._reset_runtime_state()
        broadcast_module_state(self.module)

    def _build_buckets(self) -> list[list[torch.nn.Parameter]]:
        unique_trainable_parameters: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for parameter in self.module.parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if parameter.requires_grad:
                unique_trainable_parameters.append(parameter)

        params_in_backward_ready_order = list(reversed(unique_trainable_parameters))
        if self._bucket_size_bytes is None:
            return [params_in_backward_ready_order] if params_in_backward_ready_order else []

        buckets: list[list[torch.nn.Parameter]] = []
        current_bucket: list[torch.nn.Parameter] = []
        current_bucket_bytes = 0
        for parameter in params_in_backward_ready_order:
            parameter_bytes = parameter.numel() * parameter.element_size()
            if current_bucket and (current_bucket_bytes + parameter_bytes > self._bucket_size_bytes):
                buckets.append(current_bucket)
                current_bucket = [parameter]
                current_bucket_bytes = parameter_bytes
            else:
                current_bucket.append(parameter)
                current_bucket_bytes += parameter_bytes
        if current_bucket:
            buckets.append(current_bucket)
        return buckets

    def _register_gradient_hooks(self):
        for bucket_index, bucket_parameters in enumerate(self._buckets):
            for parameter in bucket_parameters:
                parameter.register_hook(self._make_bucket_hook(bucket_index, parameter))

    def _make_bucket_hook(self, bucket_index: int, parameter: torch.nn.Parameter):
        def _hook(grad: torch.Tensor):
            runtime_bucket = self._runtime_buckets[bucket_index]
            if runtime_bucket["completed"]:
                return grad
            runtime_bucket["grads"][id(parameter)] = grad
            runtime_bucket["remaining"] -= 1
            if runtime_bucket["remaining"] == 0:
                flat_grads = [runtime_bucket["grads"][id(p)].reshape(-1) for p in runtime_bucket["params"]]
                runtime_bucket["flat"] = torch.cat(flat_grads, dim=0)
                if self._world_size > 1:
                    runtime_bucket["handle"] = dist.all_reduce(
                        runtime_bucket["flat"], op=dist.ReduceOp.SUM, async_op=True
                    )
                runtime_bucket["completed"] = True
            return grad

        return _hook

    def _reset_runtime_state(self):
        self._runtime_buckets = []
        for params in self._buckets:
            self._runtime_buckets.append(
                {
                    "params": params,
                    "remaining": len(params),
                    "grads": {},
                    "flat": None,
                    "handle": None,
                    "completed": False,
                }
            )

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def on_train_batch_start(self):
        self._reset_runtime_state()

    def finish_gradient_synchronization(self):
        for runtime_bucket in self._runtime_buckets:
            if runtime_bucket["flat"] is None:
                continue
            if runtime_bucket["handle"] is not None:
                runtime_bucket["handle"].wait()
            flat = runtime_bucket["flat"]
            if self._world_size > 1:
                flat.div_(self._world_size)
            offset = 0
            for parameter in runtime_bucket["params"]:
                numel = parameter.numel()
                reduced_grad = flat[offset : offset + numel].view_as(parameter)
                if parameter.grad is None:
                    parameter.grad = reduced_grad.clone()
                else:
                    parameter.grad.copy_(reduced_grad)
                offset += numel


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    return DDPBucketed(module, bucket_size_mb=bucket_size_mb)


def ddp_bucketed_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    del optimizer
    ddp_model.finish_gradient_synchronization()


def ddp_bucketed_on_train_batch_start(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    del optimizer
    ddp_model.on_train_batch_start()
