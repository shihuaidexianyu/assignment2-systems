from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from .common import broadcast_module_state, is_dist_initialized


class DDPIndividualParameters(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._world_size = dist.get_world_size() if is_dist_initialized() else 1
        self._trainable_parameters: list[torch.nn.Parameter] = []
        self._pending_allreduces: dict[int, tuple[dist.Work, torch.Tensor, torch.Tensor]] = {}
        seen: set[int] = set()
        for parameter in self.module.parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if parameter.requires_grad:
                self._trainable_parameters.append(parameter)
                parameter.register_hook(self._make_grad_hook(parameter))
        broadcast_module_state(self.module)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _make_grad_hook(self, parameter: torch.nn.Parameter):
        def _hook(grad: torch.Tensor):
            if self._world_size == 1:
                return grad
            reduced_grad = grad if grad.is_contiguous() else grad.contiguous()
            handle = dist.all_reduce(reduced_grad, op=dist.ReduceOp.SUM, async_op=True)
            self._pending_allreduces[id(parameter)] = (handle, grad, reduced_grad)
            return grad

        return _hook

    def finish_gradient_synchronization(self):
        if self._world_size == 1:
            return
        for parameter in self._trainable_parameters:
            if parameter.grad is None:
                continue
            work_and_grad = self._pending_allreduces.get(id(parameter))
            if work_and_grad is None:
                reduced_grad = parameter.grad if parameter.grad.is_contiguous() else parameter.grad.contiguous()
                dist.all_reduce(reduced_grad, op=dist.ReduceOp.SUM)
                reduced_grad.div_(self._world_size)
                if reduced_grad.data_ptr() != parameter.grad.data_ptr():
                    parameter.grad.copy_(reduced_grad)
                continue
            handle, original_grad, reduced_grad = work_and_grad
            handle.wait()
            reduced_grad.div_(self._world_size)
            if reduced_grad.data_ptr() != original_grad.data_ptr():
                original_grad.copy_(reduced_grad)
            if parameter.grad.data_ptr() != original_grad.data_ptr():
                parameter.grad.copy_(original_grad)
        self._pending_allreduces.clear()


def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    return DDPIndividualParameters(module)


def ddp_individual_parameters_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    del optimizer
    ddp_model.finish_gradient_synchronization()
