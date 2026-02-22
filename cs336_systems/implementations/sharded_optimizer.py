from __future__ import annotations

from typing import Type

import torch
import torch.distributed as dist

from .common import is_dist_initialized


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs):
        self._optimizer_cls = optimizer_cls
        self._optimizer_kwargs = kwargs
        self._rank = dist.get_rank() if is_dist_initialized() else 0
        self._world_size = dist.get_world_size() if is_dist_initialized() else 1
        super().__init__(params, defaults={})
        self._rebuild_local_optimizer()

    def _iter_unique_parameters(self):
        seen: set[int] = set()
        for group in self.param_groups:
            for parameter in group["params"]:
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                yield parameter

    def _rebuild_local_optimizer(self):
        ordered_parameters = list(self._iter_unique_parameters())
        self._owners: dict[int, int] = {id(p): idx % self._world_size for idx, p in enumerate(ordered_parameters)}

        local_param_groups = []
        for group in self.param_groups:
            local_parameters = [p for p in group["params"] if self._owners[id(p)] == self._rank]
            if not local_parameters:
                continue
            local_group = {k: v for k, v in group.items() if k != "params"}
            local_group["params"] = local_parameters
            local_param_groups.append(local_group)

        if local_param_groups:
            self._local_optimizer = self._optimizer_cls(local_param_groups, **self._optimizer_kwargs)
        else:
            self._local_optimizer = None

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._local_optimizer is not None:
            local_loss = self._local_optimizer.step(**kwargs)
            if loss is None:
                loss = local_loss

        if is_dist_initialized() and self._world_size > 1:
            for parameter in self._iter_unique_parameters():
                owner_rank = self._owners[id(parameter)]
                dist.broadcast(parameter.data, src=owner_rank)

        return loss

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        self._rebuild_local_optimizer()


def get_sharded_optimizer(params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    return ShardedOptimizer(params=params, optimizer_cls=optimizer_cls, **kwargs)
