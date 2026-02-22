from __future__ import annotations

import torch.distributed as dist
import torch.nn as nn


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def broadcast_module_state(module: nn.Module):
    if not is_dist_initialized():
        return
    for parameter in module.parameters():
        dist.broadcast(parameter.data, src=0)
    for buffer in module.buffers():
        dist.broadcast(buffer.data, src=0)
