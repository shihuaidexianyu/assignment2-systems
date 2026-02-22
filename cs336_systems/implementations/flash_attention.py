from __future__ import annotations

from typing import Type

import torch


def _attention_and_lse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool):
    d = q.shape[-1]
    scale = 1.0 / (d**0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        mask = torch.arange(n_queries, device=scores.device)[:, None] >= torch.arange(
            n_keys, device=scores.device
        )[None, :]
        scores = torch.where(mask, scores, torch.full_like(scores, -1e6))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    lse = torch.logsumexp(scores, dim=-1)
    return out, lse


class FlashAttentionAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
        out, lse = _attention_and_lse(q, k, v, is_causal=is_causal)
        ctx.save_for_backward(q, k, v, lse)
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, _lse = ctx.saved_tensors
        with torch.enable_grad():
            q_req = q.detach().requires_grad_(True)
            k_req = k.detach().requires_grad_(True)
            v_req = v.detach().requires_grad_(True)
            out, _ = _attention_and_lse(q_req, k_req, v_req, is_causal=ctx.is_causal)
            dq, dk, dv = torch.autograd.grad(
                outputs=out,
                inputs=(q_req, k_req, v_req),
                grad_outputs=dout,
                allow_unused=False,
            )
        return dq, dk, dv, None


def get_flashattention_autograd_function_pytorch() -> Type:
    return FlashAttentionAutogradFunction


def get_flashattention_autograd_function_triton() -> Type:
    return FlashAttentionAutogradFunction
