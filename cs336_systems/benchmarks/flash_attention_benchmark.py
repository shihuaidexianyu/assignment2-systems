from __future__ import annotations

import argparse
import math

import torch
import triton.testing

from cs336_systems.implementations.flash_attention import (
    get_flashattention_autograd_function_pytorch,
    get_flashattention_autograd_function_triton,
)


def regular_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-2, -1)) * scale
    if causal:
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        mask = torch.arange(n_queries, device=q.device)[:, None] >= torch.arange(n_keys, device=q.device)[None, :]
        scores = torch.where(mask, scores, torch.full_like(scores, -1e6))
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def _make_input(batch_size: int, seq_len: int, d_model: int, dtype: torch.dtype, device: str):
    q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    return q, k, v


def _run_one(
    impl_name: str,
    attn_fn,
    batch_size: int,
    seq_len: int,
    d_model: int,
    dtype: torch.dtype,
    causal: bool,
    device: str,
) -> dict[str, str]:
    q, k, v = _make_input(batch_size, seq_len, d_model, dtype, device)

    def fwd():
        _ = attn_fn(q, k, v, causal)

    def bwd():
        q1, k1, v1 = _make_input(batch_size, seq_len, d_model, dtype, device)
        out = attn_fn(q1, k1, v1, causal)
        out.sum().backward()

    try:
        fwd_ms = triton.testing.do_bench(fwd, warmup=25, rep=100)
        bwd_ms = triton.testing.do_bench(bwd, warmup=25, rep=100)
        return {
            "impl": impl_name,
            "seq_len": str(seq_len),
            "d_model": str(d_model),
            "dtype": str(dtype).replace("torch.", ""),
            "causal": str(causal),
            "forward_ms": f"{fwd_ms:.3f}",
            "backward_ms": f"{bwd_ms:.3f}",
            "end_to_end_ms": f"{(fwd_ms + bwd_ms):.3f}",
            "status": "ok",
        }
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        return {
            "impl": impl_name,
            "seq_len": str(seq_len),
            "d_model": str(d_model),
            "dtype": str(dtype).replace("torch.", ""),
            "causal": str(causal),
            "forward_ms": "-",
            "backward_ms": "-",
            "end_to_end_ms": "-",
            "status": "oom",
        }


def _to_markdown_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "| metric | value |\n| --- | --- |"
    headers = list(rows[0].keys())
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(row[h]) for h in headers) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark regular attention vs FlashAttention implementation.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048,4096")
    parser.add_argument("--d-models", type=str, default="16,32,64,128")
    parser.add_argument("--dtypes", type=str, default="bf16,float32")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--non-causal", dest="causal", action="store_false")
    parser.set_defaults(causal=True)
    parser.add_argument("--use-triton-entrypoint", action="store_true")
    parser.add_argument("--output-markdown", type=str, default=None)
    return parser.parse_args()


def _parse_dtypes(values: str, device: str) -> list[torch.dtype]:
    mapping = {"float32": torch.float32, "fp32": torch.float32, "bf16": torch.bfloat16, "float16": torch.float16}
    dtypes: list[torch.dtype] = []
    for v in values.split(","):
        key = v.strip().lower()
        if not key:
            continue
        dtype = mapping[key]
        if device == "cpu" and dtype != torch.float32:
            continue
        dtypes.append(dtype)
    return dtypes or [torch.float32]


def main():
    args = parse_args()
    if not args.device.startswith("cuda"):
        raise RuntimeError("FlashAttention benchmark requires CUDA device.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable.")

    flash_cls = (
        get_flashattention_autograd_function_triton()
        if args.use_triton_entrypoint
        else get_flashattention_autograd_function_pytorch()
    )
    flash_apply = flash_cls.apply

    rows: list[dict[str, str]] = []
    seq_lens = [int(v) for v in args.seq_lens.split(",") if v.strip()]
    d_models = [int(v) for v in args.d_models.split(",") if v.strip()]
    dtypes = _parse_dtypes(args.dtypes, args.device)

    for dtype in dtypes:
        for seq_len in seq_lens:
            for d_model in d_models:
                rows.append(
                    _run_one(
                        impl_name="regular_attention",
                        attn_fn=regular_attention,
                        batch_size=args.batch_size,
                        seq_len=seq_len,
                        d_model=d_model,
                        dtype=dtype,
                        causal=args.causal,
                        device=args.device,
                    )
                )
                rows.append(
                    _run_one(
                        impl_name="flash_attention",
                        attn_fn=flash_apply,
                        batch_size=args.batch_size,
                        seq_len=seq_len,
                        d_model=d_model,
                        dtype=dtype,
                        causal=args.causal,
                        device=args.device,
                    )
                )

    table = _to_markdown_table(rows)
    print(table)
    if args.output_markdown:
        with open(args.output_markdown, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"\nSaved markdown table to: {args.output_markdown}")


if __name__ == "__main__":
    main()
