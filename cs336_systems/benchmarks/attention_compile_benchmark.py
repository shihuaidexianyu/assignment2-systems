from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttentionBenchmarkConfig:
    batch_size: int
    seq_lens: list[int]
    d_models: list[int]
    warmup_steps: int
    measure_steps: int
    device: str
    dtype: torch.dtype
    include_compile: bool
    causal: bool
    compile_backend: str


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _make_causal_mask(seq_len_q: int, seq_len_k: int, device: str) -> torch.Tensor:
    q_idx = torch.arange(seq_len_q, device=device)[:, None]
    k_idx = torch.arange(seq_len_k, device=device)[None, :]
    return q_idx >= k_idx


def regular_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = q @ k.transpose(-2, -1)
    scores = scores * scale
    if causal:
        mask = _make_causal_mask(q.shape[-2], k.shape[-2], q.device)
        scores = torch.where(mask, scores, torch.full_like(scores, -1e6))
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def _benchmark_forward(
    fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    warmup_steps: int,
    measure_steps: int,
    device: str,
) -> tuple[float, float]:
    for _ in range(warmup_steps):
        _ = fn(q, k, v, causal)
        _sync(device)

    samples: list[float] = []
    for _ in range(measure_steps):
        _sync(device)
        t0 = time.perf_counter()
        _ = fn(q, k, v, causal)
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = statistics.mean(samples)
    std_ms = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return mean_ms, std_ms


def _benchmark_backward(
    fn,
    batch_size: int,
    seq_len: int,
    d_model: int,
    causal: bool,
    warmup_steps: int,
    measure_steps: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[float, float, float]:
    def _build_inputs():
        q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
        return q, k, v

    for _ in range(warmup_steps):
        q, k, v = _build_inputs()
        out = fn(q, k, v, causal)
        out.sum().backward()
        _sync(device)

    samples: list[float] = []
    mem_samples_mb: list[float] = []
    for _ in range(measure_steps):
        q, k, v = _build_inputs()
        out = fn(q, k, v, causal)
        if device.startswith("cuda"):
            mem_samples_mb.append(torch.cuda.memory_allocated(device=device) / (1024**2))
        else:
            mem_samples_mb.append(float("nan"))
        _sync(device)
        t0 = time.perf_counter()
        out.sum().backward()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = statistics.mean(samples)
    std_ms = statistics.stdev(samples) if len(samples) > 1 else 0.0
    mean_mem_mb = statistics.mean(mem_samples_mb)
    return mean_ms, std_ms, mean_mem_mb


def _to_markdown_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "| metric | value |\n| --- | --- |"
    headers = list(rows[0].keys())
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(row[h]) for h in headers) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def run_attention_benchmarks(cfg: AttentionBenchmarkConfig) -> str:
    variants: list[tuple[str, callable]] = [("regular", regular_attention)]
    if cfg.include_compile:
        compiled = torch.compile(regular_attention, backend=cfg.compile_backend, dynamic=False)
        variants.append(("regular_compiled", compiled))

    rows: list[dict[str, str]] = []
    for seq_len in cfg.seq_lens:
        for d_model in cfg.d_models:
            q = torch.randn(cfg.batch_size, seq_len, d_model, device=cfg.device, dtype=cfg.dtype)
            k = torch.randn(cfg.batch_size, seq_len, d_model, device=cfg.device, dtype=cfg.dtype)
            v = torch.randn(cfg.batch_size, seq_len, d_model, device=cfg.device, dtype=cfg.dtype)
            for variant_name, fn in variants:
                try:
                    fwd_mean, fwd_std = _benchmark_forward(
                        fn=fn,
                        q=q,
                        k=k,
                        v=v,
                        causal=cfg.causal,
                        warmup_steps=cfg.warmup_steps,
                        measure_steps=cfg.measure_steps,
                        device=cfg.device,
                    )
                    bwd_mean, bwd_std, bwd_mem_mb = _benchmark_backward(
                        fn=fn,
                        batch_size=cfg.batch_size,
                        seq_len=seq_len,
                        d_model=d_model,
                        causal=cfg.causal,
                        warmup_steps=cfg.warmup_steps,
                        measure_steps=cfg.measure_steps,
                        device=cfg.device,
                        dtype=cfg.dtype,
                    )
                    rows.append(
                        {
                            "variant": variant_name,
                            "batch_size": str(cfg.batch_size),
                            "seq_len": str(seq_len),
                            "d_model": str(d_model),
                            "fwd_mean_ms": f"{fwd_mean:.3f}",
                            "fwd_std_ms": f"{fwd_std:.3f}",
                            "bwd_mean_ms": f"{bwd_mean:.3f}",
                            "bwd_std_ms": f"{bwd_std:.3f}",
                            "bwd_mem_before_mb": "nan" if math.isnan(bwd_mem_mb) else f"{bwd_mem_mb:.3f}",
                            "status": "ok",
                        }
                    )
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    if cfg.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    rows.append(
                        {
                            "variant": variant_name,
                            "batch_size": str(cfg.batch_size),
                            "seq_len": str(seq_len),
                            "d_model": str(d_model),
                            "fwd_mean_ms": "-",
                            "fwd_std_ms": "-",
                            "bwd_mean_ms": "-",
                            "bwd_std_ms": "-",
                            "bwd_mem_before_mb": "-",
                            "status": "oom",
                        }
                    )
    return _to_markdown_table(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attention benchmark with optional torch.compile.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-lens", type=str, default="256,1024,4096,8192,16384")
    parser.add_argument("--d-models", type=str, default="16,32,64,128")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--include-compile", action="store_true")
    parser.add_argument(
        "--compile-backend",
        type=str,
        default="inductor",
        choices=["inductor", "aot_eager", "eager"],
        help="Backend passed to torch.compile when --include-compile is enabled.",
    )
    parser.add_argument("--output-markdown", type=str, default=None)
    return parser.parse_args()


def _parse_dtype(dtype: str, device: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    resolved = mapping[dtype]
    if device == "cpu" and resolved != torch.float32:
        return torch.float32
    return resolved


def main():
    args = parse_args()
    cfg = AttentionBenchmarkConfig(
        batch_size=args.batch_size,
        seq_lens=[int(v) for v in args.seq_lens.split(",") if v.strip()],
        d_models=[int(v) for v in args.d_models.split(",") if v.strip()],
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        device=args.device,
        dtype=_parse_dtype(args.dtype, args.device),
        include_compile=args.include_compile,
        causal=args.causal,
        compile_backend=args.compile_backend,
    )
    markdown = run_attention_benchmarks(cfg)
    print(markdown)
    if args.output_markdown:
        with open(args.output_markdown, "w", encoding="utf-8") as f:
            f.write(markdown + "\n")
        print(f"\nSaved markdown table to: {args.output_markdown}")


if __name__ == "__main__":
    main()
