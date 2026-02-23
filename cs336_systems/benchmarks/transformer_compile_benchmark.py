from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM


TABLE1_MODEL_SIZES = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


@dataclass(frozen=True)
class CompileBenchmarkConfig:
    model_name: str
    batch_size: int
    context_length: int
    vocab_size: int
    warmup_steps: int
    measure_steps: int
    lr: float
    device: str
    dtype: torch.dtype
    mode: str  # forward_only | train_step
    compile_backend: str


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _build_model(cfg: CompileBenchmarkConfig) -> BasicsTransformerLM:
    m = TABLE1_MODEL_SIZES[cfg.model_name]
    model = BasicsTransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=m["d_model"],
        d_ff=m["d_ff"],
        num_layers=m["num_layers"],
        num_heads=m["num_heads"],
        rope_theta=10000.0,
    ).to(device=cfg.device, dtype=cfg.dtype)
    if cfg.mode == "forward_only":
        model.eval()
    else:
        model.train()
    return model


def _run_single_variant(cfg: CompileBenchmarkConfig, use_compile: bool) -> tuple[float, float]:
    model = _build_model(cfg)
    if use_compile:
        model = torch.compile(model, backend=cfg.compile_backend, dynamic=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.context_length), device=cfg.device)
    y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.context_length), device=cfg.device)

    for _ in range(cfg.warmup_steps):
        if cfg.mode == "forward_only":
            with torch.no_grad():
                _ = model(x)
            _sync(cfg.device)
        else:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
            _sync(cfg.device)

    step_ms: list[float] = []
    for _ in range(cfg.measure_steps):
        _sync(cfg.device)
        t0 = time.perf_counter()
        if cfg.mode == "forward_only":
            with torch.no_grad():
                _ = model(x)
        else:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
        _sync(cfg.device)
        step_ms.append((time.perf_counter() - t0) * 1000.0)

    return statistics.mean(step_ms), statistics.stdev(step_ms) if len(step_ms) > 1 else 0.0


def _to_markdown_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "| metric | value |\n| --- | --- |"
    headers = list(rows[0].keys())
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(row[h]) for h in headers) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _summarize_runtime_error(err: RuntimeError) -> tuple[str, str]:
    msg = str(err)
    msg_lower = msg.lower()
    if "out of memory" in msg_lower:
        return "oom", "out_of_memory"
    if "inductorerror" in msg_lower or "compilation subprocess exited unexpectedly" in msg_lower:
        return "error", "inductor_compile_crash"
    return "error", "runtime_error"


def run_compile_benchmark(cfg: CompileBenchmarkConfig) -> str:
    rows: list[dict[str, str]] = []
    baseline_mean = None
    baseline_std = None
    baseline_status = "ok"
    baseline_error = "-"
    compiled_mean = None
    compiled_std = None
    compiled_status = "ok"
    compiled_error = "-"

    try:
        baseline_mean, baseline_std = _run_single_variant(cfg, use_compile=False)
    except RuntimeError as e:
        baseline_status, baseline_error = _summarize_runtime_error(e)
        if cfg.device.startswith("cuda"):
            torch.cuda.empty_cache()

    try:
        compiled_mean, compiled_std = _run_single_variant(cfg, use_compile=True)
    except RuntimeError as e:
        compiled_status, compiled_error = _summarize_runtime_error(e)
        if cfg.device.startswith("cuda"):
            torch.cuda.empty_cache()

    speedup = "-"
    if baseline_mean is not None and compiled_mean is not None:
        speedup = f"{(baseline_mean / max(compiled_mean, 1e-9)):.3f}"

    rows.append(
        {
            "variant": "eager",
            "mode": cfg.mode,
            "model": cfg.model_name,
            "mean_ms": "-" if baseline_mean is None else f"{baseline_mean:.3f}",
            "std_ms": "-" if baseline_std is None else f"{baseline_std:.3f}",
            "speedup_vs_eager": "-" if baseline_mean is None else "1.000",
            "status": baseline_status,
            "error": baseline_error,
        }
    )
    rows.append(
        {
            "variant": "compiled",
            "mode": cfg.mode,
            "model": cfg.model_name,
            "mean_ms": "-" if compiled_mean is None else f"{compiled_mean:.3f}",
            "std_ms": "-" if compiled_std is None else f"{compiled_std:.3f}",
            "speedup_vs_eager": speedup,
            "status": compiled_status,
            "error": compiled_error,
        }
    )
    return _to_markdown_table(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Transformer eager vs torch.compile.")
    parser.add_argument("--model", type=str, default="small", choices=list(TABLE1_MODEL_SIZES.keys()))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--mode", type=str, default="forward_only", choices=["forward_only", "train_step"])
    parser.add_argument(
        "--compile-backend",
        type=str,
        default="inductor",
        choices=["inductor", "aot_eager", "eager"],
        help="Backend passed to torch.compile for the compiled variant.",
    )
    parser.add_argument(
        "--compile-threads",
        type=int,
        default=1,
        help="Set TORCHINDUCTOR_COMPILE_THREADS (1 is more stable on some environments).",
    )
    parser.add_argument("--output-markdown", type=str, default=None)
    return parser.parse_args()


def _parse_dtype(value: str, device: str) -> torch.dtype:
    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = mapping[value]
    if device == "cpu" and dtype != torch.float32:
        return torch.float32
    return dtype


def main():
    args = parse_args()
    if args.compile_threads >= 1:
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(args.compile_threads)
    cfg = CompileBenchmarkConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        lr=args.lr,
        device=args.device,
        dtype=_parse_dtype(args.dtype, args.device),
        mode=args.mode,
        compile_backend=args.compile_backend,
    )
    table = run_compile_benchmark(cfg)
    print(table)
    if args.output_markdown:
        with open(args.output_markdown, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"\nSaved markdown table to: {args.output_markdown}")


if __name__ == "__main__":
    main()
