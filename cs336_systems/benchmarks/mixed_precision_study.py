from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM


TABLE1_MODEL_SIZES = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def _sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _to_markdown_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "| metric | value |\n| --- | --- |"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def run_mixed_precision_accumulation(num_terms: int = 20000) -> dict[str, float]:
    # Build a numerically sensitive sum: one large value + many tiny values.
    x32 = torch.ones(num_terms, dtype=torch.float32) * 1e-3
    x32[0] = 1.0

    fp32_sum = x32.sum(dtype=torch.float32).item()

    x16 = x32.to(torch.float16)
    fp16_sum = x16.sum(dtype=torch.float16).item()

    mixed_input_fp16_acc_fp32 = x16.sum(dtype=torch.float32).item()

    # Chunked mixed accumulation (simulates partial reductions in fp16 then fp32 merge).
    chunk = 256
    partials = []
    for i in range(0, x16.numel(), chunk):
        partials.append(x16[i : i + chunk].sum(dtype=torch.float16))
    mixed_chunked = torch.stack(partials).sum(dtype=torch.float32).item()

    return {
        "fp32_sum": fp32_sum,
        "fp16_sum": fp16_sum,
        "mixed_input_fp16_acc_fp32_sum": mixed_input_fp16_acc_fp32,
        "mixed_chunked_sum": mixed_chunked,
    }


class ToyAutocastModel(nn.Module):
    def __init__(self, d_in: int = 64, d_hidden: int = 128, d_out: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.ln = nn.LayerNorm(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor):
        h = self.fc1(x)
        n = self.ln(h)
        logits = self.fc2(n)
        return h, n, logits


def run_toy_autocast_dtype_report(device: str) -> dict[str, str]:
    if not device.startswith("cuda"):
        raise RuntimeError("FP16 autocast dtype report requires CUDA.")

    torch.manual_seed(42)
    model = ToyAutocastModel().to(device=device, dtype=torch.float32)
    x = torch.randn(8, 64, device=device, dtype=torch.float32)
    y = torch.randint(0, 10, (8,), device=device, dtype=torch.long)

    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        fc1_out, ln_out, logits = model(x)
        loss = F.cross_entropy(logits, y)
    loss.backward()

    grad_dtype = next(p.grad.dtype for p in model.parameters() if p.grad is not None)
    return {
        "parameters_dtype_in_autocast": str(next(model.parameters()).dtype),
        "fc1_output_dtype": str(fc1_out.dtype),
        "layernorm_output_dtype": str(ln_out.dtype),
        "logits_dtype": str(logits.dtype),
        "loss_dtype": str(loss.dtype),
        "gradients_dtype": str(grad_dtype),
    }


@dataclass(frozen=True)
class MPBenchmarkConfig:
    model_name: str
    precision: str  # fp32 / bf16_autocast
    batch_size: int
    context_length: int
    vocab_size: int
    warmup_steps: int
    measure_steps: int
    lr: float


def _build_model(model_name: str, vocab_size: int, context_length: int, device: str) -> BasicsTransformerLM:
    cfg = TABLE1_MODEL_SIZES[model_name]
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=cfg["d_model"],
        d_ff=cfg["d_ff"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        rope_theta=10000.0,
    )
    model = model.to(device=device, dtype=torch.float32)
    model.train()
    return model


def _timed_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    device: str,
    precision: str,
    vocab_size: int,
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    _sync(device)
    f0 = time.perf_counter()
    if precision == "bf16_autocast":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
    else:
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
    _sync(device)
    f_ms = (time.perf_counter() - f0) * 1000.0

    b0 = time.perf_counter()
    loss.backward()
    _sync(device)
    b_ms = (time.perf_counter() - b0) * 1000.0
    optimizer.step()
    return f_ms, b_ms


def run_bf16_mixed_precision_benchmark(device: str, model_names: list[str], warmup_steps: int, measure_steps: int) -> str:
    if not device.startswith("cuda"):
        raise RuntimeError("BF16 mixed precision benchmark requires CUDA.")
    rows: list[dict[str, str]] = []

    for model_name in model_names:
        for precision in ["fp32", "bf16_autocast"]:
            cfg = MPBenchmarkConfig(
                model_name=model_name,
                precision=precision,
                batch_size=4,
                context_length=128,
                vocab_size=10000,
                warmup_steps=warmup_steps,
                measure_steps=measure_steps,
                lr=1e-4,
            )
            try:
                model = _build_model(cfg.model_name, cfg.vocab_size, cfg.context_length, device=device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
                x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.context_length), device=device)
                y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.context_length), device=device)

                f_vals: list[float] = []
                b_vals: list[float] = []
                for i in range(cfg.warmup_steps + cfg.measure_steps):
                    f_ms, b_ms = _timed_step(model, optimizer, x, y, device, cfg.precision, cfg.vocab_size)
                    if i >= cfg.warmup_steps:
                        f_vals.append(f_ms)
                        b_vals.append(b_ms)

                rows.append(
                    {
                        "model": cfg.model_name,
                        "precision": cfg.precision,
                        "fwd_mean_ms": f"{statistics.mean(f_vals):.3f}",
                        "bwd_mean_ms": f"{statistics.mean(b_vals):.3f}",
                        "status": "ok",
                    }
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    rows.append(
                        {
                            "model": cfg.model_name,
                            "precision": cfg.precision,
                            "fwd_mean_ms": "-",
                            "bwd_mean_ms": "-",
                            "status": "oom",
                        }
                    )
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
                else:
                    raise

    return _to_markdown_table(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mixed precision studies for assignment 2.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-accumulation", action="store_true")
    parser.add_argument("--run-dtype-report", action="store_true")
    parser.add_argument("--run-bf16-benchmark", action="store_true")
    parser.add_argument("--models", type=str, default="small,medium,large,xl,2.7B")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=3)
    parser.add_argument("--output-markdown", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    sections: list[str] = []

    if args.run_accumulation:
        values = run_mixed_precision_accumulation()
        rows = [{"metric": k, "value": f"{v:.8f}"} for k, v in values.items()]
        text = "## mixed_precision_accumulation\n\n" + _to_markdown_table(rows)
        print(text)
        sections.append(text)

    if args.run_dtype_report:
        values = run_toy_autocast_dtype_report(args.device)
        rows = [{"metric": k, "value": v} for k, v in values.items()]
        text = "## autocast_dtype_report\n\n" + _to_markdown_table(rows)
        print("\n" + text)
        sections.append(text)

    if args.run_bf16_benchmark:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]
        table = run_bf16_mixed_precision_benchmark(
            device=args.device,
            model_names=model_names,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
        )
        text = "## bf16_benchmark\n\n" + table
        print("\n" + text)
        sections.append(text)

    if args.output_markdown and sections:
        with open(args.output_markdown, "w", encoding="utf-8") as f:
            f.write("\n\n".join(sections) + "\n")
        print(f"\nSaved markdown report to: {args.output_markdown}")


if __name__ == "__main__":
    main()

