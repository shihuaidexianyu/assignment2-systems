from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

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
class MemoryProfileConfig:
    model_name: str
    context_lengths: list[int]
    batch_size: int
    vocab_size: int
    warmup_steps: int
    measure_steps: int
    lr: float
    device: str
    mixed_precision: bool
    output_dir: Path
    record_history: bool


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _to_markdown_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "| metric | value |\n| --- | --- |"
    headers = list(rows[0].keys())
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(row[h]) for h in headers) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _build_model(model_name: str, context_length: int, vocab_size: int, device: str) -> BasicsTransformerLM:
    m = TABLE1_MODEL_SIZES[model_name]
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=m["d_model"],
        d_ff=m["d_ff"],
        num_layers=m["num_layers"],
        num_heads=m["num_heads"],
        rope_theta=10000.0,
    ).to(device=device, dtype=torch.float32)
    model.train()
    return model


def _run_forward_only(model, x, use_mixed_precision: bool, device: str):
    with torch.no_grad():
        if use_mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model(x)
        else:
            _ = model(x)
    _sync(device)


def _run_train_step(model, optimizer, x, y, vocab_size: int, use_mixed_precision: bool, device: str):
    optimizer.zero_grad(set_to_none=True)
    if use_mixed_precision:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
    else:
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
    loss.backward()
    optimizer.step()
    _sync(device)


def _profile_phase(
    model_name: str,
    context_length: int,
    phase: str,
    cfg: MemoryProfileConfig,
) -> dict[str, str]:
    model = _build_model(model_name=model_name, context_length=context_length, vocab_size=cfg.vocab_size, device=cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, context_length), device=cfg.device, dtype=torch.long)
    y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, context_length), device=cfg.device, dtype=torch.long)

    if cfg.record_history:
        torch.cuda.memory._record_memory_history(
            enabled="all",
            context="all",
            stacks="all",
            max_entries=200000,
            clear_history=True,
        )

    torch.cuda.reset_peak_memory_stats(cfg.device)
    peaks_mb: list[float] = []

    for i in range(cfg.warmup_steps + cfg.measure_steps):
        if phase == "forward_only":
            _run_forward_only(model, x, cfg.mixed_precision, cfg.device)
        else:
            _run_train_step(model, optimizer, x, y, cfg.vocab_size, cfg.mixed_precision, cfg.device)
        if i >= cfg.warmup_steps:
            peaks_mb.append(torch.cuda.max_memory_allocated(cfg.device) / (1024**2))

    snapshot_path = ""
    if cfg.record_history:
        snapshot_path = str(cfg.output_dir / f"memory_snapshot_{phase}_ctx{context_length}.pickle")
        torch.cuda.memory._dump_snapshot(snapshot_path)
        torch.cuda.memory._record_memory_history(enabled=None)

    return {
        "model": model_name,
        "phase": phase,
        "context_length": str(context_length),
        "mixed_precision": str(cfg.mixed_precision),
        "peak_mean_mb": f"{statistics.mean(peaks_mb):.3f}",
        "peak_max_mb": f"{max(peaks_mb):.3f}",
        "snapshot_path": snapshot_path if snapshot_path else "-",
    }


def run_memory_profile(cfg: MemoryProfileConfig) -> str:
    if not cfg.device.startswith("cuda"):
        raise RuntimeError("Memory profiling script requires CUDA device.")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for context_length in cfg.context_lengths:
        rows.append(_profile_phase(cfg.model_name, context_length, "forward_only", cfg))
        rows.append(_profile_phase(cfg.model_name, context_length, "train_step", cfg))
    return _to_markdown_table(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch memory profiling for Transformer forward/train phases.")
    parser.add_argument("--model", type=str, default="2.7B", choices=list(TABLE1_MODEL_SIZES.keys()))
    parser.add_argument("--context-lengths", type=str, default="128,256,512")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable bf16 autocast for measured phase.")
    parser.add_argument(
        "--record-history",
        action="store_true",
        help="Enable torch.cuda.memory history recording and dump memory_snapshot_*.pickle.",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/memory")
    parser.add_argument("--output-markdown", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = MemoryProfileConfig(
        model_name=args.model,
        context_lengths=[int(v) for v in args.context_lengths.split(",") if v.strip()],
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        lr=args.lr,
        device=args.device,
        mixed_precision=args.mixed_precision,
        output_dir=Path(args.output_dir),
        record_history=args.record_history,
    )
    table = run_memory_profile(cfg)
    print(table)
    if args.output_markdown:
        with open(args.output_markdown, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"\nSaved markdown table to: {args.output_markdown}")


if __name__ == "__main__":
    main()
