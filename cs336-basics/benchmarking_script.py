"""
Write a script to perform basic end-to-end benchmarking of the forward and backward passes in
your model. Specifically, your script should support the following:
• Given hyperparameters (e.g., number of layers), initialize a model.
• Generate a random batch of data.
• Call torch.cuda.synchronize() after each step.
• Run warm-up steps (before you start measuring time), then time the execution of n steps
  (either only forward, or both forward and backward passes, depending on an argument). For
  timing, you can use the Python timeit module (e.g., either using the timeit function, or
  using timeit.default_timer(), which gives you the system's highest resolution clock, thus
  a better default for benchmarking than time.time()).

Deliverable: A script that will initialize a basics Transformer model with the given hyperpa
rameters, create a random batch of data, and time forward and backward passes.
"""

from __future__ import annotations

import argparse
import statistics
import timeit
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency fallback
    pd = None

from cs336_basics.model import BasicsTransformerLM


@dataclass(frozen=True)
class BenchmarkConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float
    batch_size: int
    steps: int
    warmup_steps: int
    forward_only: bool
    device: str
    dtype: torch.dtype
    output_markdown: str | None


class Benchmarking:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model = self._build_model()
        self.input_ids, self.targets = self._build_batch()

    def _build_model(self) -> BasicsTransformerLM:
        cfg = self.config
        if cfg.d_model % cfg.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        model = BasicsTransformerLM(
            vocab_size=cfg.vocab_size,
            context_length=cfg.context_length,
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            d_ff=cfg.d_ff,
            rope_theta=cfg.rope_theta,
        )
        model.to(device=cfg.device, dtype=cfg.dtype)
        if cfg.forward_only:
            model.eval()
        else:
            model.train()
        return model

    def _build_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        input_ids = torch.randint(
            0,
            cfg.vocab_size,
            (cfg.batch_size, cfg.context_length),
            device=cfg.device,
            dtype=torch.long,
        )
        targets = torch.randint(
            0,
            cfg.vocab_size,
            (cfg.batch_size, cfg.context_length),
            device=cfg.device,
            dtype=torch.long,
        )
        return input_ids, targets

    def _synchronize(self) -> None:
        if torch.cuda.is_available() and self.config.device.startswith("cuda"):
            torch.cuda.synchronize()

    def _forward_only_step(self) -> None:
        with torch.no_grad():
            _ = self.model(self.input_ids)
        self._synchronize()

    def _forward_backward_step(self) -> None:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(self.input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            self.targets.reshape(-1),
        )
        loss.backward()
        self._synchronize()

    def _timed_forward_only_step(self) -> float:
        self._synchronize()
        start = timeit.default_timer()
        with torch.no_grad():
            _ = self.model(self.input_ids)
        self._synchronize()
        return (timeit.default_timer() - start) * 1000.0

    def _timed_forward_backward_step(self) -> tuple[float, float]:
        self.model.zero_grad(set_to_none=True)
        self._synchronize()
        forward_start = timeit.default_timer()
        logits = self.model(self.input_ids)
        self._synchronize()
        forward_ms = (timeit.default_timer() - forward_start) * 1000.0

        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            self.targets.reshape(-1),
        )
        backward_start = timeit.default_timer()
        loss.backward()
        self._synchronize()
        backward_ms = (timeit.default_timer() - backward_start) * 1000.0
        return forward_ms, backward_ms

    @staticmethod
    def _to_markdown_table_fallback(rows: list[dict[str, str | int | float]]) -> str:
        if not rows:
            return "| metric | value |\n|---|---|"

        headers = list(rows[0].keys())
        separator = ["---"] * len(headers)

        def _cell(value: object) -> str:
            text = str(value)
            return text.replace("|", "\\|").replace("\n", " ")

        header_row = "| " + " | ".join(headers) + " |"
        separator_row = "| " + " | ".join(separator) + " |"
        body_rows = [
            "| " + " | ".join(_cell(row.get(h, "")) for h in headers) + " |"
            for row in rows
        ]
        return "\n".join([header_row, separator_row, *body_rows])

    def _to_markdown_table(self, rows: list[dict[str, str | int | float]]) -> str:
        if pd is not None:
            try:
                return pd.DataFrame(rows).to_markdown(index=False)
            except ImportError:
                pass
        return self._to_markdown_table_fallback(rows)

    def run(self) -> str:
        step_fn = self._forward_only_step if self.config.forward_only else self._forward_backward_step

        for _ in range(self.config.warmup_steps):
            step_fn()

        mode = "forward" if self.config.forward_only else "forward+backward"
        tokens_per_step = self.config.batch_size * self.config.context_length
        print(f"Mode: {mode}")
        print(f"Steps: {self.config.steps} (warmup: {self.config.warmup_steps})")
        rows: list[dict[str, str | int | float]] = [
            {"metric": "mode", "value": mode},
            {"metric": "steps", "value": self.config.steps},
            {"metric": "warmup_steps", "value": self.config.warmup_steps},
        ]

        if self.config.forward_only:
            forward_ms_values = [self._timed_forward_only_step() for _ in range(self.config.steps)]
            avg_forward_ms = statistics.mean(forward_ms_values)
            std_forward_ms = statistics.stdev(forward_ms_values) if len(forward_ms_values) > 1 else 0.0
            tokens_per_sec = tokens_per_step * 1000.0 / avg_forward_ms
            print(f"Forward mean: {avg_forward_ms:.3f} ms")
            print(f"Forward std: {std_forward_ms:.3f} ms")
            print(f"Throughput: {tokens_per_sec:,.0f} tokens/s")

            rows.extend(
                [
                    {"metric": "forward_mean_ms", "value": f"{avg_forward_ms:.3f}"},
                    {"metric": "forward_std_ms", "value": f"{std_forward_ms:.3f}"},
                    {"metric": "tokens_per_sec", "value": f"{tokens_per_sec:,.0f}"},
                ]
            )
        else:
            forward_ms_values: list[float] = []
            backward_ms_values: list[float] = []
            for _ in range(self.config.steps):
                forward_ms, backward_ms = self._timed_forward_backward_step()
                forward_ms_values.append(forward_ms)
                backward_ms_values.append(backward_ms)

            avg_forward_ms = statistics.mean(forward_ms_values)
            std_forward_ms = statistics.stdev(forward_ms_values) if len(forward_ms_values) > 1 else 0.0
            avg_backward_ms = statistics.mean(backward_ms_values)
            std_backward_ms = statistics.stdev(backward_ms_values) if len(backward_ms_values) > 1 else 0.0
            avg_step_ms = avg_forward_ms + avg_backward_ms
            tokens_per_sec = tokens_per_step * 1000.0 / avg_step_ms
            print(f"Forward mean: {avg_forward_ms:.3f} ms")
            print(f"Forward std: {std_forward_ms:.3f} ms")
            print(f"Backward mean: {avg_backward_ms:.3f} ms")
            print(f"Backward std: {std_backward_ms:.3f} ms")
            print(f"Average step time: {avg_step_ms:.3f} ms")
            print(f"Throughput: {tokens_per_sec:,.0f} tokens/s")

            rows.extend(
                [
                    {"metric": "forward_mean_ms", "value": f"{avg_forward_ms:.3f}"},
                    {"metric": "forward_std_ms", "value": f"{std_forward_ms:.3f}"},
                    {"metric": "backward_mean_ms", "value": f"{avg_backward_ms:.3f}"},
                    {"metric": "backward_std_ms", "value": f"{std_backward_ms:.3f}"},
                    {"metric": "avg_step_ms", "value": f"{avg_step_ms:.3f}"},
                    {"metric": "tokens_per_sec", "value": f"{tokens_per_sec:,.0f}"},
                ]
            )

        markdown_table = self._to_markdown_table(rows)
        print("\nMarkdown table:\n")
        print(markdown_table)

        if self.config.output_markdown:
            output_path = Path(self.config.output_markdown)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(markdown_table + "\n", encoding="utf-8")
            print(f"\nSaved markdown table to: {output_path}")

        return markdown_table


def _parse_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_str}'. Choose from: {', '.join(mapping)}")
    return mapping[dtype_str]


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Benchmark BasicsTransformerLM forward/backward.")
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--forward-only", action="store_true", help="Only run forward pass.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--output-markdown",
        type=str,
        default=None,
        help="Optional path to save benchmark results as a Markdown table.",
    )
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = _parse_dtype(args.dtype)
    if device == "cpu" and dtype != torch.float32:
        dtype = torch.float32

    return BenchmarkConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        batch_size=args.batch_size,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        forward_only=args.forward_only,
        device=device,
        dtype=dtype,
        output_markdown=args.output_markdown,
    )


def main() -> None:
    config = parse_args()
    bench = Benchmarking(config)
    bench.run()


if __name__ == "__main__":
    main()
