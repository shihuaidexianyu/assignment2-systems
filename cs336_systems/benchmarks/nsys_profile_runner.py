from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


TABLE1_MODEL_SIZES = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


@dataclass(frozen=True)
class NsysJob:
    model_name: str
    context_length: int
    mode: str
    warmup_steps: int
    measure_steps: int


def _build_benchmark_command(job: NsysJob) -> list[str]:
    m = TABLE1_MODEL_SIZES[job.model_name]
    cmd = [
        "uv",
        "run",
        "python",
        "cs336-basics/benchmarking_script.py",
        "--vocab-size",
        "10000",
        "--context-length",
        str(job.context_length),
        "--d-model",
        str(m["d_model"]),
        "--num-layers",
        str(m["num_layers"]),
        "--num-heads",
        str(m["num_heads"]),
        "--d-ff",
        str(m["d_ff"]),
        "--batch-size",
        "4",
        "--warmup-steps",
        str(job.warmup_steps),
        "--steps",
        str(job.measure_steps),
        "--device",
        "cuda",
        "--dtype",
        "float32",
    ]
    if job.mode == "forward":
        cmd.append("--forward-only")
    return cmd


def _build_nsys_command(job: NsysJob, output_dir: Path) -> list[str]:
    profile_name = f"{job.model_name}_ctx{job.context_length}_{job.mode}"
    return [
        "nsys",
        "profile",
        "-t",
        "cuda,nvtx,osrt",
        "--sample",
        "none",
        "--force-overwrite",
        "true",
        "-o",
        str(output_dir / profile_name),
        *(_build_benchmark_command(job)),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or run nsys profiles for transformer benchmark modes.")
    parser.add_argument("--models", type=str, default="small,medium,large,xl,2.7B")
    parser.add_argument("--contexts", type=str, default="128,256,512,1024")
    parser.add_argument("--modes", type=str, default="forward,train")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="artifacts/nsys")
    parser.add_argument("--run", action="store_true", help="Run nsys profile jobs; otherwise print commands only.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[NsysJob] = []
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    contexts = [int(v) for v in args.contexts.split(",") if v.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for model_name in models:
        for context_length in contexts:
            for mode in modes:
                jobs.append(
                    NsysJob(
                        model_name=model_name,
                        context_length=context_length,
                        mode=mode,
                        warmup_steps=args.warmup_steps,
                        measure_steps=args.measure_steps,
                    )
                )

    has_nsys = shutil.which("nsys") is not None
    if args.run and not has_nsys:
        raise RuntimeError("`nsys` is not installed or not found in PATH.")

    for job in jobs:
        cmd = _build_nsys_command(job, output_dir)
        printable = " ".join(cmd)
        if not args.run:
            print(printable)
            continue
        print(f"Running: {printable}")
        subprocess.run(cmd, check=True)

    if not args.run:
        print(f"\nGenerated {len(jobs)} nsys commands.")
        if not has_nsys:
            print("Note: `nsys` not found in PATH on this machine.")
        print(f"Profiles will be written under: {output_dir}")


if __name__ == "__main__":
    main()
