# benchmarking_script

## a
`cs336-basics/benchmarking_script.py`

## b
Setup:
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)
- Device/dtype: `cuda`, `float16`
- Batch size: `4`
- Vocab size: `10000`
- Context length: `128`
- Warmup: `5` steps
- Measurement: `10` steps
- Script: `cs336-basics/benchmarking_script.py`

| Size | d_model | d_ff | num_layers | num_heads | Forward mean (ms) | Forward std (ms) | Backward mean (ms) | Backward std (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| small | 768 | 3072 | 12 | 12 | 19.920 | 2.175 | 23.206 | 3.082 |
| medium | 1024 | 4096 | 24 | 16 | 52.056 | 5.215 | 63.543 | 3.584 |
| large | 1280 | 5120 | 36 | 20 | 77.149 | 23.159 | 131.873 | 0.581 |
| xl | 1600 | 6400 | 48 | 25 | 103.368 | 4.547 | 2239.672 | 17.477 |
| 2.7B | 2560 | 10240 | 32 | 32 | 5305.990 | 174.848 | 48300.726 | 2803.290 |

Variability notes:
- Standard deviation is generally small to moderate relative to mean for most runs.
- `large` forward pass had high variability (std/mean ~= 30%), suggesting occasional outlier steps.
- The largest models show very long backward times on this hardware, but relative variation is still modest in most cases.

# c
Setup (same as part b, only warmup changed):
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)
- Device/dtype: `cuda`, `float16`
- Batch size: `4`
- Vocab size: `10000`
- Context length: `128`
- Measurement: `10` steps

| Size | Warmup steps | Forward mean ± std (ms) | Backward mean ± std (ms) |
|---|---:|---:|---:|
| small | 0 | 54.895 ± 103.579 | 38.504 ± 43.400 |
| small | 1 | 21.779 ± 3.050 | 27.148 ± 5.358 |
| small | 2 | 21.485 ± 2.118 | 25.143 ± 1.869 |
| small | 5 | 19.920 ± 2.175 | 23.206 ± 3.082 |
| medium | 0 | 72.413 ± 98.527 | 70.682 ± 26.522 |
| medium | 1 | 43.484 ± 5.508 | 61.311 ± 1.343 |
| medium | 2 | 45.230 ± 3.047 | 62.480 ± 1.895 |
| medium | 5 | 52.056 ± 5.215 | 63.543 ± 3.584 |
| large | 0 | 101.146 ± 108.551 | 142.805 ± 23.597 |
| large | 1 | 78.032 ± 24.142 | 137.020 ± 3.851 |
| large | 2 | 68.659 ± 18.033 | 132.744 ± 2.199 |
| large | 5 | 77.149 ± 23.159 | 131.873 ± 0.581 |
| xl | 0 | 147.227 ± 116.770 | 2838.276 ± 443.291 |
| xl | 1 | 131.035 ± 32.839 | 3608.187 ± 1453.018 |
| xl | 2 | 103.445 ± 7.546 | 2445.984 ± 487.236 |
| xl | 5 | 103.368 ± 4.547 | 2239.672 ± 17.477 |
| 2.7B | 0 | 5440.966 ± 985.767 | 52875.512 ± 5676.078 |
| 2.7B | 1 | 6559.964 ± 1949.146 | 75058.446 ± 21757.590 |
| 2.7B | 2 | 4800.032 ± 821.181 | 46554.128 ± 6760.723 |
| 2.7B | 5 | 5305.990 ± 174.848 | 48300.726 ± 2803.290 |

Observations:
- Removing warmup (`warmup=0`) significantly increases variability. Example: `small` forward std is `103.579 ms` for a `54.895 ms` mean; `medium` forward std is `98.527 ms` for a `72.413 ms` mean.
- `warmup=1` or `2` improves stability for many settings, but large models still show instability, especially backward (`xl`, `2.7B`).
- `warmup=5` is generally the most stable in this setup and gives tighter standard deviations.

Why this happens:
- First iterations include one-time costs: CUDA context and kernel startup, allocator pool growth, and lazy initialization/autotuning effects.
- GPU frequency/power state also ramps up from idle, so early steps can be systematically slower or noisier.
- With only `1-2` warmup steps, these transients may not be fully gone, especially for larger models.
- In this environment, larger models are memory-stressed, so early-step memory behavior can differ from later steady-state behavior, making `1-2` warmup runs still different from `5`.
