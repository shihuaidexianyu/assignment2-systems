# benchmarking_script

## a

`cs336-basics/benchmarking_script.py`

## b

**设置：**

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)
- 设备/数据类型: `cuda`、`float16`
- 批次大小: `4`
- 词汇表大小: `10000`
- 上下文长度: `128`
- 预热: `5` 步
- 测量: `10` 步
- 脚本: `cs336-basics/benchmarking_script.py`

| 规模 | d_model | d_ff | num_layers | num_heads | 前向传播均值 (ms) | 前向传播标准差 (ms) | 反向传播均值 (ms) | 反向传播标准差 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| small | 768 | 3072 | 12 | 12 | 19.920 | 2.175 | 23.206 | 3.082 |
| medium | 1024 | 4096 | 24 | 16 | 52.056 | 5.215 | 63.543 | 3.584 |
| large | 1280 | 5120 | 36 | 20 | 77.149 | 23.159 | 131.873 | 0.581 |
| xl | 1600 | 6400 | 48 | 25 | 103.368 | 4.547 | 2239.672 | 17.477 |
| 2.7B | 2560 | 10240 | 32 | 32 | 5305.990 | 174.848 | 48300.726 | 2803.290 |

**变异性说明：**

- 大多数运行的标准差相对于均值来说属于小到中等水平。
- `large` 模型的前向传播表现出高变异性（标准差/均值 ≈ 30%），表明存在偶发的离群步骤。
- 在此硬件上，最大模型的反向传播时间非常长，但在大多数情况下相对变异性仍然较小。

## c

**设置（与 b 部分相同，仅改变了预热步数）：**

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)
- 设备/数据类型: `cuda`、`float16`
- 批次大小: `4`
- 词汇表大小: `10000`
- 上下文长度: `128`
- 测量: `10` 步

| 规模 | 预热步数 | 前向传播均值 ± 标准差 (ms) | 反向传播均值 ± 标准差 (ms) |
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

**观察结论：**

- 移除预热（`warmup=0`）会显著增加变异性。例如：对于 `54.895 ms` 的均值，`small` 模型的前向传播标准差达到 `103.579 ms`；对于 `72.413 ms` 的均值，`medium` 模型的前向传播标准差为 `98.527 ms`。
- `warmup=1` 或 `2` 可以改善许多配置的稳定性，但大型模型仍表现出不稳定性，尤其是反向传播（`xl`、`2.7B`）。
- 在此设置下，`warmup=5` 通常是最稳定的，能够产生更紧凑的标准差。

**原因分析：**

- 前几次迭代包含一次性开销：CUDA 上下文和内核启动、分配器池增长，以及延迟初始化/自动调优效应。
- GPU 频率/电源状态也需要从空闲状态逐步提升，因此早期的步骤可能系统性更慢或噪声更大。
- 仅用 `1-2` 步预热时，这些瞬态效应可能尚未完全消除，尤其是对于大型模型。
- 在此环境中，大型模型的内存压力较大，因此早期步骤的内存行为可能与后续稳态行为不同，使得 `1-2` 步预热仍与 `5` 步预热存在差异。

# implementation_progress（截至 2026-02-22）

## 已完成代码项

- FlashAttention（PyTorch 版，含 causal mask 与 backward，测试通过）
  - 文件：`cs336_systems/implementations/flash_attention.py`
  - 说明：`get_flashattention_autograd_function_triton` 当前是 PyTorch fallback，不是 Triton kernel。
- DDP individual parameters（测试通过）
  - 文件：`cs336_systems/implementations/ddp_individual.py`
  - 说明：实现了按参数粒度异步 all-reduce；在 `finish_gradient_synchronization()` 统一等待并写回梯度。
- DDP bucketed（测试通过）
  - 文件：`cs336_systems/implementations/ddp_bucketed.py`
  - 说明：按 bucket 聚合梯度并做异步 all-reduce，`finish_gradient_synchronization()` 统一等待与回写。
- Optimizer state sharding（测试通过）
  - 文件：`cs336_systems/implementations/sharded_optimizer.py`
  - 说明：按 rank 分片参数 owner，local step 后由 owner 广播参数。

## 测试结果

- 命令：`uv run pytest -q tests`
- 结果：`16 passed, 1 warning`

包含通过的关键测试：

- `tests/test_attention.py`
- `tests/test_ddp_individual_parameters.py`
- `tests/test_ddp.py`
- `tests/test_sharded_optimizer.py`

## 尚未完成（写作/实验维度）

- `nsys_profile`：脚本已完成，当前机器无 `nsys` 二进制，未产出 trace
- `memory profiling`：2.7B + 多 context 已跑齐（含 OOM 记录），但 memory_viz 图和写作题仍待补齐
- `FlashAttention Triton kernel`：当前 Triton adapter 仍是 PyTorch fallback（功能正确但非融合 kernel）
- DDP / sharded optimizer：已完成 2xRTX5090 的 XL 实测；Nsight trace 与部分写作题仍待补齐

## distributed_benchmarking_scripts（新增）

已新增脚本：

- `cs336_systems/benchmarks/distributed_allreduce_single_node.py`
  - 覆盖 4.1 的参数维度：backend/world_size/tensor size sweep。
- `cs336_systems/benchmarks/naive_ddp_equivalence.py`
  - 覆盖 4.2：逐参数 all-reduce 的 naive DDP 与单进程结果一致性验证。
- `cs336_systems/benchmarks/ddp_strategy_benchmark.py`
  - 覆盖 4.4/4.6/4.8 的实验框架：
    - naive / flat / overlap_individual / overlap_bucketed
    - 输出 `mean_step_ms`、`mean_comm_wait_ms`、`comm_wait_ratio`
    - 支持 bucket size sweep

本机 smoke run（Gloo + CPU）示例：

### 4.1 All-Reduce 示例结果

| backend | world_size | tensor_mb | mean_ms | std_ms | effective_gbps |
| --- | --- | --- | --- | --- | --- |
| gloo | 2 | 1 | 1.241 | 0.225 | 0.845 |

### DDP 策略对比示例结果（tiny 配置）

| strategy | model_size | bucket_size_mb | mean_step_ms | mean_comm_wait_ms | comm_wait_ratio |
| --- | --- | --- | --- | --- | --- |
| naive | tiny | - | 56.044 | 38.714 | 0.6908 |
| flat | tiny | - | 20.447 | 7.565 | 0.3700 |
| overlap_individual | tiny | - | 23.335 | 8.489 | 0.3638 |
| overlap_bucketed | tiny | 1.0 | 20.020 | 4.391 | 0.2193 |

简要观察（仅 smoke run）：

- flatten 后通信等待时间显著低于 naive（`7.565ms` vs `38.714ms`）。
- overlap+bucket 进一步降低了 `finish_gradient_synchronization()` 的等待开销。
- 以上仅用于验证脚本与趋势；作业最终结果仍需按题目要求在指定设置（如单机 2 GPU、XL、bucket sweep）完整运行。

## distributed_benchmarking_results（2xRTX5090 实测，2026-02-23）

环境说明：

- 机器：`lfs-dev`
- GPU：`2 x NVIDIA GeForce RTX 5090`
- 分布式后端：`NCCL`（4.1 同时包含 `Gloo`）

### 4.1 all-reduce（world_size=2）

| backend | world_size | tensor_mb | mean_ms | std_ms | effective_gbps |
| --- | --- | --- | --- | --- | --- |
| gloo | 2 | 1 | 1.177 | 0.193 | 0.891 |
| gloo | 2 | 10 | 4.145 | 0.330 | 2.530 |
| gloo | 2 | 100 | 40.147 | 0.648 | 2.612 |
| gloo | 2 | 1024 | 433.904 | 18.193 | 2.475 |
| nccl | 2 | 1 | 0.069 | 0.003 | 15.302 |
| nccl | 2 | 10 | 0.541 | 0.681 | 19.393 |
| nccl | 2 | 100 | 3.319 | 0.023 | 31.591 |
| nccl | 2 | 1024 | 31.793 | 0.602 | 33.773 |

简要结论：

- `NCCL` 在所有消息大小上都显著优于 `Gloo`，大消息优势最明显（`1024MB`：`33.773 Gbps` vs `2.475 Gbps`）。
- 小消息更受固定通信开销影响，大消息时有效带宽更接近链路上限。

### 4.3/4.4/4.6（XL, world_size=2, bf16+sgd）

说明：`XL + AdamW` 在本环境会触发 optimizer-state OOM，因此 DDP 策略对比使用 `bf16 + SGD`（同一配置下各策略可公平比较）。

| strategy | model_size | bucket_size_mb | mean_step_ms | mean_comm_wait_ms | comm_wait_ratio |
| --- | --- | --- | --- | --- | --- |
| naive | xl | - | 283.717 | 139.948 | 0.4933 |
| flat | xl | - | 275.381 | 137.146 | 0.4980 |
| overlap_individual | xl | - | 271.323 | 42.633 | 0.1571 |

简要结论：

- `naive` 与 `flat` 的 step time 接近，但两者通信等待占比都接近 `0.5`。
- `overlap_individual` 把通信等待占比从 `~0.49` 显著压到 `~0.16`，说明通信被较好地隐藏在反向传播中。

### 4.8（bucket sweep, XL, world_size=2, bf16+sgd）

| strategy | model_size | bucket_size_mb | mean_step_ms | mean_comm_wait_ms | comm_wait_ratio |
| --- | --- | --- | --- | --- | --- |
| overlap_bucketed | xl | 1.0 | 230.696 | 7.770 | 0.0337 |
| overlap_bucketed | xl | 10.0 | 221.662 | 8.048 | 0.0363 |
| overlap_bucketed | xl | 100.0 | 224.568 | 10.617 | 0.0473 |
| overlap_bucketed | xl | 1000.0 | 264.168 | 12.227 | 0.0463 |

简要结论：

- 本次实验中 `10MB` bucket 最优（`221.662 ms/iter`）。
- bucket 过大（`1000MB`）会退化，原因是通信启动更晚、overlap 变弱。
- bucket 过小会增加通信次数和 per-collective 固定开销。

## mixed_precision_study（新增）

脚本：`python -m cs336_systems.benchmarks.mixed_precision_study ...`

### accumulation（1.3）

| metric | value |
| --- | --- |
| fp32_sum | 20.99900246 |
| fp16_sum | 21.00000000 |
| mixed_input_fp16_acc_fp32_sum | 21.00708771 |
| mixed_chunked_sum | 21.00686646 |

简要结论：

- FP16 在该构造下会产生可见舍入误差。
- “输入 FP16、累加 FP32”比纯 FP16 更稳定，但仍受输入量化误差影响。

### autocast dtype report（1.4a）

| metric | value |
| --- | --- |
| parameters_dtype_in_autocast | torch.float32 |
| fc1_output_dtype | torch.float16 |
| layernorm_output_dtype | torch.float32 |
| logits_dtype | torch.float16 |
| loss_dtype | torch.float32 |
| gradients_dtype | torch.float32 |

### BF16 benchmark（1.4c，small smoke run）

| model | precision | fwd_mean_ms | bwd_mean_ms | status |
| --- | --- | --- | --- | --- |
| small | fp32 | 67.520 | 83.271 | ok |
| small | bf16_autocast | 53.663 | 116.377 | ok |

## attention_and_compile（新增）

脚本：`python -m cs336_systems.benchmarks.attention_compile_benchmark ...`

### attention regular vs compile（2.1/2.2a，单点 smoke run）

| variant | batch_size | seq_len | d_model | fwd_mean_ms | fwd_std_ms | bwd_mean_ms | bwd_std_ms | bwd_mem_before_mb | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regular | 8 | 256 | 16 | 0.194 | 0.077 | 0.447 | 0.054 | 19.125 | ok |
| regular_compiled | 8 | 256 | 16 | 0.220 | 0.047 | 0.787 | 0.030 | 19.125 | ok |

### transformer eager vs compile（2.2b，small smoke run）

脚本：`python -m cs336_systems.benchmarks.transformer_compile_benchmark ...`

forward-only:

| variant | mode | model | mean_ms | std_ms | speedup_vs_eager |
| --- | --- | --- | --- | --- | --- |
| eager | forward_only | small | 68.587 | 11.122 | 1.000 |
| compiled | forward_only | small | 35.850 | 2.447 | 1.913 |

train-step:

| variant | mode | model | mean_ms | std_ms | speedup_vs_eager |
| --- | --- | --- | --- | --- | --- |
| eager | train_step | small | 206.769 | 55.713 | 1.000 |
| compiled | train_step | small | 151.916 | 3.278 | 1.361 |

## memory_profiling（新增）

脚本：`python -m cs336_systems.benchmarks.memory_profile_transformer --record-history ...`

2xRTX5090 实测（2.7B，batch_size=4，AdamW）：

FP32:

| model | phase | context_length | mixed_precision | optimizer | peak_mean_mb | peak_max_mb | status | error | snapshot_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.7B | forward_only | 128 | False | adamw | 13212.807 | 13212.807 | ok | - | artifacts/memory_fp32/memory_snapshot_forward_only_ctx128.pickle |
| 2.7B | train_step | 128 | False | adamw | - | - | oom | out_of_memory | - |
| 2.7B | forward_only | 256 | False | adamw | 13319.170 | 13319.170 | ok | - | artifacts/memory_fp32/memory_snapshot_forward_only_ctx256.pickle |
| 2.7B | train_step | 256 | False | adamw | - | - | oom | out_of_memory | - |
| 2.7B | forward_only | 512 | False | adamw | 13753.580 | 13753.580 | ok | - | artifacts/memory_fp32/memory_snapshot_forward_only_ctx512.pickle |
| 2.7B | train_step | 512 | False | adamw | - | - | oom | out_of_memory | - |

BF16 autocast:

| model | phase | context_length | mixed_precision | optimizer | peak_mean_mb | peak_max_mb | status | error | snapshot_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.7B | forward_only | 128 | True | adamw | 19599.135 | 19599.135 | ok | - | artifacts/memory_bf16/memory_snapshot_forward_only_ctx128.pickle |
| 2.7B | train_step | 128 | True | adamw | - | - | oom | out_of_memory | - |
| 2.7B | forward_only | 256 | True | adamw | 19624.338 | 19624.338 | ok | - | artifacts/memory_bf16/memory_snapshot_forward_only_ctx256.pickle |
| 2.7B | train_step | 256 | True | adamw | - | - | oom | out_of_memory | - |
| 2.7B | forward_only | 512 | True | adamw | 19857.580 | 19857.580 | ok | - | artifacts/memory_bf16/memory_snapshot_forward_only_ctx512.pickle |
| 2.7B | train_step | 512 | True | adamw | - | - | oom | out_of_memory | - |

简要结论：

- 在本机 32GB 显存条件下，`2.7B + AdamW` 的完整 `train_step` 在 `ctx=128/256/512` 均 OOM。
- `forward_only` 可稳定运行，且峰值随 context length 增长（FP32: `13212 -> 13754 MB`）。
- 本次 `BF16 autocast` 的 forward 峰值高于 FP32；这是因为参数仍为 FP32，且混精路径引入额外中间缓冲/工作区，未能在该配置下带来峰值显存下降。

## flash_attention_benchmark（新增）

脚本：`python -m cs336_systems.benchmarks.flash_attention_benchmark ...`

| impl | seq_len | d_model | dtype | causal | forward_ms | backward_ms | end_to_end_ms | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regular_attention | 128 | 16 | bfloat16 | True | 0.066 | 0.149 | 0.215 | ok |
| flash_attention | 128 | 16 | bfloat16 | True | 0.071 | 0.893 | 0.963 | ok |

说明：当前 `flash_attention` 路径仍为 PyTorch fallback，因此未体现 Triton fused kernel 的预期加速。

## sharded_optimizer_accounting（新增）

脚本：`python -m cs336_systems.benchmarks.sharded_optimizer_accounting ...`

`world_size=1` smoke run（用于脚本自检）：

| variant | world_size | model | init_alloc_mb | pre_step_alloc_mb | post_step_alloc_mb | peak_alloc_mb | param_mb | optimizer_state_mb_post | other_mb_post | step_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1 | small | 256.060 | 1058.584 | 1058.584 | 1317.082 | 245.333 | 490.667 | 322.584 | 22.643 |
| sharded | 1 | small | 256.060 | 1058.584 | 1058.584 | 1317.082 | 245.333 | 490.667 | 322.584 | 22.669 |

说明：`world_size=1` 时两者应一致；2 GPU 场景下才会出现 optimizer state 分片收益。

### 5.2 实测（XL, world_size=2, bf16）

| variant | world_size | model | init_alloc_mb | pre_step_alloc_mb | post_step_alloc_mb | peak_alloc_mb | param_mb | optimizer_state_mb_post | other_mb_post | step_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 2 | xl | 3901.390 | 15622.678 | 15622.678 | 19530.686 | 3811.331 | 7622.664 | 4188.683 | 55.792 |
| sharded | 2 | xl | 3901.390 | 11721.312 | 11721.312 | 13678.635 | 3811.331 | 3811.332 | 4098.648 | 125.828 |

简要结论：

- sharding 将 optimizer state 从 `7622.664MB` 降到 `3811.332MB`（约减半），峰值显存也从 `19530.686MB` 降到 `13678.635MB`。
- 代价是 step time 上升（`55.792ms -> 125.828ms`），反映了每步额外参数同步通信开销。
- 与 ZeRO stage 1 的关系：本实现与 ZeRO-1 在目标上相同（分片 optimizer states 以省显存），但工程细节和通信调度更简化，未做完整 ZeRO 体系中的优化（如更细粒度 overlap/融合策略）。

## nsys_profile_runner（新增）

脚本：`python -m cs336_systems.benchmarks.nsys_profile_runner ...`

当前机器输出：`nsys` 不在 PATH，因此脚本已验证为“命令生成模式”；待在安装 Nsight Systems 的机器执行 `--run` 产出 trace。

## 提交前说明

- 当前 `handout.md` 已包含 1.1 benchmarking_script 的结果和分析，以及本次代码实现进度。
- 最终提交前建议将本文整理为 `writeup.pdf`，并补齐上述未完成实验题。
