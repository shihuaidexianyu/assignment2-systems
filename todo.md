# CS336 Assignment 2（Systems）— TODO 清单

> 目标：提升单 GPU 训练速度 + 扩展到多 GPU（数据并行 + 优化器状态分片）。  
> 交付：`writeup.pdf`（所有书面题） + `code.zip`（所有代码）。用 `test_and_make_submission.sh` 生成 `code.zip`。

---

## 0. 项目/环境与提交

- [x] Clone 代码仓库：`github.com/stanford-cs336/assignment2-systems`
- [x] 确认能导入 Basics（A1）模型：`uv run python` → `import cs336_basics`
- [x] 熟悉目录结构
  - `cs336-basics/`：A1 staff solution（供你复用/对照）
  - `cs336_systems/`：你主要写代码的模块（默认是空的）
  - `tests/*.py`：必须通过的测试（通过 `tests/adapters.py` 的 hook 调你写的实现）
- [x] 实现/对接 `tests/adapters.py` 中要求的 adapter（不同题目会点名）
- [x] 本地跑测试（建议多跑几次，尤其是分布式相关）
- [ ] 准备提交：
  - [ ] `writeup.pdf`：回答所有 written questions（排版/latex/markdown 转 pdf 都行，要求“typeset”）
  - [x] `code.zip`：包含你写的全部代码（用 `test_and_make_submission.sh` 生成）

进度快照（2026-02-22）：`uv run pytest -q tests` 已通过（`16 passed, 1 warning`）。

---

## 1. Profiling & Benchmarking（单 GPU）

### 1.1 Problem: `benchmarking_script`（4 pts）

- [x] (a) 写一个端到端 benchmark 脚本（建议命令行参数可配）
  - [x] 根据超参初始化模型
  - [x] 随机生成 batch 数据
  - [x] 先 warm-up `w` 步，再测量 `n` 步（支持只 forward 或 forward+backward）
  - [x] 每一步后 `torch.cuda.synchronize()`
  - [x] 计时用 `timeit`（推荐 `timeit.default_timer()`）
- [x] (b) 对表 1 的模型尺寸：`w=5`，测量 `n=10`，报告均值+标准差（forward / backward 各自）
- [x] (c) 复现「不 warm-up」以及「1~2 步 warm-up」的结果差异，并解释原因（缓存/编译/内存分配等）

### 1.2 Problem: `nsys_profile`（5 pts）

- [ ] 用 `nsys` 分别 profile：forward / backward / optimizer step
  - [ ] 模型：表 1 所有尺寸
  - [ ] context length：128 / 256 / 512 / 1024（大模型可能 OOM 就在报告里说明）
- [x] 已补脚本：`python -m cs336_systems.benchmarks.nsys_profile_runner ...`（可批量生成/执行命令）
  - 说明：当前机器未安装 `nsys`，尚未产出 trace 文件
- [ ] (a) forward 总耗时？是否与 Python 计时一致？
- [ ] (b) forward 中累计 GPU 时间最长的 CUDA kernel 是哪个？单次 forward 调用多少次？做 forward+backward 时是否同一个 kernel 最耗时？
- [ ] (c) 除 matmul 外，forward 中还有哪些 kernel 贡献了非 trivial 的 CUDA runtime？
- [ ] (d) profile 完整训练一步（fwd + loss + bwd + AdamW step）：matmul 占比相较 inference（仅 fwd）如何变化？其他 kernel 呢？
- [ ] (e) 在 self-attention 层内，对比 softmax vs matmul 的 runtime；它们的 runtime 差距和 FLOPs 差距是否一致？

### 1.3 Problem: `mixed_precision_accumulation`（1 pt）

- [x] 跑题目给的 4 段代码（FP32/FP16/混合累加）并评论数值准确性（2~3 句）
  - 脚本：`python -m cs336_systems.benchmarks.mixed_precision_study --run-accumulation ...`

### 1.4 Problem: `benchmarking_mixed_precision`（2 pts）

- [x] (a) ToyModel + autocast(FP16)：写出以下各项的 dtype
  - [x] 参数（autocast context 内）
  - [x] fc1 输出
  - [x] layernorm 输出
  - [x] logits
  - [x] loss
  - [x] gradients
- [ ] (b) 为什么 layernorm 对混精敏感？换 BF16 还需要特殊处理吗？为什么？
- [x] (c) 给你的端到端 benchmark 脚本加 BF16 mixed precision 开关
  - [ ] 表 1 所有模型尺寸：对比 full precision vs mixed precision 的 fwd/bwd 耗时
  - [ ] 给 2~3 句趋势总结（随模型变大是否更划算等）
  - 脚本：`python -m cs336_systems.benchmarks.mixed_precision_study --run-bf16-benchmark ...`

### 1.5 Problem: `memory_profiling`（4 pts）

- [x] 给 profiling/benchmark 脚本加一个「开启 PyTorch memory profiler」选项
  - [x] 记录 memory history → dump `memory_snapshot.pickle` → 用 `pytorch.org/memory_viz` 看
  - 脚本：`python -m cs336_systems.benchmarks.memory_profile_transformer --record-history ...`
- [ ] 只做 2.7B 模型：context length 128 / 256 / 512
- [ ] (a) 产出两张图（memory_viz 的 Active memory timeline）：
  - [ ] 仅 forward
  - [ ] 完整训练一步（fwd+bwd+optimizer step）
  - [ ] 外加 2~3 句解释：不同阶段的峰值能否区分？
- [ ] (b) 表格：每个 context length 的 peak memory（forward-only vs full training step）
- [ ] (c) 混精下（mixed precision）2.7B 的 peak memory（forward / full step）是否显著变化？（2~3 句）
- [ ] (d) 在参考超参下：residual stream 的 activation tensor（FP32）大小（MB）推导
- [ ] (e) memory_viz 把 Detail 降到只剩最大分配：最大分配有多大？从 stack trace 看来源是哪儿？

---

## 2. Attention 与 torch.compile

### 2.1 Problem: `pytorch_attention`（2 pts）

- [x] 写 attention benchmark 脚本（只测 attention，不测整模型）
  - [x] batch 固定 8；不使用 multihead（去掉 head 维度）
  - [x] d_model ∈ {16, 32, 64, 128}
  - [x] seq_len ∈ {256, 1024, 4096, 8192, 16384}
  - [x] 随机生成 Q/K/V
  - [x] 计时：100 次 forward
  - [x] backward：测量 backward 前的 memory in use，并计时 100 次 backward
  - [x] warmup + 每次后 `torch.cuda.synchronize()`
  - 脚本：`python -m cs336_systems.benchmarks.attention_compile_benchmark ...`
- [ ] 汇报：timings/或 OOM；找最小能 OOM 的配置并做 attention memory usage 的手算；讨论 seq_len 增大时 backward 需要保存的 memory 如何变；怎么消掉这笔成本？

### 2.2 Problem: `torch_compile`（2 pts）

- [x] (a) 在 attention benchmark 脚本中加入 `torch.compile` 版本 attention，并与未编译对比（fwd/bwd 表格）
- [x] (b) 在端到端 benchmark 脚本里对整个 Transformer `torch.compile(model)`，对比：
  - [x] 仅 forward
  - [x] forward+backward+optimizer step
  - [x] 输出对比表
  - 脚本：`python -m cs336_systems.benchmarks.transformer_compile_benchmark ...`

---

## 3. FlashAttention-2（Triton）

### 3.1 Problem: `flash_forward`（15 pts）

- [ ] (a) 写纯 PyTorch（no Triton）的 FlashAttention-2 **forward**（autograd.Function）
  - [x] 输入：Q, K, V, `is_causal`（这一步可忽略 causal）
  - [x] 输出：O + logsumexp L（forward 返回 O；保存 L,Q,K,V,O 供 backward 用）
  - [ ] backward 先 `raise NotImplementedError`
  - [ ] tile ≥ 16×16；测试维度保证是 2 的幂且 ≥16（无需越界处理）
  - [x] 对接 adapter：`adapters.get_flashattention_autograd_function_pytorch`
  - [x] 测试：`uv run pytest -k test_flash_forward_pass_pytorch`
- [ ] (b) 写 FlashAttention-2 forward 的 Triton kernel（融合 kernel）+ autograd.Function wrapper
  - [ ] launch grid = (Tq, batch_size)
  - [ ] kernel 只有一层 loop：遍历 key tiles（1..Tk）
  - [ ] on-chip buffers (O_i, l, m) 用 `tl.float32`；accumulate 用 `acc=...`
  - [ ] P̃ 乘 V 前做 dtype cast；写回 O 前 cast 回输出 dtype
  - [ ] 对接 adapter：`adapters.get_flash_autograd_function_triton`
  - [x] 测试：`uv run pytest -k test_flash_forward_pass_triton`
- [ ] (c) 增加 causal masking 开关（最后一个参数，默认 False）
  - [ ] Triton kernel 增加 `is_causal: tl.constexpr`
  - [x] 对被 mask 的 score 加 `-1e6`
  - [x] 保存 `ctx.is_causal = is_causal`
  - [x] 保证 (a)/(b) 的测试仍通过（默认 False 不影响）

### 3.2 Problem: `flash_backward`（5 pts）

- [ ] 用 PyTorch + `torch.compile` 实现 FA2 backward（不写 Triton backward）
  - [x] 输入：Q, K, V, O, dO, L
  - [x] 输出：dQ, dK, dV
  - [ ] 记得计算并使用 D 向量（rowsum(O ◦ dO)）
  - [x] 测试：`uv run pytest -k test_flash_backward`

### 3.3 Problem: `flash_benchmarking`（5 pts）

- [x] 写 benchmark 脚本（用 `triton.testing.do_bench`）比较：
  - [x] 你的 FA2（forward/backward/端到端 fwd+bwd）
  - [x] 纯 PyTorch regular attention（同样的 forward/backward/端到端）
  - 脚本：`python -m cs336_systems.benchmarks.flash_attention_benchmark ...`
- [ ] 设定：
  - [ ] 单卡 H100
  - [ ] batch size = 1
  - [ ] causal masking = True
  - [ ] sweep：seq_len = 2^k，从 128 到 65536；d ∈ 2^k，从 16 到 128；dtype ∈ {bf16, fp32}
  - [ ] 可能需要按输入尺寸调 tile size
- [ ] 输出：对比表（forward/backward/end-to-end latency）

### 3.4 Optional（不计分但很建议）：Leaderboard / Triton backward

- [ ] 参加 leaderboard：优化 forward+backward（必须 Triton，不能用 CUDA；接口不能改；BF16+causal）
- [ ] 可选优化方向（示例）：
  - [ ] tile size autotune、调 Triton config
  - [ ] Triton 实现 backward（两趟：dQ 与 dK/dV 分开以避免原子/同步）
  - [ ] causal 下跳过全零 tiles / 分离非 mask tiles 与对角 tiles
  - [ ] 利用 H100 TMA 等

---

## 4. Distributed / DDP（多 GPU）

### 4.1 Problem: `distributed_communication_single_node`（5 pts）

- [x] 写脚本 benchmark **单机多进程** all-reduce
  - [x] backend + device：Gloo+CPU、NCCL+GPU
  - [x] tensor size：1MB / 10MB / 100MB / 1GB（float32）
  - [x] 进程数：2 / 4 / 6
  - [x] 资源：最多 6 GPUs；每个 benchmark run < 5 分钟
- [ ] 输出：对比 plot/table + 2~3 句总结（这些因素如何交互）
  - 脚本：`python -m cs336_systems.benchmarks.distributed_allreduce_single_node ...`

### 4.2 Problem: `naive_ddp`（5 pts）

- [x] 写一个朴素 DDP 训练脚本：
  - [x] backward 之后对 **每个参数 grad** 做 all-reduce
  - [x] 用随机数据训练一个 toy model，验证它的 weights 与单进程训练一致
  - 脚本：`python -m cs336_systems.benchmarks.naive_ddp_equivalence --world-size 2 --steps 3 --local-batch-size 8`

### 4.3 Problem: `naive_ddp_benchmarking`（3 pts）

- [x] benchmark 朴素 DDP 训练你的语言模型（脚本支持）
  - [ ] setting：单机 2 GPU；模型：XL
  - [x] 测量：每步训练总时间 + 通信占比（梯度 all-reduce 占用）
  - 脚本：`python -m cs336_systems.benchmarks.ddp_strategy_benchmark ... --strategies naive`
- [ ] 输出：benchmark 设置描述 + 每步耗时 + 通信耗时

### 4.4 Problem: `minimal_ddp_flat_benchmarking`（2 pts）

- [x] 改造 minimal DDP：把所有 grad flatten 成一个大 tensor，做 **一次** batched all-reduce
- [ ] 对比：原「每个参数一个 all-reduce」 vs 「单次 batched all-reduce」
  - [ ] 指标：time/iter + time communicating gradients
  - [ ] 1~2 句结论
  - 脚本：`python -m cs336_systems.benchmarks.ddp_strategy_benchmark ... --strategies naive,flat`

### 4.5 Problem: `ddp_overlap_individual_parameters`（5 pts）

- [x] 实现 DDP wrapper 类：边反传边通信（**按参数粒度**异步 all-reduce）
  - [x] 训练前 broadcast 权重（保证各 rank 初始一致）
  - [x] interface：
    - [x] `__init__(module: nn.Module)`
    - [x] `forward(*inputs, **kwargs)`
    - [x] `finish_gradient_synchronization()`：在 optimizer.step() 前等待通信 handle
  - [x] 对接 adapter：
    - [x] `adapters.get_ddp_individual_parameters`
    - [x] `adapters.ddp_individual_parameters_on_after_backward`（可选）
  - [x] 测试：`uv run pytest tests/test_ddp_individual_parameters.py`（建议跑 5 次）

### 4.6 Problem: `ddp_overlap_individual_parameters_benchmarking`（1 pt）

- [ ] (a) benchmark overlap 版本（单机 2 GPU，XL）
  - [ ] 与之前两种设置对比：逐参 all-reduce / flatten batched all-reduce
  - [ ] 输出：time/iter + 1~2 句对比
  - 已实现 benchmark 脚本：`python -m cs336_systems.benchmarks.ddp_strategy_benchmark ... --strategies naive,flat,overlap_individual`
- [ ] (b) Nsight 对比 trace：
  - [ ] 初始 DDP（不 overlap）
  - [ ] overlap DDP（有 overlap）
  - [ ] 输出：2 张截图（能看出是否把通信隐藏在 backward 里）

### 4.7 Problem: `ddp_overlap_bucketed`（8 pts）

- [x] 实现 bucketed overlap DDP（既 bucket 降低通信次数，又尽量 overlap）
  - [x] `__init__(module, bucket_size_mb)`
  - [x] bucket 内参数总大小 ≤ bucket_size_mb
  - [x] 推荐按 `reversed(list(model.parameters()))` 分桶（更贴近 grad ready 顺序）
  - [x] 其他接口同上：forward / finish_gradient_synchronization
  - [x] 对接 adapter：
    - [x] `adapters.get_ddp_bucketed`
    - [x] `adapters.ddp_bucketed_on_after_backward`（可选）
    - [x] `adapters.ddp_bucketed_on_train_batch_start`（可选）
  - [x] 测试：`pytest tests/test_ddp.py`（建议跑 5 次）

### 4.8 Problem: `ddp_bucketed_benchmarking`（3 pts）

- [ ] (a) benchmark bucket_size_mb ∈ {1, 10, 100, 1000}（单机 2 GPU，XL）
  - [ ] 对比：不 bucket 的结果是否符合预期？不符合就解释（必要时用 PyTorch profiler）
  - [ ] 你认为要怎么改实验设置才更符合预期？
  - [ ] 输出：各 bucket size 的 time/iter + 3~4 句分析
  - 已实现 sweep 脚本：`python -m cs336_systems.benchmarks.ddp_strategy_benchmark ... --strategies overlap_bucketed --bucket-sizes-mb 1,10,100,1000`
- [ ] (b) 建模：
  - [ ] 给出 DDP 通信 overhead 的公式：用参数总大小 s（bytes）、all-reduce 带宽 w、每次通信开销 o（seconds）、bucket 数 n_b
  - [ ] 推导最优 bucket size（使 overhead 最小）

### 4.9 Problem: `communication_accounting`（10 pts）

- [ ] XXL config：d_model=16384，d_ff=53248，num_blocks=126（只算 FFN：两层 linear；忽略 attention/embedding/output）
  - [ ] 激活与通信 BF16；累计梯度/主权重/optimizer state FP32；不做 activation checkpointing
- [ ] (a) 单设备：master weights + accumulated grads + optimizer states（FP32）占用多少？saved-for-backward（BF16）多少？折合多少张 H100 80GB？
- [ ] (b) FSDP：master weights、optimizer state、grads、以及一半 activations（比如每隔一层）分片到 N 台设备：每设备内存表达式？N 至少多少使其 < TPU v5p 单卡 95GB？
- [ ] (c) 只看 forward：用 TPU v5p 的 W_ici 与 C（Scaling Book 给定），mesh 设定：M_X=2, M_Y=1（3D mesh），X=16（FSDP 维度），Y=4（TP 维度）
  - [ ] 在什么 per-device batch size 下 compute-bound？
  - [ ] 该 setting 下 overall batch size 是多少？
- [ ] (d) 实际希望 overall batch size 尽量小且仍 compute-bound：还能用哪些 trick 降 batch size 又保持吞吐？（一段话，带引用/公式）

---

## 5. Optimizer State Sharding

### 5.1 Problem: `optimizer_state_sharding`（15 pts）

- [x] 实现 sharded optimizer wrapper（包装任意 `torch.optim.Optimizer`）
  - [x] 训练中各 rank 只持有自己那份参数及对应 optimizer state
  - [x] 每次 `step()` 后同步 updated parameters（让各 rank 模型参数一致/可继续训练）
  - [x] interface：
    - [x] `__init__(params, optimizer_cls: Type[Optimizer], **kwargs)`
      - [x] params（或 param groups）需要在 ranks 间分片
      - [x] 调用 `torch.optim.Optimizer` 的 super-class constructor
    - [x] `step(closure, **kwargs)`：调用底层 optimizer.step()，之后跨 rank 同步参数
    - [x] `add_param_group(param_group)`：支持训练中追加/重新分片参数
- [x] 对接 adapter：`adapters.get_sharded_optimizer`
- [x] 测试：`uv run pytest tests/test_sharded_optimizer.py`（建议跑 5 次）

### 5.2 Problem: `optimizer_state_sharding_accounting`（5 pts）

- [ ] 标准配置：单机 2 GPU + XL
- [x] (a) 写脚本 profile peak memory（有/无 optimizer state sharding）
  - [x] report：初始化后、optimizer step 前、optimizer step 后
  - [x] 分解：参数/optimizer states/其它（2~3 句）
- [x] (b) 比较训练速度：有/无 sharding 的 time/iter（2~3 句）
- [ ] (c) 和 ZeRO stage 1（ZeRO-DP Pos）差异？（尤其 memory 与通信量；2~3 句）
  - 脚本：`python -m cs336_systems.benchmarks.sharded_optimizer_accounting ...`

---

## 表 1：模型尺寸（用于多处 benchmark/profile）

| size | d_model | d_ff | num_layers | num_heads |
|---|---:|---:|---:|---:|
| small | 768 | 3072 | 12 | 12 |
| medium | 1024 | 4096 | 24 | 16 |
| large | 1280 | 5120 | 36 | 20 |
| xl | 1600 | 6400 | 48 | 25 |
| 2.7B | 2560 | 10240 | 32 | 32 |

默认：vocab size=10,000，batch size=4，context length 在题目要求范围内变化。

---

## 必跑测试清单（按题目点名的）

- Flash forward（PyTorch）：`uv run pytest -k test_flash_forward_pass_pytorch`
- Flash forward（Triton）：`uv run pytest -k test_flash_forward_pass_triton`
- Flash backward：`uv run pytest -k test_flash_backward`
- DDP（individual-parameter overlap）：`uv run pytest tests/test_ddp_individual_parameters.py`
- DDP（bucketed overlap）：`pytest tests/test_ddp.py`
- Sharded optimizer：`uv run pytest tests/test_sharded_optimizer.py`
