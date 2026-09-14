# Code LLM Project — 踩坑记录 & 经验教训

> 整理自聊天记录、训练日志、checkpoint 历史和记忆片段。
> 用于补充 TECHNICAL.md 和 presentation 的"challenges"部分。

---

## 阶段 1: 环境选择 — JAX → WSL2 → PyTorch Windows

### 1.1 JAX on Windows 不支持 GPU

- **时间**: Day 1-2
- **问题**: 最初计划用 JAX 做训练（熟悉 Flax/Haiku 生态），但 JAX 在 Windows 上**不支持 GPU 加速**——JAX 的 CUDA 后端只支持 Linux。Windows 上只能用 CPU，训练速度不可接受。
- **教训**: 框架选择必须先确认平台支持。JAX 官方明确 "Windows support is experimental, CPU-only"。

### 1.2 WSL2: 网络 + GPU 碎片化双重打击

- **时间**: Day 2
- **问题**:
  - WSL2 内部网络不通，无法直接访问 HuggingFace 下载数据
  - GPU 通过 WSL2 直通后，JAX 出现 CUDA BFC (Best-Fit with Coalescing) 内存碎片化 bug → OOM 即使显存明明够用
  - WSL2 文件系统性能差（跨 OS 文件访问慢）
- **决策**: 放弃 JAX/WSL2 路线，回到 Windows 原生 PyTorch
- **教训**: WSL2 做 ML 训练不适合生产级任务。GPU 直通虽然能用，但内存管理、网络配置都是额外的坑。

### 1.3 PyTorch 版本地狱: sm_120 + Python 3.14

- **时间**: Day 3 凌晨
- **问题**:
  - RTX 5090 是 Blackwell 架构 (sm_120)，**PyTorch 2.6 stable 不认识这个架构**，无法编译 CUDA kernel
  - 需要 PyTorch nightly cu128 版本才有 sm_120 支持
  - Python 3.14 不兼容 PyTorch nightly → 必须降级到 3.12
- **最终环境**: Python 3.12 + PyTorch 2.12 nightly + CUDA 12.8
- **教训**: 最新硬件（尤其是新架构 GPU）+ 最新软件 = 兼容性地狱。给未来项目的建议：GPU 发布后等 2-3 个月等 PyTorch stable 跟上。

---

## 阶段 2: 实验阶段 — 模型规模 & 超参数搜索

### 2.1 模型规模: 9B 参考 → 353M 自训练

- **背景**: 最初参考了 Qwen 3.5 9B 级别的开源代码模型（32层, d_model=4096, 32头, d_ff=12288, 词表248K），参数量是我们的 **25×**
- **现实**: 9B 模型单卡 RTX 5090 (32GB) 根本跑不动（BF16 训练需 70GB+，必须多卡）。即使量化后推理也勉强。
- **决策链**:
  - 9B → 不可行（单卡放不下，训练需多卡）
  - 400M → 考察发现 VRAM 有富裕（200M 只用了 24GB）
  - 200M (v1) → 验证可行性，76K tok/s
  - 353M (v2) → 最大化单卡利用率，46K tok/s，VRAM 29GB/32GB
- **核心对比**:

| 参数 | 353M (Ours) | Qwen 3.5 9B | 倍数 |
|------|------------|-------------|------|
| 总参数 | 353M | 9B | 25× |
| 层数 | 18 | 32 | 1.8× |
| d_model | 1,024 | 4,096 | 4× |
| 注意力头 | 16 | 32 | 2× |
| d_ff (SwiGLU) | 3,840 | 12,288 | 3.2× |
| 词表 | 32K | ~248K | 7.8× |
| 训练VRAM | 29 GB | ~70+ GB (多卡) | — |

- **关键指标**: 参数量 1.76×（v1→v2），速度只降到 0.60×，**优于线性缩放**
- **核心主张**: 25× 参数差距，但我们的 353M 模型已经在 7.78B tokens 后涌现了递归推理能力——小模型在消费级硬件上也能捕捉非平凡的代码语义
- **教训**: 不要直接对标大模型规模。先用最小可用模型验证 pipeline，再逐步 push 硬件极限。

### 2.2 batch_size / grad_accum 组合搜索

- **问题**: batch_size 太小→训练不稳定（梯度噪声大），太大→OOM
- **搜索过程**:
  - 先固定 grad_accum=1，测 batch_size 上限：14 是 BF16 下不 OOM 的最大值
  - grad_accum=4 把有效 batch 扩大到 56，跟 Chinchilla 建议的 ~50-100 范围一致
  - 最终: `--batch_size 14 --grad_accum 4`（有效 batch=56，每步 57,344 tokens）
- **教训**: 先找到 memory limit，再通过 grad_accum 达到目标有效 batch。不要反过来（先定 batch 再调 grad_accum 容易反复 OOM）。

### 2.3 seq_len 权衡: 256 vs 512 vs 1024

- **考察**: 更长的 seq_len → 更多上下文，但 VRAM 增长是 O(seq_len²)（attention 部分）
- **选择 1024 的理由**:
  - 1024 tokens ≈ 200-300 行 Python，覆盖绝大多数函数定义和文件片段
  - 512 经常截断函数体，256 只能看函数签名
  - VRAM 开销在可接受范围（约 29GB total）
  - 46K tok/s 吞吐刚好匹配 tokenize 速度，无 CPU 瓶颈
- **教训**: seq_len 的选择 = 任务需求（覆盖率）∩ 硬件约束（VRAM）∩ 速度约束（tokenize 匹配）

---

## 阶段 3: 训练阶段 — 数据 & 学习率踩坑

### 3.1 训练数据被反复使用（最隐蔽的坑）

- **来源**: 聊天记录 2026-07-17 02:54 (`session_20260717_025422_1a34af`)
- **发现过程**: 用户主动提问："checkpoint恢复后，会用新的没用过的数据训练吗？是怎么记录的？"
- **根因**: `train_pt.py` 慢路径不做数据级别的 resume。每次 resume 都从 `parquet_files[0]` 重新开始读取。
- **实际影响**（助手诊断）:
  - "文件 1-300 可能被训了 3 次，文件 301-400 训了 1-2 次，文件 401-800 0 次"
  - 最长一次 session 跑了 1.6B tokens → 约占总量 16%。1126 个文件中，前 180 个被反复训过，后 946 个从未碰过。
- **用户解决方案**: 根据训练日志算出最长 session 22239 秒 × 49K tok/s ≈ 1.09B tokens → 约 87 个文件被消耗，加安全 margin → `--start-file-index 90`
- **关键对话**:
  > 用户: "所以一直是重复数据训练？那800个文件有什么意义？"
  > 助手: "每次 resume 都在重新咀嚼前 43%"
  > 用户: "不要擅自主张，按我说的改。我一共只下载了800个文件"
- **教训**:
  - 数据管理比模型架构更重要。训练前一次性准备好完整数据集
  - 分批次下载数据时，必须追踪"哪些文件已经被模型见过"
  - Resume 逻辑必须记录数据消费位置，不能只记录模型状态

### 3.2 强行修改学习率，忽略优化器动量（最严重的质量波动原因）

- **来源**: 聊天记录 2026-07-17 (`session_20260717_025422_1a34af`)
- **用户报告退化** (line 4102):
  > "这次Fibonacci更不靠谱了。也可能是因为现在不是换成新数据了吗，而且我把学习率上调到2.8e-04了"
- **根因分析**:
  1. CosineAnnealingLR 在 resume 时，`base_lrs` 是从 **optimizer 的 state_dict** 读取的（`[group['lr'] for group in optimizer.param_groups]`），**不是**从 `args.lr` 读取
  2. 代码验证: `CosineAnnealingLR.__init__` 中 `self.base_lrs = [group['lr'] for group in optimizer.param_groups]`
  3. Resume 时 optimizer 从 checkpoint 恢复 → `param_groups[0]['lr'] = 1.62e-04`（已衰减值）；新建 scheduler → `base_lrs = [1.62e-04]`——**不是** 2.4e-04
  4. 用户强行改 LR 到 2.8e-04（+73%）但不 reset optimizer：
     - 实际更新 = 新 LR × 旧动量方向
     - 历史梯度的影响被放大 73%
     - 参数更新方向偏离正确梯度方向
  5. 直接后果：loss 从 1.16 跳到 1.58（step 80K→100K），生成质量退化（Fibonacci 输出"更不靠谱"）
- **修复**: 实现 `--lr-override` 参数（用户要求 2.2e-04），配合 `--reset-optimizer` 使用
- **关键 insight**: 改 LR 不 reset optimizer = 往正在行驶的车上突然换挡。动量、二阶矩估计都是上下文相关的。
- **教训**: 优化器状态 ≠ 可独立修改的参数。Fine-tuning 阶段天然需要新 LR → 强制 reset optimizer

### 3.3 生成质量波动与 loss 不完全相关

- **观察**:
  - 训练 loss 持续下降（1.5 → 1.3 → 1.2），但生成质量不是单调提升
  - 某个 checkpoint 的 loss 更低，但生成的代码反而更差（更多重复、更少多样性）
  - 这可能跟 checkpoint 保存时恰好处于"过拟合某个 batch pattern"的状态有关
- **应对**:
  - 加了 repetition_penalty=1.1, min_p=0.0, temperature=0.2 来抑制重复
  - 但这些是**推理时的修正**，不解决训练问题
  - 根本原因是模型容量 (353M) 相对于代码语义的复杂度还不够
- **教训**: Loss 是训练指标，不是质量指标。对生成任务，**必须定期人工抽样评估**，不能只看 loss 曲线。

---

## 阶段 4: Fine-tuning Phase 1 — 数据污染灾难（实际发生）

### 4.1 目标与计划

- 从 800 个 parquet 中提取短 Python 函数（50-1000 chars，需有 def，3+ 有效行）
- 过滤噪音: copyright、test、setup、shebang
- 结果: 90,000 训练 + 10,000 验证条，预分词后 22.8M tokens
- 训练: LR=8e-5, t_max=500, 训练到 val_loss=1.3244（step 300）→ 看起来不错

### 4.2 灾难性结果

- **20 次生成（4 种算法 × 5 seeds），0 次正确**
- 所有输出都是一个固定模式:
  ```
  def some_func(x):          ← prompt
  # -*- coding: utf-8 -*-    ← 模型立即跳到 Django
  from django.db import migrations
  def forwards(apps, schema_editor):
      ...
  class Migration(migrations.Migration):
      ...
  import _plotly_utils.basevalidators   ← 再接 Plotly
  class TicktextsrcValidator(...):
  ```

- binary_search → Django ×5
- two_sum → Django + Plotly ×5
- is_palindrome → Django + Plotly ×5
- inorder_traversal → 全部错误（base 模型 seed 456 反而写对了！）

### 4.3 根因: 提取过滤器严重不足

- 旧过滤器只拦截了: copyright、test、setup、shebang
- **遗漏的大类**:
  - Django migrations（`migrations.RunPython`, `apps.get_model`, `class Migration`）
  - Django models（`models.CharField`, `OneToOneField`, `class Meta`, `verbose_name`）
  - Plotly validators（`_plotly_utils.basevalidators`, `plotly_name=`, `edit_type=`）
  - Python 2 遗留（`# -*- coding:`, `from __future__ import`）
  - 纯 import 文件（>60% 行是 import）
  - 纯类定义文件（>70% class/decorator）
  - 连续重复行（模板/配置文件特征）

- 这些代码短（<1000 chars）、有 def、没有 test/setup 标记 → 全部通过旧过滤器
- 22.8M tokens 中可能一半以上是这类"看起来像函数"的垃圾模板
- 模型学会了最强模式: "看到 def 签名 → 输出 Django 迁移"

### 4.4 修复: 增强过滤器

新增 30+ 条精确污染模式 + 5 项质量指标:
- **精确匹配**: Django migrations, Django ORM, Django admin, Plotly validators, Python 2 遗留, config/boilerplate
- **文件名过滤**: migrations/, tests/, setup.py, conftest.py, __init__.py
- **质量指标**: import 比例 >60% 丢弃, class/decorator 比例 >70% 丢弃, 连续重复行检测, def 行占比, 逻辑行存在性检查
- **NAS 全长重提取**: 从 800 个 parquet 重新提取干净数据

### 4.5 教训

1. **数据质量 >> 数据量**: 22.8M 干净数据 > 100M 污染数据
2. **Loss 是危险的误导指标**: val_loss=1.32 看起来很好，但模型实际在学"输出 Django 模板"
3. **清洗规则必须基于实际数据抽样**: 不能拍脑袋列几条规则就完事
4. **微调前必须验证 base 模型**: 对比 base vs FT 的生成质量，才能发现退化

---

## 按 presentation 用途的速查表

| 挑战类别 | 具体问题 | 一句话概括 | 对应 Slide |
|---------|---------|-----------|-----------|
| 环境适配 | JAX/WSL2/PyTorch | 最新硬件+最新软件=兼容性地狱 | Slide 7 |
| 规模决策 | 9B→353M | 不追大模型，最大化单卡利用率 | Slide 2/4 |
| 超参数 | batch/grad_accum/seq_len | 三个参数互相制约，需联合搜索 | Slide 4 |
| 数据管理 | Resume 不追踪文件位置→重复训练前300个文件 | 数据管道比模型架构更影响结果 | Slide 3/7 |
| 优化器 | 改LR 1.62→2.8e-4 不reset→loss 1.16→1.58 | 优化器状态不是独立变量 | Slide 7 |
|| 评估 | loss下降≠质量提升（6.3B tokens 仍语法对语义错） | 生成任务必须人工抽样 | Slide 6 |
|| 数据污染 | Django/Plotly 模板污染训练数据 → FT 后模型退化 | 数据清洗不足 = 变相毒化 | Slide 8 |
|| Fine-tuning | 小数据微调，质量不如 base 模型 | 22.8M 垃圾数据 < 0 微调 | Slide 8 |

---

## 给报告的建议叙事线

按照因果链组织：

```
我想训练一个代码 LLM
  → 选什么框架？JAX（不支持Windows）→ WSL2（网络+GPU bug）→ PyTorch native ✓
  → 多大规模？9B（单卡放不下）→ 200M（验证可行）→ 353M（推满硬件）✓
  → 什么超参数？batch=14, grad_accum=4, seq_len=1024（VRAM极限+速度平衡）✓
  → 数据怎么管？分批次下载→文件被反复使用（过拟合）→ start-file-index 显式跳过 ✓
  → 学习率怎么调？强行改LR→没reset optimizer→动量错乱→loss波动→生成退化 ✓
  → 怎么评估？loss≠质量，2.6B tokens 输出 (2*n*n)，7.7B tokens 输出 F(n-1)+F(n-2) ✓
  → Fine-tuning：清洗不足→Django/Plotly污染→模型退化→Base模型反超FT ✓
  → 结论：数据质量>>数据量，loss≠能力，清洗规则需基于数据抽样
```

## 关键转折点: Fibonacci 突破

- **2.59B tokens**: `return (2*n*n)` — 纯统计噪声
- **7.74B tokens**: `return fibonacci(n-1) + fibonacci(n-2)` — 正确递归结构
- **Multi-seed 验证 (14 seeds)**: 46K→50%, 70K→14% (数据切换毁掉), 135K→79%, 141.6K→86%可用
- Loss 只从 ~1.3 降到 ~1.2，但能力发生了质的飞跃
- **loss ≠ 能力**: best loss 0.91 ≠ best generation
- 验证 Chinchilla 假说在小模型上依然成立

## 新增发现

- **Seed 不稳定性**: 同一 checkpoint 不同 seed 可差 30%+ 正确率 — 单次评估不可靠
- **FT eval bug**: `evaluate_loss` 未处理 dict 格式数据，已修复
- **Benchmark 扩展**: 新增 MBPP sanitized + CodeContests，正在下载
- **Epoch 支持**: `train_ft.py` 新增 `--epochs` 参数，支持小数据集多轮训练

---

*最后更新: 2026-07-17*
