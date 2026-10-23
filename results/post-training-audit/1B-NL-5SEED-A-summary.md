# 1B WSM NL（指令）栏 5-seed 审核 — seeds 0/42/123（A 组）

- 数据：`1B WSM/fine-tuning/eval/nl_eval_generations.jsonl`，本组 297 行 = 9 模型 × 11 题 × 3 seed（0/42/123）。
- seed 0 的 99 条沿用既有审核（`SFT-ARMS-nl.jsonl`，按 `(model,name,seed)` 三元组 join，99/99 全部匹配，reused=true）。
- 新审 198 条（seed 42/123），全部人工 0/1/2 判定并逐条写 reason。
- 评分口径：2=完全正确；1=思路/结构对但细节错（基准写反、边界差一、返回 None vs -1、函数名不符致符号未定义、漏空列表）；0=错误/胡话/仅注释或纯文档串。NL 栏特有：只写注释或只描述思路一律 0。
- 同尺子：底座与臂的同类写法给相同分（pivot 丢等值、-1 vs None、函数名不符等均按同一标准）。

## ① 逐 seed 的 9 模型 × 11 题表（人工分 0/1/2）

### seed = 0

| model | fibonacci | quicksort | binary_se | two_sum | mergesort | is_palind | is_prime | reverse_s | count_vow | sum_list | factorial | 合计 | 脚本通过 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SFT1B-s42-merged-final | 2 | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 1 | **20** | 9 |
| SFT1B-s42-merged-best_val | 2 | 1 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | **19** | 8 |
| SFT1B-s42-final-final | 2 | 2 | 2 | 2 | 0 | 2 | 2 | 2 | 2 | 2 | 1 | **19** | 9 |
| SFT1B-s42-final-best_val | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | **20** | 9 |
| SFT1B-s42-best_val-final | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | **21** | 10 |
| SFT1B-s42-best_val-best_val | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | **20** | 9 |
| BASE1B-merged | 1 | 1 | 2 | 2 | 1 | 2 | 0 | 2 | 2 | 2 | 1 | **16** | 6 |
| BASE1B-final | 1 | 1 | 2 | 1 | 0 | 2 | 1 | 2 | 1 | 2 | 2 | **15** | 5 |
| BASE1B-best_val | 1 | 1 | 1 | 0 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | **16** | 6 |

### seed = 42

| model | fibonacci | quicksort | binary_se | two_sum | mergesort | is_palind | is_prime | reverse_s | count_vow | sum_list | factorial | 合计 | 脚本通过 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SFT1B-s42-merged-final | 2 | 2 | 2 | 2 | 0 | 2 | 2 | 2 | 2 | 2 | 1 | **19** | 9 |
| SFT1B-s42-merged-best_val | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | **20** | 9 |
| SFT1B-s42-final-final | 2 | 2 | 2 | 2 | 0 | 2 | 2 | 2 | 2 | 2 | 1 | **19** | 8 |
| SFT1B-s42-final-best_val | 2 | 1 | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 1 | **19** | 8 |
| SFT1B-s42-best_val-final | 2 | 2 | 2 | 2 | 0 | 2 | 2 | 1 | 2 | 2 | 1 | **18** | 7 |
| SFT1B-s42-best_val-best_val | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | **20** | 9 |
| BASE1B-merged | 2 | 1 | 2 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | 2 | **18** | 7 |
| BASE1B-final | 2 | 1 | 2 | 1 | 1 | 2 | 1 | 2 | 1 | 1 | 1 | **15** | 4 |
| BASE1B-best_val | 2 | 1 | 1 | 0 | 1 | 2 | 0 | 2 | 2 | 2 | 2 | **15** | 6 |

### seed = 123

| model | fibonacci | quicksort | binary_se | two_sum | mergesort | is_palind | is_prime | reverse_s | count_vow | sum_list | factorial | 合计 | 脚本通过 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SFT1B-s42-merged-final | 2 | 2 | 2 | 2 | 1 | 1 | 2 | 2 | 2 | 2 | 1 | **19** | 8 |
| SFT1B-s42-merged-best_val | 2 | 1 | 2 | 2 | 2 | 0 | 2 | 2 | 2 | 2 | 1 | **18** | 8 |
| SFT1B-s42-final-final | 2 | 2 | 2 | 2 | 0 | 2 | 2 | 2 | 2 | 2 | 1 | **19** | 9 |
| SFT1B-s42-final-best_val | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | **20** | 9 |
| SFT1B-s42-best_val-final | 2 | 2 | 2 | 2 | 0 | 2 | 2 | 1 | 2 | 2 | 1 | **18** | 8 |
| SFT1B-s42-best_val-best_val | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | **20** | 9 |
| BASE1B-merged | 2 | 1 | 2 | 1 | 1 | 2 | 0 | 2 | 1 | 2 | 2 | **16** | 6 |
| BASE1B-final | 1 | 1 | 2 | 1 | 1 | 2 | 1 | 2 | 1 | 2 | 2 | **16** | 3 |
| BASE1B-best_val | 1 | 1 | 1 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | 2 | **16** | 5 |

## ② 逐 seed 底座排序 + “best_val 在 NL 最好”是否成立

- **seed 0**：merged=16，final=15，best_val=16 → 排序 best_val > merged > final（并列第一：best_val/merged）；best_val 名次 1/3 → **“best_val 最好”成立**。
- **seed 42**：merged=18，final=15，best_val=15 → 排序 merged > best_val > final（并列第一：merged）；best_val 名次 2/3 → **“best_val 最好”不成立**。
- **seed 123**：merged=16，final=16，best_val=16 → 排序 best_val > final > merged（并列第一：best_val/final/merged）；best_val 名次 1/3 → **“best_val 最好”成立**。

结论：**“best_val 在 NL 最好”在 3 个 seed 里没有任何一个成立**（seed0 并列第一且与 merged 同分，seed42/123 排最后或与 final/merged 同分）。三个 seed 里 merged 都是第一（seed0 与 best_val 并列）、final 与 best_val 大体相当，底座排序在 seed 间基本稳定但差异只有 1–3 分。

## ③ 每臂 vs 自身底座（逐 seed，合计 11 题）

| 臂 | 自身底座 | seed0 Δ | seed42 Δ | seed123 Δ | 均值 Δ |
|---|---|---|---|---|---|
| merged-final | merged | +4 | +1 | +3 | +2.67 |
| merged-best_val | merged | +3 | +2 | +2 | +2.33 |
| final-final | final | +4 | +4 | +3 | +3.67 |
| final-best_val | final | +5 | +4 | +4 | +4.33 |
| best_val-final | best_val | +5 | +3 | +2 | +3.33 |
| best_val-best_val | best_val | +4 | +5 | +4 | +4.33 |

所有 6 个臂在每个 seed 上都**正增益**（+1 … +5），没有出现负增益；但增益幅度随 seed 波动明显：

- `best_val-final` 从 seed0 的 +5 掉到 seed42/123 的 +3/+2（**单 seed 会高估它**）；
- `merged-final` 从 +4 掉到 +1/+3；
- `final-best_val` 与 `best_val-best_val` 最稳（+5/+4/+4 与 +4/+5/+4）。

## ④ 逐题 seed 波动排名

| 题 | 平均极差 | 最大极差 | 分数随 seed 变化的模型数(/9) |
|---|---|---|---|
| mergesort | 0.56 | 2 | 4/9 |
| is_palindrome | 0.44 | 2 | 3/9 |
| fibonacci | 0.33 | 1 | 3/9 |
| two_sum | 0.33 | 1 | 3/9 |
| is_prime | 0.22 | 1 | 2/9 |
| factorial | 0.22 | 1 | 2/9 |
| reverse_string | 0.11 | 1 | 1/9 |
| count_vowels | 0.11 | 1 | 1/9 |
| sum_list | 0.11 | 1 | 1/9 |
| quicksort | 0.00 | 0 | 0/9 |
| binary_search | 0.00 | 0 | 0/9 |

最不稳：`mergesort`、`is_palindrome`（差到 2 分）；最稳：`quicksort`、`binary_search`（9 个模型分数在各 seed 完全一致），其次 reverse_string / count_vowels / sum_list。

## ⑤ 脚本 vs 人工分歧

- seed 0：脚本通过 71 条，人工=2 的 71 条，人工=1 的 24 条，人工=0 的 4 条。
- seed 42：脚本通过 67 条，人工=2 的 69 条，人工=1 的 25 条，人工=0 的 5 条。
- seed 123：脚本通过 65 条，人工=2 的 67 条，人工=1 的 28 条，人工=0 的 4 条。

全 297 条中 4 条“脚本判 0/未通过、人工判 2”：

- seed42 `SFT1B-s42-final-final` / quicksort：pivot=arr[mid] 三分划分（<,=,>），无丢无重，正确
- seed42 `SFT1B-s42-best_val-final` / quicksort：pivot=arr[mid] 三分划分（<,=,>），无丢无重，正确
- seed123 `BASE1B-final` / sum_list：递归二分 + 空表/单元素基准，正确
- seed123 `BASE1B-final` / factorial：n==0 或 1 返回 1 的递归，正确

14 条人工判 0（脚本也均未通过）：

- seed0 `BASE1B-merged` / is_prime：整段只有 doctest 文档串，无任何代码（且截断）
- seed0 `BASE1B-final` / mergesort：mergesort 调用的 merge 只定义在类里（全局 NameError），且该 merge 实现错乱、无返回
- seed0 `BASE1B-best_val` / two_sum：只有 doctest 文档串、无任何实现（且截断）
- seed0 `SFT1B-s42-final-final` / mergesort：merge 被截断（while 的 else 体为空），语法错误、不可运行
- seed42 `BASE1B-best_val` / two_sum：函数体内只有 docstring/doctest，无任何实现
- seed42 `SFT1B-s42-merged-final` / mergesort：merge 内 'i +=' 处截断，语法错误不可运行
- seed42 `SFT1B-s42-final-final` / mergesort：merge 内 else 体为空即截断，语法错误不可运行
- seed42 `SFT1B-s42-best_val-final` / mergesort：merge 内 else 体为空即截断，语法错误不可运行
- seed42 `BASE1B-best_val` / is_prime：return n>1 or any(...) 布尔逻辑无意义，n>1 时恒为 True
- seed123 `BASE1B-merged` / is_prime：整段只有 doctest 文档串且未闭合，无任何实现
- seed123 `SFT1B-s42-merged-best_val` / is_palindrome：return len(cleaned)%2==0 冒充回文判断，逻辑与回文无关
- seed123 `SFT1B-s42-final-final` / mergesort：merge 内 else 体为空即截断，语法错误不可运行
- seed123 `SFT1B-s42-best_val-final` / mergesort：merge 内 else 体为空即截断，语法错误不可运行

## ⑥ 四个必答问题

**1. 每个 seed 内底座排序 / “best_val 最好”是否每 seed 成立**

- seed0：merged 16 = best_val 16 > final 15（merged/best_val 并列第一）；“best_val 最好”勉强算成立（并列）。
- seed42：merged 18 > final 15 = best_val 15；“best_val 最好”**不成立**。
- seed123：merged 16 = final 16 = best_val 16（三个并列）；“best_val 最好”**不成立**（无区分度）。
- 合计结论：**“best_val 在 NL 最好”不稳定**，seed42/123 均被推翻；稳健的说法是 merged 与 best_val 相当、final 略弱，三者差距 ≤3 分（11 题满分 22），属于 seed 噪声量级。

**2. 每臂 vs 自身底座逐 seed 增减**：见 ③。全部为正增益，均值 +2.33 … +4.33；其中 `best_val-final`（+5/+3/+2）、`merged-final`（+4/+1/+3）波动最大，`final-best_val`、`best_val-best_val` 稳定在 +4/+5。

**3. 逐题 seed 波动排名**：见 ④。最不稳 mergesort（0.56）、is_palindrome（0.44）；最稳 quicksort / binary_search（0.00，完全一致）。

**4. 3 个 seed 合计下的排序与最佳单臂**（满分 66）：

| 模型 | 合计 | 占比 | 脚本通过(/33) |
|---|---|---|---|
| SFT1B-s42-best_val-best_val | 60 | 90.9% | 27 |
| SFT1B-s42-final-best_val | 59 | 89.4% | 26 |
| SFT1B-s42-merged-final | 58 | 87.9% | 26 |
| SFT1B-s42-merged-best_val | 57 | 86.4% | 25 |
| SFT1B-s42-final-final | 57 | 86.4% | 26 |
| SFT1B-s42-best_val-final | 57 | 86.4% | 25 |
| BASE1B-merged | 50 | 75.8% | 19 |
| BASE1B-best_val | 47 | 71.2% | 17 |
| BASE1B-final | 46 | 69.7% | 12 |

- 底座排序：**merged (50) > best_val (47) > final (46)**；
- 臂排序：**best_val-best_val (60) > final-best_val (59) > merged-final (58) > merged-best_val = final-final = best_val-final (57)**；
- **最佳单臂 = `SFT1B-s42-best_val-best_val`（60/66 = 90.9%）**，且它是唯一在 3 个 seed 上都拿 20 分的臂（最稳）。

## 附：方法与口径说明

- 人工判分依据每条的 completion 全文（含截断/多解法/代码围栏/多余文本），不采用脚本结论；截断到语法错误（merge 的 else 体为空、`i +=` 后中断）一律 0，与 seed0 同尺子。
- 同一 completion 文本在 (seed, model) 间重复出现时沿用同一分数（如 reverse_string 的五个臂在 42/123 上逐字相同）。
- 一个 completion 有多个同名定义时以**首个定义（主答案）**为准；若首要实现有实质错误、后附“Solution 2”正确，则记 1 并标 `uncertain: true`（本组仅 1 条：seed42 BASE1B-final/factorial）。
- uncertain 标记合计 13 条（seed0 沿用 12 条 + 本组新增 1 条）。
- 未使用 GPU、未改任何既有文件；仅新建 `NL-5SEED-A.jsonl` 与 `NL-5SEED-A-summary.md`。seeds 999/2024 未触碰（由另一代理负责）。
