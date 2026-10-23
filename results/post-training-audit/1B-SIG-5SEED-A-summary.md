# SIG-5SEED-A — 1B WSM 签名栏（续写代码）seeds 0/42/123 人工审核

审核人：子代理 A（负责 seeds 0 / 42 / 123）。另一个代理负责 seeds 999 / 2024，本文件不含那两组。
数据源：`1B WSM/fine-tuning/eval/eval_sft_generations.jsonl`（495 行，本组取 297 行 = 9 模型 × 11 题 × 3 seed）。
产物：`SIG-5SEED-A.jsonl`（297 行，字段 `i,model,name,seed,script_passed,score,reason,uncertain,reused`）。

## 0. 口径与审计前提（务必先读）

- 地面真值 = 人工 0/1/2。`script_passed` 只记录、不定分。
- `reused: true` 共 **66** 行：seed 0 的 6 个臂 × 11 题，直接沿用 `SFT-ARMS-sig.jsonl`（按 `(model,name,seed)` 三元组 join，已抽检 6 条：reason 与 seed-0 completion 逐条吻合，script_passed 与 generations 文件一致）。
- **修正一处任务前提**：`SFT-ARMS-sig.jsonl` 只有 **66** 行（不是 99 行），它只覆盖 seed 0 的 6 个臂；seed 0 的 **3 个底座（33 行）此前无人审**。因此本组新审 = 33 + 198 = **231 行**，加沿用 66 行 = 297 行。
- **接口口径（关键，已统一用于底座与臂）**：
  - `binary_search` 规范契约取 LeetCode 口径：未找到 `return -1`。臂清一色 `return None` → 一律 **1**（与 seed-0 既有审核一致）。
  - `fibonacci(n)` 的裸签名以底座家族产出的契约为准：返回**第 n 个斐波那契数（int）**。臂在 seed 42/123 改成"返回前 n 项列表"（11 行）→ 按接口漂移记 **1** 并标 `uncertain: true`（裸 prompt 也容许列表读法；只有 `SFT1B-s42-final-best_val` seed 123 给的是规范 int → 2）。
  - **敏感性**：若这 11 行改记 2，仅使各臂均值 +0.09~0.18，底座不变；本报告全部排序与增/减结论**不受影响**（merged 臂的负增益来自 count_vowels / factorial / quicksort / two_sum，与 fibonacci 无关）。
- 缩进错乱（completion 以 1 空格起、与 prompt 的 4 空格缩进冲突，导致 30/45 行不可解析）在底座与臂上同等出现，属模型风格问题而非实现质量，评分时忽略缩进本身，只看函数体逻辑；相关行 `reason` 中另有说明。

## 1. 各 seed 的 9 模型 × 11 题评分表

列顺序：fibonacci, quicksort, binary_search, two_sum, mergesort, is_palindrome, is_prime, reverse_string, count_vowels, sum_list, factorial

### seed 0
| model | fib | qs | bs | ts | ms | pal | prime | rev | cv | sum | fact | 合计/22 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE1B-merged | 2 | 2 | 2 | 1 | 2 | 1 | 1 | 2 | 1 | 0 | 1 | 15 |
| BASE1B-final | 1 | 1 | 1 | 0 | 0 | 2 | 1 | 0 | 1 | 0 | 0 | 7 |
| BASE1B-best_val | 1 | 1 | 1 | 1 | 1 | 2 | 1 | 0 | 1 | 0 | 0 | 9 |
| SFT1B-s42-merged-best_val | 2 | 2 | 1 | 1 | 2 | 2 | 0 | 2 | 2 | 0 | 0 | 14 |
| SFT1B-s42-merged-final | 2 | 1 | 1 | 1 | 2 | 2 | 1 | 2 | 2 | 0 | 0 | 14 |
| SFT1B-s42-final-best_val | 2 | 1 | 1 | 1 | 2 | 2 | 1 | 0 | 2 | 0 | 0 | 12 |
| SFT1B-s42-final-final | 2 | 2 | 1 | 1 | 2 | 2 | 1 | 0 | 2 | 0 | 0 | 13 |
| SFT1B-s42-best_val-best_val | 1 | 1 | 1 | 1 | 2 | 0 | 1 | 0 | 2 | 0 | 0 | 9 |
| SFT1B-s42-best_val-final | 2 | 1 | 1 | 1 | 2 | 0 | 2 | 0 | 2 | 0 | 0 | 11 |

### seed 42
| model | fib | qs | bs | ts | ms | pal | prime | rev | cv | sum | fact | 合计/22 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE1B-merged | 1 | 2 | 1 | 0 | 2 | 2 | 1 | 2 | 2 | 0 | 2 | 15 |
| BASE1B-final | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | 2 | 9 |
| BASE1B-best_val | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 2 | 7 |
| SFT1B-s42-merged-best_val | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 2 | 0 | 0 | 0 | 12 |
| SFT1B-s42-merged-final | 1 | 1 | 1 | 1 | 2 | 2 | 1 | 2 | 0 | 0 | 0 | 11 |
| SFT1B-s42-final-best_val | 1 | 1 | 1 | 1 | 2 | 0 | 1 | 0 | 2 | 0 | 0 | 9 |
| SFT1B-s42-final-final | 1 | 2 | 1 | 0 | 2 | 2 | 1 | 0 | 2 | 0 | 0 | 11 |
| SFT1B-s42-best_val-best_val | 1 | 1 | 1 | 1 | 2 | 2 | 1 | 0 | 2 | 0 | 0 | 11 |
| SFT1B-s42-best_val-final | 1 | 2 | 1 | 1 | 2 | 2 | 1 | 0 | 2 | 0 | 0 | 12 |

### seed 123
| model | fib | qs | bs | ts | ms | pal | prime | rev | cv | sum | fact | 合计/22 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE1B-merged | 2 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| BASE1B-final | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 8 |
| BASE1B-best_val | 1 | 1 | 0 | 0 | 1 | 2 | 1 | 2 | 1 | 0 | 0 | 9 |
| SFT1B-s42-merged-best_val | 1 | 1 | 1 | 1 | 2 | 2 | 0 | 2 | 2 | 0 | 2 | 14 |
| SFT1B-s42-merged-final | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 2 | 2 | 0 | 2 | 16 |
| SFT1B-s42-final-best_val | 2 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 2 | 0 | 0 | 8 |
| SFT1B-s42-final-final | 1 | 2 | 1 | 0 | 2 | 2 | 0 | 0 | 2 | 0 | 2 | 12 |
| SFT1B-s42-best_val-best_val | 1 | 1 | 1 | 1 | 2 | 2 | 0 | 0 | 2 | 0 | 0 | 10 |
| SFT1B-s42-best_val-final | 1 | 2 | 1 | 1 | 2 | 2 | 0 | 0 | 2 | 0 | 0 | 11 |

## 2. 逐 seed 底座排序：merged > final > best_val 是否稳定

| seed | merged | final | best_val | 实际排序 | merged>final>best_val |
|---|---|---|---|---|---|
| 0 | 1.36 | 0.64 | 0.82 | merged > **best_val > final** | 不成立 |
| 42 | 1.36 | 0.82 | 0.64 | merged > final > best_val | 成立 |
| 123 | 0.55 | 0.73 | 0.82 | **best_val > final > merged** | 不成立（完全倒置） |

- **只有 seed 42 成立。3 个 seed 里 2 个破坏排序，且破坏方式不同。**
- seed 0 破坏点：`BASE1B-final` 掉到末位（7/22），被 `best_val`（9/22）反超。掉分题：mergesort 0（merge 自递归无进展）、two_sum 0（实现成"查 target 是否在数组里"）、reverse_string 0、factorial 0（全部只剩注释）。
- seed 123 破坏点：`BASE1B-merged` 跌到末位（6/22），被 final、best_val 双双反超。掉分题：is_palindrome 0（长度+正则逐字符检查冒充回文判定）、is_prime 0（`(n-1)%7 or (n-1)%11` 当作素性判据，属胡话）、reverse_string 0、count_vowels 0、factorial 0（TODO+`return 1` 占位）——5 个 0。
- 结论：**节选底座不可靠且与 seed 强相关**；merged 在 seed 0/42 是最强底座（并列 15/22），在 seed 123 却最弱。

## 3. 每臂 vs 自身底座：逐 seed 增益

臂命名 `SFT1B-s42-<底座>-<变体>`，与其同名前缀底座 `BASE1B-<底座>` 逐题比较（同一 seed 的生成结果）。

| 臂 | seed 0 | seed 42 | seed 123 | 3-seed 合计 |
|---|---|---|---|---|
| merged-best_val | -1 | -3 | +8 | +4 |
| merged-final | -1 | -4 | +10 | +5 |
| final-best_val | +5 | 0 | 0 | +5 |
| final-final | +6 | +2 | +4 | +12 |
| best_val-best_val | 0 | +4 | +1 | +5 |
| best_val-final | +2 | +5 | +2 | +9 |

- 6 个臂的 **3-seed 合计全部为正**；但**逐 seed 不稳定**：merged 系两臂在 seed 0 与 seed 42 都是**负增益**（-1/-1、-3/-4），只在 seed 123 大赚。
- merged 系负增益的成因：底座在这些 seed 本身很强（seed 0 拿 15/22、seed 42 拿 15/22），而臂在 **count_vowels**（seed 42 把"数元音"写成 `char.isalpha() and char not in vowels`，数成非元音 → 0）与 **factorial**（seed 0/42 退化成只剩示例注释 → 0）上崩塌；seed 0 另有 binary_search、is_prime。
- final 系与 best_val 系增益为正但同样有 seed 翻转：final-best_val 在 seed 42/123 增益恰为 0（无净变化），best_val-best_val 在 seed 0 为 0。
- **"SFT 对签名栏是净增益"在 3 个 seed 上不成立（不 stable）**：合计为正，但按 seed 看，6 个臂里有 2 个在 2/3 的 seed 上是负或零增益。

## 4. 逐题 seed 波动排名（同一模型同题跨 3 seed 的分差）

| 排名 | 题 | 平均极差 | 9 模型中出现 seed 翻转的个数 | 总极差 |
|---|---|---|---|---|
| 1 | factorial | 1.33 | 6 | 12 |
| 2 | is_palindrome | 1.22 | 6 | 11 |
| 3 | is_prime | 0.89 | 6 | 8 |
| 4 | count_vowels | 0.78 | 4 | 7 |
| 5 | fibonacci | 0.67 | 6 | 6 |
| 6 | two_sum | 0.56 | 5 | 5 |
| 7 | reverse_string | 0.56 | 3 | 5 |
| 8 | quicksort | 0.33 | 3 | 3 |
| 8 | mergesort | 0.33 | 3 | 3 |
| 10 | binary_search | 0.22 | 2 | 2 |
| 11 | sum_list | 0.00 | 0 | 0 |

- 最不稳：**factorial、is_palindrome、is_prime**（同一模型常从 2 掉到 0 或从 0 跳到 2，是"会写/不会写"的抽样波动，不是细节差一）。
- 最稳：**sum_list（9 模型 × 3 seed 全 0，无人写实现）**、binary_search（所有臂恒定 1，底座 0/1 抖动小）——稳定但**稳定地差**。

## 5. 脚本 vs 人工分歧（297 行）

| 类别 | 行数 |
|---|---|
| 脚本判过 且 人工=2 | 28 |
| **脚本判失败 但 人工=2（脚本漏判）** | **61** |
| **脚本判过 但 人工<2（脚本误判）** | **1** |
| 脚本判失败 且 人工<2 | 207 |

- 漏判比误判严重得多（61 : 1），复现了项目已知结论（脚本对 signature 续写有系统性低估）。典型：mergesort 的原地归并、is_palindrome 的净化+反转、count_vowels 的生成器计数——实现正确但脚本测不过。
- 唯一误判：`BASE1B-final / binary_search / seed 42`（递归切片版左右分支写反、返回切片内局部下标，脚本却判 True）→ 人工 1。该行正是脚本最不可信的形态。

## 6. 结论（回答 4 问）

**Q1 每个 seed 内底座排序？merged > final > best_val 是否每 seed 成立？**
- seed 42：merged(1.36) > final(0.82) > best_val(0.64)，**成立**。
- seed 0：merged(1.36) > best_val(0.82) > final(0.64)，**不成立**（final 被 best_val 反超，掉在 mergesort/two_sum/reverse_string/factorial）。
- seed 123：best_val(0.82) > final(0.73) > merged(0.55)，**不成立且完全倒置**（merged 在 is_palindrome/is_prime/reverse_string/count_vowels/factorial 五题拿 0）。
- 即：**3 个 seed 里只有 1 个成立**，底座强弱随 seed 大幅重排。

**Q2 每臂 vs 自身底座增减？SFT 净增益是否 3 seed 稳定？**
- 幅度：final-final 最大（3-seed 合计 +12），其次 best_val-final(+9)，其余在 +4~+5。
- 稳定性：合计全部为正，但**逐 seed 不稳定**。merged-best_val / merged-final 在 seed 0 与 seed 42 均为负（-1/-3、-1/-4），仅 seed 123 为正（+8/+10）；final-best_val 在 seed 42/123 恰为 0。→ **不能宣称"SFT 对签名栏稳定净增益"**；只能说"3 seed 合计为正，且增益主要由 seed 123 贡献"。

**Q3 逐题 seed 波动最大者？**
factorial(1.33) > is_palindrome(1.22) > is_prime(0.89) > count_vowels(0.78) > fibonacci(0.67) > two_sum = reverse_string(0.56) > quicksort = mergesort(0.33) > binary_search(0.22) > sum_list(0.00)。

**Q4 3-seed 合计排序与最佳单臂？**
- 模型均值（3 seed × 11 题 = 33 条）：`SFT1B-s42-merged-final` 1.242 > `SFT1B-s42-merged-best_val` 1.212 > **`BASE1B-merged` 1.091 = `SFT1B-s42-final-final` 1.091** > `SFT1B-s42-best_val-final` 1.030 > `SFT1B-s42-best_val-best_val` 0.909 > `SFT1B-s42-final-best_val` 0.879 > `BASE1B-best_val` 0.758 > `BASE1B-final` 0.727。
- **最佳单臂 = `SFT1B-s42-merged-final`**（1.242，比最强底座 BASE1B-merged 高 +0.15）。注意这是"合计最优"，它在 seed 0/42 相对自身底座是负的，靠 seed 123 的 +10 拉起来。

## 7. 评分分布与不确定项

- 分数分布（297 行）：0 分 91 行，1 分 117 行，2 分 89 行。
- `reused: true` 66 行；`uncertain: true` 17 行 = 我判的 11 行（fibonacci 列表契约，接口漂移存疑）+ 2 行（BASE1B-merged 的 factorial/is_palindrome seed 0 属 0/1 边界）+ 沿用文件自带 4 行。
- 覆盖情况：seeds 0/42/123 的全部 297 行均已给分，无遗漏、无编造；seeds 999/2024 未触碰。
