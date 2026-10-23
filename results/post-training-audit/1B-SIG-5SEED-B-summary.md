# 1B 三底座 × SFT 签名栏审核（seeds 999 / 2024）

- 数据源：`1B WSM/fine-tuning/eval/eval_sft_generations.jsonl`（495 行 = 9 模型 × 11 题 × 5 seed）
- 本次范围：**seeds 999、2024**，共 **198 条**（seeds 0/42/123 由另一代理负责，未触碰）
- 评分口径：2 = 完全正确；1 = 结构正确但细节错（丢重复元素、边界差一、返回 None vs -1、顺序反了等）；0 = 无实现/仅注释/仅示例调用/结果恒错
- 人工分与 `script_passed` 分开记录；`script_passed` 不参与定分
- 尺子一致性：未找到时返回 `-1`（底座口径）视为 2；返回 `None`（臂口径）视为 1 —— 底座与臂用同一把尺子
- 排序函数口径：原地排序且确实完成排序（mergesort 原地版）视为 2；原地版实际不写回/丢弃结果（quicksort 部分臂）视为 1
- GPU 未使用；未修改任何现有文件；产物均为新建

## 一、逐 seed 评分表（9 模型 × 11 题，满分 22）

### seed 999（9 模型合计 102 / 198）

| model | fibona | binary | two_su | quicks | merges | is_pal | is_pri | revers | count_ | sum_li | factor | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE1B-merged | 2 | 1 | 2 | 1 | 2 | 1 | 0 | 2 | 0 | 0 | 0 | 11 |
| BASE1B-final | 2 | 1 | 2 | 1 | 2 | 2 | 1 | 0 | 1 | 0 | 2 | 14 |
| BASE1B-best_val | 1 | 1 | 1 | 1 | 1 | 2 | 0 | 0 | 1 | 0 | 2 | 10 |
| SFT1B-s42-merged-final | 1 | 1 | 1 | 1 | 2 | 2 | 1 | 2 | 2 | 0 | 0 | 13 |
| SFT1B-s42-merged-best_val | 1 | 1 | 1 | 2 | 2 | 2 | 1 | 2 | 2 | 0 | 0 | 14 |
| SFT1B-s42-final-final | 1 | 1 | 1 | 2 | 2 | 2 | 0 | 0 | 2 | 0 | 0 | 11 |
| SFT1B-s42-final-best_val | 1 | 1 | 1 | 1 | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 8 |
| SFT1B-s42-best_val-final | 1 | 1 | 1 | 2 | 2 | 2 | 0 | 0 | 2 | 0 | 0 | 11 |
| SFT1B-s42-best_val-best_val | 1 | 1 | 1 | 1 | 2 | 2 | 0 | 0 | 2 | 0 | 0 | 10 |
| per-task total (9 models) | 11 | 9 | 11 | 12 | 17 | 15 | 3 | 6 | 14 | 0 | 4 | 102 |

### seed 2024（9 模型合计 90 / 198）

| model | fibona | binary | two_su | quicks | merges | is_pal | is_pri | revers | count_ | sum_li | factor | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE1B-merged | 2 | 1 | 0 | 2 | 2 | 2 | 1 | 2 | 1 | 0 | 1 | 14 |
| BASE1B-final | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 8 |
| BASE1B-best_val | 1 | 0 | 0 | 2 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 6 |
| SFT1B-s42-merged-final | 1 | 1 | 1 | 1 | 2 | 2 | 0 | 2 | 1 | 0 | 0 | 11 |
| SFT1B-s42-merged-best_val | 1 | 1 | 1 | 1 | 2 | 2 | 1 | 2 | 2 | 0 | 0 | 13 |
| SFT1B-s42-final-final | 1 | 1 | 1 | 1 | 2 | 2 | 0 | 0 | 2 | 0 | 0 | 10 |
| SFT1B-s42-final-best_val | 1 | 1 | 0 | 1 | 2 | 2 | 0 | 0 | 2 | 0 | 0 | 9 |
| SFT1B-s42-best_val-final | 1 | 1 | 1 | 1 | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 8 |
| SFT1B-s42-best_val-best_val | 2 | 1 | 1 | 1 | 2 | 2 | 0 | 0 | 2 | 0 | 0 | 11 |
| per-task total (9 models) | 11 | 7 | 6 | 11 | 16 | 13 | 4 | 6 | 14 | 1 | 1 | 90 |

## 二、逐 seed 底座排序 / merged > final > best_val 是否成立

| seed | merged | final | best_val | 实际排序 | merged>final>best_val |
|---|---|---|---|---|---|
| 999 | 11 | 14 | 10 | **final > merged > best_val** | **不成立（破坏）** |
| 2024 | 14 | 8 | 6 | merged > final > best_val | 成立 |

seed 999 破坏的落点（同一题内 999 与 2024 的差异）：

- **BASE1B-merged@999 塌在“无代码”上**：is_prime、count_vowels、sum_list、factorial 四题输出的是被注释掉的代码或纯注释 → 4 个 0；is_palindrome 也只得 1。同模型 @2024 这四题变成 0/1/0/1，问题同样存在但不致命。
- **BASE1B-merged two_sum**：999 满分（哈希返回下标对），2024 只剩 docstring 无实现 → 2 → 0，单题 -2。
- **BASE1B-final factorial**：999 正确（n<2 基准）得 2，2024 整段被注释 → 2 → 0，单题 -2；同时 is_palindrome 2→1、mergesort 2→1、reverse_string 0 保持。
- **BASE1B-best_val is_palindrome**：999 是 `s == s[::-1]`（2），2024 变成相邻字符比较（0），单题 -2。
- 结论：三底座在**“是否真的写出代码”**这一维度上 seed 抖动极大（comment-only 输出在 seed 间随机出现），底座排序本身不稳定。

## 三、各臂 vs 自身底座增益（逐 seed）

| 臂 | 自身底座 | base@999 | arm@999 | Δ999 | base@2024 | arm@2024 | Δ2024 | 两 seed 均增益 |
|---|---|---|---|---|---|---|---|---|
| SFT1B-s42-merged-final | BASE1B-merged | 11 | 13 | +2 | 14 | 11 | -3 | flat/negative in 2024 |
| SFT1B-s42-merged-best_val | BASE1B-merged | 11 | 14 | +3 | 14 | 13 | -1 | flat/negative in 2024 |
| SFT1B-s42-final-final | BASE1B-final | 14 | 11 | -3 | 8 | 10 | +2 | flat/negative in 999 |
| SFT1B-s42-final-best_val | BASE1B-final | 14 | 8 | -6 | 8 | 9 | +1 | flat/negative in 999 |
| SFT1B-s42-best_val-final | BASE1B-best_val | 10 | 11 | +1 | 6 | 8 | +2 | yes |
| SFT1B-s42-best_val-best_val | BASE1B-best_val | 10 | 10 | +0 | 6 | 11 | +5 | flat/negative in 999 |

- 只有 **best_val-final**（Δ +1 / +2）在两个 seed 都为正；**merged-best_val** 总量最高（27）但 2024 为 -1。
- **final-best_val** 是唯一在 999 明显负增益的臂（Δ -6，is_palindrome 与 reverse_string 塌成 0）。
- 同底座配对汇总：merged 系臂 999 合计 27 vs 底座 22；final 系 19 vs 28；best_val 系 21 vs 20（seed 999，2 臂合计满分 44）。
- 按 seed 汇总全部 6 臂 vs 3 底座（每臂只与自身底座比，口径相同）：seed 999 臂合计 67 vs 底座合计 35；seed 2024 62 vs 28。
- 因此**“SFT 对签名栏是净增益”在总体（臂池 vs 底座池）两个 seed 都成立，但在“逐臂 vs 自身底座”的配对口径下不稳定**：6 个臂里 5 个的 Δ 符号在两个 seed 之间翻转。

## 四、逐题 seed 波动排名

波动定义：该题 9 个模型的 |score(999) − score(2024)| 之和（越大越不稳）。

| 题 | Σ|Δscore| | |Δ(9 模型合计)| |
|---|---|---|
| is_palindrome | 8 | 2 |
| two_sum | 5 | 5 |
| quicksort | 5 | 1 |
| factorial | 5 | 3 |
| is_prime | 3 | 1 |
| fibonacci | 2 | 0 |
| binary_search | 2 | 2 |
| count_vowels | 2 | 0 |
| mergesort | 1 | 1 |
| sum_list | 1 | 1 |
| reverse_string | 0 | 0 |

波动最大的四题与具体翻转：

- **is_palindrome（8）**：BASE1B-merged 1→2、BASE1B-final 2→1、BASE1B-best_val 2→0、final-best_val 0→2、best_val-final 2→0。
- **two_sum（5）**：BASE1B-merged 2→0、BASE1B-final 2→1、BASE1B-best_val 1→0、final-best_val 1→0。
- **quicksort（5）**：底座 merged/best_val 各 +1（1→2），臂 merged-best_val、final-final、best_val-final 各 -1（2→1）。
- **factorial（5）**：BASE1B-merged 0→1，BASE1B-final 2→0、BASE1B-best_val 2→0。
- 最稳三题：reverse_string（0，两 seed 完全一致）、mergesort（1）、sum_list（1）。

## 五、脚本 vs 人工分歧

- `script_passed=True` 的 24 条；人工判 2 的 58 条；一致 160/198（约 81%）。
- **假通过（脚本 True、人工 < 2）2 条**：`BASE1B-merged / binary_search / 2024`（人工 1，递归返回切片相对下标）、`SFT1B-s42-merged-best_val / is_prime / 2024`（人工 1，漏 n%(i+2)）。
- **假失败（脚本 False、人工 = 2）36 条**，占脚本判失败的 174 条中的约 21%。主要集中在 mergesort（11 条）、reverse_string（4 条）、quicksort（4 条）、count_vowels（4 条）、is_palindrome（3 条）、factorial（3 条）——多为生成侧缩进被剥离导致的 IndentationError，或“原地排序/隐式返回”被脚本的返回式断言拒绝。
- 结论与本项目铁律一致：**脚本通过率不能定分**。若按脚本，seed 999 = 12/99、seed 2024 = 12/99（合计 24，恰好等于全表脚本通过数）；人工口径下等价满分（score==2）为 33/99 与 25/99。

### 分歧明细（36 条假失败 + 2 条假通过）

| model | task | seed | script_passed | human score | 类型 |
|---|---|---|---|---|---|
| BASE1B-merged | two_sum | 999 | False | 2 | false fail |
| BASE1B-merged | mergesort | 999 | False | 2 | false fail |
| BASE1B-merged | reverse_string | 999 | False | 2 | false fail |
| BASE1B-final | mergesort | 999 | False | 2 | false fail |
| BASE1B-final | is_palindrome | 999 | False | 2 | false fail |
| BASE1B-final | factorial | 999 | False | 2 | false fail |
| BASE1B-best_val | is_palindrome | 999 | False | 2 | false fail |
| BASE1B-best_val | factorial | 999 | False | 2 | false fail |
| SFT1B-s42-merged-final | mergesort | 999 | False | 2 | false fail |
| SFT1B-s42-merged-final | reverse_string | 999 | False | 2 | false fail |
| SFT1B-s42-merged-best_val | quicksort | 999 | False | 2 | false fail |
| SFT1B-s42-merged-best_val | mergesort | 999 | False | 2 | false fail |
| SFT1B-s42-merged-best_val | reverse_string | 999 | False | 2 | false fail |
| SFT1B-s42-final-final | quicksort | 999 | False | 2 | false fail |
| SFT1B-s42-final-final | mergesort | 999 | False | 2 | false fail |
| SFT1B-s42-final-best_val | mergesort | 999 | False | 2 | false fail |
| SFT1B-s42-final-best_val | count_vowels | 999 | False | 2 | false fail |
| SFT1B-s42-best_val-final | quicksort | 999 | False | 2 | false fail |
| SFT1B-s42-best_val-final | mergesort | 999 | False | 2 | false fail |
| SFT1B-s42-best_val-final | count_vowels | 999 | False | 2 | false fail |
| SFT1B-s42-best_val-best_val | mergesort | 999 | False | 2 | false fail |
| BASE1B-merged | binary_search | 2024 | True | 1 | false pass |
| BASE1B-merged | mergesort | 2024 | False | 2 | false fail |
| BASE1B-merged | is_palindrome | 2024 | False | 2 | false fail |
| BASE1B-merged | reverse_string | 2024 | False | 2 | false fail |
| SFT1B-s42-merged-final | mergesort | 2024 | False | 2 | false fail |
| SFT1B-s42-merged-final | reverse_string | 2024 | False | 2 | false fail |
| SFT1B-s42-merged-best_val | mergesort | 2024 | False | 2 | false fail |
| SFT1B-s42-merged-best_val | is_prime | 2024 | True | 1 | false pass |
| SFT1B-s42-merged-best_val | reverse_string | 2024 | False | 2 | false fail |
| SFT1B-s42-final-final | mergesort | 2024 | False | 2 | false fail |
| SFT1B-s42-final-final | count_vowels | 2024 | False | 2 | false fail |
| SFT1B-s42-final-best_val | mergesort | 2024 | False | 2 | false fail |
| SFT1B-s42-final-best_val | count_vowels | 2024 | False | 2 | false fail |
| SFT1B-s42-best_val-final | mergesort | 2024 | False | 2 | false fail |
| SFT1B-s42-best_val-best_val | fibonacci | 2024 | False | 2 | false fail |
| SFT1B-s42-best_val-best_val | mergesort | 2024 | False | 2 | false fail |
| SFT1B-s42-best_val-best_val | count_vowels | 2024 | False | 2 | false fail |

## 六、结论（回答四问）

1. **每 seed 底座排序**：seed 2024 成立（merged 14 > final 8 > best_val 6）；**seed 999 不成立**（final 14 > merged 11 > best_val 10）。破坏点集中在 seed 999 的 BASE1B-merged——is_prime / count_vowels / sum_list / factorial 四题只输出注释或注释掉的代码（4 个 0），而 BASE1B-final 在 999 的同四题里有 3 题有可用代码。三底座的“是否写出代码”跨 seed 抖动极大，底座排序不是稳定结论。
2. **每臂 vs 自身底座增益**：仅 best_val-final（+1/+2）两 seed 均为正；merged-final（+2/-3）、merged-best_val（+3/-1）、final-final（-3/+2）、final-best_val（-6/+1）、best_val-best_val（0/+5）符号均翻转。**配对口径下“SFT 净增益”不稳定**；臂池 vs 底座池的总体口径下两个 seed 都大幅为正（999 67 vs 35，2024 62 vs 28），但该增益主要来自最弱的 best_val 底座。
3. **逐题 seed 波动**（Σ|Δ| 排序）：is_palindrome 8 > two_sum 5 = quicksort 5 = factorial 5 > is_prime 3 > fibonacci 2 = binary_search 2 = count_vowels 2 > mergesort 1 = sum_list 1 > reverse_string 0。最不稳的是 is_palindrome（5 个模型跨 seed 翻转，含底座三兄弟全翻）与 two_sum（4 个模型翻转）。
4. **两 seed 合计排序（满分 44）**：

| 名次 | model | 999 | 2024 | 合计 |
|---|---|---|---|---|
| 1 | SFT1B-s42-merged-best_val | 14 | 13 | 27 |
| 2 | BASE1B-merged | 11 | 14 | 25 |
| 3 | SFT1B-s42-merged-final | 13 | 11 | 24 |
| 4 | BASE1B-final | 14 | 8 | 22 |
| 5 | SFT1B-s42-final-final | 11 | 10 | 21 |
| 6 | SFT1B-s42-best_val-best_val | 10 | 11 | 21 |
| 7 | SFT1B-s42-best_val-final | 11 | 8 | 19 |
| 8 | SFT1B-s42-final-best_val | 8 | 9 | 17 |
| 9 | BASE1B-best_val | 10 | 6 | 16 |

   **最佳单臂 = SFT1B-s42-merged-best_val（27 / 44）**，高于其自身底座 BASE1B-merged（25）；两 seed 内它都是臂中最高分（999 与 BASE1B-final 并列 14，2024 13 为臂最高）。最佳底座 = BASE1B-merged（25）。最差 = BASE1B-best_val（16）。

### 复核提示

- 本次评分重复出现的口径判断已统一记录：`-1` vs `None` 视为 1；原地完成排序视为 2；纯注释/纯示例调用视为 0；快排返回降序视为 1。
- 标注 `uncertain: true` 的条目共 26 条，均为“1 与 0 之间”或“1 与 2 之间”的边界判断，需要第二人复核。
- 未评条目：0（198/198 全部人工评过）。
