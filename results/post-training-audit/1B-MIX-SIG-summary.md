# MIX-SIG — 1B WSM 混比臂（data_mix_gold）签名栏 5-seed 人工审核 + 与金标准臂对照

审核范围：`1B WSM/fine-tuning/eval/eval_sft_generations.jsonl` 中 `SFT1B-mix42-*` 开头的 **110 行** = 2 变体 × 11 题 × 5 seed（seeds 0/42/123/999/2024）。
产物：`MIX-SIG.jsonl`（110 行，字段 `i,model,name,seed,script_passed,score,reason,uncertain`）。
对照数据（**直接读用，未重评**）：`SIG-5SEED-A.jsonl` + `SIG-5SEED-B.jsonl`（495 行：`SFT1B-s42-*` 金标准臂 330 行、`BASE1B-*` 底座 165 行）；按 `(model, name, seed)` join，字段按 `name/task`、`score/human_score` 归一化。

## 0. 口径与审计前提（务必先读）

- 地面真值 = 人工 0/1/2；`script_passed` 只记录、不定分。锚例沿用既有审核：`fib(0)=1,fib(1)=0` → 1；`str(s)==s[::-1]` → 2；只写注释 → 0；`return len(s)%2==0` 冒充回文 → 0。
- **`binary_search` 契约**沿用既有审核：未找到应 `return -1`；混比臂 10/10 返回 `None` 或 `True/False` → 一律 **1**（与底座、金标准臂同尺子）。
- **`mergesort` 原地排序不算错**：金标准臂同款原地归并记 2，混比臂 10/10 的原地归并同样记 **2**（`script_passed=False` 只因脚本要求 `return` 列表）。
- **缺 import 是本次唯一的系统性裁量**：`math` / `re` / `functools.reduce` 未 import 时运行必报 NameError，但判据与结构本身正确。既有审核中凡含缺 import 的行从未给过 2（is_prime 的 `math 未 import` 行一律 1），故本次一律记 **1** 并标 `uncertain: true`，共 **18 行**（is_prime 9 + sum_list 7 + is_palindrome 2）。
- **敏感性**：若这 18 行改记 2，混比臂总分由 176 升到 194/220；逐 seed 对金标准臂、对底座的**全部比较结论不变**（见 §7）。
- **脚本可复现性**：按 `eval_sft.py` 的 `truncate()` + assert 口径在本机（CPU，未用 GPU）重跑 110 行，`script_passed` **逐行复现：57 通过 / 53 失败，零分歧**。
- **口径限制（必须随结论引用）**：混比臂 `data_mix_gold` = **91.78M tokens / 2801 步**；金标准臂 `data_gold` = **62.39M tokens / 1904 步**，token 量差 **1.47 倍**。两臂同底座（merged）、同 seed（42）、同超参，但**数据量不同**，所以下面所有差值都是「**配比 + 数据量**」的合成效应，不是严格单变量对照。

## 1. 7 行逐 seed 完整排名（每格 /22 = 11 题 × 2 分）

| model | seed 0 | seed 42 | seed 123 | seed 999 | seed 2024 | 均值 | 5-seed 合计 |
|---|---|---|---|---|---|---|---|
| BASE1B-merged | 15 | 15 | 6 | 11 | 14 | 12.20 | 61 |
| BASE1B-final | 7 | 9 | 8 | 14 | 8 | 9.20 | 46 |
| BASE1B-best_val | 9 | 7 | 9 | 10 | 6 | 8.20 | 41 |
| SFT1B-mix42-merged-final | 18 | 18 | 16 | 15 | 18 | 17.00 | 85 |
| SFT1B-mix42-merged-best_val | 20 | 18 | 18 | 16 | 19 | 18.20 | 91 |
| SFT1B-s42-merged-final | 14 | 11 | 16 | 13 | 11 | 13.00 | 65 |
| SFT1B-s42-merged-best_val | 14 | 12 | 14 | 14 | 13 | 13.40 | 67 |

逐 seed 名次（高到低）：
- seed 0：`SFT1B-mix42-merged-best_val`（20） > `SFT1B-mix42-merged-final`（18） > `BASE1B-merged`（15） > `SFT1B-s42-merged-final`（14） > `SFT1B-s42-merged-best_val`（14） > `BASE1B-best_val`（9） > `BASE1B-final`（7）
- seed 42：`SFT1B-mix42-merged-final`（18） > `SFT1B-mix42-merged-best_val`（18） > `BASE1B-merged`（15） > `SFT1B-s42-merged-best_val`（12） > `SFT1B-s42-merged-final`（11） > `BASE1B-final`（9） > `BASE1B-best_val`（7）
- seed 123：`SFT1B-mix42-merged-best_val`（18） > `SFT1B-mix42-merged-final`（16） > `SFT1B-s42-merged-final`（16） > `SFT1B-s42-merged-best_val`（14） > `BASE1B-best_val`（9） > `BASE1B-final`（8） > `BASE1B-merged`（6）
- seed 999：`SFT1B-mix42-merged-best_val`（16） > `SFT1B-mix42-merged-final`（15） > `BASE1B-final`（14） > `SFT1B-s42-merged-best_val`（14） > `SFT1B-s42-merged-final`（13） > `BASE1B-merged`（11） > `BASE1B-best_val`（10）
- seed 2024：`SFT1B-mix42-merged-best_val`（19） > `SFT1B-mix42-merged-final`（18） > `BASE1B-merged`（14） > `SFT1B-s42-merged-best_val`（13） > `SFT1B-s42-merged-final`（11） > `BASE1B-final`（8） > `BASE1B-best_val`（6）

要点：**5 个 seed 的第一名全部是混比臂**（best_val 变体拿 4 个，merged-final 变体在 seed 2024 与之一并列 19）；金标准臂两变体在任何 seed 上都未超过混比臂。

### 1b. 混比臂两变体逐题明细

`SFT1B-mix42-merged-final`

| task | seed 0 | seed 42 | seed 123 | seed 999 | seed 2024 | 小计/10 |
|---|---|---|---|---|---|---|
| fibonacci | 2 | 2 | 1 | 1 | 2 | 8 |
| quicksort | 2 | 2 | 2 | 2 | 2 | 10 |
| binary_search | 1 | 1 | 1 | 1 | 1 | 5 |
| two_sum | 1 | 2 | 1 | 1 | 2 | 7 |
| mergesort | 2 | 2 | 2 | 2 | 2 | 10 |
| is_palindrome | 2 | 1 | 1 | 0 | 1 | 5 |
| is_prime | 1 | 1 | 1 | 1 | 1 | 5 |
| reverse_string | 2 | 2 | 2 | 1 | 2 | 9 |
| count_vowels | 2 | 2 | 2 | 2 | 2 | 10 |
| sum_list | 1 | 1 | 1 | 2 | 1 | 6 |
| factorial | 2 | 2 | 2 | 2 | 2 | 10 |
| **合计/22** | 18 | 18 | 16 | 15 | 18 | 85/110 |

`SFT1B-mix42-merged-best_val`

| task | seed 0 | seed 42 | seed 123 | seed 999 | seed 2024 | 小计/10 |
|---|---|---|---|---|---|---|
| fibonacci | 2 | 2 | 1 | 1 | 2 | 8 |
| quicksort | 2 | 1 | 2 | 2 | 2 | 9 |
| binary_search | 1 | 1 | 1 | 1 | 1 | 5 |
| two_sum | 2 | 2 | 2 | 2 | 2 | 10 |
| mergesort | 2 | 2 | 2 | 2 | 2 | 10 |
| is_palindrome | 2 | 2 | 2 | 2 | 2 | 10 |
| is_prime | 2 | 1 | 1 | 1 | 1 | 6 |
| reverse_string | 2 | 2 | 2 | 1 | 2 | 9 |
| count_vowels | 2 | 2 | 2 | 2 | 2 | 10 |
| sum_list | 1 | 1 | 1 | 1 | 1 | 5 |
| factorial | 2 | 2 | 2 | 1 | 2 | 9 |
| **合计/22** | 20 | 18 | 18 | 16 | 19 | 91/110 |

## 2. 混比臂 vs 金标准臂（同题同 seed，变体一一配对）

配对：`SFT1B-mix42-merged-final` ↔ `SFT1B-s42-merged-final`；`SFT1B-mix42-merged-best_val` ↔ `SFT1B-s42-merged-best_val`。

| seed | 混比 final | 金标准 final | 差 | 混比 best_val | 金标准 best_val | 差 |
|---|---|---|---|---|---|---|
| 0 | 18 | 14 | **+4** | 20 | 14 | **+6** |
| 42 | 18 | 11 | **+7** | 18 | 12 | **+6** |
| 123 | 16 | 16 | **+0** | 18 | 14 | **+4** |
| 999 | 15 | 13 | **+2** | 16 | 14 | **+2** |
| 2024 | 18 | 11 | **+7** | 19 | 13 | **+6** |
| **合计/110** | **85** | 65 | **+20** | **91** | 67 | **+24** |

- 10 个配对比较：**9 正、1 平（seed 123 的 final 变体 +0）、0 负**。
- 结论表述：可以写「**每一个 seed 上混比臂都不差于金标准臂**」（两变体 5/5 seed 非负）；不能写成「每个 seed 都更好」——seed 123 的 final 变体只是打平。
- 均值口径：混比臂 176/220 = **1.60 分/条**；金标准臂 132/220 = **1.20 分/条**（两变体合计）。

## 3. 混比臂 vs 自己底座 `BASE1B-merged`（逐 seed 增益）

| seed | BASE1B-merged | 混比 final | 增益 | 混比 best_val | 增益 |
|---|---|---|---|---|---|
| 0 | 15 | 18 | **+3** | 20 | **+5** |
| 42 | 15 | 18 | **+3** | 18 | **+3** |
| 123 | 6 | 16 | **+10** | 18 | **+12** |
| 999 | 11 | 15 | **+4** | 16 | **+5** |
| 2024 | 14 | 18 | **+4** | 19 | **+5** |
| **合计/110** | **61** | **85** | **+24** | **91** | **+30** |

- **5/5 seed 增益全为正**（两变体共 10/10 严格为正），幅度 +3 到 +12。
- 相对底座的增益（+3…+12）明显大于相对金标准臂的增益（0…+7），因为金标准臂本身已把底座抬高了一截。

## 4. 逐题胜负（5 seed 合计，每格 /10，两变体分别看）

| task | 混比 final | 金标准 final | 差 | 混比 best_val | 金标准 best_val | 差 | BASE1B-merged | 判定 |
|---|---|---|---|---|---|---|---|---|
| fibonacci | 8 | 6 | +2 | 8 | 6 | +2 | 9 | 混比胜 |
| quicksort | 10 | 5 | +5 | 9 | 7 | +2 | 8 | 混比胜 |
| binary_search | 5 | 5 | +0 | 5 | 5 | +0 | 6 | ('持平/互有', None) |
| two_sum | 7 | 5 | +2 | 10 | 5 | +5 | 4 | 混比胜 |
| mergesort | 10 | 10 | +0 | 10 | 10 | +0 | 9 | ('持平/互有', None) |
| is_palindrome | 5 | 10 | -5 | 10 | 10 | +0 | 6 | 混比负 |
| is_prime | 5 | 5 | +0 | 6 | 4 | +2 | 3 | 混比胜 |
| reverse_string | 9 | 10 | -1 | 9 | 10 | -1 | 8 | 混比负 |
| count_vowels | 10 | 7 | +3 | 10 | 8 | +2 | 4 | 混比胜 |
| sum_list | 6 | 0 | +6 | 5 | 0 | +5 | 0 | 混比胜 |
| factorial | 10 | 2 | +8 | 9 | 2 | +7 | 4 | 混比胜 |

- **混比胜（7 题）**：fibonacci, quicksort, two_sum, is_prime, count_vowels, sum_list, factorial
- **混比负（2 题）**：is_palindrome, reverse_string
- **持平/互有（2 题）**：binary_search, mergesort

重点题逐 seed 对照（格式：混比final/金标准final | 混比best_val/金标准best_val | 底座merged）：

`sum_list`

| seed | 混比F/金标准F | 混比B/金标准B | 底座merged |
|---|---|---|---|
| 0 | 1/0 | 1/0 | 0 |
| 42 | 1/0 | 1/0 | 0 |
| 123 | 1/0 | 1/0 | 0 |
| 999 | 2/0 | 1/0 | 0 |
| 2024 | 1/0 | 1/0 | 0 |

`factorial`

| seed | 混比F/金标准F | 混比B/金标准B | 底座merged |
|---|---|---|---|
| 0 | 2/0 | 2/0 | 1 |
| 42 | 2/0 | 2/0 | 2 |
| 123 | 2/2 | 2/2 | 0 |
| 999 | 2/0 | 1/0 | 0 |
| 2024 | 2/0 | 2/0 | 1 |

`is_palindrome`

| seed | 混比F/金标准F | 混比B/金标准B | 底座merged |
|---|---|---|---|
| 0 | 2/2 | 2/2 | 1 |
| 42 | 1/2 | 2/2 | 2 |
| 123 | 1/2 | 2/2 | 0 |
| 999 | 0/2 | 2/2 | 1 |
| 2024 | 1/2 | 2/2 | 2 |

`binary_search`

| seed | 混比F/金标准F | 混比B/金标准B | 底座merged |
|---|---|---|---|
| 0 | 1/1 | 1/1 | 2 |
| 42 | 1/1 | 1/1 | 1 |
| 123 | 1/1 | 1/1 | 1 |
| 999 | 1/1 | 1/1 | 1 |
| 2024 | 1/1 | 1/1 | 1 |

- **`sum_list`（历史最难，金标准臂 0/20 全线只写注释）**：混比臂首次写出可判分的实现（10 行里 1 行真 2 分、9 行 1 分），**+11 分是本次最大单题增益**；但 9/10 行都倒在同一个坑——用了 `reduce`/`operator` 却没 import。
- **`factorial`（金标准臂 4/20）**：混比臂 19/20（只有一个 seed 的基例少写 n==0），是**质量最干净的一题**，说明混比数据确实把「递归基例」这类代码形状教进去了。
- **`is_palindrome`（唯一明显倒退）**：merged-final 变体 5/10（金标准 10/10）。退步原因是两行 `re.sub` 未 import + 一行恒返回 True + 一行写成 `all(c.isalnum())`；但 best_val 变体 10/10 追平金标准臂。**倒退只在 final 变体上出现**。
- **`binary_search`（两臂完全打平 5/10）**：混比臂 10/10 都写成 `return None`/`True-False`，没有一行守住 `-1` 契约，底座 merged 反而 6/10。这是**混比数据没有修复的短板**，也说明「接口契约」类知识不是靠加签名数据能自动获得的。

## 5. 脚本 vs 人工分歧（110 行）

| 类别 | 行数 |
|---|---|
| 脚本通过 且 人工 = 2 | 57 |
| 脚本失败 但 人工 = 2（脚本过严/漏判） | **10** |
| 脚本通过 但 人工 < 2（脚本误判） | **0** |
| 脚本失败 且 人工 < 2 | 43 |

- 分歧 **10/110**，全部集中在脚本过严一侧（10 : 0），**无一行脚本误判**，与项目已知结论（脚本对签名续写系统性低估）一致。
- 脚本过严的 10 行**全部是 mergesort**（两变体各 5 行）：原地排序不 `return`，脚本断言要求返回列表；人工按「原地归并、合并逻辑完整」记 2（与金标准臂同款行同尺子，不是本次新开的口径）。
- 脚本失败且有真实缺陷的行，失败原因已逐条复现：
  - `NameError: name 're' is not defined`（is_palindrome 2 行）、`name 'math' is not defined`（is_prime 9 行）、`name 'reduce' is not defined`（sum_list 7 行）→ 缺 import，人工记 1。
  - `AssertionError`（binary_search 10、mergesort 10、sum_list 2、fibonacci 4、is_palindrome 2、two_sum 3、reverse_string 2、quicksort 1）→ 接口漂移或逻辑错，人工记 1；is_palindrome merged-final seed 2024 属恒返回 True 的真错误，人工记 1。
  - `ValueError`（factorial best_val seed 999，`factorial(0)` 递归到 -1 抛错）→ 人工记 1。
- 修正记录：`SFT1B-mix42-merged-final / is_palindrome / seed 2024` 初判 2，用 `eval_sft.py` 口径实跑后确认 `is_palindrome('hello')` 返回 True（`rev = c + rev` 逐字符前置把原串拼了回来），改判 **1**。同批另排查了 reverse_string 的两处「拼接方向」写法，结论与旧审核一致（返回原串 → 1，正确反转 → 2）。

## 6. 结论（回答 5 问）

**Q1 混比臂 vs 金标准臂（同题同 seed）逐 seed 差值？是否每个 seed 都更好，还是翻转？**

差值：seed 0 = +4/+6，seed 42 = +7/+6，seed 123 = +0/+4，seed 999 = +2/+2，seed 2024 = +7/+6（两变体）。
**没有任何 seed 翻转成负**：10 个配对里 9 正 1 平。可写「每个 seed 非负/不差于」，不可写「每个 seed 都更好」。

**Q2 混比臂 vs 自己底座 merged 逐 seed 增益？是否 5/5 非负？**

增益：seed 0 = +3/+5，seed 42 = +3/+3，seed 123 = +10/+12，seed 999 = +4/+5，seed 2024 = +4/+5。
**是，5/5 seed 两个变体全部为正**（共 10/10），最小增益 +3，最大 +12。

**Q3 逐题看，混比在哪几题赢/输？**

赢 7 题：sum_list(+11 合计，最大)、factorial(+15)、quicksort(+7)、two_sum(+7)、count_vowels(+5)、fibonacci(+4)、is_prime(+2)。
输 2 题：is_palindrome(-5，只在 final 变体)、reverse_string(-2，两变体各 -1)。
打平/互有 2 题：binary_search(两臂同为 5/10，都卡在 `-1` 契约)、mergesort(两臂都是满分 10/10，已封顶)。

**Q4 逐 seed 完整排名（7 行）**

| seed | 排名（高到低，/22） |
|---|---|
| 0 | `SFT1B-mix42-merged-best_val` 20 > `SFT1B-mix42-merged-final` 18 > `BASE1B-merged` 15 > `SFT1B-s42-merged-final` 14 > `SFT1B-s42-merged-best_val` 14 > `BASE1B-best_val` 9 > `BASE1B-final` 7 |
| 42 | `SFT1B-mix42-merged-final` 18 > `SFT1B-mix42-merged-best_val` 18 > `BASE1B-merged` 15 > `SFT1B-s42-merged-best_val` 12 > `SFT1B-s42-merged-final` 11 > `BASE1B-final` 9 > `BASE1B-best_val` 7 |
| 123 | `SFT1B-mix42-merged-best_val` 18 > `SFT1B-mix42-merged-final` 16 > `SFT1B-s42-merged-final` 16 > `SFT1B-s42-merged-best_val` 14 > `BASE1B-best_val` 9 > `BASE1B-final` 8 > `BASE1B-merged` 6 |
| 999 | `SFT1B-mix42-merged-best_val` 16 > `SFT1B-mix42-merged-final` 15 > `BASE1B-final` 14 > `SFT1B-s42-merged-best_val` 14 > `SFT1B-s42-merged-final` 13 > `BASE1B-merged` 11 > `BASE1B-best_val` 10 |
| 2024 | `SFT1B-mix42-merged-best_val` 19 > `SFT1B-mix42-merged-final` 18 > `BASE1B-merged` 14 > `SFT1B-s42-merged-best_val` 13 > `SFT1B-s42-merged-final` 11 > `BASE1B-final` 8 > `BASE1B-best_val` 6 |

**Q5 结论（含 1.47× token 干扰的正确说法）**

可以说的：
- 在同底座（merged）、同 seed（42）、同超参下，**混比臂（data_mix_gold）在签名栏上一致优于纯指令式金标准臂**：7 行排名里 5/5 seed 由混比臂占据第一，逐 seed 差值 9 正 1 平 0 负；相对自身底座 5/5 seed 全为正。
- 增益的**结构**很清楚，不是均匀涨一点，而是补上了金标准臂的整块空白：`sum_list`（金标准臂 0/20 → 混比 11/20）与 `factorial`（4/20 → 19/20），即历史最难的两题。
不能说的（失实风险点）：
- **不能说这是「配比的净效应」**：混比臂多跑了 **1.47× tokens**（91.78M vs 62.39M，2801 vs 1904 步）。所以正确表述是「**配比 + 1.47× 数据量的合成增益**」；差值里有多少来自 5:1 混比、多少来自多见的 47% 数据，本实验**无法分离**。
- **不能说「每个 seed 都更好」**：seed 123 的 final 变体恰好打平（+0），且 `is_palindrome` 在 final 变体上倒退（-5）——混比能补难例，但在个别简单题上会把已学会的能力换出去（两行 `re.sub` 未 import、一行恒返回 True）。
- **不能说签名栏短板已解决**：`binary_search` 两臂同为 5/10，10/10 行都没守住 `-1` 契约，底座 merged 反而 6/10；接口契约类知识是本实验**未被混比修复的残余短板**。
- 消融建议（若要把它写成单变量结论）：用金标准臂把步数/epoch 拉到与混比臂相同的 token 量再比一次，或在两臂各自训练的等步数中途 checkpoint 上比；否则只能报告为「在该 token 预算下的整体方案更优」。

## 7. 评分分布、不确定项与敏感性

- 分数分布（110 行）：0 分 1 行、1 分 42 行、2 分 67 行；总分 **176/220**（80.0%）。
- 分变体：`SFT1B-mix42-merged-final` **85/110**、`SFT1B-mix42-merged-best_val` **91/110**（best_val 变体略优且更稳）。
- `uncertain: true` **18 行**，全部是「缺 import、结构本身正确」这一条裁量（is_prime 9 + sum_list 7 + is_palindrome 2）；其余 92 行判定确定。
- 敏感性：这 18 行若改记 2，总分 194/220（88.2%），逐 seed 排序与「混比不低于金标准、不低于底座」的结论**全部不变**——即结论不依赖这条裁量。
- 覆盖情况：110/110 混比臂行已逐条给分，无遗漏、无编造；`BASE1B-*` 165 行与 `SFT1B-s42-*` 330 行**直接沿用既有 5-seed 人工分，未重评**。
- 全程未使用 GPU、未 git push、未修改任何既有文件（只新建 `MIX-SIG.jsonl` 与 `MIX-SIG-summary.md`）。
