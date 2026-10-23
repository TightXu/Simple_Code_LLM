# 混比臂 NL（指令）栏 人工评审汇总

评审人：Hermes subagent｜评审日期：2026-10-14｜口径：2=完全正确，1=结构对/细节错（含符号未定义、函数名不符、漏边界），0=错误或冒充实现
数据：`nl_eval_generations.jsonl` 中 `SFT1B-mix42-*` 110 条（2 checkpoint × 11 题 × 5 seed），逐条打分见 `MIX-NL.jsonl`（含 `reason` / `script_passed` / `uncertain` / `verdict`）
对照：`NL-5SEED-A.jsonl` + `NL-5SEED-B.jsonl` 共 495 条人工分，按 `(model, name|task, seed)` 归一化 join；底座 `BASE1B-*` 未重评
覆盖：110/110 全部评完（无未评条目）

---

## ① 7 行逐 seed 完整排名（每 seed 满分 22 = 11 题 × 2）

| seed | 排名（分数从高到低，同分并列） |
|---|---|
| 0 | `mix42-merged-final` 22 > `mix42-merged-best_val` 20 = `s42-merged-final` 20 > `s42-merged-best_val` 19 > `BASE1B-best_val` 16 = `BASE1B-merged` 16 > `BASE1B-final` 15 |
| 42 | `mix42-merged-best_val` 20 = `s42-merged-best_val` 20 > `mix42-merged-final` 19 = `s42-merged-final` 19 > `BASE1B-merged` 18 > `BASE1B-best_val` 15 = `BASE1B-final` 15 |
| 123 | `mix42-merged-final` 20 > `mix42-merged-best_val` 19 = `s42-merged-final` 19 > `s42-merged-best_val` 18 > `BASE1B-best_val` 16 = `BASE1B-final` 16 = `BASE1B-merged` 16 |
| 999 | `s42-merged-best_val` 20 > `mix42-merged-best_val` 19 > `mix42-merged-final` 18 > `s42-merged-final` 17 > `BASE1B-best_val` 16 > `BASE1B-merged` 15 > `BASE1B-final` 13 |
| 2024 | `s42-merged-best_val` 20 = `s42-merged-final` 20 > `mix42-merged-best_val` 19 = `mix42-merged-final` 19 > `BASE1B-merged` 18 > `BASE1B-final` 17 > `BASE1B-best_val` 15 |

5-seed 合计（满分 110）：`mix42-merged-final` 98｜`mix42-merged-best_val` 97｜`s42-merged-best_val` 97｜`s42-merged-final` 95｜`BASE1B-merged` 83｜`BASE1B-best_val` 78｜`BASE1B-final` 76。

## ② 混比臂 vs 金标准臂（同 checkpoint 后缀、同题同 seed，满分 22）

| seed | mix-final vs s42-final | mix-best_val vs s42-best_val | 两行合计 |
|---|---|---|---|
| 0 | 22 vs 20 → **+2** | 20 vs 19 → **+1** | **+3** |
| 42 | 19 vs 19 → 0 | 20 vs 20 → 0 | **0** |
| 123 | 20 vs 19 → **+1** | 19 vs 18 → **+1** | **+2** |
| 999 | 18 vs 17 → **+1** | 19 vs 20 → **-1** | **0** |
| 2024 | 19 vs 20 → **-1** | 19 vs 20 → **-1** | **-2** |

**结论：不是每个 seed 都更好。** 3/5 个 seed 上混比合计不劣（0/+2/+3），seed 999 打平，seed 2024 两行同时 -1（合计 -2）。5-seed 合计差仅 +1/+0（98 vs 95、97 vs 97），量级落在单条 0.5 分以内，且方向随 seed 翻转 → 混比对指令栏**没有稳定的净增益或净损失**，属 seed 噪声带内的平手。

## ③ 混比臂 vs 自己的底座 `BASE1B-merged`（满分 22）

| seed | mix-final | mix-best_val | 底座 merged | 增益 |
|---|---|---|---|---|
| 0 | 22 | 20 | 16 | **+6 / +4** |
| 42 | 19 | 20 | 18 | **+1 / +2** |
| 123 | 20 | 19 | 16 | **+4 / +3** |
| 999 | 18 | 19 | 15 | **+3 / +4** |
| 2024 | 19 | 19 | 18 | **+1 / +1** |

**5/5 非负，且全部严格为正**（10/10 个 (checkpoint, seed) 单元为正增益，范围 +1…+6）。SFT（不论金标准还是混比）在指令栏对底座是实打实的抬升，这一点与配比无关。

## ④ 逐题胜负（每题 5 seed × 2 checkpoint 配对 = 10 个单元比较，5-seed 合计满分 10）

| 题 | mix-final | mix-best_val | gold-final | gold-best_val | 底座 merged | 混比 胜/平/负 | 备注 |
|---|---|---|---|---|---|---|---|
| fibonacci | 10 | 10 | 10 | 10 | 7 | 0/10/0 | 全对，两臂持平 |
| **quicksort** | 10 | 9 | 10 | 5 | 5 | **4/6/0** | 混比不劣，全线赢 gold-best_val（后者 pivot 取首元素 + 严格不等号丢重复元素） |
| binary_search | 10 | 10 | 10 | 10 | 10 | 0/10/0 | 全对 |
| two_sum | 9 | 9 | 10 | 9 | 7 | 1/7/2 | 混比两处输：两边各 1 个 seed 用了「未排序数组 + 双指针 + 1-based 下标」 |
| mergesort | 7 | 5 | 4 | 10 | 5 | 3/2/5 | 混比 8/10 条把 merge 写成未定义符号（NameError）→ 1 分；gold-best_val 5/5 原地归并拿满 |
| is_palindrome | 8 | 9 | 7 | 8 | 10 | 2/7/1 | 混比净胜；但双方各有 1 个 seed 出 0 分（gre 未 import → 1；`all(c==c[::-1])` 恒真 → 0） |
| is_prime | 8 | 10 | 10 | 10 | 2 | 0/8/2 | 混比两处输：`if n == 2 or n % 2 == 0: return True` 使 is_prime(4) 返 True（脚本仍误过） |
| reverse_string | 10 | 10 | 10 | 10 | 10 | 0/10/0 | 全对 |
| count_vowels | 10 | 10 | 10 | 10 | 8 | 0/10/0 | 全对 |
| **sum_list** | 10 | 10 | 9 | 10 | 10 | **1/9/0** | 混比 10/10 全对；gold-final 1 条 999 seed 只累加正数（-1+5 算成 5）→ 1 |
| **factorial** | 6 | 5 | 5 | 5 | 9 | **1/9/0** | 混比略优但两臂同病：几乎全部写成 `factorial_iterative`，题面 `factorial` 未定义 → 1（这题底座反而 9 分） |

重点题：`sum_list` 混比全对、金标准丢 1 分（混比小胜）；`quicksort` 混比明显优于金标准（金标准 best_val 只拿 5/10）；`factorial` 两臂几乎同分（都吃函数名不符的 1 分），差异不构成优势。

## ⑤ 脚本 `passed` vs 人工分 分歧表

| 类型 | 条目 | 说明 |
|---|---|---|
| **false_pass（脚本过、人工判错）** | `SFT1B-mix42-merged-final` / is_prime / seed 42 与 2024（2 条，均判 1 分） | `is_prime(4)` 返 True；官方测试集只含 2/17/15/1，偶数合数盲区未覆盖 |
| false_fail（脚本错、人工判对） | 0 条 | — |
| agree | 108 条 | 其中 8 条「脚本失败 + 人工 1 分」（merge 未定义、双指针 1-based、丢重复元素） |

- 混比臂脚本通过率 **80.0%**（88/110），人工满分率 **78.2%**（86/110）→ 脚本口径向乐观偏 ~1.8 个百分点。
- 对照：同一栏金标准 `s42-merged-final` 脚本 42/55、人工 42/55；`s42-merged-best_val` 脚本 43/55、人工 43/55（历史评审近似无偏）。故本次 2 条 false_pass 是评测集覆盖不足，不是评分口径偏移。
- `uncertain: true` 仅 1 条（`mix42-merged-best_val` / two_sum / seed 123，循环条件为 `(current_sum - target) in {0, n}` 的胡话式双指针，1 分与 0 分之间可争，按「骨架可辨」判 1）。

---

## ⑥ 五个问题的回答

1. **混比 vs 金标准（同题同 seed）**：逐 seed 差值 = seed 0 **+3**、seed 42 **0**、seed 123 **+2**、seed 999 **0**（final +1 / best_val -1 对冲）、seed 2024 **-2**。**混比不是每个 seed 都更好**：3 个 seed 不劣、1 个打平、1 个明确更差；5-seed 合计 +1/+0。
2. **混比 vs 自己底座 merged**：**5/5 非负且全为正**（+6/+4、+1/+2、+4/+3、+3/+4、+1/+1）。
3. **7 行逐 seed 排名**：见 ①。混比两行在 seed 0/42/123 占据榜首或并列榜首；seed 999/2024 由金标准 best_val 反超。底座三行 5 个 seed 全部垫底。
4. **逐题胜负**：见 ④。混比赢面集中在 `quicksort`（4 胜 0 负）、`is_palindrome`（2 胜 1 负）、`sum_list`（1 胜 0 负）、`mergesort` 部分 seed（3 胜 5 负，主要是 gold-best_val 那种原地归并模板太稳）；输面集中在 `is_prime`（2 负）、`two_sum`（2 负）、`mergesort`（5 负）。
5. **结论口径（含 1.47× token 干扰）**：
   - 可以说：**混比臂在指令栏不弱于金标准臂，同 seed 配对下 5-seed 合计 +1/+0（98 vs 95、97 vs 97，满分 110）**，且**两臂都在底座之上 5/5 全为正增益**（+1…+6）。
   - 不能说：混比**提升**了指令栏。两臂差值（+1 分 / 110 分的 0.9%）远小于 seed 波动（同一臂不同 seed 极差 4 分：18↔22），且方向在 seed 间翻转（seed 0 +3 vs seed 2024 -2）→ 差分落在噪声内。
   - 必须声明干扰：混比臂 **91.78M tokens vs 金标准 62.39M，1.47×**，本对比是「配比 + 数据量」的合成效应。因此正确表述是「**在付出 1.47× token 的前提下，混比臂的指令栏没有掉下来（持平），签名栏的收益（另见 SIG 栏）不是以指令栏为代价换来的**」——而不是「混比按比例提升了指令栏」。若要把配比效益与数据量效益分开，需要补一臂「金标准数据重复到 91.78M tokens」或对混比臂做 token 匹配的降采样重训。
   - 附带观察（非结论）：混比臂输出同质度更高——11 题里 fibonacci 5 个 seed 完全同一份实现、reverse_string 只有 2 种变体（110 条中 22 条与同模型同题的另一 seed 逐字相同），金标准臂的变体更散，这与「5:1 签名数据」的模板化倾向一致，但本任务未在同一口径下量化金标准臂的多样性，故只作提示。
