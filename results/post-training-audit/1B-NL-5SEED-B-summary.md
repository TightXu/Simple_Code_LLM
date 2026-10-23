# 1B 三底座 × 六臂 —— NL（指令）栏人工审核 seeds 999 / 2024

- 数据源：`1B WSM/fine-tuning/eval/nl_eval_generations.jsonl`（495 行 = 9 模型 × 11 题 × 5 seed）
- 本次范围：**seeds 999、2024，共 198 条**（seeds 0/42/123 归另一代理，未触碰）
- 逐条落盘：`NL-5SEED-B.jsonl`（198 行，每条含 `human_score` / `reason` / `uncertain` / `script_passed` / `verdict`）
- 评分口径：**2 = 完全正确；1 = 思路接近/结构对但细节错（基准写反、边界差一、返回 None vs -1、接口/命名不符导致符号未定义、漏元素）；0 = 错误/胡话/仅注释或纯文档串/无关**
- NL 栏特有：只写注释/只描述思路的一律 0；`script_passed` 只记录，**不参与定分**
- 未使用 GPU；未修改任何现有文件（两个产物均为新建）

**审核前钉死的判分契约**（读 `eval_sft.ALGORITHMS + SIMPLE_TASKS`，NL 栏复用同一 dict 与同一 `tests` 对象）：
`fibonacci(0)=0,(1)=1,(10)=55,(20)=6765`；`quicksort/mergesort` 返回**新升序表**（[] 与 [5] 也要对）；
`binary_search` 未命中 `-1`、`binary_search([],x)=-1`；`two_sum` 返回**下标**（用 `sorted()` 比较，`[3,2,4],6 -> [1,2]`）；
`is_palindrome('')=True`；`is_prime(2)=T,(17)=T,(15)=F,(1)=F`；`count_vowels` 只测小写；`sum_list([-1,5])=4`；`factorial(0)=1`。

**本次单列的口径取舍**（与签名栏 B 审核保持一致，写明以便复核）：
1. 命名不符（题面要 `factorial` 却写 `factorial_iterative/recursive`；要 `mergesort` 却写 `merge_sort`）→ **1**（签名栏 B 同口径；两栏都是 1，不是 0）。
2. 无解时返回 `None` / `[]` / `[-1,-1]` 不扣分（题面未约定，官方断言只测有解）。
3. `count_vowels` 多认大写或 y → 与本栏单 seed 审核同尺子：**筛元音条件本身合理不扣分**；但 `sum(字符)` 而非计数、或主实现把 y 计入导致 `'xyz'` 得 1 → 扣到 1。
4. **原地排序且确实写回并 `return arr`** → 2（与签名栏 B “原地版完成排序视为 2” 同口径）。
5. **被 `max_new` 截断只砍掉函数尾部**（merge 的收尾 `extend/return` 缺失，可见部分逻辑正确）→ 1（技能口径：判“可见的最长版本”；设为 0 的敏感性见第 ⑦ 节）。截断到“根本没有实现”（停在 docstring 内）→ 0。
6. 一个 completion 内出现多个同名定义时，**以第一个定义为主答案**；主实现有错但后面的“Solution 2”完全正确 → 1 并标 `uncertain`。

---

## 一、逐 seed 评分表（9 模型 × 11 题，满分 22）

题序与签名栏 B 一致：fibona=fibonacci，binary=binary_search，two_su=two_sum，quicks=quicksort，merges=mergesort，is_pal=is_palindrome，is_pri=is_prime，revers=reverse_string，count_=count_vowels，sum_li=sum_list，factor=factorial。

### seed 999（9 模型合计 **159 / 198**）

| model | fibona | binary | two_su | quicks | merges | is_pal | is_pri | revers | count_ | sum_li | factor | total | 脚本 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE1B-merged | 1 | 2 | 1 | 1 | 1 | 2 | 0 | 2 | 1 | 2 | 2 | **15** | 6/11 |
| BASE1B-final | 1 | 1 | 2 | 1 | 0 | 2 | 1 | 2 | 1 | 1 | 1 | **13** | 1/11 |
| BASE1B-best_val | 2 | 1 | 1 | 1 | 1 | 2 | 1 | 2 | 1 | 2 | 2 | **16** | 5/11 |
| SFT-s42-merged-final | 2 | 2 | 2 | 2 | 1 | 0 | 2 | 2 | 2 | 1 | 1 | **17** | 7/11 |
| SFT-s42-merged-best_val | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | **20** | 9/11 |
| SFT-s42-final-final | 2 | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 1 | **20** | 9/11 |
| SFT-s42-final-best_val | 2 | 2 | 2 | 1 | 2 | 1 | 2 | 2 | 2 | 2 | 1 | **19** | 8/11 |
| SFT-s42-best_val-final | 2 | 2 | 2 | 2 | 1 | 2 | 2 | 1 | 2 | 2 | 1 | **19** | 8/11 |
| SFT-s42-best_val-best_val | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | **20** | 9/11 |
| per-task total | 16 | 16 | 16 | 12 | 11 | 15 | 14 | 17 | 15 | 16 | 11 | 159 | 62/99 |

### seed 2024（9 模型合计 **166 / 198**）

| model | fibona | binary | two_su | quicks | merges | is_pal | is_pri | revers | count_ | sum_li | factor | total | 脚本 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE1B-merged | 1 | 2 | 2 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | 2 | **18** | 7/11 |
| BASE1B-final | 1 | 2 | 2 | 1 | 1 | 2 | 1 | 2 | 1 | 2 | 2 | **17** | 6/11 |
| BASE1B-best_val | 1 | 1 | 0 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | 2 | **15** | 5/11 |
| SFT-s42-merged-final | 2 | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 1 | **20** | 9/11 |
| SFT-s42-merged-best_val | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | **20** | 9/11 |
| SFT-s42-final-final | 2 | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 1 | **20** | 9/11 |
| SFT-s42-final-best_val | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | **20** | 9/11 |
| SFT-s42-best_val-final | 2 | 2 | 2 | 2 | 1 | 2 | 2 | 1 | 2 | 2 | 1 | **19** | 8/11 |
| SFT-s42-best_val-best_val | 2 | 2 | 2 | 1 | 2 | 0 | 2 | 2 | 2 | 2 | 1 | **17** | 7/11 |
| per-task total | 15 | 17 | 15 | 12 | 12 | 16 | 15 | 17 | 17 | 18 | 12 | 166 | 69/99 |

分数分布（198 条）：**2 分 132、1 分 61、0 分 5**；标 `uncertain: true` 28 条。

五个 0 分（逐条）：`BASE1B-merged/is_prime/999`、`BASE1B-final/mergesort/999`、`SFT-s42-merged-final/is_palindrome/999`、`BASE1B-best_val/two_sum/2024`、`SFT-s42-best_val-best_val/is_palindrome/2024`。

---

## 二、逐 seed 底座排序 / “best_val 在 NL 最好”是否成立

| seed | merged | final | best_val | 实际排序 | “best_val 最好” |
|---|---|---|---|---|---|
| 999 | 15 | 13 | 16 | **best_val > merged > final** | **成立（靠 1 分）** |
| 2024 | 18 | 17 | 15 | **merged > final > best_val** | **不成立（best_val 垫底）** |

- **单 seed（seed 0）结论 “best_val 在 NL 最好” 在 2 个 seed 里只有 1 个成立，且当年那 1 分优势实际来自同分并列。** 999 里 best_val 只领先 merged 1 分（16 vs 15），而同一批底座在 2024 里 best_val 反而**落后** merged 3 分——排名翻转。
- 翻转的落点（同题跨 seed 差异，均有逐条理由）：
  - `BASE1B-best_val/two_sum`：999 是“双指针但返回 [left+1,right+1]”得 1；2024 **整段只有 docstring 且被 max_new 截断**、无实现 → 0（单题 -1）。
  - `BASE1B-merged/is_prime`：999 只有 docstring（0）→ 2024 有试除实现（仅缺 import，1），单题 +1。
  - `BASE1B-merged/count_vowels`：999 `sum(字符)` 运行 TypeError（1）→ 2024 正确布尔计数（2），单题 +1。
  - `BASE1B-final/binary_search`：999 漏 `return -1`（1）→ 2024 完整（2）；`BASE1B-final/sum_list`：999 的怪递归 RecursionError（1）→ 2024 首元素递归正确（2）。
  - 也就是说底座的差异主要来自**“这次有没有真的写出可运行代码 / 有没有漏掉 return、import、边界”**，与模型能力排序无关，跨 seed 抖动 1–2 分/题。
- 附：底座三兄弟在两 seed 的平均是 merged 16.5 > final 15.0 > best_val 15.5 —— **两个 seed 合起来 merged 最好，best_val 不是。**

---

## 三、每臂 vs 自身底座（逐 seed 增/减）

| 臂 | 自身底座 | base@999 | arm@999 | Δ999 | base@2024 | arm@2024 | Δ2024 | 符号稳定 |
|---|---|---|---|---|---|---|---|---|
| SFT-s42-merged-final | BASE1B-merged | 15 | 17 | **+2** | 18 | 20 | **+2** | 稳定为正 |
| SFT-s42-merged-best_val | BASE1B-merged | 15 | 20 | **+5** | 18 | 20 | **+2** | 稳定为正 |
| SFT-s42-final-final | BASE1B-final | 13 | 20 | **+7** | 17 | 20 | **+3** | 稳定为正 |
| SFT-s42-final-best_val | BASE1B-final | 13 | 19 | **+6** | 17 | 20 | **+3** | 稳定为正 |
| SFT-s42-best_val-final | BASE1B-best_val | 16 | 19 | **+3** | 15 | 19 | **+4** | 稳定为正 |
| SFT-s42-best_val-best_val | BASE1B-best_val | 16 | 20 | **+4** | 15 | 17 | **+2** | 稳定为正 |

- **6/6 臂在两个 seed 都是正增益，符号一次都没有翻转**（与签名栏“6 臂中 5 臂翻转”形成鲜明对比）。
- 逐 seed 净额：999 臂池 115 vs 底座池 44（+71）；2024 臂池 116 vs 底座池 50（+66）。逐配对平均：999 +4.5、2024 +2.7。
- 增益大小的规律是**回归到 17–20 的臂平台**，而不是“哪个底座更强”：底座越弱（final@999 = 13）增益越大（+7/+6），底座越强（best_val@999 = 16）增益越小（+3/+4）。**用“臂比底座涨多少”来挑底座，本质是在挑最弱的底座**，不可作为选型依据。
- 逐题看，臂的失分点跨 seed 完全一致，都是**接口/命名与少量实现细节**，没有能力层面的翻转：`factorial` 12/12 条（两 seed 全部）因命名写成 `factorial_iterative/recursive` 而 -1；`quicksort` 降序实现 4 条（两 seed 各 2 条）；`mergesort` 被截断 6 条；`sum_list` 只加正数 1 条；`reverse_string` 插空格 1 条。

---

## 四、逐题 seed 波动排名

波动定义：该题 9 个模型的 `|score(999) − score(2024)|` 之和（越大越不稳）。

| 题 | Σ\|Δ\| | Δ(9 模型合计) |
|---|---|---|
| is_palindrome | **5** | +1 |
| two_sum | **3** | -1 |
| sum_list | 2 | +2 |
| count_vowels | 2 | +2 |
| mergesort | 1 | +1 |
| is_prime | 1 | +1 |
| fibonacci | 1 | -1 |
| factorial | 1 | +1 |
| binary_search | 1 | +1 |
| reverse_string | 0 | 0 |
| quicksort | 0 | 0 |

- **最不稳：`is_palindrome`（5）** —— `SFT-merged-final` 0→2、`SFT-final-best_val` 1→2、`SFT-best_val-best_val` 2→0 三处跨 seed 翻转，其中两个 0 分都是“恒为 True/恒等自比”的冒充实现。
- 次不稳：**`two_sum`（3）** —— `BASE1B-merged` 1→2、`BASE1B-best_val` 1→0、`SFT-best_val-best_val` 2→1。
- 最稳：**`reverse_string` 与 `quicksort` 全表 0 波动**（9 个模型两 seed 逐题完全一致）；`factorial` 表面也是 1 分（12/12 条两 seed 全是命名错），但两 seed 都在 1 分，其“稳定”是稳定地错。
- 结论：**11 题里只有 2 题波动 = 0**，其余 9 题都有 1–5 分摆动；`is_palindrome` 与 `two_sum` 的摆动幅度（5、3）与模型间差异（底座 13–18）同量级，这两题单独看不足以定胜负。

---

## 五、脚本 vs 人工分歧

| script_passed | 人工 2 | 人工 1 | 人工 0 |
|---|---|---|---|
| True（131 条） | 130 | 1 | 0 |
| False（67 条） | 2 | 60 | 5 |

- 脚本通过 131/198（66%），人工满分 132/198（67%）——**总数几乎相同，但分布完全不同**：脚本把“结构对但细节错”的 60 条一律算 0。
- **假失败（脚本 False、人工 = 2）只有 2 条**（与签名栏 B 的 36 条相比小得多）：
  - `BASE1B-final / two_sum / 999`：哈希表一次遍历，实现完全正确；脚本 False 是因为该 completion **以注释行 `# Solution 1: Hash Table` 开头**，`truncate()` 的停止串 `\n# ` 命中下标 0，把整段代码截成空串。
  - `BASE1B-final / reverse_string / 999`：`''.join(reversed(s))` 正确；同样因为开头是 `# Solution 1: Reversing strings` 被整段截空。
  - 即本栏的假失败机制**与签名栏不同**：不是缩进/原地返回，而是 `truncate` 对“注释打头”的 completion 清零。
- **假通过（脚本 True、人工 < 2）只有 1 条**：`BASE1B-merged / fibonacci / 999` —— 官方断言只测 `fib(0)/(1)/(10)/(20)`，而它的基准 `n <= 2: return n` 让 `fib(2)=2`（应为 1），实测 `[0,1,2,2,3,5,8,13]`，脚本照样通过。
- 单向偏差仍是“脚本更严”：脚本判 fail 的 67 条里 62 条人工给 ≥1 分（60 条 1 分 + 2 条 2 分），占 93%。按题统计这 60 条：factorial 13、quicksort 12、mergesort 11、is_prime 5、count_vowels 4、fibonacci 4、two_sum 3、binary_search 3、sum_list 2、reverse_string 2、is_palindrome 1。
- **本栏结论**：脚本数字（如“SFT 臂 8–10/11 vs 底座 4–8/11”）方向恰好与人工一致，但细则完全不同（脚本漏看降序快排、正数才求和、插空格反转；又误判缺 import/命名不符为 0），**仍不能用于定分**。

---

## 六、回答四个问题

### 6.1 每个 seed 内的底座排序？“best_val 在 NL 最好”是否在该 seed 成立？
- seed 999：**best_val 16 > merged 15 > final 13** → 该 seed **成立**（但只领先 1 分，落在同尺子的 ±1 噪声内）。
- seed 2024：**merged 18 > final 17 > best_val 15** → 该 seed **不成立**，best_val 垫底。
- **合起来：不成立。** “best_val 在 NL 最好”只活了 1/2 个 seed；两 seed 合计 merged 16.5 > best_val 15.5 > final 15.0。

### 6.2 每个臂 vs 自身底座：逐 seed 增/减多少？
- Δ999 / Δ2024：merged-final +2/+2；merged-best_val +5/+2；final-final +7/+3；final-best_val +6/+3；best_val-final +3/+4；best_val-best_val +4/+2。
- **6/6 臂两 seed 全为正，符号零翻转**（签名栏为 5/6 翻转）——NL 栏“SFT 带来净增益”是**唯一在两栏都稳住的结论**。
- 但幅度与底座强弱反相关（底座越弱涨得越多），逐配对平均 999 +4.5 / 2024 +2.7，所以这个增益是**回归到 17–20 的臂平台**，不是“某底座更适合 SFT”。

### 6.3 逐题 seed 波动排名（最不稳/最稳）
- 最不稳：`is_palindrome` 5 > `two_sum` 3 > `sum_list` 2 = `count_vowels` 2 > 其余（fibonacci / binary_search / mergesort / is_prime / factorial 各 1）。
- 最稳：`reverse_string` 0 = `quicksort` 0（9 模型两 seed 逐题一模一样）。
- 波动来源集中在“恒真冒充回文”“docstring-only 被截断”“返回 None/-1”三类写法上，与模型强弱无关。

### 6.4 两个 seed 合计下的排序与最佳单臂（满分 44）

| 名次 | model | 999 | 2024 | 合计 |
|---|---|---|---|---|
| 1 | SFT-s42-merged-best_val | 20 | 20 | **40** |
| 1 | SFT-s42-final-final | 20 | 20 | **40** |
| 3 | SFT-s42-final-best_val | 19 | 20 | 39 |
| 4 | SFT-s42-best_val-final | 19 | 19 | 38 |
| 5 | SFT-s42-merged-final | 17 | 20 | 37 |
| 5 | SFT-s42-best_val-best_val | 20 | 17 | 37 |
| 7 | BASE1B-merged | 15 | 18 | 33 |
| 8 | BASE1B-best_val | 16 | 15 | 31 |
| 9 | BASE1B-final | 13 | 17 | 30 |

- **最佳单臂是并列第一的 `SFT-s42-merged-best_val` 与 `SFT-s42-final-final`（均 40/44）**；两者都高于各自底座（+7 / +9 合计）。**必须写“并列”，不能择一**：它们各自在某个 seed 也有失分（merged-best_val 两 seed 都栽在降序 quicksort；final-final 两 seed 都栽在截断的 mergesort），40 = 40 在 ±1 噪声内。
- 最佳底座（两 seed 合计）= **BASE1B-merged（33）**，不是 best_val（31）。
- 六个臂（37–40）全部高于三个底座（30–33），臂之间 3 分的差距**小于**逐题波动幅度（is_palindrome 5、two_sum 3），所以**臂的名次不能细排**，只能说“六个臂同档，全部优于底座”。

---

## 七、不确定与敏感性

标 `uncertain: true` 共 **28 条**，全部集中在 0/1 或 1/2 边界，且 reason 里都写明了事实：

- 无限递归的快排（`BASE1B-merged` ×2 seed、`BASE1B-best_val` ×2 seed）：按签名栏同尺子记为 1（“重复 pivot 致崩”）而非 0。
- `BASE1B-final/quicksort/999`（`if not high` 直接原样返回输入，从不排序）与 `BASE1B-final/fibonacci/999`（基准+递推双错，实测 [1,1,1,2,3,2,3,5]）——1 与 0 之间。
- `BASE1B-final/sum_list/999`（自递归 → RecursionError）1 与 0 之间；`BASE1B/factorial/999`（fenced def 缺基准，factorial(0)=0）1 与 0 之间。
- `BASE1B-best_val/mergesort/2024`（基准 `len==0` 使单元素自递归 → RecursionError）1 与 0 之间。
- `BASE1B-merged/count_vowels/999`、`BASE1B-final/count_vowels/999`、`BASE1B-final/count_vowels/2024`（`sum(字符)` 运行 TypeError；最后一个还多认 y）。
- `BASE1B-best_val/count_vowels/999`（主实现把 y 算进去 → `'xyz'`=1，仅“Solution 2”正确，按主实现记 1）。
- `BASE1B-best_val/two_sum/999`（双指针条件写反且返回 +1 下标，实测 `[1,3]`）1 与 0 之间。
- 6 条截断的 mergesort（`merged-final`、`final-final`、`best_val-final`，两 seed 各 3）→ 1 与 0 之间。
- 3 条“冒充回文”：`SFT-merged-final/is_palindrome/999`（恒真）、`SFT-best_val-best_val/is_palindrome/2024`（两边同表达式恒真）、`SFT-final-best_val/is_palindrome/999`（expand-around-center 判据错，'racecar' 返 False）→ 0 与 1 之间。
- `SFT-best_val-best_val/two_sum/2024`（用 set 存补数却调 `.index()`，AttributeError）1 与 0 之间。

**敏感性 1（截断的 mergesort 记 1）：** 若改记 0，臂合计变为 merged-best_val 40、final-final 38、final-best_val 39、best_val-final 36、merged-final 35、best_val-best_val 37 —— **最佳单臂从“并列”变成 `merged-best_val` 单独第一（40）**，但“六臂全部高于三底座”与“6/6 臂两 seed 正增益”两条主结论不变。
**敏感性 2（命名不符记 1，非 0）：** 12 条 `factorial` 若改记 0，六个臂各 -2（两 seed 各 -1），臂池从 37–40 降到 35–38，**臂间名次与“臂>底座”结论不变**。
**敏感性 3（原地排序记 2）：** 若把“原地写回并 return arr”的 4 条 mergesort 记 1，则 merged-best_val（两 seed 各 -1）、final-best_val、best_val-best_val 各降分，`merged-best_val` 会跌出并列第一 —— 该口径影响第 6.4 的排名，已在第 一/三 节逐条留痕。

## 八、复核提示

- 198/198 全部人工评过，无未评条目；每条的 `reason` 都指向具体缺陷（可逐条对照 completion 复核）。
- `NL-5SEED-B.jsonl` 字段：`model, column, task, seed, human_score, reason, uncertain, script_passed, verdict`（verdict ∈ agree / false_fail / false_pass），按 seed → model → 题序排列，行数恒定 198。
- 与签名栏 B 的口径已对齐（命名错=1、原地排序=2、无解返回值不扣分、只写注释=0），故 A/B 两个 seed 组可以合并汇总。
