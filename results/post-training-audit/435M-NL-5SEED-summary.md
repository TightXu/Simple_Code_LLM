# 435M SFT NL（自然语言指令）栏 —— 5-seed 人工逐条重审

- 数据：`435M-v2/fine-tuning/eval/nl_eval_generations.jsonl`，330 行 = 6 臂 x 11 题 x 5 seed（[0,42,123,999,2024]），**330/330 全部人工评审**，script_passed 原样记录。
- 口径：2 = 完全正确；1 = 思路/结构对但细节错（基准写反、边界差一、返回 None vs -1、命名不符致符号未定义、漏空表、丢弃等值元素、降序、截断只砍尾部而逻辑正确）；0 = 错误/胡话/仅注释或纯文档串/无关/运行崩溃。
- NL 栏特有：只写注释/只描述思路一律 0。`script_passed` 只记录，**不参与定分**（地面真值 = 人工 0/1/2）。
- 臂名：`EXPB-g9-<底座>-<初始化>`；本文“自身底座”取 `<底座>`（merged / final / best_val）。
- 未使用 GPU；未改动仓库任何现有文件（两个产物均为新建）。

总分布：2 分 **243** 条，1 分 **69** 条，0 分 **18** 条；`uncertain: true` **22** 条。

> **底座对照：无。** 在 `_docs/expB/review/` 下只找到 `BASE-merged/final/best_val.jsonl` 三个底座审核文件，经核其 prompt 模板是**签名栏**（`def fibonacci(n):` 开头），**不是 NL 栏**（NL 模板为 `### Task` + `### Code`）；源文件 `eval_cross_generations.jsonl` 亦无 column 字段且用签名模板。因此 **不存在 435M 的 NL 栏底座审核**，本报告只能给臂内排序，**不给每臂 vs 自身底座的增益结论**（按硬要求不得用签名栏底座分冒充 NL 栏）。

---

## ① 逐 seed 的 6 臂 x 11 题评分表（11 题满分 22）

### seed 0（6 臂合计 114 / 132）

| 臂 | fib | quick | bin | two | merge | pal | prime | rev | vowl | sum | fact | 合计 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| merged-final | 2 | 1 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | **20** |
| merged-best_val | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | **21** |
| final-final | 2 | 1 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 1 | 2 | **19** |
| final-best_val | 2 | 1 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | **20** |
| best_val-final | 2 | 1 | 2 | 2 | 1 | 0 | 0 | 2 | 2 | 2 | 2 | **16** |
| best_val-best_val | 2 | 1 | 2 | 1 | 2 | 0 | 2 | 2 | 2 | 2 | 2 | **18** |
| 逐题合计 | 12 | 6 | 12 | 11 | 8 | 8 | 10 | 12 | 12 | 11 | 12 | 114 |

### seed 42（6 臂合计 109 / 132）

| 臂 | fib | quick | bin | two | merge | pal | prime | rev | vowl | sum | fact | 合计 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| merged-final | 1 | 2 | 2 | 2 | 1 | 1 | 2 | 2 | 2 | 1 | 2 | **18** |
| merged-best_val | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | **21** |
| final-final | 0 | 1 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 1 | 2 | **17** |
| final-best_val | 2 | 1 | 2 | 2 | 1 | 2 | 2 | 0 | 2 | 2 | 2 | **18** |
| best_val-final | 2 | 2 | 0 | 2 | 1 | 0 | 2 | 2 | 2 | 2 | 2 | **17** |
| best_val-best_val | 2 | 1 | 2 | 1 | 2 | 0 | 2 | 2 | 2 | 2 | 2 | **18** |
| 逐题合计 | 9 | 8 | 10 | 11 | 8 | 7 | 12 | 10 | 12 | 10 | 12 | 109 |

### seed 123（6 臂合计 113 / 132）

| 臂 | fib | quick | bin | two | merge | pal | prime | rev | vowl | sum | fact | 合计 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| merged-final | 1 | 0 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | **18** |
| merged-best_val | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | **21** |
| final-final | 1 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 1 | **17** |
| final-best_val | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | **21** |
| best_val-final | 2 | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 0 | 1 | **18** |
| best_val-best_val | 2 | 1 | 2 | 1 | 2 | 0 | 2 | 2 | 2 | 2 | 2 | **18** |
| 逐题合计 | 10 | 6 | 11 | 11 | 9 | 10 | 12 | 12 | 12 | 10 | 10 | 113 |

### seed 999（6 臂合计 109 / 132）

| 臂 | fib | quick | bin | two | merge | pal | prime | rev | vowl | sum | fact | 合计 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| merged-final | 2 | 1 | 2 | 1 | 1 | 2 | 2 | 0 | 0 | 1 | 2 | **14** |
| merged-best_val | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 0 | 2 | 2 | 2 | **19** |
| final-final | 2 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | 2 | 1 | 2 | **18** |
| final-best_val | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | **21** |
| best_val-final | 2 | 1 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 1 | 2 | **19** |
| best_val-best_val | 2 | 1 | 2 | 1 | 2 | 0 | 2 | 2 | 2 | 2 | 2 | **18** |
| 逐题合计 | 12 | 6 | 11 | 10 | 9 | 10 | 12 | 8 | 10 | 9 | 12 | 109 |

### seed 2024（6 臂合计 110 / 132）

| 臂 | fib | quick | bin | two | merge | pal | prime | rev | vowl | sum | fact | 合计 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| merged-final | 1 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | 2 | 1 | 2 | **17** |
| merged-best_val | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 2 | **20** |
| final-final | 1 | 1 | 2 | 2 | 1 | 2 | 2 | 0 | 2 | 1 | 2 | **16** |
| final-best_val | 2 | 1 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | **20** |
| best_val-final | 2 | 2 | 2 | 2 | 1 | 2 | 0 | 0 | 2 | 2 | 2 | **17** |
| best_val-best_val | 2 | 1 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | **20** |
| 逐题合计 | 10 | 7 | 11 | 11 | 8 | 12 | 10 | 8 | 12 | 9 | 12 | 110 |

---

## ② 逐 seed 排序与稳定性

| seed | 排序（总分，满分 22） | 头名 | 全臂极差 |
|---|---|---|---|
| 0 | merged-best_val 21 = merged-final 20 = final-best_val 20 = final-final 19 = best_val-best_val 18 = best_val-final 16 | merged-best_val | 5 |
| 42 | merged-best_val 21 = merged-final 18 = final-best_val 18 = best_val-best_val 18 = final-final 17 = best_val-final 17 | merged-best_val | 4 |
| 123 | merged-best_val 21 = final-best_val 21 = merged-final 18 = best_val-final 18 = best_val-best_val 18 = final-final 17 | merged-best_val | 4 |
| 999 | final-best_val 21 = merged-best_val 19 = best_val-final 19 = final-final 18 = best_val-best_val 18 = merged-final 14 | final-best_val | 7 |
| 2024 | merged-best_val 20 = final-best_val 20 = best_val-best_val 20 = merged-final 17 = best_val-final 17 = final-final 16 | merged-best_val | 4 |

| 臂 | 逐 seed 总分 | 5-seed 合计(满分110) | 逐 seed 标准差 | 极差 | 各 seed 名次 |
|---|---|---|---|---|---|
| merged-best_val | [21, 21, 21, 19, 20] | **102** | 0.80 | 2 | [1, 1, 1, 2, 1] |
| final-best_val | [20, 18, 21, 21, 20] | **100** | 1.10 | 3 | [3, 3, 2, 1, 2] |
| best_val-best_val | [18, 18, 18, 18, 20] | **92** | 0.80 | 2 | [5, 4, 5, 5, 3] |
| merged-final | [20, 18, 18, 14, 17] | **87** | 1.96 | 6 | [2, 2, 3, 6, 4] |
| final-final | [19, 17, 17, 18, 16] | **87** | 1.02 | 3 | [4, 5, 6, 4, 6] |
| best_val-final | [16, 17, 18, 19, 17] | **87** | 1.02 | 3 | [6, 6, 4, 3, 5] |

- **不一致。** 头名在 5 个 seed 里在 merged-best_val 与 final-best_val 之间换手（merged-best_val 单独/并列第一 4 次 + 第二 1 次；final-best_val 单独第一 1 次 + 并列第一 1 次）；全臂极差 4-7 分，与臂间分差同量级，逐 seed 无法细排。
- **最稳的臂：merged-best_val**（21/21/21/19/20，极差 2、sd 0.80），5/5 seed 都排前二，是唯一“每个 seed 都不塌”的臂；best_val-best_val 同 sd 0.80 但整体低一档。
- **最不稳的臂：merged-final**（20/18/18/14/17，极差 6、sd 1.96），seed 999 掉到 14（count_vowels/reverse_string 各 0，two_sum/mergesort/sum_list 各 1）。
- 次不稳：final-final（19/17/17/18/16）与 final-best_val（20/18/21/21/20），极差均 3。

---

## ③ 每臂 vs 自身底座

**结论：无底座对照，本项不可计算。** 理由见文首灰框——`_docs/expB/review/` 下三个 BASE 文件（及其 summary/备份）均为**签名栏**底座审核，NL 栏从未跑过底座评测。因此：
- 不给任何“逐 seed 增益 / 符号是否翻转”的数字（不得用签名栏底座分冒充）。
- 若后续补跑 435M NL 栏底座（3 个 checkpoint x 11 题 x 5 seed），可直接套本报告 ①/② 同表相减得到每臂增益。

---

## ④ 逐题 seed 波动排名（6 臂 x 5 seed = 30 个单元）

| 排名(最不稳->最稳) | 题 | 均值 | 标准差 | 单臂内最大 seed 摆幅 | Σ|相邻 seed 差| | 0 分数 |
|---|---|---|---|---|---|---|
| 1 | is_palindrome | 1.57 | 0.80 | 2 | 6 | 6 |
| 2 | reverse_string | 1.67 | 0.75 | 2 | 16 | 5 |
| 3 | sum_list | 1.63 | 0.55 | 2 | 10 | 1 |
| 4 | is_prime | 1.87 | 0.50 | 2 | 4 | 2 |
| 5 | fibonacci | 1.77 | 0.50 | 2 | 8 | 1 |
| 6 | mergesort | 1.40 | 0.49 | 1 | 2 | 0 |
| 7 | binary_search | 1.83 | 0.45 | 2 | 7 | 1 |
| 8 | two_sum | 1.80 | 0.40 | 1 | 2 | 0 |
| 9 | quicksort | 1.10 | 0.40 | 2 | 7 | 1 |
| 10 | count_vowels | 1.93 | 0.36 | 2 | 4 | 1 |
| 11 | factorial | 1.93 | 0.25 | 1 | 4 | 0 |

- **最不稳：is_palindrome**（6 个 0，全来自 best_val 两臂的“len 相等恒真”“与回文无关的字母表判断”写法，跨 seed 在 0 与 2 之间跳）。
- 其次：**reverse_string**（5 个 0：`char[::-1]` 拼回原串、`' '.join(s.split()[::-1])` 反转词序、`' '.join(result)` 不反转、手动拼装错末位）。
- **sum_list**（1 个 0、多个 1：“只累加正数”“空表返回 None”“仅字符串输入才求和”）。
- **is_prime / binary_search / fibonacci** 各有 1-2 个 0（sieve 判据胡写、旋转数组式二分误返 -1、未定义 fib_numbers / 基准移一位）。
- **最稳：factorial（sd 0.25）> count_vowels（0.36）> two_sum（0.40）**；factorial 30 单元仅 2 处 1（n<=0 即 raise 的边界），无 0。
- **quicksort 均值最低（1.10）但波动小**：30 条中 25 条 1（降序或丢弃等值元素）、3 条 2（中位三路）、1 条 2、1 条 0——“稳定地只拿 1”，非抖动。

---

## ⑤ 脚本 vs 人工分歧

| script_passed | 人工 2 | 人工 1 | 人工 0 | 行计 |
|---|---|---|---|---|
| True | 242 | 5 | 0 | 247 |
| False | 1 | 64 | 18 | 83 |
| 列计 | 243 | 69 | 18 | 330 |

- 脚本通过 247/330，人工满分 243/330——总数接近，但脚本仍会把“结构对细节错”判死。
- **假失败（脚本 False、人工 = 2）仅 1 条**：
  - best_val-final/factorial/0：阶乘正确，factorial(0)=1
- **假通过（脚本 True、人工 < 2）5 条**（全部为 1 分）：
  - merged-final/two_sum/999：只在 nums[j:] 查补数却固定返回 [i,i+1]，j 非命中位置（[5,1,2,3],7 误返 [0,1]）
  - merged-final/binary_search/2024：比较用 arr[left] 而非 arr[mid]，binary_search([1,3,5,7,9],1) 误返 -1
  - merged-best_val/quicksort/42：less/greater 命名颠倒，greater 组前置，输出降序
  - final-final/binary_search/123：套用旋转数组判断，binary_search([1,3,5,7,9],9) 误返 -1
  - final-final/binary_search/999：旋转数组式判断，target=9 误返 -1
- 与签名栏相比，本 NL 栏**假失败极少（1 条）**：NL 栏补全多是可运行实现，脚本直接判活，不再像签名栏那样被“注释打头/缩进/未导入”清零。真正与人工的错位集中在**“形状对、过了给定断言但语义错”**的 5 条假通过（旋转数组式二分、返回固定 [i,i+1] 的两数之和、降序快排）。
- 结论不变：**脚本通过率不可作选型依据**（既漏 5 条假通过、也漏 1 条假失败），地面真值只取人工 0/1/2。

---

## ⑥ 回答五个必答问题

### 6.1 逐 seed 的 6 臂总分排序是否跨 seed 一致？哪个臂最稳？
- **不一致。** 见 ② 表：头名在 merged-best_val 与 final-best_val 之间换手，全臂极差 4-7 分。
- **最稳的臂 = merged-best_val**（并列第一 4 次 + 第二 1 次，极差 2、sd 0.80）。
### 6.2 每个臂 vs 自身底座：逐 seed 增益？
- **无底座对照，无法计算**（base_comparison_available=false；`_docs/expB/review/` 下只有签名栏底座，NL 栏从未跑底座）。按硬要求不冒充。
### 6.3 三底座排序（merged > final > best_val）在 SFT 后是否仍成立？
- 无**真底座**对照，只能按臂所属底座分组看“该底座系最佳臂”：

| seed | merged 系最佳 | final 系最佳 | best_val 系最佳 | 是否 merged>final>best_val |
|---|---|---|---|---|
| 0 | merged-best_val 21 | final-best_val 20 | best_val-best_val 18 | 成立 |
| 42 | merged-best_val 21 | final-best_val 18 | best_val-best_val 18 | 成立 |
| 123 | merged-best_val 21 | final-best_val 21 | best_val-final 18 | 成立 |
| 999 | merged-best_val 19 | final-best_val 21 | best_val-final 19 | 不成立 |
| 2024 | merged-best_val 20 | final-best_val 20 | best_val-best_val 20 | 不成立 |

- **逐 seed 仅 seed 0 成立**（merged 21 > final 20 > best_val 18）；其余 4 个 seed 均不成立（seed42 merged 21 > final=best_val 18；seed123 merged=final 21 > best_val 18；seed999 final 21 > merged=best_val 19；seed2024 三者并列 20）。
- **5-seed 合计成立**：merged 系 102（merged-best_val）> final 系 100（final-best_val）> best_val 系 92（best_val-best_val）——次序对、间距薄，逐 seed 会翻转，合计才稳。
### 6.4 逐题 seed 波动排名
- 最不稳：is_palindrome > reverse_string > sum_list > is_prime > fibonacci > binary_search > quicksort > two_sum > count_vowels > factorial（见 ④）。
- 最稳：**factorial > count_vowels > two_sum**；波动集中在“冒充回文（len 恒等）/ 反转不成反转 / 只加正数 / 空表返 None”几类写法，与模型强弱无关。
### 6.5 5-seed 合计总排序与最佳单臂

| 名次 | 臂 | 5-seed 合计(/110) |
|---|---|---|
| 1 | merged-best_val | **102** |
| 2 | final-best_val | **100** |
| 3 | best_val-best_val | **92** |
| 4 | merged-final | **87** |
| 5 | final-final | **87** |
| 6 | best_val-final | **87** |

- **最佳单臂 = merged-best_val（102/110）**，也是全栏唯一 5/5 seed 都居前二的臂。
- 但要按规矩说明：**merged-best_val(102) 与 final-best_val(100) 之间“分不出来”**——差 2 分/110，落在逐 seed 摆幅（2-3 分）内。
- 明显低一档是 **best_val-best_val 92**；**merged-final = final-final = best_val-final = 87** 三条并列垫底（与头部差 15 分/110，可稳定分出；三者互相“分不出来”）。
- 故可写“**merged-best_val 稳居第一集团且最稳，final-best_val 紧随**”，但不能写 merged-best_val 显著优于 final-best_val。

---

## 附：不确定条目（uncertain=true，共 22 行）

全部落在 0/1 或 1/2 边界，reason 已写明事实：
- merged-final/is_palindrome/42（人工 1）：双指针不推进下标，回退辅助函数对字符串原地赋值会报错，无法真正判定
- merged-final/quicksort/123（人工 0）：对测试输入 left_half.remove(pivot) 抛 ValueError（pivot 不在左半），运行崩溃
- merged-final/two_sum/999（人工 1）：只在 nums[j:] 查补数却固定返回 [i,i+1]，j 非命中位置（[5,1,2,3],7 误返 [0,1]）
- merged-final/quicksort/2024（人工 1）：用未导入的 random.randint，且左右半互换后递归，运行崩溃/输出无序
- merged-final/binary_search/2024（人工 1）：比较用 arr[left] 而非 arr[mid]，binary_search([1,3,5,7,9],1) 误返 -1
- final-final/fibonacci/42（人工 0）：用未定义的列表 fib_numbers（NameError），且逻辑并非求第 n 项
- final-final/quicksort/123（人工 1）：left_half 未按 pivot 过滤，输出未排序
- final-final/binary_search/123（人工 1）：套用旋转数组判断，binary_search([1,3,5,7,9],9) 误返 -1
- final-final/factorial/123（人工 1）：n<=0 即 raise，factorial(0) 抛异常（应为 1）
- final-final/binary_search/999（人工 1）：旋转数组式判断，target=9 误返 -1
- final-final/quicksort/2024（人工 1）：左右半不按 pivot 过滤且 pivot 丢失，merge 又被截断，输出错误
- final-best_val/mergesort/0（人工 1）：调用未定义的 mergeSort（与 mergesort 大小写不符，符号未定义）
- final-best_val/mergesort/42（人工 1）：调用未定义的 mergeSort（符号未定义）
- final-best_val/mergesort/2024（人工 1）：调用未定义的 mergeSort（符号未定义）
- best_val-final/quicksort/0（人工 1）：pivot 取中位但左右半未按 pivot 过滤、pivot 被复制进右半，输出未排序
- best_val-final/is_palindrome/0（人工 0）：整段为无意义的奇偶分支与字符双写且末尾截断，对任何输入都不能正确判定
- best_val-final/is_prime/0（人工 0）：以筛法所得素数个数是否等于 n 为判据，is_prime(2)=F/(17)=F，全错
- best_val-final/quicksort/42（人工 2）：中位 pivot 三路快排，正确
- best_val-final/binary_search/42（人工 0）：命中后 mid 不等于 0 且未越界时不推进左右界，进入死循环；尾部逻辑无意义
- best_val-final/quicksort/999（人工 1）：左右半未过滤、以 mid_index*[pivot] 拼接，输出错误（未排序且元素错乱）
- best_val-final/is_prime/2024（人工 0）：n==2 分支写成 return False，is_prime(2) 误返 False
- best_val-best_val/is_palindrome/0（人工 0）：all(char in 小写字母表) 与回文无关，'hello' 也返回 True

## 复核提示
- 330/330 全部人工评过，无未评条目；每条 reason 指向具体缺陷，可逐条对照 completion 复核。
- `NL-5SEED.jsonl` 字段：`model, column, type, task, seed, human_score, reason, uncertain, script_passed`，按 arm -> seed -> 题序排列，行数恒定 330。