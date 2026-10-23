# 435M SFT 签名栏 5-seed 人工重审（SIG-5SEED）

- 数据：`435M-v2/fine-tuning/eval/eval_sft_generations.jsonl`，330 行 = 6 臂 x 11 题 x 5 seed（[0,42,123,999,2024]），**330/330 全部逐条人工评审**，script_passed 原样记录。
- 口径：与本目录底座审核同尺子（2 = 完全正确；1 = 思路/结构对但细节错：基准情形写反、边界差一、接口漂移、丢弃等值元素、原地排序不返回等；0 = 错误/胡话/与任务无关/仅注释/运行崩溃）。
- 臂名约定（推断）：`EXPB-g9-<底座>-<初始化>`. 本文"自身底座"取 `<底座>`：`merged-final`/`merged-best_val` 对 BASE-merged，依此类推。
- 评审方式：先按 (题, 补全文本) 归并出 150 个唯一补全，逐个静态审读 + 实际执行观察行为（仅用于判断，不采用脚本通过率作为地面真值），再回填 330 行；每条均写 reason，分批落盘。

总分布：2 分 183 条，1 分 130 条，0 分 17 条；uncertain 标记 12 条。

## ① 逐 seed 的 6 臂 x 11 题评分表

题序：fib QUICK bin TWO merge PAL prime rev VOWL SUM fact

### seed 0

| 臂 | fib | quick | bin | two | merge | pal | prime | rev | vowl | sum | fact | 合计 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| merged-final | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 2 | 2 | 1 | 2 | **18** |
| merged-best_val | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | **15** |
| final-final | 2 | 1 | 1 | 0 | 2 | 0 | 2 | 2 | 2 | 2 | 2 | **16** |
| final-best_val | 2 | 1 | 1 | 2 | 1 | 1 | 2 | 0 | 2 | 2 | 2 | **16** |
| best_val-final | 1 | 2 | 1 | 1 | 2 | 1 | 2 | 0 | 2 | 2 | 2 | **16** |
| best_val-best_val | 1 | 1 | 1 | 1 | 1 | 0 | 2 | 0 | 2 | 2 | 2 | **13** |

底座同题同 seed（对照）：BASE-merged 10 分; BASE-final 6 分; BASE-best_val 5 分

### seed 42

| 臂 | fib | quick | bin | two | merge | pal | prime | rev | vowl | sum | fact | 合计 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| merged-final | 2 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 2 | 2 | **17** |
| merged-best_val | 0 | 1 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | **16** |
| final-final | 2 | 1 | 2 | 2 | 1 | 1 | 2 | 1 | 2 | 1 | 2 | **17** |
| final-best_val | 2 | 1 | 2 | 1 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | **17** |
| best_val-final | 1 | 2 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | **18** |
| best_val-best_val | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 2 | 2 | **16** |

底座同题同 seed（对照）：BASE-merged 12 分; BASE-final 7 分; BASE-best_val 2 分

### seed 123

| 臂 | fib | quick | bin | two | merge | pal | prime | rev | vowl | sum | fact | 合计 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| merged-final | 2 | 1 | 1 | 0 | 2 | 0 | 2 | 0 | 2 | 2 | 2 | **14** |
| merged-best_val | 2 | 1 | 1 | 1 | 1 | 2 | 2 | 1 | 2 | 2 | 2 | **17** |
| final-final | 2 | 2 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | **19** |
| final-best_val | 2 | 1 | 2 | 1 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | **17** |
| best_val-final | 1 | 2 | 1 | 0 | 2 | 1 | 2 | 0 | 2 | 2 | 2 | **15** |
| best_val-best_val | 1 | 2 | 1 | 1 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | **18** |

底座同题同 seed（对照）：BASE-merged 14 分; BASE-final 10 分; BASE-best_val 5 分

### seed 999

| 臂 | fib | quick | bin | two | merge | pal | prime | rev | vowl | sum | fact | 合计 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| merged-final | 1 | 1 | 2 | 1 | 2 | 0 | 2 | 2 | 2 | 1 | 2 | **16** |
| merged-best_val | 2 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 2 | 2 | **17** |
| final-final | 2 | 1 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | **18** |
| final-best_val | 1 | 1 | 1 | 2 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | **16** |
| best_val-final | 1 | 2 | 1 | 1 | 2 | 1 | 2 | 0 | 2 | 2 | 2 | **16** |
| best_val-best_val | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | **17** |

底座同题同 seed（对照）：BASE-merged 10 分; BASE-final 9 分; BASE-best_val 3 分

### seed 2024

| 臂 | fib | quick | bin | two | merge | pal | prime | rev | vowl | sum | fact | 合计 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| merged-final | 2 | 1 | 2 | 1 | 2 | 0 | 2 | 2 | 2 | 2 | 2 | **18** |
| merged-best_val | 0 | 1 | 2 | 1 | 1 | 1 | 2 | 2 | 2 | 2 | 2 | **16** |
| final-final | 2 | 1 | 2 | 2 | 2 | 1 | 2 | 1 | 2 | 1 | 2 | **18** |
| final-best_val | 2 | 1 | 1 | 2 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | **17** |
| best_val-final | 1 | 2 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | 1 | 2 | **17** |
| best_val-best_val | 1 | 1 | 1 | 1 | 1 | 0 | 2 | 2 | 2 | 2 | 2 | **15** |

底座同题同 seed（对照）：BASE-merged 11 分; BASE-final 8 分; BASE-best_val 3 分

## ② 逐 seed 臂排序 + 跨 seed 稳定性

| seed | 排序（总分，11 题满分 22） |
|---|---|
| 0 | merged-final 18 > final-final 16 > final-best_val 16 > best_val-final 16 > merged-best_val 15 > best_val-best_val 13 |
| 42 | best_val-final 18 > merged-final 17 > final-final 17 > final-best_val 17 > merged-best_val 16 > best_val-best_val 16 |
| 123 | final-final 19 > best_val-best_val 18 > merged-best_val 17 > final-best_val 17 > best_val-final 15 > merged-final 14 |
| 999 | final-final 18 > merged-best_val 17 > best_val-best_val 17 > merged-final 16 > final-best_val 16 > best_val-final 16 |
| 2024 | merged-final 18 > final-final 18 > final-best_val 17 > best_val-final 17 > merged-best_val 16 > best_val-best_val 15 |

| 臂 | 逐 seed 总分 | 5-seed 合计 | 逐 seed 标准差 | 极差 | 排名（各 seed） |
|---|---|---|---|---|---|
| merged-final | [18, 17, 14, 16, 18] | **83** | 1.50 | 4 | [1, 2, 6, 4, 1] |
| merged-best_val | [15, 16, 17, 17, 16] | **81** | 0.75 | 2 | [5, 5, 3, 2, 5] |
| final-final | [16, 17, 19, 18, 18] | **88** | 1.02 | 3 | [2, 2, 1, 1, 1] |
| final-best_val | [16, 17, 17, 16, 17] | **83** | 0.49 | 1 | [2, 2, 3, 4, 3] |
| best_val-final | [16, 18, 15, 16, 17] | **82** | 1.02 | 3 | [2, 1, 5, 4, 3] |
| best_val-best_val | [13, 16, 18, 17, 15] | **79** | 1.72 | 5 | [6, 5, 2, 2, 6] |

**跨 seed 一致性**：没有任何一个臂在 5/5 seed 里保持同一名次，逐 seed 排序不一致（seed0 头名 merged-final，seed42 头名 best_val-final，seed123/999 头名 final-final，seed2024 merged-final 与 final-final 并列头名）。
**最稳的臂**：final-best_val（16/17/17/16/17，极差 1，sd 0.49）在"分数稳定"意义上最稳，但从未进过头名；merged-best_val 次之（极差 2）。
**最不稳的臂**：best_val-best_val（13/16/18/17/15，极差 5，sd 1.72）；merged-final 也偏抖（14~18，sd 1.50）。
**最常居首的臂**：final-final（seed123、999 单独第一，seed2024 并列第一，seed0/42 并列第二；且从未低于 16 分）。

## ③ 每个臂 vs 自身底座（同题同 seed 总分差）

| 臂 | 自身底座 | 底座逐 seed 总分 | 臂逐 seed 总分 | 逐 seed 增益 | 5/5 为正？ |
|---|---|---|---|---|---|
| merged-final | BASE-merged | [10, 12, 14, 10, 11] | [18, 17, 14, 16, 18] | +8/+5/+0/+6/+7 (合计 +26) | 4/5 为正，seed123 为 0 |
| merged-best_val | BASE-merged | [10, 12, 14, 10, 11] | [15, 16, 17, 17, 16] | +5/+4/+3/+7/+5 (合计 +24) | 是 |
| final-final | BASE-final | [6, 7, 10, 9, 8] | [16, 17, 19, 18, 18] | +10/+10/+9/+9/+10 (合计 +48) | 是 |
| final-best_val | BASE-final | [6, 7, 10, 9, 8] | [16, 17, 17, 16, 17] | +10/+10/+7/+7/+9 (合计 +43) | 是 |
| best_val-final | BASE-best_val | [5, 2, 5, 3, 3] | [16, 18, 15, 16, 17] | +11/+16/+10/+13/+14 (合计 +64) | 是 |
| best_val-best_val | BASE-best_val | [5, 2, 5, 3, 3] | [13, 16, 18, 17, 15] | +8/+14/+13/+14/+12 (合计 +61) | 是 |

**结论：符号不翻转。** 6 个臂对自身底座，5/5 seed 全部非负，其中 5 个臂 5/5 严格为正；只有 merged-final 在 seed123 增益为 0（该 seed 它 14 分 = 底座 merged 的 14 分）。这与 1B 签名栏的"符号翻转"不同，接近 1B 指令栏的稳定净增益。
增益幅度（5-seed 合计，绝对值不可跨底座直接比，但方向一致）：best_val-final +64 > best_val-best_val +61 > final-final +48 > final-best_val +43 > merged-final +26 > merged-best_val +24 —— 底座起点越低，SFT 抬升越大。

## ④ 三底座排序在 SFT 之后是否仍为 merged > final > best_val

**被打破。** 底座（人工已核）5/5 seed 稳定 merged > final > best_val；SFT 之后：

| seed | 各底座系最佳臂得分（merged / final / best_val） | 是否仍 merged>final>best_val |
|---|---|---|
| 0 | 18 / 16 / 16 | 不成立 |
| 42 | 17 / 17 / 18 | 不成立 |
| 123 | 17 / 19 / 18 | 不成立 |
| 999 | 17 / 18 / 17 | 不成立 |
| 2024 | 18 / 18 / 17 | 不成立 |

5-seed 合计（各底座系最佳臂）：merged 83（merged-final）、final 88（final-final）、best_val 82（best_val-final）→ final > merged > best_val。

打破点与致因（按 seed 具体到题）：
- seed 42：merged-final 在 fibonacci 上给出 n<=0 返回字符串、数列位移一位的写法（人工 0，而 final 系同题拿 2），merged 系最佳 17 分被 final 系 17 分打平、被 best_val 系 18 分反超。
- seed 123（决定性）：merged-final 掉到 14 分——两数之和写了三数之和式乱写、回文和字符串反转都写成 Caesar 位移（三题各 0 分），而 final-final 19 分、best_val-best_val 18 分。此 seed 是 merged 系最差 seed。
- seed 999 / 2024：final 系在 mergesort、quicksort 上更稳（final-final 两条都 18），merged 系与 best_val 系同题多为 16~17，"final 反超 merged" 依然成立。

## ⑤ 逐题 seed 波动排名（30 个 (臂,seed) 单元）

| 排名(最不稳->最稳) | 题 | 均值 | 标准差 | 分域 | 0 分数 | 单臂内最大 seed 摆幅 |
|---|---|---|---|---|---|---|
| 1 | reverse_string | 1.33 | 0.79 | 0-2 | 6 | 2 |
| 2 | fibonacci | 1.43 | 0.62 | 0-2 | 2 | 2 |
| 3 | is_palindrome | 0.90 | 0.54 | 0-2 | 6 | 2 |
| 4 | two_sum | 1.07 | 0.51 | 0-2 | 3 | 2 |
| 5 | mergesort | 1.47 | 0.50 | 1-2 | 0 | 1 |
| 6 | binary_search | 1.27 | 0.44 | 1-2 | 0 | 1 |
| 7 | quicksort | 1.23 | 0.42 | 1-2 | 0 | 1 |
| 8 | sum_list | 1.83 | 0.37 | 1-2 | 0 | 1 |
| 9 | is_prime | 2.00 | 0.00 | 2-2 | 0 | 0 |
| 10 | count_vowels | 2.00 | 0.00 | 2-2 | 0 | 0 |
| 11 | factorial | 2.00 | 0.00 | 2-2 | 0 | 0 |

最不稳：fibonacci、two_sum、is_palindrome、reverse_string（单臂内跨 seed 摆幅可达 2 分，且 0 分集中）。
最稳：is_prime、count_vowels、factorial（5x6 单元全 2 分，零波动）；sum_list 次稳（仅"只累加正整数"写法扣到 1）。

## ⑥ 脚本判定 vs 人工判定

| 臂 | 脚本通过 | 人工总分 | 人工 2 分条数 |
|---|---|---|---|
| merged-final | 29/55 | 83/110 | 33 |
| merged-best_val | 28/55 | 81/110 | 28 |
| final-final | 31/55 | 88/110 | 35 |
| final-best_val | 29/55 | 83/110 | 29 |
| best_val-final | 26/55 | 82/110 | 31 |
| best_val-best_val | 27/55 | 79/110 | 27 |

分歧共 13 行（全部是同向：脚本判失败、人工判 2 分，即脚本漏判），没有"脚本通过但人工 <2"的行。
13 行全部是 mergesort 的"返回新列表"递归实现（merged-final 4 行、final-final 4 行、best_val-final 5 行）：这些补全显式返回排好序的新列表，人工核对结果为正确（2 分），而脚本判 False——与底座审核中"脚本通过率不可信"的结论一致，只是方向相反（这次是脚本偏严）。
其余 317 行脚本与人工一致（其中脚本判通过、人工给 1 分的正是口径里的接口漂移/丢等值元素/返回 bool 之类）。

## ⑦ 回答五个必答问题

1. **逐 seed 臂排序是否一致**：不一致。6 臂在任何 seed 内的最大分差 5 分（11 题、满分 22），摆幅与分差同量级；头名在 4 个 seed 里换了 3 个臂。最稳的臂：分数意义上 final-best_val（极差 1）；最常居首的臂：final-final。
2. **每臂 vs 自身底座**：全部为稳定净增益，无符号翻转。逐 seed 增益 merged-final +8/+5/0/+6/+7、merged-best_val +5/+4/+3/+7/+5、final-final +10/+10/+9/+9/+10、final-best_val +10/+10/+7/+7/+9、best_val-final +11/+16/+10/+13/+14、best_val-best_val +8/+14/+13/+14/+12。只有 merged-final@seed123 为 0，其余 29/30 个 (臂,seed) 严格为正；底座越弱（best_val 系）增益越大。
3. **三底座排序**：底座 5/5 稳定 merged > final > best_val；**SFT 后被打破**，5/5 seed 均不成立：final 系（final-final 88）反超 merged 系（83），best_val 系（82）追平 merged 系。决定性 seed 是 123（merged-final 14 分 vs final-final 19），其次 seed 42；致因题：fibonacci、two_sum、is_palindrome、reverse_string（merged-final 在此命题上出现"字符串返回值/ Caesar 位移/三数之和"类退化写法）。
4. **逐题波动排名**：最不稳 fibonacci > two_sum > is_palindrome > reverse_string > quicksort/binary_search/mergesort/sum_list > is_prime = count_vowels = factorial（后三题 30 个单元全 2 分，零波动）。
5. **5 seed 合计总排序与最佳单臂**：final-final 88 > merged-final 83 = final-best_val 83 > best_val-final 82 > merged-best_val 81 > best_val-best_val 79。最佳单臂 **final-final**（88/110，2 分 35 条）。但要按规矩说明：**第一集团（final-final 88、merged-final 83、final-best_val 83、best_val-final 82、merged-best_val 81）互相"分不出来"**——逐 seed 标准差 0.49~1.50，臂间均值差 1~1.8 分/seed 落在自身 seed 摆幅（1~5 分）之内；只有 best_val-best_val（79，均值 15.80，sd 1.72）与 final-final（88，均值 17.60，sd 1.02）相差 1.8 分/seed，约 2.0 个合并标准误（t≈2.0）可勉强分出高下。因此可写"final-final 最常居首、最稳地保持高位"，但不能写"final-final 显著优于 merged-final"。

## 附：不确定条目（uncertain=true，共 12 行）

- merged-final fibonacci seed=999 script=False 人工=1：返回前 n 项数列而非常规第 n 项：接口漂移，算法本身正确
- merged-final two_sum seed=0 script=False 人工=1：哈希集合骨架正确但 num 只在命中时入集合，实测恒返回空列表
- merged-final reverse_string seed=123 script=False 人工=0：做 +26 的 Caesar 位移等于恒等变换，与反转无关
- final-final is_palindrome seed=999 script=False 人工=1：只比较首字符与自身，对 'abca' 也返回 True
- best_val-final is_palindrome seed=0 script=False 人工=1：reversed_s 按正序追加等于原串，多字符恒 True、单字符/空串又判 False
- best_val-final is_palindrome seed=42 script=False 人工=1：reversed_s 按正序追加等于原串，多字符恒 True、单字符/空串又判 False
- best_val-final is_palindrome seed=2024 script=False 人工=1：reversed_s 按正序追加等于原串，多字符恒 True、单字符/空串判 False
- best_val-final reverse_string seed=999 script=False 人工=0：+26 恒等位移并丢弃非字母，与反转无关
- best_val-best_val fibonacci seed=42 script=False 人工=1：数列算法正确但 n<=0 返回字符串、n=1/2 返回 int 而 n>=3 返回 list，返回类型不一致
- best_val-best_val fibonacci seed=123 script=False 人工=1：数列算法正确但返回类型随 n 变化（字符串/int/list）
- best_val-best_val fibonacci seed=999 script=False 人工=1：数列算法正确但 n<=0 返回字符串、n=1/2 返回 int 而 n>=3 返回 list，返回类型不一致
- best_val-best_val fibonacci seed=2024 script=False 人工=1：数列算法正确但返回类型随 n 变化（字符串/int/list）

另有若干结构性判断已按底座同尺子统一定档（不算 uncertain），主要三条：
- 返回 bool 的二分（该返回索引）一律 1 分（接口漂移），与本项目"return None vs 底座 return -1 记 1"一致。
- two_sum 返回"数值对列表"而非索引一律 1 分；返回 [complement, index] 亦为 1 分（首元素是值不是索引）。
- 原地归并排序不 return（返回 None）记 1 分，与底座 quicksort 原地排序返回 None 记 1 分同尺子。